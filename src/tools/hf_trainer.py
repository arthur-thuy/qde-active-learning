"""Module for HuggingFace trainer."""

import logging
import os
from copy import deepcopy
from functools import partial
from typing import Any, Union

import baal
import numpy as np
import structlog
from baal.active import ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import MCDropoutModule
from baal.transformers_trainer_wrapper import BaalTransformersTrainer
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    IntervalStrategy,
    TrainingArguments,
)
from yacs.config import CfgNode

from heuristic.build import build_heuristic
from model.build import build_hf_model
from tools.constants import TF_TEXT
from tools.metrics import compute_metrics_classification, compute_metrics_regression

# set up logger
structlog.stdlib.recreate_defaults(log_level=logging.WARNING)
logger = structlog.get_logger("qdet")
# NOTE: log attributes finds baal logger
baal.transformers_trainer_wrapper.log.setLevel(logging.WARNING)


ROUND_FLOAT = 4


def huggingface_al_procedure(
    cfg: CfgNode,
    datasets,
    init_active_idx: NDArray,
    device: Any,
    dry_run: bool,
) -> Union[dict[str, Union[int, float, list[Any]]], tuple[dict, list, dict]]:
    """Run HF active learning procedure for single config, single run.

    Parameters
    ----------
    cfg : CfgNode
        Config object
    datasets : _type_
        HF datasets
    init_active_idx : NDArray
        Initial active indices
    device : Any
        Compute device (GPU or CPU)
    dry_run : bool
        Run only one AL epoch

    Returns
    -------
    Union[dict[str, Union[int, float, list[Any]]], tuple[dict, list, dict]]
        Output metrics
    """
    # create HF model + tokenizer
    model, tokenizer = build_hf_model(cfg.MODEL)
    model = model.to(device)

    if cfg.TRAIN.FREEZE_BASE:
        logger.info("Freezing base model")
        # freeze base model
        for param in model.base_model.parameters():
            param.requires_grad = False
    else:
        logger.info("Fine-tuning base model")

    # tokenize dataset
    def _preprocess_function(examples, tokenizer, max_length):
        return tokenizer(
            examples[TF_TEXT], truncation=True, max_length=max_length, padding=True
        )

    tokenized_dataset = datasets.map(
        _preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.MODEL.MAX_LENGTH},
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # set up ALDataset + labeling
    al_dataset = ActiveLearningDataset(dataset=tokenized_dataset["train"])
    if cfg.AL.FULL_DATA:
        al_dataset.label_randomly(al_dataset.n_unlabelled.item())
    else:
        al_dataset.label(init_active_idx)

    # store weights to use for resetting model (need deepcopy!!!)
    init_weights = deepcopy(model.state_dict())

    # create trainer and loop objects
    logging_eval_save_steps = cfg.EVAL.LOGGING_STEPS

    training_args = TrainingArguments(
        # logging
        output_dir=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
        overwrite_output_dir=True,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=logging_eval_save_steps,  # train loss calculated per logging_steps # noqa
        save_strategy=IntervalStrategy.STEPS,
        save_steps=logging_eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=(
            "eval_discrete_rmse" if cfg.LOADER.REGRESSION else "eval_accuracy"
        ),
        greater_is_better=False if cfg.LOADER.REGRESSION else True,
        save_total_limit=1,  # keep only the best model
        # convergence
        num_train_epochs=cfg.TRAIN.EPOCHS,
        max_steps=2 if dry_run else cfg.TRAIN.MAX_STEPS,
        learning_rate=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        adam_epsilon=cfg.TRAIN.ADAM_EPSILON,
        per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
        warmup_ratio=cfg.TRAIN.WARMUP_RATIO,
        bf16=cfg.MODEL.BF16,
        tf32=True,  # NOTE: already set in main script
        # evaluation
        per_device_eval_batch_size=cfg.EVAL.BATCH_SIZE,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=logging_eval_save_steps,  # eval loss calculated per eval_steps
    )

    trainer = BaalTransformersTrainer(
        model=model,
        args=training_args,
        train_dataset=al_dataset,
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=(
            partial(compute_metrics_regression, dataset_name=cfg.LOADER.NAME)
            if cfg.LOADER.REGRESSION
            else partial(
                compute_metrics_classification, num_classes=cfg.MODEL.NUM_CLASSES
            )
        ),
        # early stopping configs (patience is number of evaluation calls)
        callbacks=(
            [
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.TRAIN.PATIENCE,
                    early_stopping_threshold=1e-4,
                )
            ]
            if cfg.TRAIN.EARLY_STOPPING
            else None
        ),
        replicate_in_memory=False,  # NOTE: set to False to avoid memory issues
    )

    # create AL loop
    active_loop = ActiveLearningLoop(
        dataset=al_dataset,
        get_probabilities=trainer.predict_on_dataset,
        heuristic=build_heuristic(cfg.AL),
        query_size=cfg.AL.QUERY_SIZE,
        iterations=cfg.EVAL.MC_SAMPLES,
        max_sample=cfg.AL.SUBSET_POOL if not dry_run else 1000,
    )

    labelling_progress = al_dataset._labelled.copy().astype(np.uint16)
    metrics = {}
    for epoch in tqdm(range(cfg.AL.EPOCHS)):
        print("\n", "-" * 10, f"Epoch {epoch}", "-" * 10)
        ##### TRAINING #####

        logger.info("Train - start", epochs=cfg.TRAIN.EPOCHS, dataset=len(al_dataset))
        print(
            f"=> Training {'with' if cfg.TRAIN.EARLY_STOPPING else 'without'} "
            "early stopping"
        )
        train_metrics = trainer.train()

        logger.info(  # TODO: find way to avoid second TQDM line
            "Train - end",
            train_loss=round(train_metrics.training_loss, ROUND_FLOAT),
            epochs_trained=train_metrics.metrics["epoch"],
        )
        metrics[len(al_dataset)] = {
            **dict(train_metrics.metrics),
            "dataset_size": len(al_dataset),
            "epochs_trained": train_metrics.metrics[
                "epoch"
            ],  # NOTE: true amount, corrected for max_steps
            "convergence": trainer.state.log_history,
        }

        ##### EVALUATION #####
        logger.info(
            "Evaluate - start",
            test_size=len(tokenized_dataset["test"]),
            test_samples=1,
        )
        # evaluate
        eval_results = trainer.predict(
            tokenized_dataset["test"], metric_key_prefix="test"
        )
        eval_metrics = eval_results.metrics
        eval_pred_label = (
            eval_results.predictions.squeeze(),
            eval_results.label_ids,
        )

        print("Test metrics: ", eval_metrics)
        if cfg.LOADER.REGRESSION:
            logger.info(
                "Evaluate - end",
                test_loss=round(eval_metrics["test_loss"], ROUND_FLOAT),
                test_rmse=round(eval_metrics["test_rmse"], ROUND_FLOAT),
                test_discrete_rmse=round(
                    eval_metrics["test_discrete_rmse"], ROUND_FLOAT
                ),
            )
        else:
            logger.info(
                "Evaluate - end",
                test_loss=round(eval_metrics["test_loss"], ROUND_FLOAT),
                test_acc=round(eval_metrics["test_accuracy"], ROUND_FLOAT),
            )
        metrics[len(al_dataset)].update(
            {
                **eval_metrics,
                "test_pred_label": eval_pred_label,
            }
        )

        ##### ACQUISITION #####

        # do not compute acquisitions if last epoch
        if (epoch == (cfg.AL.EPOCHS - 1) and not dry_run) or cfg.AL.FULL_DATA:
            logger.info("Acquisition skipped")
            break

        logger.info(
            "Acquire - start",
            pool_size=(
                al_dataset.n_unlabelled.item()
                if cfg.AL.SUBSET_POOL == -1
                else cfg.AL.SUBSET_POOL
            ),
            pool_samples=cfg.EVAL.MC_SAMPLES,
        )
        # label the most uncertain samples according to our heuristic

        if cfg.MODEL.MC_DROPOUT:
            print("=> using MC Dropout")
            with MCDropoutModule(model) as _mc_dropout_model:  # noqa: F841
                should_continue = active_loop.step()

            # NOTE: MC caching is not faster
            # with MCCachingModule(model) as _mc_caching_model:
            #     with MCDropoutModule(_mc_caching_model) as _mc_dropout_model:  # noqa: F841
            #         should_continue = active_loop.step()
        else:
            should_continue = active_loop.step()

        # track progress
        # fmt: off
        labelling_progress += (
            al_dataset._labelled.astype(np.uint16)
        )
        # fmt: on
        logger.info("Acquire - end")

        # stop if needed
        if not should_continue:
            break
        if dry_run:
            break

        # reset model weights and lr scheduler for next AL epoch
        trainer.load_state_dict(init_weights)
        trainer.lr_scheduler = None

    # save model, labelling progress, logs, config
    # model_weight = model.state_dict() # NOTE: do not use because take lot of memory
    model_weight = None
    # metrics = wrapper.active_learning_metrics

    return model_weight, labelling_progress, metrics
