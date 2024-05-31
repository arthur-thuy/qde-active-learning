"""Module for active learning."""

# standard library imports
import argparse
import logging
import os
import subprocess
import time

import structlog

# related third party imports
import torch
import torch.backends

# local application/library specific imports
from data_loader.build import build_hf_dataset
from tools.configurator import check_cfg, check_dataset_info, load_configs, save_config
from tools.hf_trainer import huggingface_al_procedure
from tools.random_trainer import baseline_procedure
from tools.utils import (
    get_init_active_idx,
    print_elapsed_time,
    save_checkpoint,
    set_device,
    set_seed,
)

# set up logger
logger = structlog.get_logger("qdet")
logger.setLevel(logging.INFO)

# allow TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description="PyTorch Active Learning")
parser.add_argument(
    "config",
    type=str,
    help="config file path",
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="run a single AL epoch"
)


def main() -> None:
    """Run active learning experiment."""
    args = parser.parse_args()

    # config
    configs = load_configs(args.config)

    # remove previous contents (take dir form first cfg)
    subprocess.run(
        f"rm -r {configs[0].OUTPUT_DIR}",
        shell=True,
        check=False,
    )

    # logical checks before start running
    for cfg in configs:
        check_cfg(cfg)

    # get datasets
    # NOTE: need same input for all configs in a single run
    _ = check_dataset_info(configs)
    datasets_runs = {}
    for run_n in range(1, configs[0].RUNS + 1):
        datasets_runs[run_n] = build_hf_dataset(
            configs[0].LOADER, configs[0].MODEL.NUM_LABELS, configs[0].SEED + run_n
        )

    for cfg in configs:
        print("\n", "=" * 10, f"Config: {cfg.ID}", "=" * 10)
        # device
        device = set_device(cfg.DEVICE.NO_CUDA, cfg.DEVICE.NO_MPS)

        # start AL experiment loop
        for run_n in range(1, cfg.RUNS + 1):
            start_time = time.time()
            print("\n", "*" * 10, f"Run: {run_n}/{cfg.RUNS}", "*" * 10)

            # seed
            if cfg.SEED is not None:
                set_seed(cfg.SEED + run_n)

            if cfg.MODEL.NAME in ["random", "majority"]:
                # NOTE: random model
                metrics = baseline_procedure(
                    cfg=cfg,
                    datasets=datasets_runs[run_n],
                    method=cfg.MODEL.NAME,
                )
                model_weight = None
                labelling_progress = None
            else:
                # NOTE: transformers
                # get initial active indices
                init_active_idx = get_init_active_idx(
                    dataset=datasets_runs[run_n]["train"],
                    init_size=cfg.AL.INIT_ACTIVE_SIZE,
                    num_classes=cfg.MODEL.NUM_LABELS,
                    balanced=cfg.AL.INIT_ACTIVE_BALANCED,
                    seed=cfg.SEED + run_n,
                )
                model_weight, labelling_progress, metrics = huggingface_al_procedure(
                    cfg=cfg,
                    datasets=datasets_runs[run_n],
                    init_active_idx=init_active_idx,
                    device=device,
                    dry_run=args.dry_run,
                )

            save_checkpoint(
                {
                    "model": model_weight,
                    "labelling_progress": labelling_progress,
                    "metrics": metrics,
                    "max_ds_size": datasets_runs[run_n]["train"].num_rows,
                },
                save_dir=os.path.join(cfg.OUTPUT_DIR, cfg.ID),
                fname=f"run_{run_n}",
            )
            print_elapsed_time(start_time, run_n, cfg.ID)

        save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID)


if __name__ == "__main__":
    main()
