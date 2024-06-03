"""Build file for models."""

# standard library imports
# /

# related third party imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from yacs.config import CfgNode

# local application/library specific imports
# /


def build_hf_model(model_cfg: CfgNode) -> tuple:
    """Build the HuggingFace model and tokenizer.

    Parameters
    ----------
    model_cfg : CfgNode
        Model config object

    Returns
    -------
    tuple
        Tuple of model and tokenizer
    """
    print(f"=> creating model and tokenizer '{model_cfg.NAME}'")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_cfg.NAME,
        num_labels=model_cfg.NUM_LABELS,  # NOTE: regression task if num_labels == 1
        # NOTE: do not use "torch_dtype=torch.bfloat16" because causes bad convergence!
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_cfg.NAME
    )
    return model, tokenizer
