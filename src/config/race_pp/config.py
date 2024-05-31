"""Default config file."""

from yacs.config import CfgNode as CN

_C = CN()

# seed for reproducibility
_C.SEED = 42  # run 1: 42+1, run 2: 43+1, run 3: 43+2, run 4: 45+1, run 5: 45+2

# number of runs
_C.RUNS = 5  # 5

# Trainer type
# _C.TRAINER_TYPE = "transformer"

_C.LOADER = CN()
# dataset name
_C.LOADER.NAME = "race_pp"
# number of data loading workers
# _C.LOADER.WORKERS = 4 # NOTE: unused because warning
# regression problem
_C.LOADER.REGRESSION = True
# use smaller dev set
_C.LOADER.SMALL_DEV = True

# model architecture
_C.MODEL = CN()
# model name
_C.MODEL.NAME = "distilbert-base-uncased"
# number of labels (1 for regression)
_C.MODEL.NUM_LABELS = 1
# apply MC Dropout
_C.MODEL.MC_DROPOUT = True
# max sequence length
_C.MODEL.MAX_LENGTH = 256
# BF16
_C.MODEL.BF16 = True


_C.TRAIN = CN()
# number of total epochs to run
_C.TRAIN.EPOCHS = 10
# max number of training steps
_C.TRAIN.MAX_STEPS = -1  # 250
# mini-batch size
_C.TRAIN.BATCH_SIZE = 64
# initial learning rate
_C.TRAIN.LR = 2e-5
# weight decay
_C.TRAIN.WEIGHT_DECAY = 0.05
# Adam epsilon
_C.TRAIN.ADAM_EPSILON = 1e-8
# freeze base
_C.TRAIN.FREEZE_BASE = False
# warmup ratio
_C.TRAIN.WARMUP_RATIO = 0.1
# early stopping
_C.TRAIN.EARLY_STOPPING = False
# patience for early stopping
_C.TRAIN.PATIENCE = 6  # 10

_C.EVAL = CN()
# batch size
_C.EVAL.BATCH_SIZE = 256
# MC Dropout samples
_C.EVAL.MC_SAMPLES = 10  # NOTE: set to 1 if no MC_DROPOUT
# logging and eval steps
_C.EVAL.LOGGING_STEPS = 0.02

_C.DEVICE = CN()
# disables CUDA training
_C.DEVICE.NO_CUDA = False
# disables macOS GPU training
_C.DEVICE.NO_MPS = False

_C.AL = CN()
# number of active learning epochs
_C.AL.EPOCHS = 96
# query size
_C.AL.QUERY_SIZE = 100
# pool subset size
_C.AL.SUBSET_POOL = 5000
# initial active size
_C.AL.INIT_ACTIVE_SIZE = 500
# initial active balanced
_C.AL.INIT_ACTIVE_BALANCED = False
# heuristic
_C.AL.HEURISTIC = "powervariance"  # "random"  #
# temperature for PowerBALD or PowerVariance
_C.AL.TEMPERATURE = 1.0
# train on all data
_C.AL.FULL_DATA = False


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
