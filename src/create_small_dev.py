"""Module to sample a smaller development set from the regular development set."""

# standard library imports
import argparse
import os

# related third party imports
import pandas as pd
import structlog

# local application/library specific imports
# /

# set up logger
logger = structlog.get_logger()

parser = argparse.ArgumentParser(description="Sample smaller dev set")
parser.add_argument(
    "dataset",
    type=str,
    help="dataset name",
)
parser.add_argument(
    "--size",
    type=int,
    default=1000,
    help="size of the smaller dev set",
)

# NOTE: for race_pp, do:
# $ python create_small_dev.py race_pp --size 1000


def main() -> None:
    """Create small dev sets."""
    args = parser.parse_args()

    # read dev set
    dev_set = pd.read_csv(
        os.path.join("../data/processed", f"tf_{args.dataset}_text_difficulty_dev.csv")
    )

    # sample smaller dev set
    small_dev_set = dev_set.sample(n=args.size, random_state=42)

    # save smaller dev set
    small_dev_set.to_csv(
        os.path.join(
            "../data/processed", f"tf_{args.dataset}_text_difficulty_small_dev.csv"
        ),
        index=False,
    )

    logger.info(
        f"Saved smaller {args.dataset} dev set with {small_dev_set.shape[0]} samples"
    )


if __name__ == "__main__":
    main()
