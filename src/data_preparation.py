import structlog

from tools.constants import WRITE_DIR, RACE_PP
from tools.data_manager import RaceDatamanager

# set logger
log = structlog.get_logger()


def main():
    # RACE++
    log.info("Starting preparation RACE++")
    race_data_dir = "../data/raw/RACE"
    race_c_data_dir = "../data/raw/race-c-master/data"
    race_pp_dm = RaceDatamanager()
    dataset = race_pp_dm.get_racepp_dataset(race_data_dir, race_c_data_dir, WRITE_DIR)
    # whole RACE++
    race_pp_dm.convert_to_transformers_format_and_store_dataset(
        dataset, WRITE_DIR, RACE_PP, skip_answers_texts=False
    )


if __name__ == "__main__":
    main()
