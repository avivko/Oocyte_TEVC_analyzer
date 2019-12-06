import logging
import os.path


def initialize_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "analysis.log"), "a", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s : %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
