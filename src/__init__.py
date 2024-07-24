import logging
import os


logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s : %(name)s - %(message)s",
    level=os.getenv("NBA_PREDICTION_LOG_LEVEL", logging.INFO),
    datefmt="%Y-%m-%d %H:%M:%S",
)
