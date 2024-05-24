import os
import logging
import logging.config
import yaml

os.environ["KERAS_BACKEND"] = "tensorflow"
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)
logger = logging.getLogger("my_logger")
logger.info("This is an info message %s", 1)