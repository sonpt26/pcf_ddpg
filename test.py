from network import NetworkEnv
import os
import logging
import logging.config
import yaml
import time

os.environ["KERAS_BACKEND"] = "tensorflow"
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)
logger = logging.getLogger("my_logger")

for i in range(4):
    env = NetworkEnv(
        "setting/episode_" + str(i + 1) + "/generator.yaml",
        "setting/episode_" + str(i + 1) + "/processor.yaml",
    )
    observation = env.reset()
    # action = env.action_space.sample()
    action = [0, 0, 0]
    logger.info("action %s", action)
    observation, reward, done, term, _ = env.step(action)
    logger.info("observation %s", observation)
    # env.reset()
    time.sleep(1)
    env.close()
