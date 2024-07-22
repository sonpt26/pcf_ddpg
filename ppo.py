import os
import json
import shutil
import threading
import sys
import argparse
import logging
import logging.config
import yaml

# Create work dir
base_work_dir = "./ppo_result"
if os.path.exists(base_work_dir):
    shutil.rmtree(base_work_dir)
os.mkdir(base_work_dir)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Config logging
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)
logger = logging.getLogger("my_logger")

import keras
from keras import layers

import numpy as np
import tensorflow as tf
import gymnasium as gym
import scipy.signal
from network import NetworkEnv
import time
from pathlib import Path


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(
        self, observation_dimensions, action_shape, size, gamma=0.99, lam=0.95
    ):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, *observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros((size, *action_shape), dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=keras.activations.tanh, output_activation=None):
    # Build a feedforward neural network
    print("sizes", sizes)
    print("sizes -1", sizes[-1])
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = keras.ops.log_softmax(logits)
    logprobability = keras.ops.sum(
        keras.ops.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


seed_generator = keras.random.SeedGenerator(1337)


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = keras.ops.squeeze(
        keras.random.categorical(logits, 1, seed=seed_generator), axis=1
    )
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = keras.ops.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = keras.ops.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -keras.ops.mean(
            keras.ops.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = keras.ops.mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = keras.ops.sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = keras.ops.mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# parser argument
parser = argparse.ArgumentParser()
parser.add_argument("--clear_queue", help="Clear queue before each step", default=False)
parser.add_argument(
    "--episode", help="Number of iteration for each scenario", default=100, type=int
)
args = parser.parse_args()
clear_queue_step = False
if args.clear_queue:
    clear_queue_step = True
    logger.info("Clear queue before step is ON")
if args.episode:
    total_episodes = int(args.episode)
    logger.info("Total iteration is %s", total_episodes)

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

#
env = NetworkEnv(clear_queue_step=clear_queue_step)
state_shape = env.get_state_shape()
action_shape = env.get_action_shape()
num_states = env.observation_space.shape[0]
logger.info("Shape of State Space ->  %s", state_shape)
num_actions = env.action_space.shape[0]
logger.info("Shape of Action Space ->  %s", action_shape)

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

logger.info("Max Value of Action ->  %s", upper_bound)
logger.info("Min Value of Action ->  %s", lower_bound)
env.close()

# True if you want to render the environment
render = False

# Initialize the buffer
buffer = Buffer(state_shape, action_shape, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=state_shape, dtype="float32")
logits = mlp(observation_input, list(hidden_sizes) + [np.prod(action_shape)])
actor = keras.Model(inputs=observation_input, outputs=logits)
value = keras.ops.squeeze(mlp(observation_input, list(hidden_sizes) + [action_shape]))
critic = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
observation, _ = env.reset()
episode_return, episode_length = 0, 0

directory_path = "./setting"
specific_dir = Path(directory_path)
# List all directories in the specified directory
folders = [
    name
    for name in os.listdir(directory_path)
    if os.path.isdir(os.path.join(directory_path, name))
]


def build_path(base_path, *sub_paths):
    path = Path(base_path)
    for sub_path in sub_paths:
        path /= sub_path
    return path


logger.info("Folders in directory %s: %s", directory_path, folders)
for folder in folders:
    # To store reward history of each episode
    ep_record = {}
    ep_loss = {}
    ep_reward_list = []
    ep_latency_list = {}
    ep_revenue_list = []
    ep_throughput_list = {}
    ep_queue_load_list = {}

    gen_setting = specific_dir / folder / "generator.yaml"
    proc_setting = specific_dir / folder / "processor.yaml"
    env = NetworkEnv(gen_setting, proc_setting, clear_queue_step)
    observation, _ = env.reset()
    logger.info("==================TRAINING EPISODE %s==================", folder)
    count = 0
    init_action = False
    retry = 0
    ep_record = {"reward": 0, "revenue": 0}
    ep_loss = {"actor_loss": [], "critic_loss": []}

    while True:
        observation = observation.reshape(1, *state_shape)
        logits, action = sample_action(observation)
        action = action.numpy().reshape(action_shape)
        observation_new, reward, done, terminal, _ = env.step(action)
        episode_return += reward
        episode_length += 1

        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        buffer.store(observation, action, reward, value_t, logprobability_t)

        observation = observation_new

        latency = env.get_last_step_latency()
        for clas, value in latency.items():
            if clas not in ep_latency_list:
                ep_latency_list[clas] = []
            ep_latency_list[clas].append(value)

        throughput = env.get_last_step_throughput()
        for tc, val in throughput.items():
            if tc not in ep_throughput_list:
                ep_throughput_list[tc] = {}
            for tech, tps in val.items():
                if tech not in ep_throughput_list[tc]:
                    ep_throughput_list[tc][tech] = []
                ep_throughput_list[tc][tech].append(tps)

        revenue = env.get_last_step_revenue()
        ep_revenue_list.append(revenue)

        queue_load = env.get_last_step_queue_load()
        for tech, val in queue_load.items():
            if tech not in ep_queue_load_list:
                ep_queue_load_list[tech] = []
            ep_queue_load_list[tech].append(val)

        ep_reward_list.append(reward)
        if reward > ep_record["reward"]:
            ep_record["reward"] = reward
            ep_record["revenue"] = revenue
            ep_record["action"] = action.tolist()
            ep_record["latency"] = latency
            ep_record["queue_load"] = queue_load

        retaind_rev = env.get_last_retained_revenue()
        logger.info(
            "Episode %s. Iteration %s. Loss %s. Latency %s. Revenue: %s$. Reward: %s. Retained: %s",
            folder,
            count,
            buffer.get_last_loss(),
            latency,
            revenue,
            reward,
            round(retaind_rev, 2),
        )

        if terminal or (done and retaind_rev > 0.9):
            last_value = 0 if done else critic(observation.reshape(1, *state_shape))
            buffer.finish_trajectory(last_value)
            observation, _ = env.reset()

        if count > total_episodes:
            if reward > 0:
                break
            if reward < 0:
                init_action = True
                count = int(total_episodes / 2)
                retry += 1
            else:
                break

    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            break

    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)
