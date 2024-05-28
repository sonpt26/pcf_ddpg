import os
import logging
import logging.config
from tkinter.messagebox import RETRY
import yaml
import numpy as np
import json
import shutil
import threading
import sys
import argparse

fig_size = (30,20)
lw = 0.5

# Create work dir
if os.path.exists("./result"):
    shutil.rmtree("./result")
os.mkdir("./result")
os.environ["KERAS_BACKEND"] = "tensorflow"

# Config logging
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)
logger = logging.getLogger("my_logger")

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

import keras
from keras.layers import Input, Dense, Concatenate, Flatten
from keras.models import Model
import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
from network import NetworkEnv
import time
from pathlib import Path

tf.config.run_functions_eagerly(True)

def numpy_representer(dumper, data):
    return dumper.represent_list(data.tolist())


def save_array_data_to_file(file_path, array):
    with open(file_path, "w") as file:
        yaml.dump(array, file)


yaml.add_representer(np.ndarray, numpy_representer)
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


class OUActionNoise:

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:

    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, *state_shape))
        self.action_buffer = np.zeros((self.buffer_capacity, *action_shape))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *state_shape))
        
    def set_loss_dict(self, ep_dict):
        self.loss_dict = ep_dict
        
    def get_last_loss(self):
        return self.last_loss

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = np.array(obs_tuple[0])
        self.action_buffer[index] = np.array(obs_tuple[1])
        self.reward_buffer[index] = np.array(obs_tuple[2])
        self.next_state_buffer[index] = np.array(obs_tuple[3])

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        self.last_loss = {
            "critic_loss": 0,
            "actor_loss": 0
        }
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))
            closs = critic_loss.numpy()
            self.loss_dict["critic_loss"].append(closs)
            self.last_loss["critic_loss"] = closs
            # logger.info("critic_loss %s", closs)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)
            aloss = actor_loss.numpy()
            self.loss_dict["actor_loss"].append(aloss)
            self.last_loss["actor_loss"] = aloss
            # logger.info("actor_loss %s", aloss)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


def get_actor():
    last_init = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    inputs = Input(shape=state_shape)
    x = Flatten()(inputs)  # Flatten the input if needed
    x = Dense(256, activation="tanh")(x)
    x = Dense(256, activation="tanh")(x)
    outputs = Dense(action_shape[0], activation="linear", kernel_initializer=last_init)(x)
    model = Model(inputs, outputs)
    return model


def get_critic():
    state_input = Input(shape=state_shape)
    action_input = Input(shape=action_shape)

    state_x = Flatten()(state_input)  # Flatten the state input if needed
    state_x = Dense(16, activation="tanh")(state_x)
    state_x = Dense(32, activation="tanh")(state_x)

    action_x = Dense(32, activation="tanh")(action_input)

    concat = Concatenate()([state_x, action_x])
    x = Dense(256, activation="tanh")(concat)
    x = Dense(256, activation="tanh")(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model([state_input, action_input], outputs)
    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions = keras.ops.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)


"""
## Training hyperparameters
"""

std_dev = 0.3
# x_initial =  np.random.uniform(0,1, action_shape)
# logger.info("x_inital %s", x_initial)
ou_noise = OUActionNoise(
    mean=np.zeros(action_shape), 
    std_deviation=float(std_dev) * np.ones(action_shape),
    # dt=4,
    # theta=2,
    # x_initial=x_initial
)

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.001
actor_lr = 0.001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = 0.95
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)
"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""


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
    prev_state, _ = env.reset()
    logger.info("==================TRAINING EPISODE %s==================", folder)
    count = 0
    init_action = False;
    retry = 0
    ep_record = {
        "reward": 0,
        "revenue": 0
    }
    ep_loss = {
        "actor_loss": [],
        "critic_loss": []
    }
    buffer.set_loss_dict(ep_loss)
    while True:
        count +=1;
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        if init_action:
            action = np.random.uniform(0,1, action_shape)
            init_action = False
        else:
            action = policy(tf_prev_state, ou_noise)
        
        array = np.array(action)
        contains_nan = np.isnan(array).any()
        if contains_nan:
            logger.warn("=============NaN action %s. Retry=============", action)
            env.reset()
            break
        logger.info("state: %s", tf_prev_state)
        logger.info("action: %s", action)
        # Recieve state and reward from environment.
        state, reward, done, terminated, _ = env.step(action)
        
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

        buffer.record((prev_state, action, reward, state))        
        buffer.learn()

        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        # End this episode when `done` or `truncated` is True
        # if done or terminated or count > total_episodes:
        #     break
        retaind_rev = env.get_last_retained_revenue()
        logger.info("Episode %s. Iteration %s. Loss %s. Latency %s. Revenue: %s$. Reward: %s. Retained: %s", 
                    folder, count, buffer.get_last_loss(), latency, revenue, reward, round(retaind_rev, 2))       
        if (done and retaind_rev > 0.9) or retry > 3:
            break
        
        if count > total_episodes:
            if reward > 0:
                break
            if reward < 0 :
                init_action = True
                count = int(total_episodes/2)
                retry += 1         
            else:
                break
        
        prev_state = state        

    time.sleep(2)
    env.close()
    # Plotting graph
    mypath = build_path("./result/", folder)
    os.makedirs(mypath, exist_ok=True)
    basepath = str(mypath)

    # Episodes versus Avg. Rewards    
    plt.figure(figsize=fig_size)
    plt.plot(ep_reward_list, marker='o', linewidth=lw)
    plt.xlabel("Iteration")
    plt.ylabel("Episodic Reward")
    plt.savefig(basepath + "/reward.png")
    # plt.show()

    plt.figure(figsize=fig_size)
    for tc, val in ep_latency_list.items():
        plt.plot(val, label=tc, marker='o', linewidth=lw)

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Latency")
    plt.savefig(basepath + "/latency.png")

    plt.figure(figsize=fig_size)
    plt.plot(ep_revenue_list, marker='o', linewidth=lw)
    plt.xlabel("Iteration")
    plt.ylabel("Revenue")
    plt.savefig(basepath + "/revenue.png")

    plt.figure(figsize=fig_size)
    for tech, val in ep_queue_load_list.items():
        plt.plot(val, label=tech, marker='o', linewidth=lw)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Queue Load")
    plt.savefig(basepath + "/queue_load.png")
    
    plt.figure(figsize=fig_size)
    for typ, los in ep_loss.items():        
        plt.plot(los, label=typ, marker='o', linewidth=lw)            
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel(folder + " loss")
    plt.savefig(basepath + "/loss.png")

    save_array_data_to_file(basepath + "/tps.yaml", json.dumps(ep_throughput_list))
    save_array_data_to_file(basepath + "/queue.yaml", json.dumps(ep_queue_load_list))
    save_array_data_to_file(basepath + "/record.yaml", json.dumps(ep_record))
    save_array_data_to_file(basepath + "/revenue.yaml", ep_revenue_list)
    for tc, val in ep_latency_list.items():
        save_array_data_to_file(basepath + "/latency" + tc + ".yaml", val)

# Save the weights
actor_model.save_weights("./result/pcf_dqn_actor.weights.h5")
critic_model.save_weights("./result/pcf_dqn_critic.weights.h5")

target_actor.save_weights("./result/pcf_dqn_target_actor.weights.h5")
target_critic.save_weights("./result/pcf_dqn_target_critic.weights.h5")
# sys.exit("Finish")
