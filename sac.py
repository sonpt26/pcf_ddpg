import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model

# Set random seeds for reproducibility
RANDOM_SEED = 3
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Hyperparameters
BUFFER_CAPACITY = 50000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
LEARNING_RATE = 3e-4

# Create the environment
env = gym.make("Pendulum-v1")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

class GaussianPolicy(Model):
    def __init__(self, action_dim, state_dim):
        super(GaussianPolicy, self).__init__()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.mean = Dense(action_dim)
        self.log_std = Dense(action_dim)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = tf.exp(log_std)
        normal_sample = tf.random.normal(shape=mean.shape)
        action = tf.tanh(mean + std * normal_sample)
        log_prob = self._log_prob(mean, std, normal_sample, action)
        return action, log_prob

    def _log_prob(self, mean, std, normal_sample, action):
        log_prob = -0.5 * ((normal_sample ** 2) + (2 * tf.math.log(std)) + tf.math.log(2 * np.pi))
        log_prob = tf.reduce_sum(log_prob, axis=1)
        log_prob -= tf.reduce_sum(tf.math.log(1 - tf.math.tanh(action) ** 2 + 1e-6), axis=1)
        return log_prob

class Critic(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.q_value = Dense(1)

    def call(self, state_action):
        x = self.dense1(state_action)
        x = self.dense2(x)
        q_value = self.q_value(x)
        return q_value

class SACAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        # Initialize networks
        self.policy = GaussianPolicy(self.action_dim, self.state_dim)
        self.critic_1 = Critic(self.state_dim, self.action_dim)
        self.critic_2 = Critic(self.state_dim, self.action_dim)
        self.target_critic_1 = Critic(self.state_dim,self.action_dim)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim)
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        # Optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        # Replay buffer
        self.buffer_counter = 0
        self.state_buffer = np.zeros((BUFFER_CAPACITY, self.state_dim))
        self.action_buffer = np.zeros((BUFFER_CAPACITY, self.action_dim))
        self.reward_buffer = np.zeros((BUFFER_CAPACITY, 1))
        self.next_state_buffer = np.zeros((BUFFER_CAPACITY, self.state_dim))
        self.done_buffer = np.zeros((BUFFER_CAPACITY, 1))

    def update_buffer(self, state, action, reward, next_state, done):
        index = self.buffer_counter % BUFFER_CAPACITY
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done
        self.buffer_counter += 1

    def get_action(self, state, evaluate=False):
        state = state.reshape(1, -1)
        mean, log_std = self.policy(state)
        std = tf.exp(log_std)
        if evaluate:
            action = tf.tanh(mean)
        else:
            normal_sample = tf.random.normal(shape=mean.shape)
            action = tf.tanh(mean + std * normal_sample)
        return action.numpy()[0]

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def train(self, episodes):
        ep_reward_list = []

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_buffer(state, action, reward, next_state, done)

                if self.buffer_counter >= BATCH_SIZE:
                    self.update_networks()

                state = next_state
                episode_reward += reward

                if done:
                    break

            ep_reward_list.append(episode_reward)
            avg_reward = np.mean(ep_reward_list[-40:])
            print(f"Episode: {episode}, Reward: {episode_reward}, Average Reward: {avg_reward}")

            if episode % 100 == 0:
                self.policy.save_weights(f'policy_{episode}.weights.h5')
                self.critic_1.save_weights(f'critic_1_{episode}.weights.h5')
                self.critic_2.save_weights(f'critic_2_{episode}.weights.h5')
                self.target_critic_1.save_weights(f'target_critic_1_{episode}.weights.h5')
                self.target_critic_2.save_weights(f'target_critic_2_{episode}.weights.h5')

    def update_networks(self):
        indices = np.random.choice(min(self.buffer_counter, BUFFER_CAPACITY), BATCH_SIZE)
        state_batch = self.state_buffer[indices]
        action_batch = self.action_buffer[indices]
        reward_batch = self.reward_buffer[indices]
        next_state_batch = self.next_state_buffer[indices]
        done_batch = self.done_buffer[indices]

        with tf.GradientTape(persistent=True) as tape:
            next_action, next_log_prob = self.policy.sample(next_state_batch)
            next_state_action = tf.concat([next_state_batch, next_action], axis=1)

            target_q1 = tf.squeeze(self.target_critic_1(next_state_action))
            target_q2 = tf.squeeze(self.target_critic_2(next_state_action))
            target_v = tf.minimum(target_q1, target_q2) - ALPHA * tf.squeeze(next_log_prob)
            target_q = reward_batch + GAMMA * (1 - done_batch) * target_v

            state_action = tf.concat([state_batch, action_batch], axis=1)
            q1 = tf.squeeze(self.critic_1(state_action))
            q2 = tf.squeeze(self.critic_2(state_action))

            critic_1_loss = tf.reduce_mean(tf.square(q1 - target_q))
            critic_2_loss = tf.reduce_mean(tf.square(q2 - target_q))

            new_action, log_prob = self.policy.sample(state_batch)
            new_state_action = tf.concat([state_batch, new_action], axis=1)

            q1_new = tf.squeeze(self.critic_1(new_state_action))
            q2_new = tf.squeeze(self.critic_2(new_state_action))

            policy_loss = tf.reduce_mean(ALPHA * log_prob - tf.minimum(q1_new, q2_new))

        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)

        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        self.update_target(self.target_critic_1.variables, self.critic_1.variables, TAU)
        self.update_target(self.target_critic_2.variables, self.critic_2.variables, TAU)

if __name__ == "__main__":
    agent = SACAgent(env)
    agent.train(1000)
