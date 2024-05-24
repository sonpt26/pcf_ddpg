import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DDPG:
    def __init__(self, state_shape, action_shape, action_bound):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = action_bound

        # Actor model
        self.actor = self.build_actor()
        self.target_actor = self.build_actor()
        self.target_actor.set_weights(self.actor.get_weights())

        # Critic model
        self.critic = self.build_critic()
        self.target_critic = self.build_critic()
        self.target_critic.set_weights(self.critic.get_weights())

        # Optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = optimizers.Adam(learning_rate=0.002)

    def build_actor(self):
        input_layer = layers.Input(shape=self.state_shape)
        x = layers.Flatten()(input_layer)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        output_layer = layers.Dense(np.prod(self.action_shape), activation='tanh')(x)
        output_layer = layers.Reshape(self.action_shape)(output_layer)
        output_layer = layers.Lambda(lambda x: x * self.action_bound)(output_layer)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

    def build_critic(self):
        state_input = layers.Input(shape=self.state_shape)
        action_input = layers.Input(shape=self.action_shape)
        x = layers.Concatenate()([layers.Flatten()(state_input), action_input])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        output_layer = layers.Dense(1)(x)
        model = models.Model(inputs=[state_input, action_input], outputs=output_layer)
        return model

    def act(self, state):
        return self.actor.predict(state)

    def update_target_models(self, tau):
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]

        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99, tau=0.001):
        # Update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic([next_states, target_actions])
            target_values = rewards + gamma * target_q_values * (1 - dones)
            current_values = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_values - current_values))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            critic_value = self.critic([states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update target networks
        self.update_target_models(tau)

# Example usage:
if __name__ == "__main__":
    # Define environment parameters
    state_shape = (3, 5)
    action_shape = (3,)
    action_bound = 1

    # Create DDPG agent
    agent = DDPG(state_shape, action_shape, action_bound)

    # Mock data for training
    states = np.random.random((32,) + state_shape)
    actions = np.random.random((32,) + action_shape)
    rewards = np.random.random((32,))
    next_states = np.random.random((32,) + state_shape)
    dones = np.random.choice([0, 1], size=(32,))

    # Train the agent
    agent.train(states, actions, rewards, next_states, dones)
