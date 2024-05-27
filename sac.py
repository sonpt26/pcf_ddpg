import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Sequential,load_model,Model
import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.utils import normalize as normal_values
import cv2
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import math
import glob
import io
import base64
from IPython.display import HTML
from IPython.display import clear_output
from IPython import display as ipythondisplay
import tensorflow_probability as tfp

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data=''''''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

RANDOM_SEED=3
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
env = gym.make("HalfCheetah-v2")
#env=wrap_env(env)  #use this when you want to record a video of episodes
env.seed(RANDOM_SEED)
env.reset() # reset to env


num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class GaussianPolicy():
  def __init__(self,action_dim,state_dim ,reparameterize):
    self.reparameterize = reparameterize
    self.action_dim=action_dim
    self.state_dim=state_dim
    self.model = self.create_model()

  def create_model(self):
    model=Sequential()
    model.add(Input(shape=(self.state_dim)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(self.action_dim * 2, activation=None))
    return model

  def call(self, inputs):
    mean_and_log_std = self.model(inputs)
    mean, log_std = tf.split(mean_and_log_std, num_or_size_splits=2, axis=1)
    log_std = tf.clip_by_value(log_std, -20., 2.)
    
    distribution = tfp.distributions.MultivariateNormalDiag(loc=mean,scale_diag=tf.exp(log_std))
    
    raw_actions = distribution.sample()
    if not self.reparameterize:
        raw_actions = tf.stop_gradient(raw_actions)
    log_probs = distribution.log_prob(raw_actions)
    log_probs -= self._squash_correction(raw_actions)

    self.actions = tf.tanh(raw_actions)
    
    return self.actions, log_probs
    
  def _squash_correction(self, raw_actions, stable=True, eps=1e-8):
    if not stable:
      return tf.reduce_sum(tf.math.log(1. - tf.square(tf.tanh(raw_actions)) + eps), axis=1)
    else:
      return tf.reduce_sum(tf.math.log(4.) + 2. * (raw_actions - tf.nn.softplus(2. * raw_actions)), axis=1)

  def eval(self, state):        
    action, _ = self.model(state)
    return action.flatten()

class SAC:
 
  def __init__(self,upper_bound,lower_bound,buffer_capacity=50000,observing_episodes=1,alpha=0.05,reparameterize=False,p_1=None,p_2=None,p_3=None,p_4=None,p_5=None):
    self.upper_bound=upper_bound
    self.lower_bound=lower_bound
    self.observing_episodes=observing_episodes
    self.state_shape=env.observation_space.shape[0] # the state space
    self.action_shape=env.action_space.shape[0] # the action space
    self.gamma=[0.99] # decay rate of past observations
    self.learning_rate_Q_function= 3e-4
    self.learning_rate_V_function= 3e-4 
    self.learning_rate_policy= 3e-4
    self.reparameterize=reparameterize
    self.tau=0.01
    self.alpha=alpha
    self.epsilon=1.0
    self.batch_size=256
    self.index=0
    self.beta=0.001 
    if not p_1:
      self.Q_function=self._create_model('Q_function')    
      self.target_Q_function=self._create_model('Q_function')
      self.V_function=self._create_model('V_function')
      self.policy=self._create_model('Policy')  
    else:
      self.Q_function=load_model(p_1)    
      self.target_Q_function=load_model(p_2)
      self.V_function=load_model(p_3)
      self.target_V_function=load_model(p_4)
      self.policy=GaussianPolicy(self.action_shape,self.state_shape,self.reparameterize,p_5)
    
    self.buffer_capacity = buffer_capacity
    self.buffer_counter = 0

    self.states=np.zeros((self.buffer_capacity, self.state_shape))
    self.rewards=np.zeros((self.buffer_capacity,1))
    self.dones=np.zeros((self.buffer_capacity, 1))
    self.actions=np.zeros((self.buffer_capacity, 6))
    self.next_states=np.zeros((self.buffer_capacity, self.state_shape))
  
  def remember(self, state, reward,action,next_state,done):
    '''stores observations'''
    self.index = self.buffer_counter % self.buffer_capacity
    self.rewards_plot.append(reward)
    self.states_plot.append(state)
    self.actions_plot.append(action)
    self.states[self.index] = state
    self.rewards[self.index]=reward
    self.dones[self.index]=done
    self.actions[self.index]=action
    self.next_states[self.index]=next_state
    self.buffer_counter += 1
    
  def _create_model(self,model_type):
 
    ''' builds the model using keras'''
 
    state_input = Input(shape=(17,))
  
    layer_1=(Dense(256, activation="relu"))(state_input)  
    layer_2=(Dense(256, activation="relu"))(layer_1)
 
    if model_type=='Q_function':
      action_input = Input(shape=(self.action_shape))
      action_layer_1 = Dense(128, activation="relu")(action_input)
      

      concat = Concatenate()([layer_2,action_layer_1])
      concat_layer_1=Dense(256,activation="relu")(concat)

      output = Dense(1, activation=None)(concat_layer_1)

      model = Model(inputs=[state_input,action_input],outputs=[output])

    elif model_type=='V_function':
      output = Dense(1, activation=None)(layer_2)
      model = Model(inputs=[state_input],outputs=[output])
    else:
      model=GaussianPolicy(self.action_shape,self.state_shape,self.reparameterize)
    return model

def get_action(self, state,status="Training"):
    '''samples the next action based on the policy probabilty distribution 
    of the actions'''
    if random.random() > self.epsilon:
        action,_ = self.policy.call(state)
    else:
        action=(np.random.uniform(-1, 1, self.action_shape))
    
    action=action*self.upper_bound
    legal_action = np.clip(action, self.lower_bound, self.upper_bound)
    return legal_action

def policy_loss(self,states):
    if not self.reparameterize:
      actions, log_pis = self.policy.call(states)
      actions=actions*self.upper_bound
      actions=np.clip(actions, self.lower_bound, self.upper_bound)
      if self.target_Q_function is None:
        q_n = tf.squeeze(self.Q_function((states, actions)), axis=1)
      else:
        q_n = tf.minimum(tf.squeeze(self.Q_function((states, actions)), axis=1),tf.squeeze(self.target_Q_function((states, actions)), axis=1))                
        b_n = tf.squeeze(self.V_function(states), axis=1)
        policy_loss = tf.reduce_mean(log_pis * tf.stop_gradient(self.alpha * log_pis - q_n + b_n))
    else:
      actions, log_pis = self.policy.call(states)
      actions=actions*self.upper_bound
      actions=np.clip(actions, self.lower_bound, self.upper_bound)
      if self.target_Q_function is None:
        q_n = tf.squeeze(self.Q_function((states, actions)), axis=1)
      else:
        q_n = tf.minimum(tf.squeeze(self.Q_function((states, actions)), axis=1),tf.squeeze(self.target_Q_function((states, actions)), axis=1))                
        b_n = tf.squeeze(self.V_function(states), axis=1)
        policy_loss = tf.reduce_mean(self.alpha * log_pis - q_n+b_n)
    return policy_loss
  
def value_function_loss(self, states):
    actions, log_pis = self.policy.call(states)
    actions=actions*self.upper_bound
    actions=np.clip(actions, self.lower_bound, self.upper_bound)
    if self.target_Q_function is None:
      q_n = tf.squeeze(self.Q_function((states, actions)), axis=1)
    else:
      q_n = tf.minimum(tf.squeeze(self.Q_function((states, actions)), axis=1),tf.squeeze(self.target_Q_function((states, actions)), axis=1)) 
      v_n = tf.squeeze(self.V_function(states), axis=1)
      value_function_loss = tf.reduce_mean(tf.losses.mean_squared_error(q_n - self.alpha * log_pis,v_n))
    return value_function_loss
    
def q_function_loss(self, states,next_states,actions,dones,rewards):
    q_n = tf.squeeze(self.Q_function((states, actions)), axis=1)
    next_v_n = tf.squeeze(self.V_function(next_states), axis=1)
    q_function_loss = tf.reduce_mean(tf.losses.mean_squared_error(rewards + (1 - dones) * self.gamma * next_v_n,q_n))
    return q_function_loss 

def update_models(self):
    '''
    Updates the network.
    '''
    record_range = min(self.buffer_counter, self.buffer_capacity)
    batch_indices = np.random.choice(record_range,self.batch_size)

    states_mb=self.states[batch_indices]
    actions_mb=self.actions[batch_indices]
    next_states_mb=self.next_states[batch_indices]
    rewards_mb=self.rewards[batch_indices]
    dones_mb=self.dones[batch_indices]

    optimizer_Q_function = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_Q_function)
    optimizer_V_function = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_V_function)
    optimizer_policy = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_policy)
    

    def train_Q_function(states,next_states,actions,dones,rewards):
      with tf.GradientTape() as tape:
        Q_function_loss=self.q_function_loss(states,next_states,actions,dones,rewards)
        print("Q Function Loss:",Q_function_loss)
      grads = tape.gradient(Q_function_loss,self.Q_function.trainable_variables)
      optimizer_Q_function.apply_gradients(zip(grads, self.Q_function.trainable_variables))
    train_Q_function(states_mb,next_states_mb,actions_mb,dones_mb,rewards_mb)
    

    def train_V_function(states):
      with tf.GradientTape() as tape:
        V_function_loss=self.value_function_loss(states)
        print("V Function Loss:",V_function_loss)
      grads = tape.gradient(V_function_loss,self.V_function.trainable_variables)
      optimizer_V_function.apply_gradients(zip(grads, self.V_function.trainable_variables))
    train_V_function(states_mb)
    

    def train_policy(states):
      with tf.GradientTape() as tape:
        Policy_loss=self.policy_loss(states)
        print("Policy Loss:",Policy_loss)
      grads = tape.gradient(Policy_loss,self.policy.model.trainable_variables)
      optimizer_policy.apply_gradients(zip(grads, self.policy.model.trainable_variables))
    train_policy(states_mb)
    

    Q_function_weights = np.array(self.Q_function.get_weights())
    Q_function_target_weights = np.array(self.target_Q_function.get_weights())
    Q_function_target_new_weights = self.tau*Q_function_weights + (1-self.tau)*Q_function_target_weights
    self.target_Q_function.set_weights(Q_function_target_new_weights)
    
    def train(self,episodes):
        ep_reward_list=[]
        c=0
        x=0
        for episode in range(episodes):
            env = (gym.make("HalfCheetah-v2"))
            aList=list(range(100))
            seed=(random.sample(aList, 1))[0]
            #print("Seed selected:{}".format(seed))
            env.seed(seed)
            state_=env.reset().reshape((1,17))
            done=False
            episode_reward=0  #record episode reward
            while not done:
                action=self.get_action(state_)
                next_state, reward, done, info=env.step(action)
                next_state_=next_state.reshape((1,17))
                self.remember(state_,reward,action,next_state_,done)
                state_ = next_state_
                episode_reward+=reward
            print('Updating the models')
            self.update_models()
            if self.epsilon > 0.001:
                self.epsilon=self.epsilon-0.001
            ep_reward_list.append(episode_reward)
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode:{}  Reward:{} Exploration_value:{} Average_reward:{}".format(episode,episode_reward,self.epsilon,avg_reward))
            if episode%100==0 and episode!=0:
                self.Q_function.save('Q_model_{}.h5'.format(episode))  
                self.target_Q_function.save('target_Q_model_{}.h5'.format(episode))
                self.V_function.save('V_model_{}.h5'.format(episode))  
                self.policy.model.save('policy_{}.h5'.format(episode))  
                
Agent=SAC(upper_bound,lower_bound,reparameterize=True)
Agent.train(1101)