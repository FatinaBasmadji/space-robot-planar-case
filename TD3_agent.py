#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The TD3_agent class was done (with little modifications) based on the code presented in: https://towardsdatascience.com/deep-deterministic-and-twin-delayed-deep-deterministic-policy-gradient-with-tensorflow-2-x-43517b0e0185 #

import tensorflow as tf
import numpy as np
from actor-critic-networks import Actor, Critic
from replay-buffer import ReplayBuffer
from tensorflow.keras.optimizers import Adam


class TD3_agent():
    
    def __init__(self, NrOfActions=3, BatchSize=128, gamma=0.99, tau=0.005, MinAction=-0.2, MaxAction=0.2, actor_update=2, replace=5 lr=0.001, noise=0.01, stop_noise=100, memory=100000):
        
        self.BatchSize = BatchSize
        self.NrOfActions = NrOfActions
        self.gamma = gamma
        self.tau = tau
        self.MinAction = MinAction
        self.MaxAction = MaxAction
        self.lr = lr
        self.memory = memory
        self.noise = noise
        self.step=0
        self.stop_noise = stop_noise
        self.step = 0
        self.actor_update = actor_update
        self.replace = replace
      
        self.actor = Actor(NrOfActions)
        self.actor_target = Actor(NrOfActions)
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.critic_target_1 = Critic()
        self.critic_target_2 = Critic()
        
        self.actor.compile(optimizer=Adam(learning_rate=self.lr))
        self.actor_target.compile(optimizer=Adam(learning_rate=self.lr))
        self.critic_1.compile(optimizer=Adam(learning_rate=self.lr))
        self.critic_2.compile(optimizer=Adam(learning_rate=self.lr))
        self.critic_target_1.compile(optimizer=Adam(learning_rate=self.lr))
        self.critic_target_2.compile(optimizer=Adam(learning_rate=self.lr))
        
        self.memory = ReplayBuffer(memory)
        self.update_target(tau=1)
                
        
    def take_action(self, State):
        State = tf.convert_to_tensor([State], dtype=tf.float32)
        Action = self.actor(State)
        if self.step < self.stop_noise:
            Action += tf.random.normal(shape=[self.NrOfActions], mean=0.0, stddev=self.noise)*(-1)**(np.random.randint(2,size=1))       
        Action = (tf.clip_by_value(self.MaxAction * Action, self.MinAction, self.MaxAction))     
        return Action[0]
    
           
    def store(self, State, Action, NextState, Reward, Done):
        self.memory.save(State, Action, NextState, Reward, Done)
        
          
    def update_target(self, tau=None):
        if tau is None:
            tau = self.tau
        weights1 = []
        targets1 = self.actor_target.weights
        for i, weight in enumerate(self.actor.weights):
            weights1.append(weight * tau + targets1[i]*(1-tau))
        self.actor_target.set_weights(weights1)

        weights2 = []
        targets2 = self.critic_target_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights2.append(weight * tau + targets2[i]*(1-tau))
        self.critic_target_1.set_weights(weights2)

        weights3 = []
        targets3 = self.critic_target_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights3.append(weight * tau + targets3[i]*(1-tau))
        self.critic_target_2.set_weights(weights3)

        
    def train(self):
        if self.memory.Counter < self.BatchSize:
            return 
        states, actions, next_states, rewards, dones = self.memory.sample(self.BatchSize,1)
  
        next_states = np.reshape(next_states, (self.BatchSize, 12, ))
        states = np.reshape(states, (self.BatchSize, 12, ))
        states = tf.convert_to_tensor([states], dtype= tf.float32)
        next_states = tf.convert_to_tensor([next_states], dtype= tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype= tf.float32)
        actions = tf.convert_to_tensor(actions, dtype= tf.float32)
        #dones = tf.convert_to_tensor(dones, dtype= tf.bool)  
        next_states = tf.reshape([next_states],[self.BatchSize,12])
        states = tf.reshape([states],[self.BatchSize,12])

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:   
             
            target_actions = self.actor_target(next_states)
            target_actions += tf.clip_by_value(tf.random.normal(shape=[*np.shape(target_actions)], mean=0.0, stddev=self.noise), self.MinAction, self.MaxAction)
            target_actions = self.MaxAction * (tf.clip_by_value(target_actions, self.MinAction, self.MaxAction)) 
                        
            target_next_state_values = tf.squeeze(self.critic_target_1(next_states, target_actions), 1)
            target_next_state_values2 = tf.squeeze(self.critic_target_2(next_states, target_actions), 1)
    
            critic_value = tf.squeeze(self.critic_1(states, actions), 1)
            critic_value2 = tf.squeeze(self.critic_2(states, actions), 1)
          
            next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2)
          
            target_values = rewards + self.gamma * next_state_target_value * dones
            critic_loss1 = tf.keras.losses.MSE(target_values, critic_value)
            critic_loss2 = tf.keras.losses.MSE(target_values, critic_value2)
        
      
        grads1 = tape1.gradient(critic_loss1, self.critic_1.trainable_variables)
        grads2 = tape2.gradient(critic_loss2, self.critic_2.trainable_variables)
      
        self.c_opt1.apply_gradients(zip(grads1, self.critic_1.trainable_variables))
        self.c_opt2.apply_gradients(zip(grads2, self.critic_2.trainable_variables))
      
      
        self.step +=1
      
        if self.step % self.actor_update == 0:
            with tf.GradientTape() as tape3:
                new_policy_actions = self.actor(states)
                actor_loss = -self.critic_1(states, new_policy_actions)
                actor_loss = tf.math.reduce_mean(actor_loss)
          
            grads3 = tape3.gradient(actor_loss, self.actor.trainable_variables)
            self.a_opt.apply_gradients(zip(grads3, self.actor.trainable_variables))

        if self.step % self.replace == 0:
            self.update_target()
           
    

