#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

class Actor(tf.keras.Model):
    
    def __init__(self, NrOfActions):
        super(Actor, self).__init__()
        self.layer1 = tf.keras.layers.Dense(500, input_shape=(12,), activation='tanh') 
        self.layer2 = tf.keras.layers.Dense(400, activation='tanh')
        self.layer3 = tf.keras.layers.Dense(300, activation='tanh')
        self.layer4 = tf.keras.layers.Dense(NrOfActions, activation='tanh')
        
    def call(self, State):
        out = self.layer1(State)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
    
class Critic(tf.keras.Model):
    
    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = tf.keras.layers.Dense(500, activation='tanh') 
        self.layer2 = tf.keras.layers.Dense(400, activation='tanh')
        self.layer3 = tf.keras.layers.Dense(300, activation='tanh')
        self.layer4 = tf.keras.layers.Dense(1, activation='tanh')
       
    def call(self, State, Action):
        out = self.layer1(tf.concat([State,Action], axis=1))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

