#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class ReplayBuffer:
    def __init__(self, MaxSize=100000):
        self.MaxSize = MaxSize
        self.Counter = 0
        self.States = []
        self.Actions = []
        self.NextStates = []
        self.Rewards = []
        self.Dones = []
        
    def save(self, State, Action, NextState, Reward, Done):
        if self.Counter == 0:
            self.States = State
            self.Actions = Action
            self.NextStates = NextState
            self.Rewards = Reward
            self.Dones = Done
        elif self.Counter < self.MaxSize:
            self.States = np.concatenate((self.States, State), axis=0)
            self.Actions = np.concatenate((self.Actions, Action), axis=0)
            self.NextStates = np.concatenate((self.NextStates, NextState), axis=0)
            self.Rewards = np.concatenate((self.Rewards, Reward), axis=0)
            self.Dones = np.concatenate((self.Dones, Done), axis=0)  
        else:
            i = self.Counter % self.MaxSize
            idx = range(i*np.size(self.States[0]),np.size(self.States[0])+i*np.size(self.States[0]))
            np.put(self.States, idx, State)
            np.put(self.NextStates, idx, NextState)
            idx = range(i*np.size(self.Actions[0]),np.size(self.Actions[0])+i*np.size(self.Actions[0]))
            np.put(self.Actions, idx, Action)
            idx = range(i*np.size(self.Rewards[0]),np.size(self.Rewards[0])+i*np.size(self.Rewards[0]))
            np.put(self.Rewards, idx, Reward)
            np.put(self.Dones, idx, Done)
        self.Counter =  self.Counter + 1
        
    def sample(self, BatchSize=64, RandomSample=1):
        if RandomSample == 1:
            sample = np.random.choice(self.Counter, BatchSize, replace = False)
        else:
            samplelist = np.arange(0,self.Counter)
            sample = np.random.choice(samplelist[-BatchSize:],BatchSize, replace = False)
        SampledStates = self.States[sample]
        SampledActions = self.Actions[sample]
        SampledNextStates = self.NextStates[sample]
        SampledRewards = self.Rewards[sample]
        SampledDones = self.Dones[sample]
        return SampledStates, SampledActions, SampledNextStates, SampledRewards, SampledDones

