#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class ReplayBuffer:
    def __init__(self, MaxSize=100000, InputShape, NrOfActions):
        self.MaxSize = MaxSize
        self.Counter = 0
        self.States = np.zeros((self.MaxSize, *InputShape))
        self.Actions = np.zeros((self.MaxSize, NrOfActions))
        self.NextStates = np.zeros((self.MaxSize, *InputShape))
        self.Rewards = np.zeros(self.MaxSize)
        self.Dones = np.zeros(self.MaxSize)
        
    def Store(self, State, Action, NextState, Reward, Done):
        id = self.Counter % self.MaxSize
        self.States[id] = State
        self.Actions[id] = Action
        self.NextStates[id] = NextState
        self.Rewards[id] = Reward
        self.Dones[id] = Done
        self.Counter =  self.Counter + 1
        
    def Sample(self, BatchSize=64, RandomSample=1, Recent=0):
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

