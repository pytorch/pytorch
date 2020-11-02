import gym

import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_async, rpc_sync, remote

import random
import numpy as np


class ObserverBase:
    def __init__(self):
        self.id = rpc.get_worker_info().id

    def reset(self):
        state, reward = np.random.uniform(low=-1, high=1, size=(10,)), 0
        return state, reward

    def step(self, action):
        state = np.random.uniform(low=-1, high=1, size=(10,))
        reward = random.randint(0, 1)
        done = False

        return state, reward, done

    def run_ob_episode(self, agent_rref, n_steps):
        state, reward = self.reset()

        for step in range(n_steps):
            # send the state to the agent to get an action, also updating the reward in same call to save network overhead
            action = agent_rref.rpc_sync().select_action(self.id, state, reward)
            state, reward, done = self.step(action)

            if done:
                break
