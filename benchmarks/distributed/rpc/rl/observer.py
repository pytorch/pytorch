import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_async, rpc_sync, remote

import numpy as np
import random
import time

from agent import AgentBase


class ObserverBase:
    def __init__(self):
        self.id = rpc.get_worker_info().id

    def set_state(self, state_size, batch):
        self.state_size = state_size
        self.select_action = AgentBase.select_action_batch if batch else AgentBase.select_action_non_batch

    def reset(self):
        state = torch.rand(self.state_size)
        return state

    def step(self, action):
        state = torch.rand(self.state_size)
        reward = random.randint(0, 1)

        return state, reward

    def run_ob_episode(self, agent_rref, n_steps):
        state, ep_reward = self.reset(), None
        rewards = torch.zeros(n_steps)
        observer_latencies = []

        for st in range(n_steps):
            ob_latency_start = time.time()
            # action = agent_rref.rpc_sync().select_action(self.id, state)
            action = rpc_sync(agent_rref.owner(), self.select_action, args=(
                agent_rref, self.id, state))
            observer_latencies.append(time.time() - ob_latency_start)
            state, reward = self.step(action)
            rewards[st] = reward

        return [rewards, ep_reward]
