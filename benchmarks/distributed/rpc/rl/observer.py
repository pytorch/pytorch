import random
import time

import torch
import torch.distributed.rpc as rpc

from agent import AgentBase
from torch.distributed.rpc import rpc_sync


class ObserverBase:
    def __init__(self):
        r"""
        Inits observer class
        """
        self.id = rpc.get_worker_info().id

    def set_state(self, state_size, batch):
        r"""
        Further initializes observer to be aware of rpc environment
        Args:
            state_size (list): List of integers denoting dimensions of state
            batch (bool): Whether agent will be using batch select action
        """
        self.state_size = state_size
        self.select_action = (
            AgentBase.select_action_batch
            if batch
            else AgentBase.select_action_non_batch
        )

    def reset(self):
        r"""
        Resets state randomly
        """
        state = torch.rand(self.state_size)
        return state

    def step(self, action):
        r"""
        Generates random state and reward
        Args:
            action (int): Int received from agent representing action to take on state
        """
        state = torch.rand(self.state_size)
        reward = random.randint(0, 1)

        return state, reward

    def run_ob_episode(self, agent_rref, n_steps):
        r"""
        Runs single observer episode where for n_steps, an action is selected
        from the agent based on curent state and state is updated
        Args:
            agent_rref (RRef): Remote Reference to the agent
            n_steps (int): Number of times to select an action to transform state per episode
        """
        state, ep_reward = self.reset(), None
        rewards = torch.zeros(n_steps)
        observer_latencies = []
        observer_throughput = []

        for st in range(n_steps):
            ob_latency_start = time.time()
            action = rpc_sync(
                agent_rref.owner(),
                self.select_action,
                args=(agent_rref, self.id, state),
            )

            ob_latency = time.time() - ob_latency_start
            observer_latencies.append(ob_latency)
            observer_throughput.append(1 / ob_latency)

            state, reward = self.step(action)
            rewards[st] = reward

        return [rewards, ep_reward, observer_latencies, observer_throughput]
