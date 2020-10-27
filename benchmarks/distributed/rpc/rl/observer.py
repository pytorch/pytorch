import gym

import torch.distributed.rpc as rpc

import Coordinator
from Agent import AgentBase


class ObserverBase:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('CartPole-v1')

    def reset(self):
        state, reward = self.env.reset(), 0
        return state, reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        print("-->", state, reward, done, info)
        return state, reward, done

    def run_episode(self, agent_rref, n_steps):
        state, reward = self.reset()

        for step in range(n_steps):
            # send the state to the agent to get an action, also updating the reward in same call to save network overhead
            action = Coordinator.remote_method(
                AgentBase.select_action, agent_rref, self.id, state, reward)

            state, reward, done = self.step(action)

            if done:
                break
