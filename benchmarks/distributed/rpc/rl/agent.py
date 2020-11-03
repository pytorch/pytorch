from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote
from torch.distributions import Categorical

from Observer import ObserverBase

OBSERVER_NAME = "observer{}"
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, state_size, nlayers, out_features):
        super(Policy, self).__init__()
        in_features = reduce((lambda x, y: x*y), state_size)

        self.model = nn.Sequential(
            nn.Flatten(0, -1),
            nn.Linear(in_features, out_features),
            *[nn.Linear(out_features, out_features) for _ in range(nlayers)]
        )
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.model(x)
        return F.softmax(action_scores, dim=0)


class AgentBase:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.running_reward = 0
        self.eps = 1e-7

        self.ob_rrefs = []   # Observer RRef
        self.rewards = {}
        self.saved_log_probs = {}

    def set_world(self, world_size, state_size, nlayers, out_features):
        self.policy = Policy(state_size, nlayers, out_features)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.world_size = world_size
        for rank in range(2, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank))
            self.ob_rrefs.append(remote(ob_info, ObserverBase))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, observer_id, state, reward):
        self.rewards[observer_id].append(reward)

        state = state.float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs[observer_id].append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + \
            (1 - 0.05) * self.running_reward

        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return min_reward
