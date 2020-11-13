from functools import reduce
import time
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote
from torch.distributions import Categorical

OBSERVER_NAME = "observer{}"
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, state_size, nlayers, out_features, batch=True):
        super(Policy, self).__init__()
        self.in_features = reduce((lambda x, y: x*y), state_size)

        self.model = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(self.in_features, out_features),
            * [nn.Linear(out_features, out_features) for _ in range(nlayers)]
        )
        self.dim = 0

    def forward(self, x):
        action_scores = self.model(x)
        ret = F.softmax(action_scores, dim=self.dim)
        return ret


class AgentBase:
    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id
        self.running_reward = 0
        self.eps = 1e-7

        self.ob_rrefs = []   # Observer RRef
        self.rewards = {}

        self.future_actions = torch.futures.Future()
        self.lock = threading.Lock()

    def set_world(self, world_size, state_size, nlayers, out_features, batch=True):
        from observer import ObserverBase

        self.batch = batch
        self.policy = Policy(state_size, nlayers, out_features, self.batch)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.world_size = world_size
        for rank in range(2, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank))
            self.ob_rrefs.append(
                remote(ob_info, ObserverBase))
            self.rewards[ob_info.id] = []

        self.saved_log_probs = [] if self.batch else {
            k: [] for k in range(self.world_size - 2)}

        self.pending_states = self.world_size - 2
        self.state_size = state_size
        self.states = torch.zeros(self.world_size - 2, *state_size)

    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, observer_id, state):
        self = agent_rref.local_value()
        observer_id -= 2

        if self.pending_states == self.world_size - 2:
            agent_latency_start = time.time()

        self.states[observer_id].copy_(state)
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[observer_id].item()
        )

        with self.lock:
            self.pending_states -= 1
            if self.pending_states == 0:
                self.pending_states = self.world_size - 2
                probs = self.policy(self.states)
                m = Categorical(probs)
                actions = m.sample()
                self.saved_log_probs.append(m.log_prob(actions).t())
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(actions)

        return future_action

    @staticmethod
    def select_action_non_batch(agent_rref, observer_id, state):
        self = agent_rref.local_value()
        observer_id -= 2
        # self.rewards[observer_id].append(reward)

        state = state.float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs[observer_id].append(m.log_prob(action))
        return action.item()

    def finish_episode(self, rets):
        rewards = torch.stack([ret[0] for ret in rets]).t()
        # ep_rewards = sum([ret[1] for ret in rets]) / len(rets)

        if self.batch:
            probs = torch.stack(self.saved_log_probs)
        else:
            probs = [torch.stack(self.saved_log_probs[i])
                     for i in range(len(rets))]
            probs = torch.stack(probs)

        policy_loss = -probs * rewards / len(rets)
        policy_loss.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # reset variables
        self.saved_log_probs = [] if self.batch else {
            k: [] for k in range(self.world_size - 2)}
        self.states = torch.zeros(self.world_size - 2, *self.state_size)

        return None

        # calculate running rewards
        # self.running_reward = 0.5 * ep_rewards + 0.5 * self.running_reward
        # return ep_rewards, self.running_reward
