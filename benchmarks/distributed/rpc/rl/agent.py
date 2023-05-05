from functools import reduce
import time
import threading

import torch
from torch.distributions import Categorical
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


OBSERVER_NAME = "observer{}"


class Policy(nn.Module):
    def __init__(self, in_features, nlayers, out_features):
        r"""
        Inits policy class
        Args:
            in_features (int): Number of input features the model takes
            nlayers (int): Number of layers in the model
            out_features (int): Number of features the model outputs
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(in_features, out_features),
            * [nn.Linear(out_features, out_features) for _ in range(nlayers)]
        )
        self.dim = 0

    def forward(self, x):
        action_scores = self.model(x)
        return F.softmax(action_scores, dim=self.dim)


class AgentBase:
    def __init__(self):
        r"""
        Inits agent class
        """
        self.id = rpc.get_worker_info().id
        self.running_reward = 0
        self.eps = 1e-7

        self.rewards = {}

        self.future_actions = torch.futures.Future()
        self.lock = threading.Lock()

        self.agent_latency_start = None
        self.agent_latency_end = None
        self.agent_latency = []
        self.agent_throughput = []

    def reset_metrics(self):
        r"""
        Sets all benchmark metrics to their empty values
        """
        self.agent_latency_start = None
        self.agent_latency_end = None
        self.agent_latency = []
        self.agent_throughput = []

    def set_world(self, batch_size, state_size, nlayers, out_features, batch=True):
        r"""
        Further initializes agent to be aware of rpc environment
        Args:
            batch_size (int): size of batches of observer requests to process
            state_size (list): List of ints dictating the dimensions of the state
            nlayers (int): Number of layers in the model
            out_features (int): Number of out features in the model
            batch (bool): Whether to process and respond to observer requests as a batch or 1 at a time
        """
        self.batch = batch
        self.policy = Policy(reduce((lambda x, y: x * y), state_size), nlayers, out_features)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.batch_size = batch_size
        for rank in range(batch_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank + 2))

            self.rewards[ob_info.id] = []

        self.saved_log_probs = [] if self.batch else {
            k: [] for k in range(self.batch_size)}

        self.pending_states = self.batch_size
        self.state_size = state_size
        self.states = torch.zeros(self.batch_size, *state_size)

    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, observer_id, state):
        r"""
        Receives state from an observer to select action for.  Queues the observers's request
        for an action until queue size equals batch size named during Agent initiation, at which point
        actions are selected for all pending observer requests and communicated back to observers
        Args:
            agent_rref (RRef): RRFef of this agent
            observer_id (int): Observer id of observer calling this function
            state (Tensor): Tensor representing current state held by observer
        """
        self = agent_rref.local_value()
        observer_id -= 2

        self.states[observer_id].copy_(state)
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[observer_id].item()
        )

        with self.lock:
            if self.pending_states == self.batch_size:
                self.agent_latency_start = time.time()
            self.pending_states -= 1
            if self.pending_states == 0:
                self.pending_states = self.batch_size
                probs = self.policy(self.states)
                m = Categorical(probs)
                actions = m.sample()
                self.saved_log_probs.append(m.log_prob(actions).t())
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(actions)

                self.agent_latency_end = time.time()

                batch_latency = self.agent_latency_end - self.agent_latency_start
                self.agent_latency.append(batch_latency)
                self.agent_throughput.append(self.batch_size / batch_latency)

        return future_action

    @staticmethod
    def select_action_non_batch(agent_rref, observer_id, state):
        r"""
        Select actions based on observer state and communicates back to observer
        Args:
            agent_rref (RRef): RRef of this agent
            observer_id (int): Observer id of observer calling this function
            state (Tensor): Tensor representing current state held by observer
        """
        self = agent_rref.local_value()
        observer_id -= 2
        agent_latency_start = time.time()

        state = state.float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[observer_id].append(m.log_prob(action))

        agent_latency_end = time.time()
        non_batch_latency = agent_latency_end - agent_latency_start
        self.agent_latency.append(non_batch_latency)
        self.agent_throughput.append(1 / non_batch_latency)

        return action.item()

    def finish_episode(self, rets):
        r"""
        Finishes the episode
        Args:
            rets (list): List containing rewards generated by selct action calls during
            episode run
        """
        return self.agent_latency, self.agent_throughput
