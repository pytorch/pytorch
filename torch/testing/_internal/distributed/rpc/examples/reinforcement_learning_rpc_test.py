# mypy: ignore-errors

# If you need to modify this file to make this test pass, please also apply same edits accordingly to
# https://github.com/pytorch/examples/blob/master/distributed/rpc/rl/main.py
# and https://pytorch.org/tutorials/intermediate/rpc_tutorial.html

import numpy as np
from itertools import count

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical

from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture

TOTAL_EPISODE_STEP = 5000
GAMMA = 0.1
SEED = 543

def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class Policy(nn.Module):
    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class DummyEnv:
    r"""
    A dummy environment that implements the required subset of the OpenAI gym
    interface. It exists only to avoid a dependency on gym for running the
    tests in this file. It is designed to run for a set max number of iterations,
    returning random states and rewards at each step.
    """
    def __init__(self, state_dim=4, num_iters=10, reward_threshold=475.0):
        self.state_dim = state_dim
        self.num_iters = num_iters
        self.iter = 0
        self.reward_threshold = reward_threshold

    def seed(self, manual_seed):
        torch.manual_seed(manual_seed)

    def reset(self):
        self.iter = 0
        return torch.randn(self.state_dim)

    def step(self, action):
        self.iter += 1
        state = torch.randn(self.state_dim)
        reward = torch.rand(1).item() * self.reward_threshold
        done = self.iter >= self.num_iters
        info = {}
        return state, reward, done, info


class Observer:
    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.
    """
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = DummyEnv()
        self.env.seed(SEED)

    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.
        Arguments:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        state, ep_reward = self.env.reset(), 0
        for step in range(n_steps):
            # send the state to the agent to get an action
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)

            # apply the action to the environment, and get the reward
            state, reward, done, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)

            if done:
                break


class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = DummyEnv().reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(worker_name(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, state):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.

        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """
        probs = self.policy(state.unsqueeze(0))
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        r"""
        Observers call this function to report rewards.
        """
        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each observer to run n_steps.
        """
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)
                )
            )

        # wait until all observers have finished this episode
        for fut in futs:
            fut.wait()

    def finish_episode(self):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """

        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return min_reward


def run_agent(agent, n_steps):
    for i_episode in count(1):
        agent.run_episode(n_steps=n_steps)
        last_reward = agent.finish_episode()

        if agent.running_reward > agent.reward_threshold:
            print(f"Solved! Running reward is now {agent.running_reward}!")
            break


class ReinforcementLearningRpcTest(RpcAgentTestFixture):
    @dist_init(setup_rpc=False)
    def test_rl_rpc(self):
        if self.rank == 0:
            # Rank 0 is the agent.
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
            agent = Agent(self.world_size)
            run_agent(agent, n_steps=int(TOTAL_EPISODE_STEP / (self.world_size - 1)))

            # Ensure training was run. We don't really care about whether the task was learned,
            # since the purpose of the test is to check the API calls.
            self.assertGreater(agent.running_reward, 0.0)
        else:
            # Other ranks are observers that passively wait for instructions from the agent.
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        rpc.shutdown()
