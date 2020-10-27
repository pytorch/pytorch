import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_async, rpc_sync, remote

from Agent import AgentBase
from Observer import ObserverBase

import copy
import time

COORDINATOR_NAME = "coordinator"
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"


def call_method(method, rref, *args, **kwargs):
    # a helper function to call a method on the given RRef
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    # a helper function to run method on the owner of rref and fetch back the result using RPC
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


class CoordinatorBase:
    def __init__(self, world_size):
        self.world_size = world_size
        self.agent_rref = None  # Agent RRef
        self.ob_rrefs = []   # Observer RRef
        self.rewards = {}
        self.saved_log_probs = {}

        agent_info = rpc.get_worker_info(AGENT_NAME)
        self.agent_rref = remote(agent_info, AgentBase)

        for rank in range(2, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank))
            self.ob_rrefs.append(remote(ob_info, ObserverBase))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def run_episode(self, n_steps):
        # Run 1 episode, n_steps will be executed in each episode run
        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    call_method,
                    args=(ObserverBase.run_episode,
                          ob_rref, self.agent_rref, n_steps)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def run_coordinator(self, episodes, episode_steps):
        # ob_rrefs = self.ob_rrefs
        # rewards = self.rewards
        # saved_log_probs = self.saved_log_probs

        benchmarks = []
        agent = AgentBase()

        for ep in range(episodes):
            start_time = time.time()

            print("Episode ", ep)
            n_steps = int(episode_steps / (self.world_size - 2))
            self.run_episode(n_steps)

            agent.finish_episode()

            end_time = time.time()
            benchmarks.append(end_time - start_time)
