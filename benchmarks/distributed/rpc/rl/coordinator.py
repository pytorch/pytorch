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


class CoordinatorBase:
    def __init__(self, world_size, batch):
        self.world_size = world_size
        self.batch = batch

        self.agent_rref = None  # Agent RRef
        self.ob_rrefs = []   # Observer RRef

        agent_info = rpc.get_worker_info(AGENT_NAME)
        self.agent_rref = remote(agent_info, AgentBase)

        for rank in range(2, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank))
            self.ob_rrefs.append(remote(ob_info, ObserverBase))

        self.agent_rref.rpc_sync().set_world(world_size)

    def run_coordinator(self, episodes, episode_steps):
        benchmarks = []

        for ep in range(episodes):
            start_time = time.time()
            print(f"Episode {ep} - ", end='')
            n_steps = int(episode_steps / (self.world_size - 2))

            if self.batch:
                for ob_rref in self.ob_rrefs:
                    ob_rref.rpc_async().run_ob_episode(self.agent_rref, n_steps).wait()

            else:
                for ob_rref in self.ob_rrefs:
                    ob_rref.rpc_sync().run_ob_episode(self.agent_rref, n_steps)

            self.agent_rref.rpc_sync().finish_episode()

            end_time = time.time()
            episode_time = end_time - start_time

            benchmarks.append(episode_time)
            print(episode_time)
