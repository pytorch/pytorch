import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_async, rpc_sync, remote

from agent import AgentBase
from observer import ObserverBase

import time
import numpy as np

COORDINATOR_NAME = "coordinator"
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"


class CoordinatorBase:
    def __init__(self, world_size, batch, state_size, nlayers, out_features):
        self.world_size = world_size
        self.batch = batch

        self.agent_rref = None  # Agent RRef
        self.ob_rrefs = []   # Observer RRef

        agent_info = rpc.get_worker_info(AGENT_NAME)
        self.agent_rref = remote(agent_info, AgentBase)

        for rank in range(2, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank))
            ob_ref = remote(ob_info, ObserverBase)
            self.ob_rrefs.append(ob_ref)

            ob_ref.rpc_sync().set_state(state_size, batch)

        self.agent_rref.rpc_sync().set_world(
            world_size, state_size, nlayers, out_features, self.batch)

    def run_coordinator(self, episodes, episode_steps):
        for ep in range(episodes):
            ep_start_time = time.time()

            print(f"Episode {ep} - ", end='')
            n_steps = int(episode_steps / (self.world_size - 2))

            agent_start_time = time.time()

            futs = []
            for ob_rref in self.ob_rrefs:
                futs.append(ob_rref.rpc_async().run_ob_episode(
                    self.agent_rref, n_steps))

            rets = torch.futures.wait_all(futs)
            print("After waiting")
            self.agent_rref.rpc_sync().finish_episode(rets)

            ep_end_time = time.time()
            episode_time = ep_end_time - ep_start_time
            print(episode_time)
