import time

import numpy as np
from agent import AgentBase
from observer import ObserverBase

import torch
import torch.distributed.rpc as rpc


COORDINATOR_NAME = "coordinator"
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

EPISODE_STEPS = 100


class CoordinatorBase:
    def __init__(self, batch_size, batch, state_size, nlayers, out_features):
        r"""
        Coordinator object to run on worker.  Only one coordinator exists.  Responsible
        for facilitating communication between agent and observers and recording benchmark
        throughput and latency data.
        Args:
            batch_size (int): Number of observer requests to process in a batch
            batch (bool): Whether to process and respond to observer requests as a batch or 1 at a time
            state_size (list): List of ints dictating the dimensions of the state
            nlayers (int): Number of layers in the model
            out_features (int): Number of out features in the model
        """
        self.batch_size = batch_size
        self.batch = batch

        self.agent_rref = None  # Agent RRef
        self.ob_rrefs = []  # Observer RRef

        agent_info = rpc.get_worker_info(AGENT_NAME)
        self.agent_rref = rpc.remote(agent_info, AgentBase)

        for rank in range(batch_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(rank + 2))
            ob_ref = rpc.remote(ob_info, ObserverBase)
            self.ob_rrefs.append(ob_ref)

            ob_ref.rpc_sync().set_state(state_size, batch)

        self.agent_rref.rpc_sync().set_world(
            batch_size, state_size, nlayers, out_features, self.batch
        )

    def run_coordinator(self, episodes, episode_steps, queue):
        r"""
        Runs n benchmark episodes.  Each episode is started by coordinator telling each
        observer to contact the agent.  Each episode is concluded by coordinator telling agent
        to finish the episode, and then the coordinator records benchmark data
        Args:
            episodes (int): Number of episodes to run
            episode_steps (int): Number steps to be run in each episdoe by each observer
            queue (SimpleQueue): SimpleQueue from torch.multiprocessing.get_context() for
                                 saving benchmark run results to
        """

        agent_latency_final = []
        agent_throughput_final = []

        observer_latency_final = []
        observer_throughput_final = []

        for ep in range(episodes):
            ep_start_time = time.time()

            print(f"Episode {ep} - ", end="")

            n_steps = episode_steps

            futs = []
            for ob_rref in self.ob_rrefs:
                futs.append(
                    ob_rref.rpc_async().run_ob_episode(self.agent_rref, n_steps)
                )

            rets = torch.futures.wait_all(futs)
            agent_latency, agent_throughput = self.agent_rref.rpc_sync().finish_episode(
                rets
            )

            self.agent_rref.rpc_sync().reset_metrics()

            agent_latency_final += agent_latency
            agent_throughput_final += agent_throughput

            observer_latency_final += [ret[2] for ret in rets]
            observer_throughput_final += [ret[3] for ret in rets]

            ep_end_time = time.time()
            episode_time = ep_end_time - ep_start_time
            print(round(episode_time, 3))

        observer_latency_final = [t for s in observer_latency_final for t in s]
        observer_throughput_final = [t for s in observer_throughput_final for t in s]

        benchmark_metrics = {
            "agent latency (seconds)": {},
            "agent throughput": {},
            "observer latency (seconds)": {},
            "observer throughput": {},
        }

        print(f"For batch size {self.batch_size}")
        print("\nAgent Latency - ", len(agent_latency_final))
        agent_latency_final = sorted(agent_latency_final)
        for p in [50, 75, 90, 95]:
            v = np.percentile(agent_latency_final, p)
            print("p" + str(p) + ":", round(v, 3))
            p = f"p{p}"
            benchmark_metrics["agent latency (seconds)"][p] = round(v, 3)

        print("\nAgent Throughput - ", len(agent_throughput_final))
        agent_throughput_final = sorted(agent_throughput_final)
        for p in [50, 75, 90, 95]:
            v = np.percentile(agent_throughput_final, p)
            print("p" + str(p) + ":", int(v))
            p = f"p{p}"
            benchmark_metrics["agent throughput"][p] = int(v)

        print("\nObserver Latency - ", len(observer_latency_final))
        observer_latency_final = sorted(observer_latency_final)
        for p in [50, 75, 90, 95]:
            v = np.percentile(observer_latency_final, p)
            print("p" + str(p) + ":", round(v, 3))
            p = f"p{p}"
            benchmark_metrics["observer latency (seconds)"][p] = round(v, 3)

        print("\nObserver Throughput - ", len(observer_throughput_final))
        observer_throughput_final = sorted(observer_throughput_final)
        for p in [50, 75, 90, 95]:
            v = np.percentile(observer_throughput_final, p)
            print("p" + str(p) + ":", int(v))
            p = f"p{p}"
            benchmark_metrics["observer throughput"][p] = int(v)

        if queue:
            queue.put(benchmark_metrics)
