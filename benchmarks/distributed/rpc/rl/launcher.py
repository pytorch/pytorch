import argparse
import os

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from Coordinator import CoordinatorBase

COORDINATOR_NAME = "coordinator"
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

TOTAL_EPISODES = 500
TOTAL_EPISODE_STEPS = 1000

parser = argparse.ArgumentParser(description='PyTorch RPC RL Benchmark')
parser.add_argument('--world_size', type=int, default=3)
parser.add_argument('--master_addr', type=str, default='127.0.0.1')
parser.add_argument('--master_port', type=str, default='29501')
args = parser.parse_args()


def run_worker(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    if rank == 0:
        rpc.init_rpc(COORDINATOR_NAME, rank=rank, world_size=world_size)

        coordinator = CoordinatorBase(world_size)
        coordinator.run_coordinator(TOTAL_EPISODES, TOTAL_EPISODE_STEPS)

    elif rank == 1:
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)
    else:
        rpc.init_rpc(OBSERVER_NAME.format(rank),
                     rank=rank, world_size=world_size)

    rpc.shutdown()


def main():
    mp.spawn(
        run_worker,
        args=(args.world_size, args.master_addr, args.master_port,),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()
