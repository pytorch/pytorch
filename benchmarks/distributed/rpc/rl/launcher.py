import argparse
import os

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import time

from coordinator import CoordinatorBase

COORDINATOR_NAME = "coordinator"
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

TOTAL_EPISODES = 10
TOTAL_EPISODE_STEPS = 100


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch RPC RL Benchmark')
parser.add_argument('--world_size', type=int, default=5)
parser.add_argument('--master_addr', type=str, default='127.0.0.1')
parser.add_argument('--master_port', type=str, default='29501')
parser.add_argument('--batch', type=str2bool, default=True)

parser.add_argument('--state_size', type=str, default='10,20,10')
parser.add_argument('--nlayers', type=int, default=5)
parser.add_argument('--out_features', type=int, default=10)

args = parser.parse_args()


def run_worker(rank, world_size, master_addr, master_port, batch, state_size, nlayers, out_features):
    state_size = list(map(int, state_size.split(',')))
    batch_size = world_size - 2  # No. of observers

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    if rank == 0:
        rpc.init_rpc(COORDINATOR_NAME, rank=rank, world_size=world_size)

        coordinator = CoordinatorBase(
            batch_size, batch, state_size, nlayers, out_features)
        coordinator.run_coordinator(TOTAL_EPISODES, TOTAL_EPISODE_STEPS)

    elif rank == 1:
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)
    else:
        rpc.init_rpc(OBSERVER_NAME.format(rank),
                     rank=rank, world_size=world_size)

    rpc.shutdown()


def main():
    # for world_size in range(12, 13):
        # delays = []
        # for batch in [True, False]:
        #     print(world_size, batch, "==>\n")
        #     tik = time.time()
        #     mp.spawn(
        #         run_worker,
        #         args=(world_size, args.master_addr, args.master_port,
        #               batch, args.state_size, args.nlayers, args.out_features),
        #         nprocs=world_size,
        #         join=True
        #     )
        #     tok = time.time()
        #     delays.append(tok - tik)
        # print(f"Time taken - {world_size}, {delays[0]}, {delays[1]}")
    batch = True
    world_size = 12
    tik = time.time()
    mp.spawn(
        run_worker,
        args=(world_size, args.master_addr, args.master_port,
              batch, args.state_size, args.nlayers, args.out_features),
        nprocs=world_size,
        join=True
    )
    tok = time.time()
    delay = tok - tik
    print(f"Time taken - worldsize: {world_size}, delay: {delay}")


if __name__ == '__main__':
    main()
