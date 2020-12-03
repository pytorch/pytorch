import argparse
import os
import pdb

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

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
parser.add_argument('--graph_variable', type=str, default='world_size')

args = parser.parse_args()
args = vars(args)


def run_worker(rank, world_size, master_addr, master_port, batch, state_size, nlayers, out_features, queue):
    state_size = list(map(int, state_size.split(',')))
    batch_size = world_size - 2  # No. of observers

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print("running run worker")
    if rank == 0:
        rpc.init_rpc(COORDINATOR_NAME, rank=rank, world_size=world_size)

        coordinator = CoordinatorBase(
            batch_size, batch, state_size, nlayers, out_features)
        coordinator.run_coordinator(TOTAL_EPISODES, TOTAL_EPISODE_STEPS, queue)


    elif rank == 1:
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)
    else:
        rpc.init_rpc(OBSERVER_NAME.format(rank),
                     rank=rank, world_size=world_size)
    rpc.shutdown()


def main():
    GRAPH_VARIABLES = {'world_size':[7,12]} #[12,22,42,62,122,242]
    if args['graph_variable'] in GRAPH_VARIABLES.keys():
        graph_variables = GRAPH_VARIABLES[args['graph_variable']]
        ctx = mp.get_context('spawn')
        queue = ctx.SimpleQueue()
        returns = {}
        for i, graph_variable in enumerate(graph_variables): #x axis variable
            args[args['graph_variable']] = graph_variable
            print('starting process {0}'.format(i ))            
            processes = []
            for rank in range(args['world_size']):
                prc = ctx.Process(target=run_worker, args=(rank, args['world_size'], args['master_addr'], args['master_port'],
                args['batch'], args['state_size'], args['nlayers'], args['out_features'], queue))
                prc.start()
                processes.append(prc)
            benchmark_metrics = queue.get()   
            for process in processes:
                process.join()         
            returns[i] = benchmark_metrics
            print("finished process {0}, ret cxt is: {1}".format(i, returns))

        width = 0.35  # the width of the bars
        labels = graph_variables
        label_location = np.arange(len(labels))

        for i, benchmark_metric in enumerate(returns[0].keys()):
            fig, ax = plt.subplots()
            p50s = []
            p95s = []
            for i in range(len(graph_variables)):
                p50s.append(returns[i][benchmark_metric][50]) #KeyError: '50'
                p95s.append(returns[i][benchmark_metric][95])

            y1 = ax.bar(label_location - width/2, p50s, width, label='p50')
            y2 = ax.bar(label_location + width/2, p95s, width, label='p95')
            ax.set_ylabel(benchmark_metric)
            ax.set_xlabel(args['graph_variable'])
            ax.set_title('RPC Benchmarks')
            ax.set_xticks(label_location)
            ax.set_xticklabels(labels)
            ax.legend()
            fig.tight_layout()
        plt.grid()
        plt.show()
    # else: #need to add in a queue to satisfy run coordinator
        # for world_size in range(12, 13):
        # delays = []
        # for batch in [True, False]:
        #     print(world_size, batch, "==>\n")
        #     tik = time.time()
        #     mp.spawn(
        #         run_worker,
        #         args=(world_size, args['master_addr'], args['master_port'],
        #               batch, args['state_size'], args['nlayers'], args['out_features']),
        #         nprocs=world_size,
        #         join=True
        #     )
        #     tok = time.time()
        #     delays.append(tok - tik)
        # print(f"Time taken - {world_size}, {delays[0]}, {delays[1]}")
            # mp.spawn(
            #     run_worker,
            #     args=(world_size, args['master_addr'], args['master_port'],
            #           args['batch'], args['state_size'], args['nlayers'], args['out_features'], ret_cxt, i),
            #     nprocs=world_size,
            #     join=True
            # )

if __name__ == '__main__':
    main()
