import argparse
import os
import pdb

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import json
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

def run_worker(rank, world_size, master_addr, master_port, batch, state_size, nlayers, out_features, queue=None):
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
    GRAPH_VARIABLES = {
        'world_size':[10,20,40], 
        'batch': [True, False], 
        'state_size': ['10,20,10', '15,20,25'], 
        'nlayers': [5, 10], 
        'out_features': [10, 20]}
    if args['graph_variable'] in GRAPH_VARIABLES.keys():
        x_axis_name = args['graph_variable']
        x_axis_variables = GRAPH_VARIABLES[x_axis_name]
        ctx = mp.get_context('spawn')
        queue = ctx.SimpleQueue()
        benchmark_runs = []

        for i, x_axis_variable in enumerate(x_axis_variables): #run benchmark for every x axis variable
            args[x_axis_name] = x_axis_variable #set x axis variable for this iteration of benchmark run
            processes = []
            for rank in range(args['world_size']):
                prc = ctx.Process(
                    target=run_worker, 
                    args=(
                        rank, args['world_size'], args['master_addr'], args['master_port'],
                        args['batch'], args['state_size'], args['nlayers'], 
                        args['out_features'], queue
                        )
                )
                prc.start()
                processes.append(prc)
            benchmark_run_results = queue.get()   
            for process in processes:
                process.join()
            benchmark_run_results[x_axis_name] = x_axis_variable #save what the x axis value was for this benchmark run       
            benchmark_runs.append(benchmark_run_results)

        report = args
        report['x_axis_name'] = x_axis_name
        del report[x_axis_name] #x_axis_name was variable so dont save a constant in the report
        report['benchmark_results'] = benchmark_runs
        with open('report.json', 'w') as f:
            json.dump(report, f)
    else:
        start_time = time.time()
        mp.spawn(
            run_worker,
            args=(
                args['world_size'], args['master_addr'], args['master_port'],
                  args['batch'], args['state_size'], args['nlayers'], 
                  args['out_features']
            ),
            nprocs=args['world_size'],
            join=True
        )
        print(f"Time taken -, {time.time() - start_time}")

if __name__ == '__main__':
    main()
