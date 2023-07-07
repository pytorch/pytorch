import argparse
import os
import time

import json
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp


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
parser.add_argument('--world-size', '--world_size', type=str, default='10')
parser.add_argument('--master-addr', '--master_addr', type=str, default='127.0.0.1')
parser.add_argument('--master-port', '--master_port', type=str, default='29501')
parser.add_argument('--batch', type=str, default='True')

parser.add_argument('--state-size', '--state_size', type=str, default='10-20-10')
parser.add_argument('--nlayers', type=str, default='5')
parser.add_argument('--out-features', '--out_features', type=str, default='10')
parser.add_argument('--output-file-path', '--output_file_path', type=str, default='benchmark_report.json')

args = parser.parse_args()
args = vars(args)

def run_worker(rank, world_size, master_addr, master_port, batch, state_size, nlayers, out_features, queue):
    r"""
    inits an rpc worker
    Args:
        rank (int): Rpc rank of worker machine
        world_size (int): Number of workers in rpc network (number of observers +
                          1 agent + 1 coordinator)
        master_addr (str): Master address of cooridator
        master_port (str): Master port of coordinator
        batch (bool): Whether agent will use batching or process one observer
                      request a at a time
        state_size (str): Numerical str representing state dimensions (ie: 5-15-10)
        nlayers (int): Number of layers in model
        out_features (int): Number of out features in model
        queue (SimpleQueue): SimpleQueue from torch.multiprocessing.get_context() for
                             saving benchmark run results to
    """
    state_size = list(map(int, state_size.split('-')))
    batch_size = world_size - 2  # No. of observers

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
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

def find_graph_variable(args):
    r"""
    Determines if user specified multiple entries for a single argument, in which case
    benchmark is run for each of these entries.  Comma separated values in a given argument indicate multiple entries.
    Output is presented so that user can use plot repo to plot the results with each of the
    variable argument's entries on the x-axis. Args is modified in accordance with this.
    More than 1 argument with multiple entries is not permitted.
    Args:
        args (dict): Dictionary containing arguments passed by the user (and default arguments)
    """
    var_types = {'world_size': int,
                 'state_size': str,
                 'nlayers': int,
                 'out_features': int,
                 'batch': str2bool}
    for arg in var_types.keys():
        if ',' in args[arg]:
            if args.get('x_axis_name'):
                raise ValueError("Only 1 x axis graph variable allowed")
            args[arg] = list(map(var_types[arg], args[arg].split(',')))  # convert , separated str to list
            args['x_axis_name'] = arg
        else:
            args[arg] = var_types[arg](args[arg])  # convert string to proper type

def append_spaces(string, length):
    r"""
    Returns a modified string with spaces appended to the end.  If length of string argument
    is greater than or equal to length, a single space is appended, otherwise x spaces are appended
    where x is the difference between the length of string and the length argument
    Args:
        string (str): String to be modified
        length (int): Size of desired return string with spaces appended
    Return: (str)
    """
    string = str(string)
    offset = length - len(string)
    if offset <= 0:
        offset = 1
    string += ' ' * offset
    return string

def print_benchmark_results(report):
    r"""
    Prints benchmark results
    Args:
        report (dict): JSON formatted dictionary containing relevant data on the run of this application
    """
    print("--------------------------------------------------------------")
    print("PyTorch distributed rpc benchmark reinforcement learning suite")
    print("--------------------------------------------------------------")
    for key, val in report.items():
        if key != "benchmark_results":
            print(f'{key} : {val}')

    x_axis_name = report.get('x_axis_name')
    col_width = 7
    heading = ""
    if x_axis_name:
        x_axis_output_label = f'{x_axis_name} |'
        heading += append_spaces(x_axis_output_label, col_width)
    metric_headers = ['agent latency (seconds)', 'agent throughput',
                      'observer latency (seconds)', 'observer throughput']
    percentile_subheaders = ['p50', 'p75', 'p90', 'p95']
    subheading = ""
    if x_axis_name:
        subheading += append_spaces(' ' * (len(x_axis_output_label) - 1), col_width)
    for header in metric_headers:
        heading += append_spaces(header, col_width * len(percentile_subheaders))
        for percentile in percentile_subheaders:
            subheading += append_spaces(percentile, col_width)
    print(heading)
    print(subheading)

    for benchmark_run in report['benchmark_results']:
        run_results = ""
        if x_axis_name:
            run_results += append_spaces(benchmark_run[x_axis_name], max(col_width, len(x_axis_output_label)))
        for metric_name in metric_headers:
            percentile_results = benchmark_run[metric_name]
            for percentile in percentile_subheaders:
                run_results += append_spaces(percentile_results[percentile], col_width)
        print(run_results)

def main():
    r"""
    Runs rpc benchmark once if no argument has multiple entries, and otherwise once for each of the multiple entries.
    Multiple entries is indicated by comma separated values, and may only be done for a single argument.
    Results are printed as well as saved to output file.  In case of multiple entries for a single argument,
    the plot repo can be used to benchmark results on the y axis with each entry on the x axis.
    """
    find_graph_variable(args)

    # run once if no x axis variables
    x_axis_variables = args[args['x_axis_name']] if args.get('x_axis_name') else [None]
    ctx = mp.get_context('spawn')
    queue = ctx.SimpleQueue()
    benchmark_runs = []
    for i, x_axis_variable in enumerate(x_axis_variables):  # run benchmark for every x axis variable
        if len(x_axis_variables) > 1:
            args[args['x_axis_name']] = x_axis_variable  # set x axis variable for this benchmark iteration
        processes = []
        start_time = time.time()
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
        print(f"Time taken benchmark run {i} -, {time.time() - start_time}")
        if args.get('x_axis_name'):
            # save x axis value was for this iteration in the results
            benchmark_run_results[args['x_axis_name']] = x_axis_variable
        benchmark_runs.append(benchmark_run_results)

    report = args
    report['benchmark_results'] = benchmark_runs
    if args.get('x_axis_name'):
        # x_axis_name was variable so dont save a constant in the report for that variable
        del report[args['x_axis_name']]
    with open(args['output_file_path'], 'w') as f:
        json.dump(report, f)
    print_benchmark_results(report)

if __name__ == '__main__':
    main()
