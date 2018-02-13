import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER

import torch

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch Exec is a helper utiliy that "
            "spawns up multiple distributed training processes. The utility "
            "can be used in single-node distributed training that each "
            "distributed process will be operating on a single GPU. (for "
            "well-improved performance reasons). "
            "In this case, this utilily will launch a given number of "
            "processes per node (nproc_per_node) , while this number needs to "
            "be smaller than the number of GPUs (n_gpus) on the current system,"
            " and each process will be operating on a single GPU from GPU 0 to "
            "GPU nproc_per_node - 1. "

            "This utility can be further used for multi-node "
            "distributed training by spawning up multiple processes per node "
            "for well-improved distributed performance as well. This will "
            "especially be benefitial for systems with multiple Infiniband "
            "interfaces since all of them can be utilized for aggregated "
            "communication bandwidth. Please note that this utilty and "
            "multi-process/node distributed single node or multi-node "
            "training currently only supports the NCCL distributed backend. "
            "This utilty helper will require that training script is able to "
            "parse --device=X as an argument since it will be injected by this "
            "utility. "

            "In your training program, other than parsing --device=X as "
            "argument.device, you should also use the following three function "
            "calls: "
            "torch.distributed.init_process_group(backend=\"nccl\", "
            "init_method=\"env://\"), torch.cuda.set_device(arg.device), and "
            "model = torch.nn.parallel.DistributedDataParallel(model, "
            "device_ids=[arg.device]) in order to use this utility.")

    parser.add_argument("--num_node", type=int, default=1,
            help="The number of nodes to use for distributed training")
    parser.add_argument("--rank_node", type=int, default=0,
            help="The rank of the node for multi-node distributed training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
            help="The number of processes to launch on each node")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
            help="Master node (rank 0)'s address, should be either the IP "
            "address or hostname of node 0, for single node, IP can simply be "
            "127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
            help="Master node (rank 0)'s free port that needs to be used for "
            "communciation in distributed training")

    # positional
    parser.add_argument("training_script", type=str,
            help="The full path to the single GPU training program to be "
            "launched in parallel, followed by all the argument to the "
            "training script")
    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    if args.nproc_per_node > num_gpus:
        raise RuntimeError("Found: {} GPUs on host: {} with rank: {}, the "
                           "number of processes per node cannot be greater "
                           "than the number of GPUs availble on the host."
                           .format(num_gpus,
                                   socket.gethostname(),
                                   args.rank_node))

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.num_node

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []
    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.rank_node + local_rank
        current_env["RANK"] = str(dist_rank)

        # spawn the processes
        cmd = ["python",
                args.training_script,
                "--device={}".format(local_rank)] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
