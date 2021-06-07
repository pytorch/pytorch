# This file provides an experiment skeleton for examining the ZeRO optimizer
# state partitioning and for measuring the optimizer.step() latency.

# Utility Imports #
import os
import json
import csv
import argparse
import time
import numpy as np

# Torch Imports #
import torch
import torchvision
import transformers
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

# Warning Suppression #
import warnings
from transformers import logging
# Ignoring UserWarning warnings suppresses warnings about the experimental
# nature of named tensors, which are used by the ResNet implementation.
warnings.filterwarnings("ignore", category=UserWarning)
# Setting the verbosity to error suppresses warnings for unused weights at
# initialization for BERT.
logging.set_verbosity_error()

# Global Variables #
BATCH_SIZE = 32
NUM_STEP_ITERS = 30


# Model Output Functions #
# These must be defined at the top level to be pickle-able by the multi-
# processing library.
def resnet_output(rank, ddp_model):
    r"""
    Performs a forward pass with a random input through the ResNet model.

    Arguments:
        rank (int): process rank.
        ddp_model (DDP): ResNet DDP model.
    """
    return ddp_model(torch.randn(BATCH_SIZE, 3, 224, 224).to(rank))


def bert_output(rank, ddp_model):
    r"""
    Performs a forward pass with a random input through the BERT model.

    Arguments:
        rank (int): process rank.
        ddp_model (DDP): BERT DDP model.
    Returns:

    """
    return ddp_model(torch.randint(1, 10, (BATCH_SIZE, 100)).to(rank))[0]


# Experiment Setup #
class BenchmarkModels():
    r"""
    Stores a list containing the models to benchmark and their relevant
    information.

    The format per element is as follows:
        (model_name, output_fn, labels, loss_fn, find_unused_parameters)
    where:
        model_name (str): human-readable name of the model.
        output_fn (int * DDP -> Tensor): function mapping a (rank, ddp_model)
            input to a tensor output representing the forward pass.
        labels (Tensor): tensor representing example output labels.
        loss_fn (_Loss): loss function.
        find_unused_parameters (bool): True to find unused parameters; False
            otherwise; set as True if some parameters may not contribute to
            the gradient.
    """
    models = [
        (
            "ResNet50",
            torchvision.models.resnet50(pretrained=False),
            resnet_output,
            torch.randn(BATCH_SIZE, 1000),
            nn.MSELoss(),
            False,
        ), (
            "ResNet152",
            torchvision.models.resnet152(pretrained=False),
            resnet_output,
            torch.randn(BATCH_SIZE, 1000),
            nn.MSELoss(),
            False,
        ), (
            "BERT",
            transformers.BertModel.from_pretrained("bert-base-uncased"),
            bert_output,
            torch.randint(2, (BATCH_SIZE, 768)),
            nn.CrossEntropyLoss(),
            True,
        ),
    ]


def dump_experiment_info(world_sizes, models, outdir):
    r"""
    Dumps the experiment info to a .json file named "exp_info.json".

    Arguments:
        world_sizes (List[int]): list of world sizes.
        models (List[(str, int * DDP -> tensor, Tensor, _Loss, bool)]): a
            list of models as described in `get_benchmark_models()`.
        outdir (str): dirname to dump to.
    """
    experiment_info = dict()
    experiment_info["world_sizes"] = world_sizes
    experiment_info["model_names"] = [model_entry[0] for model_entry in models]
    outfile = os.path.join(outdir, "exp_info.json")
    with open(outfile, 'w') as f:
        json.dump(experiment_info, f)


# Worker Helpers #
def get_parameter_sizes(rank, partitions):
    r"""
    Gets a list of parameter sizes for the given rank.

    Arguments:
        rank (int): rank to get parameter sizes for.
        partitions (List[List[dict]]): parameter partitions returned by
            partition_parameters().
    """
    sizes = list()
    param_groups = partitions[rank]
    for param_group in param_groups:
        sizes.extend([param.numel() for param in param_group["params"]])
    return sizes


def worker_setup(rank, backend, world_size):
    r"""
    Sets up the environment and initializes the process group.

    Arguments:
        rank (int): process rank.
        backend (str): name of the backend (e.g. "nccl", "gloo", etc.).
        world_size (int): world size.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


# Worker #
def worker(rank, backend, world_size, model_name, model, outputs_fn, labels,
           loss_fn, find_unused_parameters, outdir):
    r"""
    Finds the number of parameters in this process's partition of the
    optimizer state and times the optimizer.step() latency.

    Arguments:
        rank (int): process rank.
        backend (str): name of the backend (e.g. "nccl", "gloo", etc.).
        world_size (int): world size.
        model_name (str): human-readable name of the model.
        output_fn (int * DDP -> Tensor): function mapping a (rank, ddp_model)
            input to a tensor output representing the forward pass.
        labels (Tensor): tensor representing example output labels.
        loss_fn (_Loss): loss function.
        find_unused_parameters (bool): True to find unused parameters; False
            otherwise; set as True if some parameters may not contribute to
            the gradient.
        outdir (str): dirname to write the .csv output to.
    """
    # set up environment and initialize process group
    worker_setup(rank, backend, world_size)

    # create DDP model
    ddp_model = DDP(model.to(rank), device_ids=[rank],
                    find_unused_parameters=find_unused_parameters)

    # define optimizer
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=0.01
    )

    # time the optimizer step
    optimizer_step_times = list()
    labels = labels.to(rank)
    for _ in range(NUM_STEP_ITERS):
        # perform forward/backward passes
        outputs = outputs_fn(rank, ddp_model)
        loss_fn(outputs, labels).backward()

        # perform optimizer step
        optimizer.zero_grad()
        start = time.time()
        optimizer.step()
        end = time.time()
        optimizer_step_time = end - start
        optimizer_step_times.append(optimizer_step_time)

    # ignore the first iteration's time (due to cache coldness)
    optimizer_step_times.pop(0)
    optimizer_step_time = np.mean(optimizer_step_times)
    optimizer_step_time_std = np.std(optimizer_step_times)

    # find the process's parameter count
    proc_param_sizes = get_parameter_sizes(rank,
                                           optimizer.partition_parameters())
    proc_num_params = sum(proc_param_sizes)

    # output results
    print("rank=%d: optimizer.step() took %.3f s" %
          (rank, optimizer_step_time))
    print("rank=%d:" % rank, proc_num_params, "parameters")

    outfile = os.path.join(outdir, "%s_%d_%d.csv" %
                           (model_name, world_size, rank))
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([model_name, rank, proc_num_params,
                         optimizer_step_time, optimizer_step_time_std])


# Main #
def main():
    r"""
    Defines the wrapper for launching the DDP processes and running the
    experiment.

    An example run command using NCCL backend, an output directory of
    "greedy/", and world sizes of 2 and 4:
        python zero.py -b nccl greedy 2 4
    """
    # parse arguments
    parser = argparse.ArgumentParser(description="PyTorch ZeRO experiment")
    parser.add_argument("-b", "--backend", metavar="backend", default="nccl",
                        type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("world_sizes", nargs='+', type=int)
    args = parser.parse_args()

    # check GPU availability
    num_gpus = torch.cuda.device_count()
    max_world_size = max(args.world_sizes)
    assert num_gpus >= max_world_size, \
        "Insufficient GPUs to support a world size of %d" % max_world_size

    # save relevant info
    models = BenchmarkModels.models
    if (not os.path.exists(args.outdir)):
        os.makedirs(args.outdir)
    dump_experiment_info(args.world_sizes, models, args.outdir)

    # print version info
    print("-----------------------------------")
    print("PyTorch ZeRO Experiment")
    print("-----------------------------------")
    print()
    print("* PyTorch version: {}".format(torch.__version__))
    print("* CUDA version: {}".format(torch.version.cuda))
    print("* Distributed backend: {}".format(args.backend))
    print()

    # run experiment
    for world_size in args.world_sizes:
        for model_entry in models:
            (model_name, model, output_fn, labels, loss_fn,
                find_unused_parameters) = model_entry
            print("world_size=%d model=%s" % (world_size, model_name))
            mp.spawn(worker,
                     args=(args.backend, world_size, model_name, model,
                           output_fn, labels, loss_fn, find_unused_parameters,
                           args.outdir),
                     nprocs=world_size,
                     join=True)
            print()


if __name__ == "__main__":
    main()
