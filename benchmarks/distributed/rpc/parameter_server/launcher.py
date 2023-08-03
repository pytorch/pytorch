import argparse
import json
import os
from pathlib import Path

from data import data_map
from metrics.ProcessedMetricsPrinter import ProcessedMetricsPrinter
from models import model_map
from server import server_map
from trainer import (
    criterion_map,
    ddp_hook_map,
    ddp_model_map,
    hook_state_map,
    iteration_step_map,
    preprocess_data_map,
    trainer_map,
)

import torch
import torch.distributed as c10d
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.futures import wait_all
from torch.utils.data import DataLoader


def get_name(rank, args):
    r"""
    A function that gets the name for the rank
    argument
    Args:
        rank (int): process number in the world
        args (parser): benchmark configurations
    """
    t_count = args.ntrainer + args.ncudatrainer
    s_count = args.nserver + args.ncudaserver
    if rank < t_count:
        return f"trainer{rank}"
    elif rank < (t_count + s_count):
        return f"server{rank}"
    else:
        return "master"


def get_server_rank(args, rank):
    r"""
    A function that gets the server rank for
    the rank argument.
    Args:
        args (parser): benchmark configurations
        rank (int): trainer rank
    """
    s_offset = args.ntrainer + args.ncudatrainer
    tps = args.ntrainer // args.nserver
    return rank // tps + s_offset


def get_cuda_server_rank(args, rank):
    r"""
    A function that gets the cudaserver rank for
    the rank argument.
    Args:
        args (parser): benchmark configurations
        rank (int): trainer rank
    """
    s_offset = args.ntrainer + args.ncudatrainer + args.nserver
    t_index = rank - args.ntrainer
    ctps = args.ncudatrainer // args.ncudaserver
    return t_index // ctps + s_offset


def get_server_rref(server_rank, args, extra_args):
    r"""
    A function that creates a RRef to the server.
    Args:
        server_rank (int): process number in the world
        args (parser): benchmark configurations
        extra_args (dict): configurations added by the user
    """
    server = server_map[args.server]
    name = get_name(
        server_rank,
        args
    )
    if extra_args is not None:
        server_args = extra_args.values()
    else:
        server_args = []
    if server_rank >= args.ntrainer + args.ncudatrainer + args.nserver:
        trainer_count = args.ncudatrainer / args.ncudaserver
        use_cuda_rpc = True
    else:
        trainer_count = args.ntrainer / args.nserver
        use_cuda_rpc = False
    return rpc.remote(
        name,
        server,
        args=(
            server_rank,
            trainer_count,
            use_cuda_rpc,
            *server_args,
        ),
    )


def run_trainer(
    args, extra_args, data, rank, server_rref
):
    r"""
    A function that runs obtains a trainer instance and calls
    the train method.
    Args:
        args (parser): benchmark configurations
        extra_args (dict): configurations added by the user
        data (list): training samples
        rank (int): process number in the world
        server_rref (dict): a dictionary containing server RRefs
    """
    trainer_class = trainer_map[args.trainer]
    if extra_args is not None:
        trainer_args = extra_args.values()
    else:
        trainer_args = []
    trainer_count = args.ntrainer + args.ncudatrainer
    store = c10d.FileStore(args.filestore, trainer_count)
    if args.backend == "gloo":
        process_group = c10d.ProcessGroupGloo(
            store, rank, trainer_count
        )
    elif args.backend == "nccl":
        process_group = c10d.ProcessGroupNCCL(
            store, rank, trainer_count
        )
    elif args.backend == "multi":
        process_group = c10d.ProcessGroupNCCL(
            store, rank, trainer_count
        )
        if c10d.is_initialized() is False:
            c10d.init_process_group(backend="gloo", rank=rank, world_size=trainer_count)

    model = load_model(args)
    preprocess_data = preprocess_data_map[args.preprocess_data]
    create_criterion = criterion_map[args.create_criterion]
    create_ddp_model = ddp_model_map[args.create_ddp_model]
    iteration_step = iteration_step_map[args.iteration_step]
    hook_state_class = hook_state_map[args.hook_state]
    hook = ddp_hook_map[args.ddp_hook]
    # check if this a cudatrainer
    use_cuda_rpc = rank >= args.ntrainer
    trainer = trainer_class(
        process_group,
        use_cuda_rpc,
        server_rref,
        args.backend,
        args.epochs,
        preprocess_data,
        create_criterion,
        create_ddp_model,
        hook_state_class,
        hook,
        iteration_step,
        *trainer_args
    )
    trainer.train(model, data)
    metrics = trainer.get_metrics()
    return [rank, metrics]


def call_trainers(args, extra_args, train_data, server_rrefs):
    r"""
    A function that starts the trainers. Each trainer is started
    using an rpc_async request.
    Args:
        args (parser): benchmark configurations
        extra_args (dict): configurations added by the user
        train_data (list): training samples
        server_rrefs (dict): a dictionary containing server RRefs
    """
    futs = []
    for trainer_rank in range(0, args.ntrainer + args.ncudatrainer):
        trainer_name = get_name(
            trainer_rank,
            args
        )
        server_rref = None
        if server_rrefs:
            if trainer_rank >= args.ntrainer:
                server_rank = get_cuda_server_rank(args, trainer_rank)
            else:
                server_rank = get_server_rank(args, trainer_rank)
            server_rref = server_rrefs[server_rank]
        fut = rpc.rpc_async(
            trainer_name,
            run_trainer,
            args=(
                args,
                extra_args,
                train_data[trainer_rank],
                trainer_rank,
                server_rref,
            ),
            timeout=args.rpc_timeout
        )
        futs.append(fut)
    return futs


def benchmark_warmup(
    args, extra_args, data, server_rrefs
):
    r"""
    A function that runs the training algorithm. The goal of this
    function is to warm the rpc. The server states are reset.
    Args:
        args (parser): benchmark configurations
        extra_args (dict): configurations added by the user
        data (list): training samples
        server_rrefs (dict): a dictionary containing server RRefs
    """
    futs = call_trainers(args, extra_args, data, server_rrefs)
    wait_all(futs)
    for server_rref in server_rrefs.values():
        server_rref.rpc_sync().reset_state(server_rref)
    print("benchmark warmup done\n")


def split_list(arr, n):
    r"""
    A function that splits a list into n lists
    Args:
        arr (list): training samples
        n (int): number of output lists
    """
    return [arr[i::n] for i in range(n)]


def get_server_metrics(server_rrefs):
    r"""
    A function that calls the remote server to obtain metrics
    collected during the benchmark run.
    Args:
        server_rrefs (dict): a dictionary containing server RRefs
    """
    rank_metrics = []
    for rank, server_rref in server_rrefs.items():
        metrics = server_rref.rpc_sync().get_metrics(server_rref)
        rank_metrics.append([rank, metrics])
    return rank_metrics


def run_master(rank, data, args, extra_configs, rpc_backend_options):
    r"""
    A function that runs the master process in the world. This function
    obtains remote references to initialized servers, splits the data,
    runs the trainers, and prints metrics.
    Args:
        rank (int): process number in the world
        data (list): training samples
        args (parser): benchmark configurations
        extra_configs (dict): configurations added by the user
        rpc_backend_options (rpc): configurations/options for the rpc TODO: fix
    """
    world_size = args.ntrainer + args.ncudatrainer + args.nserver + args.ncudaserver + 1
    rpc.init_rpc(
        get_name(
            rank,
            args
        ),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )
    server_rrefs = {}
    for i in range(
        args.ntrainer + args.ncudatrainer, world_size - 1
    ):
        server_rrefs[i] = get_server_rref(i, args, extra_configs["server_config"])
    train_data = split_list(
        list(DataLoader(data, batch_size=args.batch_size)),
        args.ntrainer + args.ncudatrainer
    )

    # warmup run the benchmark
    benchmark_warmup(
        args, extra_configs["trainer_config"], train_data, server_rrefs
    )
    # run the benchmark
    trainer_futs = call_trainers(
        args, extra_configs["trainer_config"], train_data, server_rrefs
    )
    # collect metrics and print
    metrics_printer = ProcessedMetricsPrinter()
    rank_metrics_list = wait_all(trainer_futs)
    metrics_printer.print_metrics("trainer", rank_metrics_list)
    rank_metrics_list = get_server_metrics(server_rrefs)
    metrics_printer.print_metrics("server", rank_metrics_list)


def run_benchmark(rank, args, data):
    r"""
    A function that runs the benchmark.
    Args:
        rank (int): process number in the world
        args (parser): configuration args
        data (list): training samples
    """

    config = load_extra_configs(args)

    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.cuda_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    world_size = args.ntrainer + args.ncudatrainer + args.nserver + args.ncudaserver + 1
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    rpc_backend_options = TensorPipeRpcBackendOptions(rpc_timeout=args.rpc_timeout)
    if rank == world_size - 1:
        # master = [ntrainer + ncudatrainer + nserver + ncudaserver, ntrainer + ncudatrainer + nserver + ncudaserver]
        run_master(rank, data, args, config, rpc_backend_options)
    elif rank >= args.ntrainer + args.ncudatrainer:
        # parameter_servers = [ntrainer + ncudatrainer, ntrainer + ncudatrainer + nserver + ncudaserver)
        rpc.init_rpc(
            get_name(
                rank,
                args
            ),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    else:
        # trainers = [0, ntrainer + ncudatrainer)
        if rank >= args.ntrainer:
            server_rank = get_cuda_server_rank(args, rank)
            server_name = get_name(server_rank, args)
            rpc_backend_options.set_device_map(
                server_name,
                {rank: server_rank}
            )
        trainer_name = get_name(
            rank,
            args
        )
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    rpc.shutdown()


def get_json_config(file_name: str, id: str):
    r"""
    A function that loads a json configuration from a file.
    Args:
        file_name (str): name of configuration file to load
        id (str): configuration that will be loaded
    """
    with open(Path(__file__).parent / file_name) as f:
        json_config = json.load(f)[id]
    return json_config


def load_extra_configs(args):
    r"""
    A function that creates a dictionary that contains any extra configurations
    set by the user. The dictionary will contain two keys trainer_config and
    server_config, with default values None.
    Args:
        args (parser): launcher configurations
    """
    trainer_config_file = args.trainer_config_path
    server_config_file = args.server_config_path
    configurations = {
        "trainer_config": None,
        "server_config": None
    }
    if args.trainer is not None and trainer_config_file is not None:
        configurations["trainer_config"] = get_json_config(trainer_config_file, args.trainer)
    if args.server is not None and server_config_file is not None:
        configurations["server_config"] = get_json_config(server_config_file, args.server)
    return configurations


def load_data(args):
    r"""
    A function that creates an instance of the data class.
    Args:
        args (parser): launcher configurations
    """
    data_config_file = args.data_config_path
    data_config = get_json_config(data_config_file, args.data)
    data_class = data_map[data_config["data_class"]]
    return data_class(**data_config["configurations"])


def load_model(args):
    r"""
    A function that creates an instance of the model class.
    Args:
        args (parser): launcher configurations
    """
    model_config_file = args.model_config_path
    model_config = get_json_config(model_config_file, args.model)
    model_class = model_map[model_config["model_class"]]
    return model_class(**model_config["configurations"])


def main(args):
    r"""
    A function that creates multiple processes to run the benchmark.
    Args:
        args (parser): launcher configurations
    """
    # CPU and RPC trainer checks
    if args.ntrainer > 0 and args.ncudatrainer > 0:
        assert args.nserver > 0 and args.ncudaserver > 0
    if args.nserver > 0:
        assert args.ntrainer > 0
        assert args.ntrainer % args.nserver == 0
    if args.ncudaserver > 0:
        assert args.ncudatrainer > 0
        assert args.ncudatrainer % args.ncudaserver == 0

    world_size = (
        args.ntrainer + args.ncudatrainer + args.nserver + args.ncudaserver + 1
    )

    data = load_data(args)

    mp.spawn(
        run_benchmark,
        args=(
            args,
            data,
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPC server Benchmark")
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        type=str,
        help="IP address of the machine that will host the process with rank 0"
    )
    parser.add_argument(
        "--master-port",
        "--master_port",
        type=str,
        help="A free port on the machine that will host the process with rank 0"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        help="trainer map key to get trainer class for benchmark run"
    )
    parser.add_argument(
        "--ntrainer",
        type=int,
        help="trainer count for benchmark run"
    )
    parser.add_argument(
        "--ncudatrainer",
        type=int,
        help="cudatrainer count for benchmark run"
    )
    parser.add_argument(
        "--filestore",
        type=str,
        help="filestore location for process group"
    )
    parser.add_argument(
        "--server",
        type=str,
        help="server map key to get trainer class for benchmark run"
    )
    parser.add_argument(
        "--nserver",
        type=int,
        help="server count for benchmark run"
    )
    parser.add_argument(
        "--ncudaserver",
        type=int,
        help="cudaserver count for benchmark run"
    )
    parser.add_argument(
        "--rpc-timeout",
        "--rpc_timeout",
        type=int,
        help="timeout in seconds to use for RPC"
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="distributed communication backend to use for benchmark run"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="epoch count for training"
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        help="number of training examples used in one iteration"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="id for data configuration"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="id for model configuration"
    )
    parser.add_argument(
        "--data-config-path",
        "--data_config_path",
        type=str,
        help="path to data configuration file"
    )
    parser.add_argument(
        "--model-config-path",
        "--model_config_path",
        type=str,
        help="path to model configuration file"
    )
    parser.add_argument(
        "--server-config-path",
        "--server_config_path",
        type=str,
        help="path to server configuration file"
    )
    parser.add_argument(
        "--trainer-config-path",
        "--trainer_config_path",
        type=str,
        help="path to trainer configuration file"
    )
    parser.add_argument(
        "--torch-seed",
        "--torch_seed",
        type=int,
        help="seed for generating random numbers to a non-deterministic random number"
    )
    parser.add_argument(
        "--cuda-seed",
        "--cuda_seed",
        type=int,
        help="seed for generating random numbers to a random number for the current GPU"
    )
    parser.add_argument(
        "--preprocess-data",
        "--preprocess_data",
        type=str,
        help="this function will be used to preprocess data before training"
    )
    parser.add_argument(
        "--create-criterion",
        "--create_criterion",
        type=str,
        help="this function will be used to create the criterion used for model loss calculation"
    )
    parser.add_argument(
        "--create-ddp-model",
        "--create_ddp_model",
        type=str,
        help="this function will be used to create the ddp model used during training"
    )
    parser.add_argument(
        "--hook-state",
        "--hook_state",
        type=str,
        help="this will be the state class used when registering the ddp communication hook"
    )
    parser.add_argument(
        "--ddp-hook",
        "--ddp_hook",
        type=str,
        default="allreduce_hook",
        help="ddp communication hook"
    )
    parser.add_argument(
        "--iteration-step",
        "--iteration_step",
        type=str,
        help="this will be the function called for each iteration of training"
    )
    args = parser.parse_args()
    print(f"{args}\n")
    main(args)
