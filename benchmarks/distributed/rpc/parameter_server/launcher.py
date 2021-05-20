import argparse
import copy
import json
import os
from pathlib import Path

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.utils.data import DataLoader

from benchmark_class_helper import (get_benchmark_data_map,
                                    get_benchmark_model_map,
                                    get_benchmark_ps_map,
                                    get_benchmark_trainer_map)
from BenchmarkConfigurations import BenchmarkConfigurations
from metrics.ProcessedMetricsPrinter import ProcessedMetricsPrinter

USE_CUDA_RPC = "use_cuda_rpc"


def get_name(rank, configs):
    t_count = configs.trainer_count
    ps_count = configs.ps_count
    if rank < t_count:
        return f"trainer{rank}"
    elif rank < (t_count + ps_count):
        return f"ps{rank}"
    else:
        return "master"


def get_parameter_server_rank(rank, config):
    # rank mod parameter server count to get parameter server number
    # add trainer_count to get parameter server rank
    rank_mod_ps_count = rank % config.ps_count
    return rank_mod_ps_count + config.trainer_count


def get_ps_rref(parameter_server_rank, config):
    ps_config = config.ps_config
    ps = get_benchmark_ps_map()[str(ps_config["ps_class"])]
    name = get_name(
        parameter_server_rank,
        config
    )
    ps_args = ps_config["configurations"].values()
    ps_trainer_count = config.trainer_count / ps_config.ps_count
    rem = config.trainer_count % ps_config.ps_count
    if parameter_server_rank - config.trainer_count < rem:
        ps_trainer_count += 1
    return rpc.remote(
        name,
        ps,
        args=(
            parameter_server_rank,
            ps_trainer_count,
            *ps_args,
        ),
    )


def run_trainer(
    config, model, data, rank, ps_rref
):
    trainer_config = config.trainer_config
    trainer_class = get_benchmark_trainer_map()[str(trainer_config["trainer_class"])]
    trainer_args = trainer_config["configurations"].values()
    trainer = trainer_class(
        rank,
        config.trainer_count,
        ps_rref,
        *trainer_args
    )
    trainer.train(model, data)
    metrics = trainer.get_metrics()
    return [rank, metrics]


def call_trainers(config, model, train_data, parameter_server_rrefs):
    futs = []
    for trainer_rank in range(0, config.trainer_count):
        trainer_name = get_name(
            trainer_rank,
            config
        )
        ps_rref = None
        if parameter_server_rrefs:
            ps_rank = get_parameter_server_rank(trainer_rank, config)
            ps_rref = parameter_server_rrefs[ps_rank]
        fut = rpc.rpc_async(
            trainer_name,
            run_trainer,
            args=(
                config,
                copy.deepcopy(model),
                train_data[trainer_rank],
                trainer_rank,
                ps_rref,
            ),
            timeout=config.rpc_async_timeout
        )
        futs.append(fut)
    return futs


def benchmark_warmup(
    config, model, data, parameter_server_rrefs
):
    if config.ps_count > 0:
        ps_config = config.ps_config
        ps = get_benchmark_ps_map()[str(ps_config["ps_class"])]
    futs = call_trainers(config, model, data, parameter_server_rrefs)
    for fut in futs:
        fut.wait()
    for ps_rref in parameter_server_rrefs.values():
        rpc.rpc_sync(
            ps_rref.owner(),
            ps.reset_state,
            args=(ps_rref,)
        )
    print("benchmark warmup done\n")


def split_list(arr, n):
    return [arr[i::n] for i in range(n)]


def run_master(rank, model, data, config, rpc_backend_options):
    world_size = config.trainer_count + config.ps_count + 1
    rpc.init_rpc(
        get_name(
            rank,
            config
        ),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )
    parameter_server_rrefs = {}
    for i in range(
        config.trainer_count, world_size - 1
    ):
        parameter_server_rrefs[i] = get_ps_rref(i, config)

    train_data = split_list(
        list(DataLoader(data, batch_size=config.batch_size)),
        config.trainer_count
    )

    # warmup run the benchmark
    benchmark_warmup(
        config, model, train_data, parameter_server_rrefs
    )
    # run the benchmark
    trainer_futs = call_trainers(
        config, model, train_data, parameter_server_rrefs
    )
    # collect metrics and print
    metrics_printer = ProcessedMetricsPrinter()
    rank_metrics_list = [fut.wait() for fut in trainer_futs]
    metrics_printer.print_metrics("trainer", rank_metrics_list)


def run_benchmark(rank, model, data, config):

    world_size = config.trainer_count + config.ps_count + 1
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = config.rpc_init_method
    if rank == world_size - 1:
        # master = [trainer_count + parameter_server_count, trainer_count + parameter_server_count]
        run_master(rank, model, data, config, rpc_backend_options)
    elif rank >= config.trainer_count:
        # parameter_servers = [trainer_count, trainer_count + parameter_server_count)
        rpc.init_rpc(
            get_name(
                rank,
                config
            ),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    else:
        # trainers = [0, trainer_count)
        trainer_config = config.trainer_config
        ps_config = config.ps_config
        if (USE_CUDA_RPC in trainer_config and
            trainer_config[USE_CUDA_RPC] and
            USE_CUDA_RPC in ps_config and
            ps_config[USE_CUDA_RPC] and
                config.ps_count > 0):
            ps_rank = get_parameter_server_rank(rank, config)
            ps_name = get_name(
                ps_rank,
                config
            )
            rpc_backend_options.set_device_map(
                ps_name,
                {rank: ps_rank}
            )
        trainer_name = get_name(
            rank,
            config
        )
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    rpc.shutdown()


def get_json_config(file_name, id):
    f = open(
        os.path.join(
            Path(__file__).parent, file_name
        ),
        "r"
    )
    json_config = json.load(f)[id]
    f.close()
    return json_config


def load_configurations(args):
    trainer_config_file = args.trainer_config_path
    ps_config_file = args.server_config_path
    benchmark_config = get_json_config(args.benchmark_config_path, args.benchmark)
    benchmark_config["trainer_config"] = get_json_config(trainer_config_file, args.trainer)
    if args.server != "None":
        benchmark_config["ps_config"] = get_json_config(ps_config_file, args.server)
    else:
        benchmark_config["ps_config"] = None
    return BenchmarkConfigurations(**benchmark_config)


def get_data(data_class, data_config):
    data_class = get_benchmark_data_map()[data_class]
    return data_class(**data_config)


def load_data(args):
    data_config_file = args.data_config_path
    data_config = get_json_config(data_config_file, args.data)
    return get_data(data_config["data_class"], data_config["configurations"])


def get_model(model_class, model_config):
    model_class = get_benchmark_model_map()[model_class]
    return model_class(**model_config)


def load_model(args):
    model_config_file = args.model_config_path
    model_config = get_json_config(model_config_file, args.model)
    return get_model(model_config["model_class"], model_config["configurations"])


def main():
    parser = argparse.ArgumentParser(description="RPC PS Benchmark")

    parser.add_argument(
        "--benchmark_config_path",
        type=str,
        default="configurations/benchmark_configurations.json",
        help="path to benchmark configuration file"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="configurations/data_configurations.json",
        help="path to data configuration file"
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="configurations/model_configurations.json",
        help="path to model configuration file"
    )
    parser.add_argument(
        "--server_config_path",
        type=str,
        default="configurations/server_configurations.json",
        help="path to server configuration file"
    )
    parser.add_argument(
        "--trainer_config_path",
        type=str,
        default="configurations/trainer_configurations.json",
        help="path to trainer configuration file"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="id for benchmark configuration"
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
        "--server",
        type=str,
        help="id for parameter server configuration"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        help="id for trainer configuration"
    )
    args = parser.parse_args()
    print(f"{args}\n")

    config = load_configurations(args)
    data = load_data(args)
    model = load_model(args)

    world_size = config.trainer_count + config.ps_count + 1

    mp.spawn(
        run_benchmark,
        args=(
            model,
            data,
            config,
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
