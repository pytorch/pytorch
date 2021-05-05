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


def get_name(rank, trainer_count, parameter_server_count):
    if rank < trainer_count:
        return f"trainer{rank}"
    elif rank < (trainer_count + parameter_server_count):
        return f"ps{rank}"
    else:
        return "master"


def get_parameter_server_rank(rank, configurations):
    # rank mod parameter server count to get parameter server number
    # add trainer_count to get parameter server rank
    rank_mod_ps_count = rank % configurations.parameter_server_count
    return rank_mod_ps_count + configurations.trainer_count


def get_ps_rref(parameter_server_rank, configurations, ps_configurations):
    ps = get_benchmark_ps_map()[str(ps_configurations["ps_class"])]
    name = get_name(
        parameter_server_rank,
        configurations.trainer_count,
        configurations.parameter_server_count
    )
    configured_args = ps_configurations["configurations"].values()
    ps_trainer_count = configurations.trainer_count / configurations.parameter_server_count
    rem = configurations.trainer_count % configurations.parameter_server_count
    if parameter_server_rank - configurations.trainer_count < rem:
        ps_trainer_count += 1
    return rpc.remote(
        name,
        ps,
        args=(
            parameter_server_rank,
            ps_trainer_count,
            *configured_args,
        ),
    )


def run_trainer(
    configurations, trainer_configurations, model, data, rank, ps_rref
):
    trainer_class = get_benchmark_trainer_map()[str(trainer_configurations["trainer_class"])]
    configured_args = trainer_configurations["configurations"].values()
    trainer = trainer_class(
        rank,
        configurations.trainer_count,
        ps_rref,
        *configured_args
    )
    trainer.train(model, data)
    metrics = trainer.get_metrics()
    return [rank, metrics]


def call_trainers(configurations, trainer_configurations, model, train_data, parameter_server_rrefs):
    futs = []
    for trainer_rank in range(0, configurations.trainer_count):
        trainer_name = get_name(
            trainer_rank,
            configurations.trainer_count,
            configurations.parameter_server_count
        )
        ps_rref = None
        if parameter_server_rrefs:
            ps_rank = get_parameter_server_rank(trainer_rank, configurations)
            ps_rref = parameter_server_rrefs[ps_rank]
        fut = rpc.rpc_async(
            trainer_name,
            run_trainer,
            args=(
                configurations,
                trainer_configurations,
                copy.deepcopy(model),
                train_data[trainer_rank],
                trainer_rank,
                ps_rref,
            ),
            timeout=configurations.rpc_async_timeout
        )
        futs.append(fut)
    return futs


def benchmark_warmup(
    configurations, trainer_configurations, ps_configurations, model, data, parameter_server_rrefs
):
    if configurations.parameter_server_count > 0:
        ps = get_benchmark_ps_map()[str(ps_configurations["ps_class"])]
    futs = call_trainers(configurations, trainer_configurations, model, data, parameter_server_rrefs)
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


def run_master(rank, model, data, configurations, trainer_configurations, ps_configurations, rpc_backend_options):
    world_size = configurations.trainer_count + configurations.parameter_server_count + 1
    rpc.init_rpc(
        get_name(
            rank,
            configurations.trainer_count,
            configurations.parameter_server_count
        ),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )
    parameter_server_rrefs = {}
    for i in range(
        configurations.trainer_count, world_size - 1
    ):
        parameter_server_rrefs[i] = get_ps_rref(i, configurations, ps_configurations)

    train_data = split_list(
        list(DataLoader(data, batch_size=configurations.batch_size)),
        configurations.trainer_count
    )

    # warmup run the benchmark
    benchmark_warmup(
        configurations, trainer_configurations, ps_configurations, model, train_data, parameter_server_rrefs
    )
    # run the benchmark
    trainer_futs = call_trainers(
        configurations, trainer_configurations, model, train_data, parameter_server_rrefs
    )
    # collect metrics and print
    metrics_printer = ProcessedMetricsPrinter()
    rank_metrics_list = [fut.wait() for fut in trainer_futs]
    metrics_printer.print_metrics("trainer", rank_metrics_list)


def run_benchmark(rank, model, data, configurations, trainer_configurations, ps_configurations):

    world_size = configurations.trainer_count + configurations.parameter_server_count + 1
    os.environ['MASTER_ADDR'] = configurations.master_addr
    os.environ['MASTER_PORT'] = configurations.master_port
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = configurations.rpc_init_method
    if rank == world_size - 1:
        # master = [trainer_count + parameter_server_count, trainer_count + parameter_server_count]
        run_master(rank, model, data, configurations, trainer_configurations, ps_configurations, rpc_backend_options)
    elif rank >= configurations.trainer_count:
        # parameter_servers = [trainer_count, trainer_count + parameter_server_count)
        rpc.init_rpc(
            get_name(
                rank,
                configurations.trainer_count,
                configurations.parameter_server_count
            ),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    else:
        # trainers = [0, trainer_count)
        if (USE_CUDA_RPC in trainer_configurations and
            trainer_configurations[USE_CUDA_RPC] and
            USE_CUDA_RPC in ps_configurations and
            ps_configurations[USE_CUDA_RPC] and
                configurations.parameter_server_count > 0):
            ps_rank = get_parameter_server_rank(rank, configurations)
            ps_name = get_name(
                ps_rank,
                configurations.trainer_count,
                configurations.parameter_server_count
            )
            rpc_backend_options.set_device_map(
                ps_name,
                {rank: ps_rank}
            )
        trainer_name = get_name(
            rank,
            configurations.trainer_count,
            configurations.parameter_server_count
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
    return json.load(f)[id]


def load_configurations(args):
    benchmark_config_file = "configurations/benchmark_configurations.json"
    benchmark_config = get_json_config(benchmark_config_file, args.bconfig_id)
    return BenchmarkConfigurations(**benchmark_config)


def get_data(data_class, data_config):
    data_class = get_benchmark_data_map()[data_class]
    return data_class(**data_config)


def load_data(args):
    data_config_file = "configurations/data_configurations.json"
    data_config = get_json_config(data_config_file, args.dconfig_id)
    return get_data(data_config["data_class"], data_config["configurations"])


def get_model(model_class, model_config):
    model_class = get_benchmark_model_map()[model_class]
    return model_class(**model_config)


def load_model(args):
    model_config_file = "configurations/model_configurations.json"
    model_config = get_json_config(model_config_file, args.mconfig_id)
    return get_model(model_config["model_class"], model_config["configurations"])


def load_trainer_configurations(args):
    trainer_config_file = "configurations/trainer_configurations.json"
    return get_json_config(trainer_config_file, args.tconfig_id)


def load_parameter_server_configurations(args):
    if args.pconfig_id == "None":
        return None
    ps_config_file = "configurations/parameter_server_configurations.json"
    return get_json_config(ps_config_file, args.pconfig_id)


def main():
    parser = argparse.ArgumentParser(description="RPC PS Benchmark")
    parser.add_argument(
        "--bconfig_id",
        type=str,
        help="id for configuration stored in benchmark_configurations.json"
    )
    parser.add_argument(
        "--dconfig_id",
        type=str,
        help="id for configuration stored in data_configurations.json"
    )
    parser.add_argument(
        "--mconfig_id",
        type=str,
        help="id for configuration stored in model_configurations.json"
    )
    parser.add_argument(
        "--pconfig_id",
        type=str,
        help="id for configuration stored in parameter_server_configurations.json"
    )
    parser.add_argument(
        "--tconfig_id",
        type=str,
        help="id for configuration stored in trainer_configurations.json"
    )
    args = parser.parse_args()
    print(f"{args}\n")

    configurations = load_configurations(args)
    data = load_data(args)
    model = load_model(args)
    trainer_configurations = load_trainer_configurations(args)
    ps_configurations = load_parameter_server_configurations(args)

    world_size = configurations.trainer_count + configurations.parameter_server_count + 1

    mp.spawn(
        run_benchmark,
        args=(
            model,
            data,
            configurations,
            trainer_configurations,
            ps_configurations,
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
