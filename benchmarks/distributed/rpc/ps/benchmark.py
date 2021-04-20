import argparse
import copy
import json
import os
import statistics
import sys
import threading
from pathlib import Path
import pandas as pd
import torch
import torch.distributed as c10d
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP


GLOO = "gloo"
NCCL = "nccl"
INDICES = "indices"
VALUES = "values"
SIZE = "size"
GRADIENT = "gradient"
HOOK_METRIC = "hook_metric"
FORWARD_METRIC = "foward_metric"
BACKWARD_METRIC = "backward_metric"
BATCH_LEVEL_METRIC = "batch_level_metric"
GRADIENT_SERVER_BATCH_METRIC = "gradient_server_batch_metric"
GRADIENT_SERVER_STRAGGLER_METRIC = "gradient_server_straggler_metric"
RANK = "rank"
TRAINER_COUNT = "trainer_count"
USE_CUDA_RPC = "use_cuda_rpc"
BATCH_MODE = "batch_mode"
CUDA_SPARSE_RPC = "cuda_sparse_rpc"
CPU_SPARSE_RPC = "cpu_sparse_rpc"
CUDA_DENSE_RPC = "cuda_dense_rpc"
CPU_DENSE_RPC = "cpu_dense_rpc"
NCCL_ALLREDUCE_DENSE = "nccl_allreduce_dense"
NCCL_ALLREDUCE = "nccl_allreduce"
GLOO_ALLREDUCE = "gloo_allreduce"
BATCH_ALL = "batch_all"
FORWARD_PASS = "forward_pass"
BACKWARD = "backward"
BP_LOC_STRAGGLER = "bp_loc_straggler"
BP_LOC_BATCH = "bp_loc_batch"


def get_name(rank, ddp_trainers, gradient_servers):
    if rank < ddp_trainers:
        return "trainer{}".format(rank)
    elif rank < (ddp_trainers + gradient_servers):
        return "gs{}".format(rank)
    else:
        return "master"


def gs_for_rank(rank, configurations):
    if (configurations.ddp_trainers % configurations.gradient_servers) != 0:
        gs_rank = rank % configurations.gradient_servers
    else:
        x = int(configurations.ddp_trainers / configurations.gradient_servers)
        gs_rank = int(rank / x)
    return gs_rank + configurations.ddp_trainers


def record_event(rank, metric_type, key, name, metrics):
    event = torch.cuda.Event(enable_timing=True)
    if metric_type not in metrics:
        metrics[metric_type] = {}
    metrics = metrics[metric_type]
    if key in metrics:
        metrics[key]["end"] = event
    else:
        metrics[key] = {
            "name": name,
            "start": event
        }
    with torch.cuda.device(rank):
        event.record()

# TODO: clean up metric processing


def process_metrics_for_elapsed_time(metrics):
    # cannot pickle 'Event' object
    metrics_processed = {}
    for metric_type in metrics.keys():
        metrics_processed[metric_type] = {}
        for metric_key in metrics[metric_type].keys():
            name = metrics[metric_type][metric_key]["name"]
            start = metrics[metric_type][metric_key]["start"]
            end = metrics[metric_type][metric_key]["end"]
            elapsed_time = start.elapsed_time(end)
            metrics_processed[metric_type][metric_key] = {
                "name": name,
                "elapsed_time": elapsed_time
            }
    return metrics_processed


def print_rank_totals(rank_type, rank_totals, file_name, print_metrics_to_dir):
    df = pd.DataFrame(
        columns=['name', 'min', 'max', 'mean', 'variance', 'stdev']
    )
    totals = {}
    for rank in rank_totals:
        for key in sorted(rank.keys()):
            if key not in totals:
                totals[key] = rank[key]
            else:
                totals[key] += rank[key]

    for key, values in totals.items():
        row = {
            "name": key,
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "variance": statistics.variance(values),
            "stdev": statistics.stdev(values)
        }
        df = df.append(row, ignore_index=True)
    print("metrics for {}".format(rank_type))
    print(tabulate(df, showindex=False, headers=df.columns, tablefmt="grid"))
    if print_metrics_to_dir:
        file_name = "data_frames/{}__{}.csv".format(rank_type, file_name)
        df.to_csv(file_name, encoding='utf-8', index=False)


def get_rank_totals_and_print(
    rank_type, rank_metrics, file_name, print_metrics_to_dir
):
    rank = rank_metrics[0]
    metrics = rank_metrics[1]
    df = pd.DataFrame(
        columns=['name', 'min', 'max', 'mean', 'variance', 'stdev']
    )
    rank_totals = {}
    for metric_type in sorted(metrics.keys()):
        values = []
        name = None
        for metric_key in sorted(metrics[metric_type].keys()):
            metric = metrics[metric_type][metric_key]
            values.append(metric["elapsed_time"])
            name = metric["name"]
        rank_totals[name] = values
        row = {
            "name": name,
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "variance": statistics.variance(values),
            "stdev": statistics.stdev(values)
        }
        df = df.append(row, ignore_index=True)
    print("metrics for {}={}".format(rank_type, rank))
    print(tabulate(df, showindex=False, headers=df.columns, tablefmt="grid"))
    if print_metrics_to_dir:
        file_name = "data_frames/{}_{}__{}.csv".format(
            rank_type, rank, file_name
        )
        df.to_csv(file_name, encoding='utf-8', index=False)
    return rank_totals


class GradientServer:

    # TODO: cuda sparse kernel
    """
        rank - the process number of the gradient server in the world

        gradient_servers - the number of processes running gradient
        servers in the world

        ddp_trainers - the number of processes running ddp_trainers
        in the world

        backend - the communication backend selected for distributed
        training

        use_cuda_rpc - indicates that the RPC should be using Cuda or CPU

        batch_mode - indicates if the gradient server should be waiting to
        sum all gradients at once to reduce Cuda kernel calls
    """
    def __init__(
        self,
        rank,
        gradient_servers,
        ddp_trainers,
        use_cuda_rpc,
        backend,
        batch_mode
    ):
        self.lock = threading.Lock()
        self.rank = rank
        self.gradient_servers = gradient_servers
        self.ddp_trainers = ddp_trainers
        self.trainer_count = ddp_trainers / gradient_servers
        remainder = (ddp_trainers % gradient_servers) != 0
        add = (rank - ddp_trainers + 1) <= (ddp_trainers % gradient_servers)
        if remainder and add:
            self.trainer_count += 1
        self.use_cuda_rpc = use_cuda_rpc
        self.backend = backend
        self.batch_mode = batch_mode
        self.futures = {}
        self.gradient = {}
        self.batch_number = 0
        self.metrics = {}

    def clear_batch_state(self):
        self.futures.clear()
        self.gradient.clear()

    def average_gradient_bp_key(self, bp_loc):
        return "{},{}".format(self.batch_number, bp_loc)

    """
        gs_rref - the shared pointer to the gradient server assigned
            to the rank

        rank - the process number in the world

        received_batch_number - ddp_trainer batch number

        backend - the communication backend selected for distributed training

        use_cuda_rpc - indicates that the RPC should be using Cuda or CPU

        batch_mode - indicates if the gradient server should
            sum all gradients at once to reduce Cuda kernel calls
    """
    @staticmethod
    @rpc.functions.async_execution
    def average_gradient(
        gs_rref,
        rank,
        received_batch_number,
        bp_loc,
        **kwargs
    ):
        sparse_gradient = INDICES in kwargs
        if not sparse_gradient:
            gradient = kwargs[GRADIENT]
        else:
            gradient = torch.sparse_coo_tensor(
                kwargs[INDICES],
                kwargs[VALUES],
                kwargs[SIZE]
            )
        self = gs_rref.local_value()
        gradient = gradient.cuda(self.rank)
        fut = torch.futures.Future()
        with self.lock:
            if self.batch_number < received_batch_number:
                self.batch_number = received_batch_number
                self.clear_batch_state()
            if bp_loc not in self.gradient:
                record_event(
                    self.rank,
                    GRADIENT_SERVER_STRAGGLER_METRIC,
                    self.average_gradient_bp_key(
                        self.batch_number
                    ),
                    BP_LOC_STRAGGLER,
                    self.metrics
                )
                record_event(
                    self.rank,
                    GRADIENT_SERVER_BATCH_METRIC,
                    self.average_gradient_bp_key(
                        self.batch_number
                    ),
                    BP_LOC_BATCH,
                    self.metrics
                )
                if self.batch_mode:
                    self.gradient[bp_loc] = [gradient]
                else:
                    self.gradient[bp_loc] = gradient
                self.futures[bp_loc] = [fut]
            else:
                if self.batch_mode:
                    self.gradient[bp_loc].append(gradient)
                else:
                    self.gradient[bp_loc] += gradient
                self.futures[bp_loc].append(fut)
            if len(self.futures[bp_loc]) == self.trainer_count:
                record_event(
                    self.rank,
                    GRADIENT_SERVER_STRAGGLER_METRIC,
                    self.average_gradient_bp_key(
                        self.batch_number
                    ),
                    BP_LOC_STRAGGLER,
                    self.metrics
                )
                if self.batch_mode:
                    # TODO: cuda kernel
                    bp_loc_avg = self.gradient[bp_loc][0]
                    for i in range(1, self.trainer_count):
                        bp_loc_avg += self.gradient[bp_loc][i]
                else:
                    bp_loc_avg = self.gradient[bp_loc]
                bp_loc_avg / (1.0 * self.trainer_count)
                if not self.use_cuda_rpc:
                    bp_loc_avg = bp_loc_avg.cpu()
                if sparse_gradient:
                    if self.backend == GLOO:
                        bp_loc_avg = bp_loc_avg.coalesce()
                    bp_loc_avg = [
                        bp_loc_avg._indices(),
                        bp_loc_avg._values(),
                        bp_loc_avg.size()
                    ]
                for cur_fut in self.futures[bp_loc]:
                    cur_fut.set_result(bp_loc_avg)
                record_event(
                    self.rank,
                    GRADIENT_SERVER_BATCH_METRIC,
                    self.average_gradient_bp_key(
                        self.batch_number
                    ),
                    BP_LOC_BATCH,
                    self.metrics
                )
                return fut
        return fut

    def rpc_warmup_call(tensor):
        return tensor

    def reset(gs_rref):
        self = gs_rref.local_value()
        self.clear_batch_state()
        self.batch_number = 0

    def get_metrics(gs_rref):
        self = gs_rref.local_value()
        torch.cuda.synchronize(self.rank)
        return [self.rank, process_metrics_for_elapsed_time(self.metrics)]


class HookState():
    def __init__(
        self,
        rank,
        process_group,
        process_group_size,
        use_cuda_rpc,
        gs_rref,
        batch_number,
        bp_location,
        hook_gradient_futures,
        metrics
    ):
        self.rank = rank
        self.process_group = process_group
        self.process_group_size = process_group_size
        self.use_cuda_rpc = use_cuda_rpc
        self.gs_rref = gs_rref
        self.batch_number = batch_number
        self.bp_location = bp_location
        self.hook_gradient_futures = hook_gradient_futures
        self.metrics = metrics

    def next_batch_state(self):
        self.bp_location = 0
        self.batch_number += 1
        self.hook_gradient_futures = []


def hook_metric_key(state):
    return "{},{}".format(state.batch_number, state.bp_location)


def get_tensors(bucket):
    parameter_tensors = bucket.get_per_parameter_tensors()
    parameter_tensors_count = len(parameter_tensors)
    if parameter_tensors_count > 0:
        return parameter_tensors
    else:
        return [bucket.get_tensor()]


def send_request_rpc(state, tensor, rpc_type, kwargs):
    tensor = tensor.cuda(state.rank) if state.use_cuda_rpc else tensor.cpu()
    record_event(
        state.rank,
        HOOK_METRIC,
        hook_metric_key(state),
        rpc_type,
        state.metrics
    )
    fut = rpc.rpc_async(
        state.gs_rref.owner(),
        GradientServer.average_gradient,
        args=(
            state.gs_rref,
            state.rank,
            state.batch_number,
            state.bp_location
        ),
        kwargs=kwargs
    )
    record_event(
        state.rank,
        HOOK_METRIC,
        hook_metric_key(state),
        rpc_type,
        state.metrics
    )
    return fut


def send_requests(state, bucket):
    tensors = get_tensors(bucket)
    tensors_len = len(tensors)
    for i in range(tensors_len):
        tensor = tensors[tensors_len - i - 1]
        kwargs = {}
        if tensor.is_sparse:
            kwargs[INDICES] = tensor._indices()
            kwargs[VALUES] = tensor._values()
            kwargs[SIZE] = tensor.size()
            if state.use_cuda_rpc:
                rpc_type = CUDA_SPARSE_RPC
            else:
                rpc_type = CPU_SPARSE_RPC
                kwargs[INDICES] = kwargs[INDICES].cpu()
                kwargs[VALUES] = kwargs[VALUES].cpu()
        else:
            kwargs[GRADIENT] = tensor
            if state.use_cuda_rpc:
                rpc_type = CUDA_DENSE_RPC
            else:
                rpc_type = CPU_DENSE_RPC
                kwargs[GRADIENT] = kwargs[GRADIENT].cpu()
        # Since the buckets are rebuilt after the first iteration,
        # should not rely on the indices at the beginning of training.
        if state.batch_number > 0:
            fut = send_request_rpc(state, tensor, rpc_type, kwargs)
            state.hook_gradient_futures.append([fut, state.bp_location])
        state.bp_location += 1


def gradient_server_hook(state, bucket):
    send_requests(state, bucket)
    # After the backward pass,
    # we can manually synchronous sparse gradients or parameters
    fut = torch.futures.Future()
    fut.set_result([bucket.get_tensor()])
    return fut


def gradient_server_sparse_and_nccl_allreduce_dense_hook(state, bucket):
    tensor = bucket.get_tensor()
    tensors_count = len(get_tensors(bucket))
    if tensor.is_sparse:
        send_requests(state, bucket)
        # After the backward pass,
        # we can manually synchronous sparse gradients or parameters
        fut = torch.futures.Future()
        fut.set_result([bucket.get_tensor()])
        return fut
    else:
        tensor = [tensor / state.process_group_size]
        record_event(
            state.rank,
            HOOK_METRIC,
            hook_metric_key(state),
            NCCL_ALLREDUCE_DENSE,
            state.metrics
        )
        fut = state.process_group.allreduce(tensor).get_future()
        record_event(
            state.rank,
            HOOK_METRIC,
            hook_metric_key(state),
            NCCL_ALLREDUCE_DENSE,
            state.metrics
        )
        state.bp_location += tensors_count
        return fut


def nccl_allreduce_hook(state, bucket):
    tensor = bucket.get_tensor()
    tensors_count = len(get_tensors(bucket))
    record_event(
        state.rank,
        HOOK_METRIC,
        hook_metric_key(state),
        NCCL_ALLREDUCE,
        state.metrics
    )
    fut = state.process_group.allreduce(tensor).get_future()
    record_event(
        state.rank,
        HOOK_METRIC,
        hook_metric_key(state),
        NCCL_ALLREDUCE,
        state.metrics
    )
    state.bp_location += tensors_count
    return fut


def gloo_allreduce_hook(state, bucket):
    tensors_count = len(get_tensors(bucket))
    record_event(
        state.rank,
        HOOK_METRIC,
        hook_metric_key(state),
        GLOO_ALLREDUCE,
        state.metrics
    )
    work = state.process_group.allreduce([bucket.get_tensor()])
    work.wait()
    fut = torch.futures.Future()
    fut.set_result([bucket.get_tensor() / state.process_group_size])
    record_event(
        state.rank,
        HOOK_METRIC,
        hook_metric_key(state),
        GLOO_ALLREDUCE,
        state.metrics
    )
    state.bp_location += tensors_count
    return fut


def run_trainer(configurations, model, data, rank, gs_rref):

    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    model.cuda(rank)

    process_group_size = configurations.ddp_trainers

    # random file or set store configuration?
    store = c10d.FileStore("/tmp/tmpn_k_8so02", process_group_size)

    if configurations.backend == GLOO:
        process_group = c10d.ProcessGroupGloo(store, rank, process_group_size)
    elif configurations.backend == NCCL:
        process_group = c10d.ProcessGroupNCCL(store, rank, process_group_size)

    ddp_model = DDP(model, device_ids=[rank], process_group=process_group)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-4)

    input, target = data.get_input_and_target()
    input = input.split(process_group_size)[rank].cuda(rank)
    target = target.split(process_group_size)[rank].cuda(rank)

    # rpc calls to gradient server warmpup
    for _ in range(30):
        rpc.rpc_sync(
            gs_rref.owner(),
            GradientServer.rpc_warmup_call,
            args=(torch.tensor([[1] * 100] * 100),)
        )

    metrics = {}
    hook_state = HookState(
        rank,
        process_group,
        process_group_size,
        configurations.use_cuda_rpc,
        gs_rref,
        0,
        0,
        [],
        metrics
    )

    hook_id = configurations.hook_id
    if hook_id == 1:
        ddp_model.register_comm_hook(
            hook_state,
            gradient_server_hook
        )
    elif hook_id == 2:
        ddp_model.register_comm_hook(
            hook_state,
            gradient_server_sparse_and_nccl_allreduce_dense_hook
        )
    elif hook_id == 3:
        raise NotImplementedError("hook_id={}".format(hook_id))
    elif hook_id == 4:
        ddp_model.register_comm_hook(
            hook_state,
            nccl_allreduce_hook
        )
    elif hook_id == 5:
        ddp_model.register_comm_hook(
            hook_state,
            gloo_allreduce_hook
        )

    # bucket index reordering
    criterion(ddp_model(input), target).backward()
    metrics.clear()
    process_group.barrier()

    ddp_model_parameters = list(ddp_model.parameters())

    for i in range(configurations.iterations):
        hook_state.next_batch_state()

        record_event(rank, BATCH_LEVEL_METRIC, i, BATCH_ALL, metrics)

        optimizer.zero_grad()

        record_event(rank, FORWARD_METRIC, i, FORWARD_PASS, metrics)
        out = ddp_model(input)
        record_event(rank, FORWARD_METRIC, i, FORWARD_PASS, metrics)

        loss = criterion(out, target)

        record_event(rank, BACKWARD_METRIC, i, BACKWARD, metrics)
        loss.backward()
        record_event(rank, BACKWARD_METRIC, i, BACKWARD, metrics)

        for fut, bp_loc in hook_state.hook_gradient_futures:
            gradient = fut.wait()
            if isinstance(gradient, list):
                indices = gradient[0].cuda(rank)
                values = gradient[1].cuda(rank)
                size = gradient[2]
                gradient = torch.sparse_coo_tensor(indices, values, size)
            gradient = gradient.cuda(rank)
            ddp_model_parameters[bp_loc].grad = gradient

        optimizer.step()

        record_event(rank, BATCH_LEVEL_METRIC, i, BATCH_ALL, metrics)

    torch.cuda.synchronize(rank)

    return [rank, process_metrics_for_elapsed_time(metrics)]


def run_benchmark(rank, model, data, configurations):
    world_size = configurations.world_size
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        rpc_backend_options = TensorPipeRpcBackendOptions()
        rpc_backend_options.init_method = 'tcp://localhost:29501'
        if rank == world_size - 1:
            rpc.init_rpc(
                get_name(
                    rank,
                    configurations.ddp_trainers,
                    configurations.gradient_servers
                ),
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options
            )
            # TODO: fix for no gradient servers
            gradient_servers = {}
            for i in range(
                configurations.ddp_trainers, configurations.world_size - 1
            ):
                gs_rref = rpc.remote(
                    get_name(
                        i,
                        configurations.ddp_trainers,
                        configurations.gradient_servers
                    ),
                    GradientServer,
                    args=(
                        i,
                        configurations.gradient_servers,
                        configurations.ddp_trainers,
                        configurations.use_cuda_rpc,
                        configurations.backend,
                        configurations.batch_mode
                    )
                )
                gradient_servers[i] = gs_rref
            futs = [
                rpc.rpc_async(
                    get_name(
                        trainer_rank,
                        configurations.ddp_trainers,
                        configurations.gradient_servers
                    ),
                    run_trainer,
                    [
                        configurations,
                        copy.deepcopy(model),
                        copy.deepcopy(data),
                        trainer_rank,
                        gradient_servers[
                            gs_for_rank(trainer_rank, configurations)
                        ]
                    ]
                )
                for trainer_rank in range(0, configurations.ddp_trainers)
            ]

            # metrics
            if not os.path.exists("./data_frames"):
                os.makedirs("./data_frames")
            rank_trainer_metric_totals = []
            for fut in futs:
                rank_metrics = fut.wait()
                rank_trainer_metric_totals.append(
                    get_rank_totals_and_print(
                        "trainer",
                        rank_metrics,
                        configurations.to_string(),
                        configurations.print_metrics_to_dir
                    )
                )
            print_rank_totals(
                "trainers",
                rank_trainer_metric_totals,
                configurations.to_string(),
                configurations.print_metrics_to_dir
            )
            rank_gradient_server_metric_totals = []
            for gs_rref in gradient_servers.values():
                gs_metrics = rpc.rpc_sync(
                    gs_rref.owner(),
                    GradientServer.get_metrics,
                    args=(gs_rref,)
                )
                if gs_metrics:
                    rank_gradient_server_metric_totals.append(
                        get_rank_totals_and_print(
                            "gradient server",
                            gs_metrics,
                            configurations.to_string(),
                            configurations.print_metrics_to_dir
                        )
                    )
            print_rank_totals(
                "gradient servers",
                rank_gradient_server_metric_totals,
                configurations.to_string(),
                configurations.print_metrics_to_dir
            )

        elif rank >= configurations.ddp_trainers:
            rpc.init_rpc(
                get_name(
                    rank,
                    configurations.ddp_trainers,
                    configurations.gradient_servers
                ),
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options
            )
        else:
            if configurations.use_cuda_rpc:
                gs_rank = gs_for_rank(rank, configurations)
                rpc_backend_options.set_device_map(
                    get_name(
                        gs_rank,
                        configurations.ddp_trainers,
                        configurations.gradient_servers
                    ),
                    {rank: gs_rank}
                )
            rpc.init_rpc(
                get_name(
                    rank,
                    configurations.ddp_trainers,
                    configurations.gradient_servers
                ),
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options
            )
    except Exception as e:
        print("error: {}".format(e))
    rpc.shutdown()


class DummyModel(nn.Module):
    def __init__(
        self,
        num_embeddings=4,
        embedding_dim=4,
        dense_input_size=4,
        dense_output_size=4,
        sparse=True
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            num_embeddings, embedding_dim, sparse=sparse
        )
        self.fc1 = nn.Linear(dense_input_size, dense_output_size)

    def forward(self, x):
        x = self.embedding(x)
        return F.softmax(self.fc1(x), dim=1)


def get_model(model_id, model_config):
    if model_id == 1:
        return DummyModel(**model_config)
    sys.exit("model_id not found")


class RandomData:
    def __init__(
        self,
        min_val: int = 0,
        max_val: int = 4,
        batch_size: int = 4,
        mult: int = 2
    ):
        self.input = torch.randint(min_val, max_val, [batch_size, mult])
        self.target = torch.randint(min_val, max_val, [batch_size])

    def get_input_and_target(self):
        return self.input, self.target


def get_data(data_id, data_config):
    if data_id == 1:
        return RandomData(**data_config)
    sys.exit("data_id not found")


class Configurations:
    def __init__(
        self,
        hook_id: int,
        backend: str = GLOO,
        ddp_trainers: int = 1,
        gradient_servers: int = 1,
        iterations: int = 2,
        batch_mode: bool = False,
        use_cuda_rpc: bool = False,
        print_metrics_to_dir: bool = False
    ):
        backend = backend.lower()
        assert backend == GLOO or backend == NCCL
        assert hook_id > 0 and hook_id < 6
        assert iterations > 0
        assert gradient_servers <= ddp_trainers

        self.backend = backend
        self.batch_mode = batch_mode
        self.hook_id = hook_id
        self.iterations = iterations
        self.world_size = ddp_trainers + gradient_servers + 1
        self.ddp_trainers = ddp_trainers
        self.gradient_servers = gradient_servers
        self.use_cuda_rpc = use_cuda_rpc
        self.print_metrics_to_dir = print_metrics_to_dir

    def to_string(self):
        output = ""
        class_items = list(self.__dict__.items())
        for i in range(len(class_items)):
            attr, value = class_items[i]
            if i > 0:
                output += "__"
            output += "{}_{}".format(attr, value)
        return output


"""
    user needs to confirm cluster or server meets GPU requirements
"""


def main():
    parser = argparse.ArgumentParser(description="RPC PS Benchmark")
    parser.add_argument(
        "--bconfig_id",
        type=str,
        default="1"
    )
    parser.add_argument(
        "--dconfig_id",
        type=str,
        default="1"
    )
    parser.add_argument(
        "--mconfig_id",
        type=str,
        default="1"

    )
    args = parser.parse_args()

    benchmark_config_file = "configurations/benchmark_configurations.json"
    benchmark_config = json.load(
        open(
            os.path.join(Path(__file__).parent, benchmark_config_file),
            "r"
        )
    )[args.bconfig_id]
    configurations = Configurations(**benchmark_config)

    data_config_file = "configurations/data_configurations.json"
    data_config = json.load(
        open(
            os.path.join(
                Path(__file__).parent, data_config_file
            ),
            "r"
        )
    )[args.dconfig_id]
    data = get_data(data_config["data_id"], data_config["configurations"])

    model_config_file = "configurations/model_configurations.json"
    model_config = json.load(
        open(
            os.path.join(
                Path(__file__).parent, model_config_file
            ),
            "r"
        )
    )[args.mconfig_id]
    model = get_model(model_config["model_id"], model_config["configurations"])

    print("{}\nbconfig_id={}\ndconfig_id={}\nmconfig_id={}\n".format(
        parser.description, args.bconfig_id, args.dconfig_id, args.mconfig_id))

    mp.spawn(
        run_benchmark,
        [model, data, configurations],
        nprocs=configurations.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
