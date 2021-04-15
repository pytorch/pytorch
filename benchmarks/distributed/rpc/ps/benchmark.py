from functools import wraps
import os
import random
import time
import threading
import copy
import sys
import json
from pathlib import Path
import argparse
import pprint
pp = pprint.PrettyPrinter(indent=4)

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as c10d
import torch.distributed.rpc as rpc

# --------------------------- constants -----------------------------------------

GLOO = "gloo"
NCCL = "nccl"
SPARSE = "sparse"
INDICES = "indices"
VALUES = "values"
SIZE = "size"
GRADIENT = "gradient"
HOOK_METRIC = "hook_metric"
FORWARD_METRIC = "foward_metric"
BACKWARD_METRIC = "backward_metric"
BATCH_LEVEL_METRIC = "batch_level_metric"
RANK = "rank"
TRAINER_COUNT = "trainer_count"
USE_CUDA_RPC = "use_cuda_rpc"
BATCH_MODE = "batch_mode"


# --------------------------- basic helpers -----------------------------------------


def get_name(rank, world_size):
    if rank < world_size - 2:
        return "trainer{}".format(rank)
    elif rank == world_size - 2:
        return "gs"
    else:
        return "master"

# --------------------------- metrics -----------------------------------------


def record_event(rank, metric_type, key, name, metrics):
    event = torch.cuda.Event(enable_timing=True)
    if metric_type not in metrics:
        metrics[metric_type] = {}
    metrics = metrics[metric_type]
    if key in metrics:
        assert "end" not in metrics[key]
        metrics[key]["end"] = event
    else:
        metrics[key] = {
            "name": name,
            "start": event
        }
    with torch.cuda.device(rank):
        event.record()

# --------------------------- Parameter Server -----------------------------------------

# TODO: fix implementation to allow for multiple gradient servers

class GradientServer:

    #TODO: find out how to use cuda kernel for sparse tensor addition / implement if there is no kernel
    #TODO: metrics for computations in average_gradient

    def __init__(self, rank, trainer_count, backend, use_cuda_rpc, batch_mode):
        self.lock = threading.Lock()

        self.rank = rank
        self.trainer_count = trainer_count
        self.use_cuda_rpc = use_cuda_rpc
        self.backend = backend
        self.batch_mode = batch_mode

        self.futures = {}
        self.gradient = {}
        self.batch_number = 0


    @staticmethod
    @rpc.functions.async_execution
    def average_gradient(gs_rref, rank, bp_loc, received_batch_number, **kwargs):
        sparse_gradient = False
        if SPARSE in kwargs and kwargs[SPARSE]:
            assert INDICES in kwargs and kwargs[INDICES] is not None
            assert VALUES in kwargs and kwargs[VALUES] is not None
            assert SIZE in kwargs and kwargs[SIZE] is not None
            sparse_gradient = True
            gradient = torch.sparse_coo_tensor(kwargs[INDICES], kwargs[VALUES], kwargs[SIZE])
        else:
            gradient = kwargs[GRADIENT]
        self = gs_rref.local_value()
        gradient = gradient.cuda(self.rank)
        fut = torch.futures.Future()
        with self.lock:
            if self.batch_number < received_batch_number[0]:
                self.batch_number = received_batch_number[0]
                self.clear_state()
            if bp_loc not in self.gradient:
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
                    bp_loc_avg = [bp_loc_avg._indices(), bp_loc_avg._values(), bp_loc_avg.size()]
                for cur_fut in self.futures[bp_loc]:
                    cur_fut.set_result(bp_loc_avg)
                return fut
        return fut

    def clear_state(self):
        self.futures.clear()
        self.gradient.clear()

    def reset(gs_rref):
        self = gs_rref.local_value()
        self.futures.clear()
        self.gradient.clear()
        self.batch_number = 0

# --------------------------- Model -----------------------------------------


class DummyModel(nn.Module):
    def __init__(self, num_embeddings=4, embedding_dim=4, dense_input_size=4, dense_output_size=4, sparse=True):
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_dim, sparse=sparse)
        self.fc1 = nn.Linear(dense_input_size, dense_output_size)

    def forward(self, x):
        x = self.embedding(x)
        return F.softmax(self.fc1(x), dim=1)

def get_model(model_id, model_config):
    if model_id == 1:
        return DummyModel(**model_config)
    sys.exit("model_id not found")

# --------------------------- Data -----------------------------------------


class RandomData:
    def __init__(self, min_val: int = 0, max_val: int = 4, batch_size: int = 4, mult: int = 2):
        self.input = torch.randint(min_val, max_val, [batch_size, mult])
        self.target = torch.randint(min_val, max_val, [batch_size])

    def get_input_and_target(self):
        return self.input, self.target

def get_data(data_id, data_config):
    if data_id == 1:
        return RandomData(**data_config)
    sys.exit("data_id not found")

# --------------------------- Hooks -----------------------------------------

#TODO look into refactoring this method
def register_hook(hook_id, ddp_model, process_group, gs_rref, world_size, rank, use_cuda_rpc, futures, bp_location, batch_number, metrics):

    def get_tensors(bucket):
        parameter_tensors = bucket.get_per_parameter_tensors()
        parameter_tensors_count = len(parameter_tensors)
        if parameter_tensors_count > 0:
            return parameter_tensors
        else:
            return [bucket.get_tensor()]        

    def send_request(tensor, sparse=True, dense=False):

        if use_cuda_rpc:
            tensor = tensor.cuda(rank)
        else:
            tensor = tensor.cpu()

        if tensor.is_sparse and sparse:
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), 
            "sparse_rpc", metrics)
            tensor_indices = tensor._indices()
            tensor_values = tensor._values()
            tensor_size = tensor.size()
            fut = rpc.rpc_async(
                gs_rref.owner(),
                GradientServer.average_gradient,
                args=(
                    gs_rref,
                    rank,
                    bp_location[0],
                    batch_number
                ),
                kwargs={
                    SPARSE: True,
                    INDICES: tensor_indices,
                    VALUES: tensor_values,
                    SIZE: tensor_size})
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "sparse_rpc", metrics)
        elif not tensor.is_sparse and dense:
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "dense_rpc", metrics)
            fut = rpc.rpc_async(
                gs_rref.owner(),
                GradientServer.average_gradient,
                args=(
                    gs_rref, 
                    rank,
                    bp_location[0],
                    batch_number
                ),
                kwargs={GRADIENT: tensor}
            )
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "dense_rpc", metrics)
        else:
            sys.exit("invalid request tensor={}, sparse={}, dense={}".format(tensor, sparse, dense))
        return fut

    def send_requests(bucket, sparse=True, dense=False):
        tensors = get_tensors(bucket)
        tensors_len = len(tensors)
        # TODO find a method to skip sending requests during warmup
        if batch_number[0] >= 0:
            for i in range(tensors_len):
                fut = send_request(tensors[tensors_len - 1 - i], sparse, dense)
                futures.append([fut, bp_location[0]])
                bp_location[0] += 1

    if hook_id == 1:
        def rpc_for_sparse_and_dense_hook(state, bucket):
            send_requests(bucket, True, True)
            # After the backward pass, we can manually synchronous sparse gradients or parameters
            fut = torch.futures.Future()
            fut.set_result([bucket.get_tensor()])
            return fut
        ddp_model.register_comm_hook(None, rpc_for_sparse_and_dense_hook)
    elif hook_id == 2:
        def rpc_for_sparse_nccl_allreduce_dense_hook(state, bucket):
            tensor = bucket.get_tensor()
            tensors_count = len(get_tensors(bucket))
            if tensor.is_sparse:
                send_requests(bucket, True, False)
                # After the backward pass, we can manually synchronous sparse gradients or parameters
                fut = torch.futures.Future()
                fut.set_result([bucket.get_tensor()])
                return fut
            else:
                tensor = [tensor / world_size]
                record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "nccl_allreduce_dense", metrics)
                fut=process_group.allreduce(tensor).get_future()
                record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "nccl_allreduce_dense", metrics)
                bp_location[0] += tensors_count
                return fut
        ddp_model.register_comm_hook(None, rpc_for_sparse_nccl_allreduce_dense_hook)
    elif hook_id == 3:
        pass
    elif hook_id == 4:
        def nccl_all_reduce_hook(state, bucket):
            tensor=bucket.get_tensor()
            tensors_count = len(get_tensors(bucket))
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "nccl_allreduce", metrics)
            fut=process_group.allreduce(tensor).get_future()
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "nccl_allreduce", metrics)
            bp_location[0] += tensors_count
            return fut
        ddp_model.register_comm_hook(None, nccl_all_reduce_hook)
    elif hook_id == 5:
        def gloo_allreduce_hook(state, bucket):
            tensor = bucket.get_tensor()
            tensors_count = len(get_tensors(bucket))
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "gloo_allreduce", metrics)
            work=process_group.allreduce([bucket.get_tensor()])
            work.wait()
            fut=torch.futures.Future()
            fut.set_result([bucket.get_tensor() / world_size])
            record_event(rank, HOOK_METRIC, "{},{}".format(batch_number[0],bp_location[0]), "gloo_allreduce", metrics)
            bp_location[0] += tensors_count
            return fut
        ddp_model.register_comm_hook(None, gloo_allreduce_hook)

# --------------------------- Run Worker -----------------------------------------


def run_trainer(configurations, model, data, rank, gs_rref):

    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    model.cuda(rank)

    process_group_size=configurations.world_size - 2

    store=c10d.FileStore("/tmp/tmpn_k_8so02", process_group_size)

    if configurations.backend == GLOO:
        process_group=c10d.ProcessGroupGloo(store, rank, process_group_size)
    elif configurations.backend == NCCL:
        process_group=c10d.ProcessGroupNCCL(store, rank, process_group_size)

    ddp_model=DDP(model, device_ids=[rank], process_group=process_group)
    criterion=nn.CrossEntropyLoss().cuda(rank)
    optimizer=torch.optim.SGD(ddp_model.parameters(), 1e-4)

    input, target=data.get_input_and_target()
    input=input.split(process_group_size)[rank].cuda(rank)
    target=target.split(process_group_size)[rank].cuda(rank)

    hook_gradient_futures=[]
    bp_location = [0]
    batch_number = [-20]
    metrics={}

    register_hook(
        configurations.hook_id,
        ddp_model,
        process_group,
        gs_rref,
        process_group_size,
        rank,
        configurations.use_cuda_rpc,
        hook_gradient_futures,
        bp_location,
        batch_number,
        metrics
    )

    # better name?
    def state_helper():
        hook_gradient_futures.clear()
        bp_location[0] = 0
        batch_number[0] += 1
    
    # rpc warmup required
    # .. warning::
    # Since the buckets are rebuilt after the first iteration, should not rely on the indices at the beginning of training.
    for _ in range(10):  
        state_helper()    
        out = ddp_model(input)
        loss=criterion(out, target)
        loss.backward()
        for fut, bp_loc in hook_gradient_futures:
            gradient=fut.wait()  
                 
    if rank == 0:
        rpc.rpc_sync(gs_rref.owner(),GradientServer.reset, args=(gs_rref,))     
    batch_number[0] = 0
    metrics.clear()
    process_group.barrier()


    ddp_model_parameters=list(ddp_model.parameters())
    ddp_model_parameters_len=len(ddp_model_parameters)

    for i in range(configurations.iterations):
        state_helper()    

        record_event(rank, BATCH_LEVEL_METRIC, i, "batch_all", metrics)

        optimizer.zero_grad()

        record_event(rank, FORWARD_METRIC, i, "forward_pass", metrics)
        out=ddp_model(input)
        record_event(rank, FORWARD_METRIC, i, "forward_pass", metrics)

        loss=criterion(out, target)

        record_event(rank, BACKWARD_METRIC, i, "backward", metrics)
        loss.backward()
        record_event(rank, BACKWARD_METRIC, i, "backward", metrics)
        
        for fut, bp_loc in hook_gradient_futures:
            gradient=fut.wait()
            if isinstance(gradient, list):
                indices=gradient[0].cuda(rank)
                values=gradient[1].cuda(rank)
                size=gradient[2]
                gradient=torch.sparse_coo_tensor(indices, values, size)
            gradient=gradient.cuda(rank)
            ddp_model_parameters[bp_loc].grad=gradient
            
        optimizer.step()

        record_event(rank, BATCH_LEVEL_METRIC, i, "batch_all", metrics)

    # need to add formatting
    # visualization option ? print to file option?
    # pp.pprint("rank={}, metrics={}".format(rank, metrics))

# --------------------------- Run Benchmark -----------------------------------------


def run_benchmark(rank, model, data, configurations):
    world_size=configurations.world_size
    assert world_size > 2
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='29500'
    rpc_backend_options=TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:29501'
    if rank == world_size - 1:
        rpc.init_rpc(
            get_name(rank, world_size),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        gs_rref=rpc.remote(
            get_name(world_size - 2, world_size),
            GradientServer,
            args=(
                world_size-2, 
                world_size-2, 
                configurations.use_cuda_rpc, 
                configurations.backend,
                configurations.batch_mode
            )
        )
        futs=[
            rpc.rpc_async(
                get_name(trainer_rank, world_size),
                run_trainer,
                [
                    configurations, copy.deepcopy(model), copy.deepcopy(data), trainer_rank, gs_rref
                ]
            )
            for trainer_rank in range(0, world_size - 2)
        ]
        for fut in futs:
            fut.wait()
    elif rank == world_size - 2:
        rpc.init_rpc(
            get_name(rank, world_size),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    else:
        if configurations.use_cuda_rpc:
            rpc_backend_options.set_device_map(get_name(world_size - 2, world_size), {rank: world_size - 2})
        rpc.init_rpc(
            get_name(rank, world_size),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    rpc.shutdown()

# --------------------------- Main -----------------------------------------


class Configurations:
    def __init__(
        self,
        hook_id: int,
        backend: str = GLOO,
        ddp_trainers: int=1,
        iterations: int=2,
        batch_mode: bool=False,
        use_cuda_rpc: bool=False,
    ):
        backend=backend.lower()
        assert backend == GLOO or backend == NCCL
        assert hook_id > 0 and hook_id < 6
        assert iterations > 0

        self.backend=backend
        self.batch_mode=batch_mode
        self.hook_id=hook_id
        self.iterations=iterations
        self.world_size=ddp_trainers + 2
        self.use_cuda_rpc=use_cuda_rpc


def main():
    parser=argparse.ArgumentParser(description="RPC PS Benchmark")
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
    args=parser.parse_args()

    benchmark_config=json.load(
        open(
            os.path.join(Path(__file__).parent, "configurations/benchmark_configurations.json"), "r"
        )
    )[args.bconfig_id]
    configurations=Configurations(**benchmark_config)

    data_config=json.load(
        open(
            os.path.join(Path(__file__).parent, "configurations/data_configurations.json"), "r"
        )
    )[args.dconfig_id]
    data=get_data(data_config["data_id"], data_config["configurations"])

    model_config=json.load(
        open(
            os.path.join(Path(__file__).parent, "configurations/model_configurations.json"), "r"
        )
    )[args.mconfig_id]
    model=get_model(model_config["model_id"], model_config["configurations"])

    print("{}\nbconfig_id={}\ndconfig_id={}\nmconfig_id={}\n".format(parser.description, args.bconfig_id, args.dconfig_id, args.mconfig_id))

    start=time.time()
    mp.spawn(
        run_benchmark,
        [model, data, configurations],
        nprocs=configurations.world_size,
        join=True
    )
    print("\nbenchmark done {}".format(time.time() - start))


if __name__ == "__main__":
    main()
