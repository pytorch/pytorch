from functools import wraps
import os
import random
import time
import threading
import copy
import sys

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

GLOO = "gloo"
NCCL = "nccl"

# --------------------------- privates -----------------------------------------


def _get_name(rank, world_size):
    if rank < world_size - 2:
        return "trainer{}".format(rank)
    elif rank == world_size - 2:
        return "gs"
    else:
        return "master"


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)

# --------------------------- Parameter Server -----------------------------------------


class GradientServer(nn.Module):

    def __init__(self, world_size):
        super().__init__()
        torch.manual_seed(0)
        self.lock = threading.Lock()
        self.rank_gradient_counters = [[0] * world_size, [0] * world_size]
        self.gradient_dicts = [{}, {}]
        self.gradient_dim = {}

    # batch processing just store the gradients
    # trainer and server update every iteration
    # batch case - 1 cuda kernel
    # iteration case - n cuda kernels

    @rpc.functions.async_execution
    def add_to_sparse_gradient(self, rank, gradient, dim):
        gradient = gradient.cuda()
        sparse_gradient_dict = self.gradient_dicts[0]
        with self.lock:
            loc = self.rank_gradient_counter[0][rank]
            if loc not in sparse_gradient_dict:
                sparse_gradient_dict[loc] = gradient
                gradient_dim[loc] = dim
            else:
                sparse_gradient_dict[loc] += gradient
            self.rank_gradient_counter[0][rank] += 1

    @rpc.functions.async_execution
    def add_to_dense_gradient(self, rank, gradient, dim):
        gradient = gradient.cuda()
        dense_gradient_dict = self.gradient_dicts[1]
        with self.lock:
            loc = self.rank_gradient_counter[1][rank]
            if loc not in sparse_gradient_dict:
                dense_gradient_dict[loc] = gradient
            else:
                dense_gradient_dict[loc] += gradient
            self.rank_gradient_counter[1][rank] += 1

    @rpc.functions.async_execution
    def get_loc_gradient(self, loc, sparse):
        index = 0 if sparse else 1
        with self.lock:
            gradient_cpu = self.gradient_dict[index][loc].cpu()
            if index == 0:
                return gradient_cpu, self.gradient_dim[loc]
            else:
                return gradient_cpu

    def reset_gradients(self):
        self.gradient_dict = {}


# --------------------------- Model -----------------------------------------

class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(4, 4, sparse=True)
        self.fc1 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.embedding(x)
        return F.softmax(self.fc1(x), dim=1)

# --------------------------- Hooks -----------------------------------------


def register_ddp_with_rpc_for_sparse_and_dense_hook(ddp_model, process_group, gs_rref, world_size):
    """
        DDP with CUDA RPC for both sparse and dense parameters.
        hook_id = 1

    """
    sparse_future_container = []
    dense_future_container = []

    def rpc_for_sparse_and_dense_hook(state, bucket):
        grad = bucket.get_tensors()[0]
        if grad.is_sparse:
            dim = grad.sparse_dim()
            cpu_tensor = grad.to_dense().cpu()
            sparse_future_container.append(
                _remote_method(
                    GradientServer.add_to_sparse_gradient,
                    gs_rref,
                    rank=rank,
                    gradient=cpu_tensor,
                    dim=dim,
                )
            )
        else:
            cpu_tensor = grad.cpu()
            dense_future_container.append(
                _remote_method(
                    GradientServer.add_to_dense_gradient,
                    gs_rref,
                    rank=rank,
                    gradient=cpu_tensor,
                )
            )
        fut = torch.futures.Future()
        fut.set_result(bucket.get_tensors()[0])
        return fut

    ddp_model.register_comm_hook(None, rpc_for_sparse_and_dense_hook)

    return sparse_future_container, dense_future_container


def register_dpp_with_rpc_for_sparse_nccl_allreduce_dense_hook(ddp_model, process_group, gs_rref, world_size):
    """
        DDP with CPU RPC for sparse parameters + NCCL AllReduce for dense parameters
        hook_id = 2

    """
    sparse_future_container = []

    def rpc_for_sparse_nccl_allreduce_dense_hook(state, bucket):
        grad = bucket.get_tensors()[0]
        if grad.is_sparse:
            dim = grad.sparse_dim()
            cpu_tensor = grad.to_dense()
            sparse_future_container.append(
                _remote_method(
                    GradientServer.add_to_sparse_gradient,
                    gs_rref,
                    rank=rank,
                    gradient=cpu_tensor,
                    dim=dim,
                )
            )
            fut = torch.futures.Future()
            fut.set_result(bucket.get_tensors())
            return fut
        else:
            tensors = [t / world_size for t in bucket.get_tensors()]
            return process_group.allreduce(tensors).get_future()

    ddp_model.register_comm_hook(None, rpc_for_sparse_nccl_allreduce_dense_hook)

    return sparse_future_container


def register_gloo_allreduce_for_sparse_and_nccl_allreduce_for_dense_hook(ddp_model, process_group, gs_rref, world_size):
    pass


def register_dpp_with_nccl_allreduce_hook(ddp_model, process_group, gs_rref, world_size):
    """
        DDP with NCCL ALLReduce for both sparse and dense gradients
        hook_id = 4
    """

    def nccl_all_reduce_hook(state, bucket):
        tensors = [t / world_size for t in bucket.get_tensors()]
        return process_group.allreduce(tensors).get_future()

    ddp_model.register_comm_hook(None, nccl_all_reduce_hook)


def register_gloo_allreduce_hook(ddp_model, process_group, gs_rref, world_size):
    """
        DDP with Gloo ALLReduce for both sparse and dense gradients
        hook_id = 5
    """

    def gloo_allreduce_hook(state, bucket):
        work = process_group.allreduce(bucket.get_tensors())
        work.wait()
        fut = torch.futures.Future()
        fut.set_result([t * world_size for t in bucket.get_tensors()])
        return fut

    ddp_model.register_comm_hook(None, gloo_allreduce_hook)

# --------------------------- Run Worker -----------------------------------------


def _run_trainer(benchmark_configurations, model, rank, gs_rref):

    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    model.cuda(rank)

    process_group_size = benchmark_configurations.world_size - 2

    store = c10d.FileStore("/tmp/tmpn_k_8so02", process_group_size)

    if benchmark_configurations.backend == GLOO:
        process_group = c10d.ProcessGroupGloo(store, rank, process_group_size)
    else:
        process_group = c10d.ProcessGroupNCCL(store, rank, process_group_size)

    ddp_model = DDP(model, device_ids=[rank], process_group=process_group)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    hook_gradient_futures = inverse_hook_map[benchmark_configurations.hook_id](
        ddp_model, process_group, gs_rref, process_group_size)

    # TODO -> add data models
    mult = 2
    batch_size = 4
    input = torch.randint(0, 4, [batch_size, 2]).split(mult)[rank].cuda(rank)
    target = torch.randint(0, 4, [batch_size]).split(mult)[rank].cuda(rank)

    for i in range(benchmark_configurations.iterations):
        optimizer.zero_grad()
        out = ddp_model(input)
        loss = criterion(out, target)
        loss.backward()

        if hook_gradient_futures is not None:
            for gradient_futures in hook_gradient_futures:
                for fut in gradient_futures:
                    fut.wait()
            process_group.barrier()
            sparse_loc = 0
            dense_loc = 0
            for param in ddp_model.parameters():
                if param.grad.is_sparse:
                    gradient_value, dim = _remote_method(
                        GradientServer.get_loc_gradient, gs_rref, loc=sparse_loc, sparse=True
                    ).wait()
                    gradient_value /= (1.0 * process_group_size)
                    gradient_value = gradient_value.to_sparse(dim)
                    gradient_value = gradient_value.cuda(rank)
                    param.grad = gradient_value
                    sparse_loc += 1
                elif len(hook_gradient_futures) > 1:
                    gradient_value = _remote_method(
                        GradientServer.get_loc_gradient, gs_rref, loc=dense_loc, sparse=False
                    ).wait()
                    gradient_value /= (1.0 * process_group_size)
                    gradient_value = gradient_value.cuda(rank)
                    param.grad = gradient_value
                    dense_loc += 1

        # batch processing ?

        optimizer.step()

# --------------------------- Run Benchmark -----------------------------------------


def run_benchmark(rank, model, benchmark_configurations):
    world_size = benchmark_configurations.world_size
    assert world_size > 2
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = 'tcp://localhost:29501'
    if rank == world_size - 1:
        rpc.init_rpc(
            _get_name(rank, world_size),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        gs_rref = rpc.remote(
            _get_name(world_size - 2, world_size),
            GradientServer,
            [world_size - 2],
        )
        futs = [
            rpc.rpc_async(
                _get_name(trainer_rank, world_size),
                _run_trainer,
                [
                    benchmark_configurations, copy.deepcopy(model), trainer_rank, gs_rref
                ]
            )
            for trainer_rank in range(0, world_size - 2)
        ]
        for fut in futs:
            fut.wait()
    elif rank == world_size - 2:
        rpc.init_rpc(
            _get_name(rank, world_size),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    else:
        if benchmark_configurations.use_rpc_cuda:
            rpc_backend_options.set_device_map(_get_name(world_size - 2, world_size), {rank: world_size - 2})
        rpc.init_rpc(
            _get_name(rank, world_size),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
    rpc.shutdown()

# --------------------------- Main -----------------------------------------


inverse_hook_map = {
    1: register_ddp_with_rpc_for_sparse_and_dense_hook,
    2: register_dpp_with_rpc_for_sparse_nccl_allreduce_dense_hook,
    3: register_gloo_allreduce_for_sparse_and_nccl_allreduce_for_dense_hook,
    4: register_dpp_with_nccl_allreduce_hook,
    5: register_gloo_allreduce_hook
}


class BenchmarkConfigurations:
    backend = GLOO
    batch_rpc = False
    batch_size = 1
    hook_id = 1
    iterations = 1
    use_rpc_cuda = False
    world_size = 4


if __name__ == "__main__":

    benchmark_configurations = BenchmarkConfigurations()
    benchmark_configurations.world_size = 4
    benchmark_configurations.use_rpc_cuda = False
    benchmark_configurations.backend = GLOO
    benchmark_configurations.hook_id = 5

    start = time.time()
    mp.spawn(
        run_benchmark,
        [MixedModel(), benchmark_configurations],
        nprocs=benchmark_configurations.world_size,
        join=True
    )
    print(time.time() - start)

