import torch
import torch.distributed as c10d
from utils import process_bucket_with_remote_server


def allreduce_hook(state, bucket):
    r"""
    A ddp communication hook that uses the process_group allreduce implementation.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    cref = state.cref
    tensor = bucket.buffer()
    tensors = [tensor / state.process_group.size()]
    key = state.get_key(bucket.get_index())
    if tensor.is_sparse:
        tensor = tensor.coalesce()
    tensor_type = "sparse" if tensor.is_sparse else "dense"
    cref.record_start(
        "hook_future_metric", key, f"{cref.backend}_{tensor_type}_allreduce"
    )
    fut = state.process_group.allreduce(tensors).get_future()

    def callback(fut):
        cref.record_end("hook_future_metric", key)
        return fut.wait()

    return fut.then(callback)


def hybrid_hook(state, bucket):
    r"""
    A ddp communication hook that uses Gloo default process
    group for sparse gradients and NCCL non-default process
    group for dense gradients.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    cref = state.cref
    tensor = bucket.buffer()
    key = state.get_key(bucket.get_index())

    if tensor.is_sparse:
        cref.record_start("hook_c10d_metric", key, "gloo_sparse_allreduce")
        tensor = tensor.coalesce()
        tensor = tensor / state.process_group.size()
        c10d.all_reduce(tensor, op=c10d.ReduceOp.SUM)
        cref.record_end("hook_c10d_metric", key)
        fut = torch.futures.Future()
        fut.set_result([tensor])
    else:
        cref.record_start("hook_future_metric", key, "nccl_dense_allreduce")
        tensors = [bucket.buffer() / state.process_group.size()]
        fut = state.process_group.allreduce(tensors).get_future()

        def callback(fut):
            cref.record_end("hook_future_metric", key)
            return fut.wait()

        fut = fut.then(callback)
    return fut


def rpc_hook(state, bucket):
    r"""
    A ddp communication hook that averages sparse and dense tensors using
    process_bucket_with_remote_server method.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    return process_bucket_with_remote_server(state, bucket)


def sparse_rpc_hook(state, bucket):
    r"""
    A ddp communication hook that uses the current backend allreduce
    implementation for dense tensors and a server for sparse tensors.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    tensor = bucket.buffer()
    if tensor.is_sparse:
        return process_bucket_with_remote_server(state, bucket)
    else:
        cref = state.cref
        tensor = [tensor / state.process_group.size()]
        key = state.get_key(bucket.get_index())
        cref.record_start("hook_future_metric", key, f"{cref.backend}_dense_allreduce")
        fut = state.process_group.allreduce(tensor).get_future()

        def callback(fut):
            cref.record_end("hook_future_metric", key)
            return fut.wait()

        return fut.then(callback)
