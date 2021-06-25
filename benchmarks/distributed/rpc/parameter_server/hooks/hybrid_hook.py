import torch
import torch.distributed as c10d


def hybrid_hook(state, bucket):
    r"""
    A ddp communication hook that uses two process groups.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    cref = state.cref
    tensor = bucket.get_tensor()
    key = state.get_key(bucket.get_index())

    if tensor.is_sparse:
        cref.record_start("hook_c10d_metric", key, "gloo_sparse_allreduce")
        tensor = tensor / state.process_group.size()
        c10d.all_reduce(tensor, op=c10d.ReduceOp.SUM)
        cref.record_end("hook_c10d_metric", key)
        fut = torch.futures.Future()
        fut.set_result([tensor])
    else:
        cref.record_start("hook_future_metric", key, "nccl_dense_allreduce")
        tensors = [bucket.get_tensor() / state.process_group.size()]
        fut = state.process_group.allreduce(tensors).get_future()

        def callback(fut):
            cref.record_end("hook_future_metric", key)
            return fut.wait()

        fut = fut.then(callback)
    return fut
