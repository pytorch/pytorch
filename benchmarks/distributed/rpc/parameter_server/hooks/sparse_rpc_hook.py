from utils import process_bucket_with_remote_server


def sparse_rpc_hook(state, bucket):
    r"""
    A ddp communication hook that uses the current backend allreduce
    implementation for dense tensors and a server for sparse tensors.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    tensor = bucket.get_tensor()
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
