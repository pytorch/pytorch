def allreduce_hook(state, bucket):
    r"""
    A ddp communication hook that uses the process_group allreduce implementation.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    cref = state.cref
    tensor = bucket.get_tensor()
    tensors = [tensor / state.process_group.size()]
    key = state.get_key(bucket.get_index())
    tensor_type = "sparse" if tensor.is_sparse else "dense"
    cref.record_start("hook_future_metric", key, f"{cref.backend}_{tensor_type}_allreduce")
    fut = state.process_group.allreduce(tensors).get_future()

    def callback(fut):
        cref.record_end("hook_future_metric", key)
        return fut.wait()

    return fut.then(callback)
