import torch

RPC_SPARSE = "rpc_sparse"
RPC_DENSE = "rpc_dense"


def sparse_tensor_to_rpc_format(sparse_tensor):
    r"""
    A helper function creates a list containing the indices, values, and size
    of a coalesced sparse tensor.
    Args:
        sparse_tensor (torch.Tensor): sparse_coo_tensor represented as a list
    """
    sparse_tensor = sparse_tensor.coalesce()
    return [sparse_tensor.indices(), sparse_tensor.values(), sparse_tensor.size()]


def sparse_rpc_format_to_tensor(sparse_rpc_format):
    r"""
    A helper function creates a sparse_coo_tensor from indices, values, and size.
    Args:
        sparse_rpc_format (list): sparse_coo_tensor represented as a list
    """
    return torch.sparse_coo_tensor(
        sparse_rpc_format[0], sparse_rpc_format[1], sparse_rpc_format[2]
    ).coalesce()


def process_bucket_with_remote_server(state, bucket):
    r"""
    Processes a gradient bucket passed by a DDP communication hook
    during .backward(). The method supports processing sparse and dense
    tensors. It records RPC future completion time metric for the trainer.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    cref = state.cref
    tensor = bucket.buffer()
    if not cref.use_cuda_rpc:
        tensor = tensor.cpu()
    sparse = tensor.is_sparse
    if sparse:
        tensor = sparse_tensor_to_rpc_format(tensor)
    b_index = bucket.get_index()
    server_args = [
        cref.server_rref,
        state.batch_number,
        b_index,
        tensor
    ]
    key = state.get_key(b_index)
    cref.record_start(
        "hook_future_metric",
        key,
        RPC_SPARSE if sparse else RPC_DENSE
    )
    fut = cref.server_rref.rpc_async().average_gradient(*server_args)

    def callback(fut):
        cref.record_end("hook_future_metric", key)
        tensor = fut.wait()
        if type(tensor) is list:
            tensor = sparse_rpc_format_to_tensor(tensor)
        tensor = tensor.cuda(cref.rank)
        return [tensor]

    return fut.then(callback)
