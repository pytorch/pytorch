import torch


def sparse_tensor_to_rpc_format(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    return [sparse_tensor.indices(), sparse_tensor.values(), sparse_tensor.size()]


def sparse_rpc_format_to_tensor(sparse_rpc_format):
    return torch.sparse_coo_tensor(
        sparse_rpc_format[0], sparse_rpc_format[1], sparse_rpc_format[2]
    ).coalesce()
