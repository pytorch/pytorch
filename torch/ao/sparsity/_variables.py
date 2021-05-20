import torch

def get_sparse_mapping():
    _sparse_mapping = dict({
        torch.nn.Linear: torch.ao.nn.sparse.Linear,
    })
    return _sparse_mapping

def get_static_sparse_quantized_mapping():
    _static_sparse_quantized_mapping = dict({
        # Dense
        torch.nn.Linear: torch.ao.nn.sparse.quantized.Linear,
        # Sparse
        torch.ao.nn.sparse.Linear: torch.ao.nn.sparse.quantized.Linear,
    })
    return _static_sparse_quantized_mapping

def get_dynamic_sparse_quantized_mapping():
    _dynamic_sparse_quantized_mapping = dict({
        # Dense
        torch.nn.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear,
        # Sparse
        torch.ao.nn.sparse.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear,
    })
    return _dynamic_sparse_quantized_mapping
