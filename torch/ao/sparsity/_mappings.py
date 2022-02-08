import torch
import torch.ao.nn

def get_static_sparse_quantized_mapping():
    _static_sparse_quantized_mapping = dict({
        torch.nn.Linear: torch.ao.nn.sparse.quantized.Linear,
    })
    return _static_sparse_quantized_mapping

def get_dynamic_sparse_quantized_mapping():
    _dynamic_sparse_quantized_mapping = dict({
        torch.nn.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear,
    })
    return _dynamic_sparse_quantized_mapping
