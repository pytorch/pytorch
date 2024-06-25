# mypy: allow-untyped-defs
__all__ = [
    "get_static_sparse_quantized_mapping",
    "get_dynamic_sparse_quantized_mapping",
]

def get_static_sparse_quantized_mapping():
    import torch.ao.nn.sparse
    _static_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.Linear,
    }
    return _static_sparse_quantized_mapping

def get_dynamic_sparse_quantized_mapping():
    import torch.ao.nn.sparse
    _dynamic_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear,
    }
    return _dynamic_sparse_quantized_mapping
