import torch

def fake_empty_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.empty_like(*args, **kwargs)

def fake_rand_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.rand_like(*args, **kwargs)

def fake_randint_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.randint_like(*args, **kwargs)

def fake_randn_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.randn_like(*args, **kwargs)

def fake_ones_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.ones_like(*args, **kwargs)

def fake_zeros_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.zeros_like(*args, **kwargs)

def fake_full_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.full_like(*args, **kwargs)
