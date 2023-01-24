import torch
from ._functions import Scatter, Gather
import warnings

__all__ = ['scatter', 'scatter_kwargs', 'gather']

def is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    warnings.warn("is_namedtuple is deprecated, please use the python checks instead")
    return _is_namedtuple(obj)

def _is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend(() for _ in range(len(kwargs) - len(inputs)))
    elif len(kwargs) < len(inputs):
        kwargs.extend({} for _ in range(len(inputs) - len(kwargs)))
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device.
    Use 'cpu' for CPU to avoid a deprecation warning.
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs]))
                             for k in out)
        if _is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res
