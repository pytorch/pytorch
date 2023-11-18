import torch
from typing import Any, Dict, List, Optional, Sequence, Union, cast
from ..modules import Module
from torch.nn.parallel.parallel_apply import parallel_apply

__all__ = ['parallel_map']

def parallel_map(
    modules: Sequence[Module],
    input: Any,
    kwargs_tup: Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
    devices: Optional[Sequence[Optional[Union[int, torch.device]]]] = None,
) -> List[Any]:
    r"""Apply each `module` in :attr:`modules` in parallel on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        input (tensor): input to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, and :attr:`devices` (if given) should 
    all have same length. :attr:`kwargs_tup` (if given) must be the same 
    length as :attr:`modules` or as that of :attr:`input`. Moreover, the
    :attr:`input` can either be a single object as the only argument
    to each module, or a collection of positional arguments.
    """
    # Determine kwargs per module
    if kwargs_tup is not None:
        if isinstance(kwargs_tup, dict):
            kwargs_tup = (kwargs_tup,) * len(modules)
        else:
            assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = (cast(Dict[str, Any], {}),) * len(modules)

    # Determine device per module
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    # Determine if gradients should be enabled
    requires_grad = None
    if isinstance(input, torch.Tensor):
        requires_grad = input.requires_grad
    else:
        requires_grad = False

    # Copy input for each module
    inputs = [input] * len(modules)
    for i in range(len(inputs)):
        inputs[i] = inputs[i].clone()  # Keep within computation graph
        if requires_grad:
            inputs[i].requires_grad = requires_grad

    # Utilize parallel_apply
    outputs = parallel_apply(
        modules=modules,
        inputs=inputs,
        kwargs_tup=kwargs_tup,
        devices=devices
    )
    return outputs
