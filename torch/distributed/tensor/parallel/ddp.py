from typing import Any, List, Tuple

import torch.nn as nn
from torch.distributed.tensor.parallel.fsdp import _flatten_tensor, _unflatten_tensor

__all__ = ["pre_dp_model_transform"]


def _get_submodule_n_params(module: nn.Module, path: str):
    """
    Get submodule and the direct path of parameter from the module
    """
    if "." in path:
        path_list = path.split(".")
        parent_module_path = ".".join(path_list[:-1])
        module = module.get_submodule(parent_module_path)
        path = path_list[-1]
    return module, path


def _update_model_param(param_list: List[Tuple[nn.Module, str, nn.Parameter]]):
    """
    Update parameters within the model
    """
    for item in param_list:
        parent_module, module_path, t = item
        assert hasattr(parent_module, module_path)
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, t)


def _reconstruct_dtensor(model: nn.Module, _input: Any):
    """
    Recontruct DTensor parameters from local tensors
    """
    param_list = []
    for name, t in model.named_parameters():
        if hasattr(t, "_st_info"):
            dtensor = _unflatten_tensor(t, t._st_info)
            param_list.append((*_get_submodule_n_params(model, name), dtensor))
    _update_model_param(param_list)  # type: ignore[arg-type]


def _localize_dtensor(model: nn.Module, _input: Any, _output: Any):
    """
    Convert DTensor parameters to local tensors
    """
    param_list = []
    for name, param in model.named_parameters():
        t, sharding_info = _flatten_tensor(param)
        if sharding_info is not None:
            t = nn.Parameter(t)
            t._st_info = sharding_info  # type: ignore[attr-defined]
            param_list.append((*_get_submodule_n_params(model, name), t))
    _update_model_param(param_list)  # type: ignore[arg-type]


def pre_dp_model_transform(model: nn.Module):
    """
    The API is to enable the composability between Tensor Parallelism (TP)
    and Data Parallelism(DP) in PyTorch. We need to convert Parameters which
    are DTensors to local tensors before wrapping with data parallelism API.
    We then register two hooks, one for converting local tensors back to DTensor
    preforward and one to convert DTensors back to tensors after Forward. By
    doing this, we can make DP api not to have special handle of DTensor parameters
    and get DTensor's gradients propogated back to DP, e.g. gradient buckets of DDP.

    For now this API is only for ``DistributedDataParallel`` and we will merge
    all other DP composability methonds into this API down the road.

    Args:
        module (:class:`nn.Module`):
            Module which has been applied TP on.

    Return:
        A :class:`nn.Module` object transformed for later-on DP.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>>
        >>> # Define the module.
        >>> m = Model(...)
        >>> m = parallelize_module(m, PairwiseParallel())
        >>> m = pre_dp_model_transform(m)
        >>> m = DDP(m)
        >>>
    """

    _localize_dtensor(model, None, None)
    model.register_forward_pre_hook(_reconstruct_dtensor)
    model.register_forward_hook(_localize_dtensor)
