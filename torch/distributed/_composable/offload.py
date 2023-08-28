import functools

from enum import auto, Enum
from typing import List

import torch

import torch.nn as nn

from torch.distributed._composable_state import _State
from .contract import contract


class _DeviceState(Enum):
    CPU = auto()
    GPU_UNSYNCED = auto()
    GPU_SYNCED = auto()


@contract()
def frozen_offload(
    module: nn.Module,
):
    state = frozen_offload.state(module)
    _check_module_params(module)
    _init_data(state, module)
    state._forward_handle = module.register_forward_hook(to_cpu)
    state._forward_pre_handle = module.register_forward_pre_hook(to_gpu)


def to_cpu(module: nn.Module, *unused_args, **unused_kwargs):
    """
    Frees the GPU parameter memory.
    """
    state = frozen_offload.state(module)
    if state is None:
        return
    if state._device_state == _DeviceState.CPU:
        return
    print(f"Resetting to CPU!")
    # Choose to not persist any changes from the GPU copy to the CPU copy
    for param, cpu_view in zip(state._params, state._cpu_views):
        param.data = cpu_view
    state._gpu_flat_tensor.untyped_storage().resize_(0)
    state._device_state = _DeviceState.CPU


def to_gpu(module: nn.Module, *unused_args, **unused_kwargs):
    """
    Allocates GPU memory for the parameters and copies them from CPU to GPU in
    a separate CUDA stream. This must be followed with a ``wait_on_copy()``
    call to ensure correct synchronization (due to the stream usage).
    """
    state = frozen_offload.state(module)
    if state is None:
        return
    if state._device_state in (_DeviceState.GPU_UNSYNCED, _DeviceState.GPU_SYNCED):
        return
    print(f"Copying to GPU!")
    state._gpu_flat_tensor.untyped_storage().resize_(
        state._gpu_flat_tensor_storage_size
    )
    with torch.cuda.stream(state._stream):
        state._gpu_flat_tensor.copy_(state._cpu_flat_tensor)
        for param, gpu_view in zip(state._params, state._gpu_views):
            param.data = gpu_view
    torch.cuda.current_stream().wait_stream(state._stream)
    # state._device_state = _DeviceState.GPU_UNSYNCED
    state._device_state = _DeviceState.GPU_SYNCED


def _init_data(state: _State, module: nn.Module) -> None:
    state._params: List[nn.Parameter] = []
    state._numels: List[int] = []
    state._sizes: List[torch.Size] = []
    state._stream = torch.cuda.Stream()
    state._device_state = _DeviceState.CPU
    total_numel = 0
    for param in module.parameters():
        state._params.append(param)
        state._numels.append(param.numel())
        state._sizes.append(param.size())
        total_numel += param.numel()
    if len(state._params) == 0:
        return
    cpu_flat_tensor = torch.empty(
        (total_numel,), device=torch.device("cpu"), dtype=state._params[0].dtype
    ).pin_memory()
    gpu_flat_tensor = torch.empty(
        (total_numel,), device=torch.device("cuda"), dtype=state._params[0].dtype
    )
    state._cpu_views: List[torch.Tensor] = []
    state._gpu_views: List[torch.Tensor] = []
    offset = 0
    for param, numel, size in zip(state._params, state._numels, state._sizes):
        state._cpu_views.append(cpu_flat_tensor[offset : offset + numel].view(size))
        state._gpu_views.append(gpu_flat_tensor[offset : offset + numel].view(size))
        param.data = state._cpu_views[-1]
        offset += numel
    state._gpu_flat_tensor_storage_size = gpu_flat_tensor.untyped_storage().size()
    gpu_flat_tensor.untyped_storage().resize_(0)
    state._cpu_flat_tensor = cpu_flat_tensor
    state._gpu_flat_tensor = gpu_flat_tensor


def _check_module_params(module: nn.Module) -> None:
    dtype = None
    for param_name, param in module.named_parameters():
        if param.requires_grad:
            raise AssertionError(
                "frozen_offload requires all parameters to be frozen "
                f"(requires_grad=False) but got {param_name} with requires_grad=True"
            )
        if dtype is not None:
            if param.dtype != dtype:
                raise NotImplementedError(
                    "frozen_offload currently only supports uniform dtype but "
                    f"got both {dtype} and {param.dtype}"
                )
            dtype = param.dtype
