"""The machine this process runs on, for judging tci constraints. It is read as
torchci's report reads it: without a device context, which tests can observe.
``python -m torch.testing._internal.tci.machine`` prints it as JSON."""

from __future__ import annotations

import ctypes
import dataclasses
import functools
import json
import sys

import torch
from torch.testing._internal import tci
from torch.testing._internal.torchci import environment


@functools.cache
def capture() -> tci.Machine:
    env = environment.capture()
    if not env.device_count:  # a CUDA build on a machine with no GPU has none
        return tci.Machine()
    memory = env.properties.get("device_memory_mib")
    # Nothing reports an Intel or an Apple GPU's memory. Elsewhere a blank is a
    # read that failed, and a guess would leave tests out without a trace
    if env.accelerator in ("cuda", "rocm") and not (env.device_name and memory):
        raise RuntimeError(f"can't read the {env.accelerator} GPU's name and memory")
    return tci.Machine(
        accelerator=env.accelerator,
        gpus=env.device_count,
        device=env.device_name,
        gpu_memory=int(memory) / 1024 if memory else None,
        capability=_cuda_capability() if env.accelerator == "cuda" else (0, 0),
    )


def _cuda_capability() -> tuple[int, int]:
    """The first visible GPU's compute capability. From NVML, as its name and its
    memory are: torch.cuda.get_device_capability() creates a CUDA context."""
    nvml = ctypes.CDLL("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
    if nvml.nvmlInit() != 0:
        raise RuntimeError("can't start NVML to read the GPU's compute capability")
    try:
        handle, major, minor = ctypes.c_void_p(), ctypes.c_int(), ctypes.c_int()
        # An int device, so CUDA_VISIBLE_DEVICES is resolved without initializing CUDA
        index = ctypes.c_uint(torch.cuda._get_nvml_device_index(0))
        found = nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))
        capability = ctypes.byref(major), ctypes.byref(minor)
        if found != 0 or nvml.nvmlDeviceGetCudaComputeCapability(handle, *capability):
            raise RuntimeError("NVML can't read the GPU's compute capability")
        return major.value, minor.value
    finally:
        nvml.nvmlShutdown()


if __name__ == "__main__":
    print(json.dumps(dataclasses.asdict(capture())))
