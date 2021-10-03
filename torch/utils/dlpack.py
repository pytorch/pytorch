from typing import Any

import torch
import enum

from torch._C import _from_dlpack
from torch._C import _to_dlpack as to_dlpack


class DLDeviceType(enum.IntEnum):
    # Enums as in DLPack specification (aten/src/ATen/dlpack.h)
    kDLCPU = 1,
    kDLGPU = 2,
    kDLCPUPinned = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLExtDev = 12,


torch._C._add_docstr(to_dlpack, r"""to_dlpack(tensor) -> PyCapsule

Returns a DLPack representing the tensor.

Args:
    tensor: a tensor to be exported

The DLPack shares the tensors memory.
Note that each DLPack can only be consumed once.
""")

# TODO: add a typing.Protocol to be able to tell Mypy that only objects with
# __dlpack__ and __dlpack_device__ methods are accepted.
def from_dlpack(ext_tensor: Any) -> torch.Tensor:
    """from_dlpack(ext_tensor) -> Tensor

    Convers a tensor from a external library into a ``torch.Tensor``
    by means of the ``__dlpack__`` protocol.

    The tensor will share the memory with the object represented
    in the DLPack.

    .. warning::
      Only call from_dlpack once per capsule. Its behavior when used
      on the same capsule multiple times is undefined.

    Args:
        ext_tensor (object with __dlpack__ attribute or DLPack capsule):
            The tensor or DLPack capsule to convert.
    """
    if hasattr(ext_tensor, '__dlpack__'):
        device = ext_tensor.__dlpack_device__()
        # device is either CUDA or ROCm, we need to pass the current
        # stream
        if device[0] in (DLDeviceType.kDLGPU, DLDeviceType.kDLROCM):
            stream = torch.cuda.current_stream('cuda:{}'.format(device[1]))
            # cuda_stream is the pointer to the stream and it is a public
            # attribute, but it is not documented
            dlpack = ext_tensor.__dlpack__(stream=stream.cuda_stream)
        else:
            dlpack = ext_tensor.__dlpack__()
    else:
        # Old versions just call the converter
        dlpack = ext_tensor
    return _from_dlpack(dlpack)
