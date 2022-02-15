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

Returns an opaque object (a "DLPack capsule") representing the tensor.

.. note::
  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns
  cannot be used for anything in Python other than use it as input to
  ``from_dlpack``. The more idiomatic use of DLPack is to call
  ``from_dlpack`` directly on the tensor object - this works when that
  object has a ``__dlpack__`` method, which PyTorch and most other
  libraries indeed have now.

.. warning::
  Only call ``from_dlpack`` once per capsule produced with ``to_dlpack``.
  Behavior when a capsule is consumed multiple times is undefined.

Args:
    tensor: a tensor to be exported

The DLPack capsule shares the tensor's memory.
""")

# TODO: add a typing.Protocol to be able to tell Mypy that only objects with
# __dlpack__ and __dlpack_device__ methods are accepted.
def from_dlpack(ext_tensor: Any) -> torch.Tensor:
    """from_dlpack(ext_tensor) -> Tensor

    Converts a tensor from an external library into a ``torch.Tensor``.

    The returned PyTorch tensor will share the memory with the input tensor
    (which may have come from another library). Note that in-place operations
    will therefore also affect the data of the input tensor. This may lead to
    unexpected issues (e.g., other libraries may have read-only flags or
    immutable data structures), so the user should only do this if they know
    for sure that this is fine.

    Args:
        ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):
            The tensor or DLPack capsule to convert.

            If ``ext_tensor`` is a tensor (or ndarray) object, it must support
            the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``
            method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is
            an opaque ``PyCapsule`` instance, typically produced by a
            ``to_dlpack`` function or method.

    Examples::

        >>> import torch.utils.dlpack
        >>> t = torch.arange(4)

        # Convert a tensor directly (supported in PyTorch >= 1.10)
        >>> t2 = torch.from_dlpack(t)
        >>> t2[:2] = -1  # show that memory is shared
        >>> t2
        tensor([-1, -1,  2,  3])
        >>> t
        tensor([-1, -1,  2,  3])

        # The old-style DLPack usage, with an intermediate capsule object
        >>> capsule = torch.utils.dlpack.to_dlpack(t)
        >>> capsule
        <capsule object "dltensor" at 0x7f6017d14300>
        >>> t3 = torch.from_dlpack(capsule)
        >>> t3
        tensor([-1, -1,  2,  3])
        >>> t3[0] = -9  # now we're sharing memory between 3 tensors
        >>> t3
        tensor([-9, -1,  2,  3])
        >>> t2
        tensor([-9, -1,  2,  3])
        >>> t
        tensor([-9, -1,  2,  3])

    """
    if hasattr(ext_tensor, '__dlpack__'):
        device = ext_tensor.__dlpack_device__()
        # device is either CUDA or ROCm, we need to pass the current
        # stream
        if device[0] in (DLDeviceType.kDLGPU, DLDeviceType.kDLROCM):
            stream = torch.cuda.current_stream('cuda:{}'.format(device[1]))
            # cuda_stream is the pointer to the stream and it is a public
            # attribute, but it is not documented
            # The array API specify that the default legacy stream must be passed
            # with a value of 1 for CUDA
            # https://data-apis.org/array-api/latest/API_specification/array_object.html?dlpack-self-stream-none#dlpack-self-stream-none  # NOQA
            is_cuda = device[0] == DLDeviceType.kDLGPU
            # Since pytorch is not using PTDS by default, lets directly pass
            # the legacy stream
            stream_ptr = 1 if is_cuda and stream.cuda_stream == 0 else stream.cuda_stream
            dlpack = ext_tensor.__dlpack__(stream=stream_ptr)
        else:
            dlpack = ext_tensor.__dlpack__()
    else:
        # Old versions just call the converter
        dlpack = ext_tensor
    return _from_dlpack(dlpack)
