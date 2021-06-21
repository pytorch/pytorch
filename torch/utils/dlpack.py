from typing import Any

import torch

from torch._C import _from_dlpack
from torch._C import _to_dlpack as to_dlpack

torch._C._add_docstr(to_dlpack, r"""to_dlpack(tensor) -> PyCapsule

Returns a DLPack representing the tensor.

Args:
    tensor: a tensor to be exported

The dlpack shares the tensors memory.
Note that each dlpack can only be consumed once.
""")


def from_dlpack(ext_tensor: Any) -> torch.Tensor:
    """from_dlpack(ext_tensor) -> Tensor

    Convers a tensor from a external library into a ``torch.Tensor``
    by means of the ``__dlpack__`` protocol.

    The tensor will share the memory with the object represented
    in the dlpack.

    In order to keep backward compatibility, this function also admits
    to pass a dlpack capsule object.
    Note that each dlpack capsule can only be consumed once.

    Args:
        ext_tensor (object with __dlpack__ attribute or dlpack capsule):
            The tensor from an external library that will be converted
            to a PyTorch one.
    """
    if hasattr(ext_tensor, '__dlpack__'):
        device = ext_tensor.__dlpack_device__()
        # device is either CUDA or ROCm, we need to pass the current
        # stream
        if device[0] in (2, 10):
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
