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


def from_dlpack(ext_tensor) -> torch.Tensor:
    """from_dlpack(dlpack) -> Tensor

    Decodes a DLPack to a tensor.

    Args:
        dlpack: a PyCapsule object with the dltensor

    The tensor will share the memory with the object represented
    in the dlpack.
    Note that each dlpack can only be consumed once.

    Args:
        ext_tensor (object with __dlpack__ attribute or dlpack capsule):
            The tensor from an external library that will be converted
            to a PyTorch one.
    """
    if hasattr(dlpack, '__dlpack__'):
        device = dlpack.__dlpack_device__()
        # device is either CUDA or ROCm, we need to pass the current
        # stream
        if device[0] in (2, 10):
            stream = torch.cuda.stream.current_stream('cuda:{}'.format(device[1]))
            # Should we pass an id? or the producer can accept a stream object
            dlpack = ext_tensor.__dlpack__(stream=stream.cuda_stream)
    else:
        # Old versions just call the converter
        dlpack = tensor
    _from_dlpack(dlpack)
