import torch

from torch._C import _from_dlpack as from_dlpack
from torch._C import _to_dlpack as to_dlpack

torch._C._add_docstr(from_dlpack, r"""from_dlpack(dlpack) -> Tensor

Decodes a DLPack to a tensor.

Args:
    dlpack: a PyCapsule object with the dltensor

The tensor will share the memory with the object represented
in the dlpack.
Note that each dlpack can only be consumed once.
""")

torch._C._add_docstr(to_dlpack, r"""to_dlpack(tensor) -> PyCapsule

Returns a DLPack representing the tensor.

Args:
    tensor: a tensor to be exported

The dlpack shares the tensors memory.
Note that each dlpack can only be consumed once.
""")
