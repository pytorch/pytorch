import builtins
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch

# Convenience aliases for common composite types that we need
# to talk about in PyTorch

_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]
_TensorOrTensorsOrGradEdge = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
    "torch.autograd.graph.GradientEdge",
    Sequence["torch.autograd.graph.GradientEdge"],
]

# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
_int = builtins.int
_float = builtins.float
_bool = builtins.bool
_complex = builtins.complex

_dtype = torch.dtype
_device = torch.device
_qscheme = torch.qscheme
_size = Union[torch.Size, List[_int], Tuple[_int, ...]]
_layout = torch.layout
_dispatchkey = Union[str, torch._C.DispatchKey]

# Meta-type for "numeric" things; matches our docs
Number = Union[_int, _float, _bool]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device = Optional[Union[_device, str, _int]]
del Optional

# Storage protocol implemented by ${Type}StorageBase classes


class Storage:
    _cdata: _int
    device: torch.device
    dtype: torch.dtype
    _torch_load_uninitialized: _bool

    def __deepcopy__(self, memo) -> "Storage":
        raise NotImplementedError

    def _new_shared(self, int) -> "Storage":
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: _bool,
        save_size: _bool,
        element_size: _int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> _int:
        raise NotImplementedError

    def is_shared(self) -> _bool:
        raise NotImplementedError

    def share_memory_(self) -> "Storage":
        raise NotImplementedError

    def nbytes(self) -> _int:
        raise NotImplementedError

    def cpu(self) -> "Storage":
        raise NotImplementedError

    def data_ptr(self) -> _int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: _bool = False,
        nbytes: _int = 0,
    ) -> "Storage":
        raise NotImplementedError

    def _new_with_file(self, f: Any, element_size: _int) -> "Storage":
        raise NotImplementedError
