import torch
from typing import Any, List, Sequence, Tuple, Union

import builtins

# Convenience aliases for common composite types that we need
# to talk about in PyTorch

_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]

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

class SymInt:
    pass

# Meta-type for "numeric" things; matches our docs
Number = Union[builtins.int, builtins.float, builtins.bool]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device = Union[_device, str, _int, None]

# Storage protocol implemented by ${Type}StorageBase classes

class Storage:
    _cdata: int
    device: torch.device
    dtype: torch.dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo) -> 'Storage':  # type: ignore[empty-body]
        ...

    def _new_shared(self, int) -> 'Storage':  # type: ignore[empty-body]
        ...

    def _write_file(self, f: Any, is_real_file: _bool, save_size: _bool, element_size: int) -> None:
        ...

    def element_size(self) -> int:  # type: ignore[empty-body]
        ...

    def is_shared(self) -> bool:  # type: ignore[empty-body]
        ...

    def share_memory_(self) -> 'Storage':  # type: ignore[empty-body]
        ...

    def nbytes(self) -> int:  # type: ignore[empty-body]
        ...

    def cpu(self) -> 'Storage':  # type: ignore[empty-body]
        ...

    def data_ptr(self) -> int:  # type: ignore[empty-body]
        ...

    def from_file(self, filename: str, shared: bool = False, nbytes: int = 0) -> 'Storage':  # type: ignore[empty-body]
        ...

    def _new_with_file(self, f: Any, element_size: int) -> 'Storage':  # type: ignore[empty-body]
        ...

    ...
