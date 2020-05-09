import torch
from typing import Union, Sequence, List, Tuple

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

_dtype = torch.dtype
_device = torch.device
_qscheme = torch.qscheme
_size = Union[torch.Size, List[_int], Tuple[_int, ...]]
_layout = torch.layout

# Meta-type for "numeric" things; matches our docs
Number = Union[builtins.int, builtins.float, builtins.bool]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device = Union[_device, str, None]
