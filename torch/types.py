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

    def __deepcopy__(self, memo) -> 'Storage':
        ...

    def _new_shared(self, int) -> 'Storage':
        ...

    def _write_file(self, f: Any, is_real_file: _bool, save_size: _bool, element_size: int) -> None:
        ...

    def element_size(self) -> int:
        ...

    def is_shared(self) -> bool:
        ...

    def share_memory_(self) -> 'Storage':
        ...

    def nbytes(self) -> int:
        ...

    def cpu(self) -> 'Storage':
        ...

    def data_ptr(self) -> int:
        ...

    def from_file(self, filename: str, shared: bool = False, nbytes: int = 0) -> 'Storage':
        ...

    def _new_with_file(self, f: Any, element_size: int) -> 'Storage':
        ...

    ...

# Expanding the provided convenience aliases for those who would like 
# increased readability. This lets users specify the type of a tensor
# that they might be using in a function signature or elsewhere. For example
# in a custom RNN architecture, the user might want to specify that arguments
# of a function are the hidden / cell states of an LSTM.
from typing import Literal, TypeAlias
from torch.nn import Module
# NOTE: devices listed by PyTorch when `torch.device(str)` raises a `RuntimeError`
TorchDeviceTypes = Union[
    Literal['cpu'], Literal['cuda'], Literal['ipu'], Literal['xpu'], 
    Literal['mkldnn'], Literal['opengl'], Literal['opencl'], Literal['ideep'], 
    Literal['hip'], Literal['ve'], Literal['fpga'], Literal['ort'], Literal['xla'], 
    Literal['lazy'], Literal['vulkan'], Literal['mps'], Literal['meta'], Literal['hpu'], 
    Literal['privateuseone']
]

TorchDevice: TypeAlias = Union[Device, TorchDeviceTypes]


TorchTensor: TypeAlias = Union[tuple(torch._tensor_classes)]
TorchTensors: TypeAlias = Sequence[TorchTensor]

# NOTE: PyTorch does not provide a type alias for `torch.nn.Module` and 
# while `nn.Loss` does inherit from `_Loss` (e.g. MSELoss --> _Loss --> Module)
# it is recommend for users to implement their own loss function with `nn.Module` rather
# than `_Loss` (see https://pytorch.org/docs/stable/generated/torch.nn._Loss.html#torch.nn._Loss)
TorchLayer: TypeAlias = torch.nn.Module
TorchLayers: TypeAlias = Sequence[TorchLayer]

TorchLoss: TypeAlias = torch.nn.Module
TorchLosses: TypeAlias = Sequence[TorchLoss]

StateDict: TypeAlias = dict
MaskTensor: TypeAlias = Union[torch.BoolTensor, torch.IntTensor]

HiddenState: TypeAlias = TorchTensor
CellState: TypeAlias = TorchTensor

GRUState: TypeAlias = HiddenState
LSTMStates: TypeAlias = Tuple[HiddenState, CellState]

RNNStates: TypeAlias = Union[GRUState, LSTMStates]