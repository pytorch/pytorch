from ctypes import c_void_p
from typing import List

from torch import Tensor

# Defined in torch/csrc/inductor/aoti_runner/pybind.cpp

# Tensor to AtenTensorHandle
def unsafe_alloc_void_ptr_from_tensors(tensors: List[Tensor]) -> List[c_void_p]: ...

# AtenTensorHandle to Tensor
def alloc_tensors_by_stealing_from_void_ptr(
    handles: List[c_void_p],
) -> List[Tensor]: ...

class AOTIModelContainerRunnerCpu: ...
class AOTIModelContainerRunnerCuda: ...
