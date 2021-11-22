from enum import Enum, auto
from typing import List

class DispatchKey(Enum):
    Undefined = 0
    CatchAll = Undefined

    CPU = auto()
    CUDA = auto()
    HIP = auto()
    FPGA = auto()
    ORT = auto()
    XLA = auto()
    Lazy = auto()
    Vulkan = auto()
    Metal = auto()
    XPU = auto()
    MKLDNN = auto()
    OpenGL = auto()
    OpenCL = auto()
    IDEEP = auto()
    QuantizedCPU = auto()
    QuantizedCUDA = auto()
    QuantizedXPU = auto()
    CustomRNGKeyId = auto()
    MkldnnCPU = auto()
    SparseCPU = auto()
    SparseCUDA = auto()
    SparseCsrCPU = auto()
    SparseCsrCUDA = auto()
    SparseHIP = auto()
    SparseXPU = auto()
    NestedTensor = auto()
    MLC = auto()
    HPU = auto()
    PrivateUse1 = auto()
    PrivateUse2 = auto()
    PrivateUse3 = auto()
    EndOfBackendKeys = PrivateUse3

    Meta = auto()
    BackendSelect = auto()
    Named = auto()
    AutogradOther = auto()
    AutogradCPU = auto()
    AutogradCUDA = auto()
    AutogradXLA = auto()
    AutogradLazy = auto()
    AutogradNestedTensor = auto()
    AutogradMLC = auto()
    AutogradHPU = auto()
    AutogradXPU = auto()
    AutogradPrivateUse1 = auto()
    AutogradPrivateUse2 = auto()
    AutogradPrivateUse3 = auto()
    Tracer = auto()
    Autocast = auto()
    Batched = auto()
    VmapMode = auto()
    TESTING_ONLY_GenericWrapper = auto()
    TESTING_ONLY_GenericMode = auto()
    NumDispatchKeys = auto()
    Autograd = auto()
    CompositeImplicitAutograd = auto()
    CompositeExplicitAutograd = auto()
    EndOfAliasKeys = CompositeExplicitAutograd

    CPUTensorId = CPU
    CUDATensorId = CUDA
    PrivateUse1_PreAutograd = AutogradPrivateUse1
    PrivateUse2_PreAutograd = AutogradPrivateUse2
    PrivateUse3_PreAutograd = AutogradPrivateUse3

    def __str__(self) -> str:
        return self.name

    def lower(self) -> str:
        return str(self).lower()

def isAliasDispatchKey(k: DispatchKey) -> bool:
    return k is DispatchKey.CompositeImplicitAutograd or k is DispatchKey.CompositeExplicitAutograd or k is DispatchKey.Autograd

def getDispatchTableIndexForDispatchKey(k: DispatchKey) -> int:
    # Note: in C++, this is also defined specially for mobile.
    # This is going to change as part of freeing up dispatch key space
    return k.value

def getAutogradKeyFromBackend(k: DispatchKey) -> DispatchKey:
    if k is DispatchKey.CPU:
        return DispatchKey.AutogradCPU
    if k is DispatchKey.XPU:
        return DispatchKey.AutogradXPU
    if k is DispatchKey.CUDA:
        return DispatchKey.AutogradCUDA
    if k is DispatchKey.XLA:
        return DispatchKey.AutogradXLA
    if k is DispatchKey.Lazy:
        return DispatchKey.AutogradLazy
    if k is DispatchKey.MLC:
        return DispatchKey.AutogradMLC
    if k is DispatchKey.HPU:
        return DispatchKey.AutogradHPU
    if k is DispatchKey.NestedTensor:
        return DispatchKey.AutogradNestedTensor
    if k is DispatchKey.PrivateUse1:
        return DispatchKey.AutogradPrivateUse1
    if k is DispatchKey.PrivateUse2:
        return DispatchKey.AutogradPrivateUse2
    if k is DispatchKey.PrivateUse3:
        return DispatchKey.AutogradPrivateUse3
    return DispatchKey.AutogradOther
