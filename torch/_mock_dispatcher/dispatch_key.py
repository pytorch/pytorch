from enum import Enum, auto

class DispatchKey(Enum):
    # "Dense" backends
    # These are "real" backends that can also have different sparse/autograd/quantized kernels
    CPUBit = 0
    CUDABit = auto()
    HIPBit = auto()
    XLABit = auto()
    LazyBit = auto()
    XPUBit = auto()
    NestedTensorBit = auto()
    MLCBit = auto()
    HPUBit = auto()
    PrivateUse1Bit = auto()
    PrivateUse2Bit = auto()
    PrivateUse3Bit = auto()

    # EndOfExtensibleBackendKeys tells us which keys should get "backend bits" in the runtime bitset.
    EndOfBackendKeys = PrivateUse3Bit

    # Undefined technically counts as a "functionality" bit - it's the lowest priority functionality.
    Undefined = auto()
    CatchAll = Undefined


    # Dense lives above the "Other" backends, mostly because the other backends should have higher priority than dense CPU.
    # e.g. torch.add(vulkan_tensor, cpu_tensor) should dispatch to the vulkan kernel
    Dense = auto()

    # "Other" backends
    # These are backends in core that currently don't need separate implementations for sparse, autograd or quantized kernels
    # Note: this means that none of these backends can customize autograd. The "AutogradOther" bit will be added to their bitset.
    FPGA = auto()
    ORT = auto()
    Vulkan = auto()
    Metal = auto()
    MKLDNN = auto()
    OpenGL = auto()
    OpenCL = auto()
    IDEEP = auto()
    CustomRNGKeyId = auto()
    MkldnnCPU = auto()
    Meta = auto()

    # "Functionality" keys start here (except for Dense, which is further up).
    Quantized = auto()
    Sparse = auto()

    # We could consider making "SparseCsr" a per-backend concept
    SparseCsrCPU = auto()
    SparseCsrCUDA = auto()

    BackendSelect = auto()
    Named = auto()

    # We need AutogradOther because we don't currently give every backend an actual bit.
    # Every key between "EndOfBackendKeys" and "Dense" corresponds to a backend that still needs to work with autograd.
    AutogradOther = auto()
    Autograd = auto()

    Tracer = auto()
    Autocast = auto()
    Batched = auto()
    VmapMode = auto()
    TESTING_ONLY_GenericWrapper = auto()
    TESTING_ONLY_GenericMode = auto()
    EndOfFunctionalityKeys = TESTING_ONLY_GenericMode

    # Per-backend functionality keys
    CPU = auto()
    CUDA = auto()
    HIP = auto()
    XLA = auto()
    Lazy = auto()
    XPU = auto()
    NestedTensor = auto()
    MLC = auto()
    HPU = auto()
    PrivateUse1 = auto()
    PrivateUse2 = auto()
    PrivateUse3 = auto()

    QuantizedCPU = auto()
    QuantizedCUDA = auto()
    QuantizedXPU = auto()

    SparseCPU = auto()
    SparseCUDA = auto()
    SparseHIP = auto()
    SparseXPU = auto()

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
    EndOfPerBackendKeys = AutogradPrivateUse3

    AutogradAlias = auto()
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

def isPerBackendFunctionalityKey(k: DispatchKey) -> bool:
    if k is DispatchKey.Dense or k is DispatchKey.Quantized or k is DispatchKey.Sparse or k is DispatchKey.Autograd:
        return True
    return False

def isRuntimeDispatchKey(k: DispatchKey) -> bool:
    # Undefined is special because it can't be added to a DispatchKeySet, but is still considered a runtime key.
    if k == DispatchKey.Undefined:
        return True
    # None of the "backend bit" keys are real runtime keys
    if k.value <= DispatchKey.EndOfBackendKeys.value:
        return False
    # Dense/Sparse/Autograd/Quantized are not runtime keys
    if isPerBackendFunctionalityKey(k):
        return False
    # Alias keys are not runtime keys
    if isAliasDispatchKey(k):
        return False
    return True

def isAliasDispatchKey(k: DispatchKey) -> bool:
    return (k is DispatchKey.CompositeImplicitAutograd
            or k is DispatchKey.CompositeExplicitAutograd
            or k is DispatchKey.AutogradAlias)

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

def toBackendKey(k: DispatchKey) -> DispatchKey:
    # SparseCsr isn't in this list because it isn't currently considered a "per-backend" functionality
    if k in [DispatchKey.CPU, DispatchKey.AutogradCPU, DispatchKey.SparseCPU, DispatchKey.QuantizedCPU]:
        return DispatchKey.CPUBit
    if k in [DispatchKey.XPU, DispatchKey.AutogradXPU, DispatchKey.QuantizedXPU, DispatchKey.SparseXPU]:
        return DispatchKey.XPUBit
    if k in [DispatchKey.CUDA, DispatchKey.AutogradCUDA, DispatchKey.SparseCUDA, DispatchKey.QuantizedCUDA]:
        return DispatchKey.CUDABit
    if k in [DispatchKey.HIP, DispatchKey.SparseHIP]:
        return DispatchKey.HIPBit
    if k in [DispatchKey.XLA, DispatchKey.AutogradXLA]:
        return DispatchKey.XLABit
    if k in [DispatchKey.Lazy, DispatchKey.AutogradLazy]:
        return DispatchKey.LazyBit
    if k in [DispatchKey.MLC, DispatchKey.AutogradMLC]:
        return DispatchKey.MLCBit
    if k in [DispatchKey.HPU, DispatchKey.AutogradHPU]:
        return DispatchKey.HPUBit
    if k in [DispatchKey.NestedTensor, DispatchKey.AutogradNestedTensor]:
        return DispatchKey.NestedTensorBit
    if k in [DispatchKey.PrivateUse1, DispatchKey.AutogradPrivateUse1]:
        return DispatchKey.PrivateUse1Bit
    if k in [DispatchKey.PrivateUse2, DispatchKey.AutogradPrivateUse2]:
        return DispatchKey.PrivateUse2Bit
    if k in [DispatchKey.PrivateUse3, DispatchKey.AutogradPrivateUse3]:
        return DispatchKey.PrivateUse3Bit
    return k

def toFunctionalityKey(k: DispatchKey) -> DispatchKey:
    assert k.value > DispatchKey.EndOfBackendKeys.value
    assert not isPerBackendFunctionalityKey(k)
    if k in [
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.HIP,
        DispatchKey.XLA,
        DispatchKey.Lazy,
        DispatchKey.XPU,
        DispatchKey.NestedTensor,
        DispatchKey.MLC,
        DispatchKey.HPU,
        DispatchKey.PrivateUse1,
        DispatchKey.PrivateUse2,
        DispatchKey.PrivateUse3
    ]:
        return DispatchKey.Dense
    if k in [
        DispatchKey.QuantizedCPU,
        DispatchKey.QuantizedCUDA,
        DispatchKey.QuantizedXPU,
    ]:
        return DispatchKey.Quantized
    if k in [
        DispatchKey.SparseCPU,
        DispatchKey.SparseCUDA,
        DispatchKey.SparseHIP,
        DispatchKey.SparseXPU,
    ]:
        return DispatchKey.Sparse
    if k in [
        DispatchKey.AutogradCPU,
        DispatchKey.AutogradCUDA,
        DispatchKey.AutogradXLA,
        DispatchKey.AutogradLazy,
        DispatchKey.AutogradXPU,
        DispatchKey.AutogradNestedTensor,
        DispatchKey.AutogradMLC,
        DispatchKey.AutogradHPU,
        DispatchKey.AutogradPrivateUse1,
        DispatchKey.AutogradPrivateUse2,
        DispatchKey.AutogradPrivateUse3
    ]:
        return DispatchKey.Autograd
    return k

NUM_BACKENDS = DispatchKey.EndOfBackendKeys.value + 1
