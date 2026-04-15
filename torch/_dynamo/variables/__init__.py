"""
This package implements variable tracking and symbolic execution capabilities for Dynamo,
which are essential for converting Python code into FX graphs. It provides a comprehensive
set of variable types that handle different Python constructs during tracing.

Each variable type (like BuiltinVariable, TensorVariable, NNModuleVariable, etc.) is responsible
for tracking and symbolically executing operations on specific Python objects. This enables
Dynamo to:
- Track the flow of values through Python code
- Maintain correct semantics during graph conversion
- Handle complex Python features like context managers, iterators, and custom objects
- Support both eager and symbolic execution modes

The VariableTracker base class provides the foundation for all variable types, with each
subclass implementing specific behavior for different Python constructs. This modular design
allows Dynamo to accurately trace and optimize Python code while preserving its semantics.
"""

from .base import VariableTracker
from .builtin import (
    BaseBuiltinVariable,
    BuiltinVariable,
    DictBuiltinVariable,
    IterBuiltinVariable,
    ListBuiltinVariable,
)
from .constant import ConstantVariable
from .ctx_manager import (
    CatchWarningsCtxManagerVariable,
    ContextWrappingVariable,
    CUDADeviceVariable,
    CudagraphOverrideVariable,
    DisabledSavedTensorsHooksVariable,
    DualLevelContextManager,
    DynamoConfigPatchVariable,
    ErrorOnGraphBreakVariable,
    FSDPParamGroupUseTrainingStateVariable,
    FxTracebackAnnotateVariable,
    GenericContextWrappingVariable,
    GradIncrementNestingCtxManagerVariable,
    GradInplaceRequiresGradCtxManagerVariable,
    GradModeVariable,
    InferenceModeVariable,
    JvpIncrementNestingCtxManagerVariable,
    SDPAKernelVariable,
    SetFwdGradEnabledContextManager,
    TemporarilyPopInterpreterStackCtxManagerVariable,
    VmapIncrementNestingCtxManagerVariable,
    WithEnterFunctionVariable,
    WithExitFunctionVariable,
)
from .dicts import (
    ConstDictVariable,
    DefaultDictVariable,
    DictItemsVariable,
    DunderDictVariable,
    MappingProxyVariable,
    NNModuleHooksDictVariable,
)
from .distributed import BackwardHookVariable, DistributedVariable
from .functions import (
    BaseUserFunctionVariable,
    BuiltinMethodVariable,
    CollectionsNamedTupleFunction,
    CreateTMADescriptorExperimentalVariable,
    CreateTMADescriptorStableVariable,
    FunctionDecoratedByContextlibContextManagerVariable,
    FunctoolsPartialVariable,
    InspectSignatureVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    NestedUserFunctionVariable,
    PolyfilledFunctionVariable,
    PyTreeGetNodeTypeFunctionVariable,
    PyTreeTreeIsLeafFunctionVariable,
    SkipFunctionVariable,
    SparseTensorCreationSkipVariable,
    TMADescriptorExperimentalVariable,
    TMADescriptorStableVariable,
    TritonSetAllocatorVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrapperUserFunctionVariable,
    WrapperUserMethodVariable,
)
from .higher_order_ops import (
    FunctionalCallVariable,
    FunctorchHigherOrderVariable,
    ReparametrizeModuleCallVariable,
    TorchHigherOrderOperatorVariable,
)
from .iter import (
    CountIteratorVariable,
    FilterVariable,
    IteratorVariable,
    ItertoolsVariable,
    MapVariable,
    ObjectIteratorVariable,
    RepeatIteratorVariable,
    ZipVariable,
)
from .lazy import LazyConstantVariable, LazyVariableTracker
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    RangeVariable,
    SliceVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .misc import (
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    CellVariable,
    DeletedVariable,
    ExceptionVariable,
    GetAttrVariable,
    LambdaVariable,
    MethodWrapperVariable,
    NewGlobalVariable,
    NumpyVariable,
    ObjectVariable,
    PythonModuleVariable,
    RandomClassVariable,
    RandomVariable,
    StringFormatVariable,
    SuperVariable,
    TorchVersionVariable,
    TracebackVariable,
    TypingVariable,
    UnknownVariable,
    WeakRefVariable,
)
from .nn_module import (
    FSDPManagedNNModuleVariable,
    NNModuleVariable,
    UnspecializedBuiltinNNModuleVariable,
    UnspecializedNNModuleVariable,
)
from .optimizer import OptimizerVariable
from .sdpa import SDPAParamsVariable
from .sets import (
    DictKeySetVariable,
    FrozensetVariable,
    OrderedSetClassVariable,
    OrderedSetVariable,
    SetVariable,
)
from .streams import (
    CudaStreamVariable,
    EventVariable,
    StreamContextVariable,
    StreamVariable,
)
from .tensor import (
    DataPtrVariable,
    FakeItemVariable,
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
    UntypedStorageVariable,
)
from .torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from .user_defined import (
    FrozenDataClassVariable,
    InspectVariable,
    MutableMappingVariable,
    NamedTupleVariable,
    OrderedDictVariable,
    RemovableHandleVariable,
    StructSequenceVariable,
    UserDefinedClassVariable,
    UserDefinedConstantVariable,
    UserDefinedDictVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
    UserDefinedListVariable,
    UserDefinedObjectVariable,
    UserDefinedSetVariable,
    UserDefinedTupleVariable,
    UserDefinedVariable,
)


__all__ = [
    "AutogradFunctionContextVariable",
    "AutogradFunctionVariable",
    "BackwardHookVariable",
    "BaseBuiltinVariable",
    "BaseListVariable",
    "BuiltinVariable",
    "CatchWarningsCtxManagerVariable",
    "ConstantVariable",
    "ConstDictVariable",
    "DictBuiltinVariable",
    "ContextWrappingVariable",
    "CountIteratorVariable",
    "CreateTMADescriptorExperimentalVariable",
    "CreateTMADescriptorStableVariable",
    "CUDADeviceVariable",
    "CudagraphOverrideVariable",
    "DataPtrVariable",
    "DefaultDictVariable",
    "DeletedVariable",
    "DictKeySetVariable",
    "DynamoConfigPatchVariable",
    "FakeItemVariable",
    "GetAttrVariable",
    "GradModeVariable",
    "InspectSignatureVariable",
    "InspectVariable",
    "IterBuiltinVariable",
    "IteratorVariable",
    "ItertoolsVariable",
    "LambdaVariable",
    "LazyConstantVariable",
    "LazyVariableTracker",
    "ListBuiltinVariable",
    "ListIteratorVariable",
    "ListVariable",
    "NestedUserFunctionVariable",
    "CellVariable",
    "NewGlobalVariable",
    "NNModuleVariable",
    "NumpyNdarrayVariable",
    "OrderedDictVariable",
    "NumpyVariable",
    "OptimizerVariable",
    "PolyfilledFunctionVariable",
    "PythonModuleVariable",
    "RangeVariable",
    "RemovableHandleVariable",
    "RepeatIteratorVariable",
    "SDPAParamsVariable",
    "ErrorOnGraphBreakVariable",
    "SkipFunctionVariable",
    "SliceVariable",
    "StringFormatVariable",
    "SuperVariable",
    "TemporarilyPopInterpreterStackCtxManagerVariable",
    "TensorVariable",
    "TMADescriptorExperimentalVariable",
    "TMADescriptorStableVariable",
    "TorchCtxManagerClassVariable",
    "TorchInGraphFunctionVariable",
    "TorchVersionVariable",
    "TupleVariable",
    "UnknownVariable",
    "UnspecializedNNModuleVariable",
    "UnspecializedPythonVariable",
    "UntypedStorageVariable",
    "UserDefinedClassVariable",
    "UserDefinedTupleVariable",
    "NamedTupleVariable",
    "StructSequenceVariable",
    "UserDefinedObjectVariable",
    "UserFunctionVariable",
    "UserMethodVariable",
    "VariableTracker",
    "WithEnterFunctionVariable",
    "WithExitFunctionVariable",
    "MappingProxyVariable",
]
