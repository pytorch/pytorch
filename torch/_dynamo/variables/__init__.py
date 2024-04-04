# mypy: ignore-errors

from .base import VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
    ContextWrappingVariable,
    DeterministicAlgorithmsVariable,
    DisabledSavedTensorsHooksVariable,
    DualLevelContextManager,
    GradIncrementNestingCtxManagerVariable,
    GradInplaceRequiresGradCtxManagerVariable,
    GradModeVariable,
    InferenceModeVariable,
    JvpIncrementNestingCtxManagerVariable,
    SetFwdGradEnabledContextManager,
    StreamContextVariable,
    StreamVariable,
    VmapIncrementNestingCtxManagerVariable,
    WithExitFunctionVariable,
)
from .dicts import (
    ConstDictVariable,
    CustomizedDictVariable,
    DataClassVariable,
    DefaultDictVariable,
    SetVariable,
)
from .distributed import BackwardHookVariable, DistributedVariable
from .functions import (
    FunctoolsPartialVariable,
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .higher_order_ops import (
    FunctorchHigherOrderVariable,
    TorchHigherOrderOperatorVariable,
)
from .iter import (
    CountIteratorVariable,
    CycleIteratorVariable,
    IteratorVariable,
    ItertoolsVariable,
    RepeatIteratorVariable,
)
from .lazy import LazyVariableTracker
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    NamedTupleVariable,
    RangeVariable,
    RestrictedListSubclassVariable,
    SliceVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .misc import (
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    ClosureVariable,
    DeletedVariable,
    GetAttrVariable,
    InspectSignatureVariable,
    LambdaVariable,
    MethodWrapperVariable,
    NewCellVariable,
    NewGlobalVariable,
    NumpyVariable,
    PythonModuleVariable,
    StopIterationVariable,
    StringFormatVariable,
    SuperVariable,
    TypingVariable,
    UnknownVariable,
)
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable

from .optimizer import OptimizerVariable
from .sdpa import SDPAParamsVariable
from .tensor import (
    FakeItemVariable,
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
    UntypedStorageVariable,
)
from .torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from .user_defined import (
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedObjectVariable,
)

__all__ = [
    "AutogradFunctionContextVariable",
    "AutogradFunctionVariable",
    "BackwardHookVariable",
    "BaseListVariable",
    "BuiltinVariable",
    "ClosureVariable",
    "ConstantVariable",
    "ConstDictVariable",
    "ContextWrappingVariable",
    "CountIteratorVariable",
    "CustomizedDictVariable",
    "CycleIteratorVariable",
    "DataClassVariable",
    "DefaultDictVariable",
    "DeletedVariable",
    "DeterministicAlgorithmsVariable",
    "EnumVariable",
    "FakeItemVariable",
    "GetAttrVariable",
    "GradModeVariable",
    "InspectSignatureVariable",
    "IteratorVariable",
    "ItertoolsVariable",
    "LambdaVariable",
    "LazyVariableTracker",
    "ListIteratorVariable",
    "ListVariable",
    "NamedTupleVariable",
    "NestedUserFunctionVariable",
    "NewCellVariable",
    "NewGlobalVariable",
    "NNModuleVariable",
    "NumpyNdarrayVariable",
    "NumpyVariable",
    "OptimizerVariable",
    "PythonModuleVariable",
    "RangeVariable",
    "RemovableHandleVariable",
    "RepeatIteratorVariable",
    "RestrictedListSubclassVariable",
    "SDPAParamsVariable",
    "SkipFunctionVariable",
    "SliceVariable",
    "StopIterationVariable",
    "StringFormatVariable",
    "SuperVariable",
    "TensorVariable",
    "TorchCtxManagerClassVariable",
    "TorchInGraphFunctionVariable",
    "TupleVariable",
    "UnknownVariable",
    "UnspecializedNNModuleVariable",
    "UnspecializedPythonVariable",
    "UntypedStorageVariable",
    "UserDefinedClassVariable",
    "UserDefinedObjectVariable",
    "UserFunctionVariable",
    "UserMethodVariable",
    "VariableTracker",
    "WithExitFunctionVariable",
]
