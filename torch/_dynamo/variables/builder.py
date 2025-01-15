# mypy: ignore-errors

import abc
import collections
import contextlib
import copy
import dataclasses
import enum
import functools
import inspect
import itertools
import logging
import math
import operator
import random
import re
import types
import warnings
import weakref
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
)

import sympy

import torch
from torch import SymInt
from torch._dynamo.utils import (
    get_metrics_context,
    is_int_specialization_case,
    is_torch_sym,
)
from torch._guards import TracingContext
from torch._higher_order_ops.torchbind import call_torchbind
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
from torch._subclasses.meta_utils import is_sparse_any, safe_grad
from torch._utils_internal import justknobs_check
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    _nested_int_aware_sort,
    DimDynamic,
    RelaxedUnspecConstraint,
    StatefulSymbolicContext,
    SubclassSymbolicContext,
    SymbolicContext,
)
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils.weak import TensorWeakRef

from .. import config, mutation_guard, replay_record, trace_rules
from ..device_interface import get_registered_device_interfaces
from ..exc import InternalTorchDynamoError, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..pgo import (
    auto_dynamic,
    auto_unset,
    FrameStateSizeEntry,
    InferStride,
    process_automatic_dynamic,
)
from ..side_effects import SideEffects
from ..source import (
    AttrProxySource,
    AttrSource,
    CallMethodItemSource,
    ConstDictKeySource,
    ConvertIntSource,
    DictGetItemSource,
    FloatTensorSource,
    GetItemSource,
    GradSource,
    is_constant_source,
    is_from_optimizer_source,
    LocalSource,
    NumpyTensorSource,
    OptimizerSource,
    RandomValueSource,
    Source,
    SubclassAttrListSource,
    TupleIteratorGetItemSource,
)
from ..utils import (
    _extract_tensor_dict,
    build_checkpoint_variable,
    build_invoke_subgraph_variable,
    clone_input,
    common_constant_types,
    dict_keys,
    get_fake_value,
    get_items_from_dict,
    get_locals_to_steal,
    get_static_address_type,
    is_frozen_dataclass,
    is_function_or_wrapper,
    is_invoke_subgraph,
    is_lru_cache_wrapped_function,
    is_namedtuple,
    is_parameter_freezing,
    is_typing,
    is_utils_checkpoint,
    is_wrapper_or_member_descriptor,
    istype,
    namedtuple_fields,
    odict_values,
    proxy_args_kwargs,
    range_iterator,
    set_example_value,
    tensor_always_has_static_shape,
    tuple_iterator,
    tuple_iterator_getitem,
    tuple_iterator_len,
    unwrap_with_attr_name_if_wrapper,
    wrap_fake_exception,
)
from .base import typestr, ValueMutationNew, VariableTracker, VariableTrackerMeta
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
    AutocastModeVariable,
    EventVariable,
    NullContextVariable,
    PreserveVersionContextVariable,
    StreamContextVariable,
    StreamVariable,
)
from .dicts import (
    ConstDictVariable,
    DefaultDictVariable,
    DictKeySetVariable,
    FrozensetVariable,
    SetVariable,
)
from .distributed import (
    DeviceMeshVariable,
    PlacementClassVariable,
    PlacementVariable,
    ProcessGroupVariable,
    WorldMetaClassVariable,
)
from .functions import (
    CollectiveFunctionRewriteVariable,
    CreateTMADescriptorVariable,
    FunctoolsPartialVariable,
    TritonKernelVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrapperUserFunctionVariable,
)
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .iter import ItertoolsVariable
from .lazy import LazyVariableTracker
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    NamedTupleVariable,
    RangeVariable,
    RestrictedListSubclassVariable,
    SizeVariable,
    SliceVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .misc import (
    AutogradEngineVariable,
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    ComptimeVariable,
    DebuggingVariable,
    DelayGraphBreakVariable,
    GetAttrVariable,
    GetSetDescriptorVariable,
    InspectSignatureVariable,
    LambdaVariable,
    LoggingLoggerVariable,
    MethodWrapperVariable,
    NumpyDTypeVariable,
    NumpyTypeInfoVariable,
    NumpyVariable,
    PythonModuleVariable,
    RandomClassVariable,
    RandomVariable,
    RegexPatternVariable,
    SavedTensorBox,
    TorchVersionVariable,
    TypingVariable,
    WeakRefVariable,
)
from .nn_module import (
    FSDPManagedNNModuleVariable,
    UnspecializedBuiltinNNModuleVariable,
    UnspecializedNNModuleVariable,
)
from .optimizer import OptimizerVariable
from .script_object import TorchScriptObjectVariable
from .sdpa import SDPAParamsVariable
from .tensor import (
    NumpyNdarrayVariable,
    supported_const_comparison_op_values,
    SymNodeVariable,
    TensorSubclassVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from .torch_function import (
    build_torch_function_fn,
    TensorWithTFOverrideVariable,
    torch_function_mode_stack_state_mgr,
    TorchFunctionModeVariable,
)
from .user_defined import (
    FrozenDataClassVariable,
    KeyedJaggedTensorVariable,
    MutableMappingVariable,
    SourcelessGraphModuleVariable,
    UserDefinedClassVariable,
    UserDefinedDictVariable,
    UserDefinedObjectVariable,
)


try:
    import numpy as np
except ModuleNotFoundError:
    np = None


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


DimList = List


def safe_has_grad(t):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The .grad attribute of a Tensor")
        return hasattr(t, "grad")


class _missing:
    pass


@dataclasses.dataclass
class GraphArg:
    source: Source
    # TODO: storing a SymInt here but not a FakeTensor is a pretty strange
    # thing to do.  Probably should have example (which stores an int) and
    # fake_example
    _example: Union[TensorWeakRef, torch.SymInt]
    # When True, this indicates that this GraphArg is a Python quantity (e.g.,
    # a float or int) which we pass to the FX graph as a Tensor.  This
    # controls how we codegen calls into the Dynamo graph: we will call
    # torch.as_tensor on the quantity before passing it in.
    #
    # Note that we typically do not pass dynamic integers as tensors, because
    # they will most frequently just be used for size computation.  But this
    # is a policy decision that we can change our mind on; in particular, when
    # an int comes from a random number generator (e.g., random.randint), we
    # DO pass it as a tensor.
    #
    # It's also worth noting that our current tracing rules for
    # pass_arg_as_tensor as subtly broken: we just pun the variable as a
    # 0d scalar Tensor and pray that the semantics are the same.  Which they
    # often are, but not necessarily.  ezyang(May 2024) plans to fix this
    # soon.
    pass_arg_as_tensor: bool
    fake_tensor: Optional[torch._subclasses.fake_tensor.FakeTensor]
    # UnspecializedPythonVariable often masquerades as a tensor.
    # We MUST NOT generate shape guard code
    # that actually tries to access tensor properties on these values.
    # is_tensor lets us tell if this graph arg actually is a tensor
    # or not.
    is_tensor: bool = True
    # Sometimes, the Tensor we pass to example is freshly allocated (smh).
    # Then we cannot only keep a weak reference to it.  This lets you
    # stash a strong reference too.
    example_strong_ref: Optional[torch.Tensor] = None

    @property
    def example(self):
        if isinstance(self._example, TensorWeakRef):
            r = self._example()
            assert r is not None
            return r
        else:
            return self._example

    def __post_init__(self):
        if isinstance(self._example, torch.Tensor):
            self._example = TensorWeakRef(self._example)
            assert is_fake(self.fake_tensor)

    def reconstruct(self, codegen):
        self.source.reconstruct(codegen)

    def erase(self):
        self._example = None
        self.example_strong_ref = None

    def __eq__(self, other):
        return self.source.name() == other.source.name()


class BackwardStateGraphArg(GraphArg):
    def __init__(self) -> None:
        super().__init__(
            source=None,
            _example=BackwardState(),
            pass_arg_as_tensor=False,
            fake_tensor=None,
            is_tensor=False,
        )

    def reconstruct(self, codegen):
        assert codegen.tx.output.backward_state_var
        codegen.add_push_null(
            lambda: codegen.load_import_from(BackwardState.__module__, "BackwardState")
        )
        codegen.call_function(0, False)
        codegen.dup_top()
        codegen.store(codegen.tx.output.backward_state_var)


# All class-based iterators in itertools
# NOTE: use id() because some objects are not hashable, it will raise error during lookup
ITERTOOLS_TYPE_IDS: FrozenSet[int] = frozenset(
    id(member)
    for name, member in vars(itertools).items()
    if not name.startswith("_") and inspect.isclass(member)
)
# Will be updated later in substitute_in_graph in torch/_dynamo/polyfills/itertools.py
ITERTOOLS_POLYFILLED_TYPE_IDS: Set[int] = set()


class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""

    def __init__(
        self,
        tx,
        source: Source,
    ) -> None:
        assert (
            source is not None
        ), "Consider SourcelessBuilder for ephemeral objects, usually objects created locally."
        assert TracingContext.try_get() is not None, "Expected active TracingContext"
        super().__init__()
        self.tx = tx
        self.source = source
        self.name = source.name()

    def __call__(self, value):
        if value in self.tx.output.side_effects:
            side_effect_result = self.tx.output.side_effects[value]
            dup_guard = make_dupe_guard(self.source, side_effect_result.source)
            if dup_guard:
                self.install_guards(dup_guard)
            return side_effect_result

        cached_vt = self.tx.output.variable_tracker_cache.lookup(value, self.source)
        if cached_vt:
            return cached_vt

        vt = self._wrap(value)
        vt.source = self.source
        if (
            self._can_lift_attrs_to_inputs(vt)
            and value not in self.tx.output.side_effects
            and not is_wrapper_or_member_descriptor(value)
        ):
            vt = self.tx.output.side_effects.track_object_existing(value, vt)

        self.tx.output.variable_tracker_cache.add(value, self.source, vt)
        return vt

    def _can_lift_attrs_to_inputs(self, vt):
        return type(vt) in {
            TensorVariable,
            TensorWithTFOverrideVariable,
            UserDefinedObjectVariable,
            NumpyNdarrayVariable,
        }

    def get_source(self):
        return self.source

    def install_guards(self, *guards):
        source = self.get_source()
        try:
            tmp = [source.make_guard(guard) for guard in guards]
        except NotImplementedError:
            return None
        install_guard(*tmp, skip=1)
        return {}

    @classmethod
    def _type_dispatch(cls):
        return cls._type_dispatch_impl(config.trace_numpy)

    @classmethod
    @functools.lru_cache(None)
    def _type_dispatch_impl(cls, trace_numpy):
        # NB: Careful not to close over self to avoid ref cycle from lru_cache
        entries = [
            (
                (
                    torch.Tensor,
                    torch.nn.Parameter,
                    torch._subclasses.FakeTensor,
                    torch._subclasses.functional_tensor.FunctionalTensor,
                ),
                cls.wrap_tensor,
            ),
            (
                (tuple, list, odict_values, collections.deque, torch.Size),
                cls.wrap_listlike,
            ),
            (tuple_iterator, cls.wrap_tuple_iterator),
            (range_iterator, cls.wrap_range_iterator),
            ((slice, range), cls.wrap_slice_range),
            (tuple(common_constant_types), cls.wrap_literal),
            (re.Pattern, cls.wrap_regex_pattern),
            (weakref.ReferenceType, cls.wrap_weakref),
            (torch.utils.hooks.RemovableHandle, cls.wrap_removable_handle),
            (torch.jit.ScriptFunction, cls.wrap_jit_function),
        ]

        if trace_numpy and np:
            entries.append((np.ndarray, cls.wrap_numpy_ndarray))

        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, tuple) else (ts,):
                assert t not in result
                result[t] = fn

        return result

    def wrap_regex_pattern(self, value: re.Pattern):
        # TODO(jansel): something like a REPR_MATCH might be more robust here
        self.install_guards(GuardBuilder.ID_MATCH)
        return RegexPatternVariable(value)

    def wrap_weakref(self, value: weakref.ReferenceType):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return WeakRefVariable.build(self.tx, value, source=self.source)

    def wrap_removable_handle(self, value):
        # This means that the removable handle was created in some other frame.
        # Our current infra requires the hook to be registered and removed in
        # the same frame. So graph break.
        # Related test - PYTORCH_TEST_WITH_DYNAMO=1 python test/test_autograd.py -k TestAutograd.test_hooks
        unimplemented("unregistered hook removable handle")

    def wrap_jit_function(self, value):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return WrapperUserFunctionVariable(
            value, "_torchdynamo_inline", source=self.source
        )

    @classmethod
    @functools.lru_cache(None)
    def _id_dispatch(
        cls,
    ) -> Dict[int, Callable[["VariableBuilder", Any], VariableTracker]]:
        from ..comptime import comptime

        entries = [
            (
                inspect.signature,
                lambda self, value: LambdaVariable(
                    InspectSignatureVariable.create,
                    source=self.source,
                    **self.install_guards(GuardBuilder.CLOSURE_MATCH),
                ),
            ),
            (comptime, lambda self, value: ComptimeVariable()),
            (
                dataclasses.fields,
                lambda self, value: LambdaVariable(
                    _dataclasses_fields_lambda,
                    source=self.source,
                    **self.install_guards(GuardBuilder.FUNCTION_MATCH),
                ),
            ),
            (torch.__version__, lambda self, value: TorchVersionVariable()),
        ]

        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, (tuple, list)) else (ts,):
                assert t not in result
                result[id(t)] = fn

        return result

    def _wrap(self, value):
        # import here to avoid circular dependencies
        from torch.utils._triton import has_triton, has_triton_tma

        if has_triton():
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction
        else:

            class JITFunction:
                pass

            class Autotuner:
                pass

        if has_triton_tma():
            from triton.tools.experimental_descriptor import (
                create_1d_tma_descriptor,
                create_2d_tma_descriptor,
            )
        else:

            def create_1d_tma_descriptor():
                pass

            def create_2d_tma_descriptor():
                pass

        # Handle exact type() match
        type_dispatch = self._type_dispatch().get(type(value))
        if type_dispatch is not None:
            return type_dispatch(self, value)

        # Handle exact id() match
        id_dispatch = self._id_dispatch().get(id(value))
        if id_dispatch is not None:
            return id_dispatch(self, value)

        # Everything else (NB: order matters!)
        if is_traceable_wrapper_subclass(value) or istype(
            value, config.traceable_tensor_subclasses
        ):
            return self.wrap_tensor(value)
        elif is_namedtuple(value):
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)
            output = [
                LazyVariableTracker.create(
                    getattr(value, name),
                    source=AttrSource(self.source, name),
                )
                for name in namedtuple_fields(type(value))
            ]
            result = NamedTupleVariable(
                output, tuple_cls=type(value), source=self.source
            )
            return result
        elif istype(value, (dict, collections.defaultdict, collections.OrderedDict)):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            all_const = all(ConstantVariable.is_literal(k) for k in value.keys())

            # For all_const, we dont have to guard on anything yet. We guard on
            # keys lazily by adding a dict_getitem entry for each accessed key.
            # For cases where we need to guard on all keys, we lazily put guards
            # during the dict call_method (check dicts.py)
            if not all_const:
                # Guard on the key order
                # This is not ideal, i.e., there is no need to guard on the key
                # order. But we guard on the key order because of the complexity
                #
                # 1) For non-constant objects, we can't save the key in the
                # guard context because it can be memory heavy. We can add
                # weakrefs but this complicates the accesses.
                #
                # 2) For non-constant objects, we also have to guard on the keys
                # (like TENSOR_MATCH on tensor). We might also have guards on
                # the attributes of the keys (like tensor.grad). To make this
                # work in tree strucutre is complicated.
                #
                # So, instead we guard on the key order. While guarding on key
                # order, we just save the indices and use it to access keys and
                # values. Indices are cheap to save.
                self.tx.output.guard_on_key_order.add(self.source.name())

            # We need all the keys to be hashable. We do this within the
            # _HashableTracker class in dicts.py
            def build_key_value(i, k, v):
                if all_const:
                    key = ConstantVariable.create(k)
                    source_key = k
                else:
                    source_key = ConstDictKeySource(self.get_source(), i)
                    key = LazyVariableTracker.create(k, source_key)

                source_value = DictGetItemSource(self.get_source(), source_key)
                value = LazyVariableTracker.create(v, source_value)

                return key, value

            # Ensure that we call dict.keys and not value.keys (which can call
            # overridden keys method). In the C++ guards, we relied on
            # PyDict_Next to traverse the dictionary, which uses the internal
            # data structure and does not call the overridden keys method.
            result = dict(
                build_key_value(i, k, v)
                for i, (k, v) in enumerate(get_items_from_dict(value))
            )

            if istype(value, collections.defaultdict):
                factory_source = AttrSource(self.source, "default_factory")
                result = DefaultDictVariable(
                    result,
                    type(value),
                    default_factory=VariableBuilder(self.tx, factory_source)(
                        value.default_factory
                    ),
                    source=self.source,
                )
            else:
                result = ConstDictVariable(
                    result, user_cls=type(value), source=self.source
                )

            return self.tx.output.side_effects.track_mutable(value, result)
        elif isinstance(value, torch.nn.Module):
            return self.wrap_module(value)
        elif ConstantVariable.is_literal(value):  # non-atomic literals
            return self.wrap_literal(value)
        elif isinstance(value, torch.overrides.TorchFunctionMode):
            var = TorchFunctionModeVariable(value, source=self.source)
            self.tx.output.side_effects.track_object_existing(value, var)
            return var
        elif istype(value, frozenset) and all(
            (
                # For DBR quantization, we could get a frozenset of torch funcs.
                (type(x) is types.BuiltinMethodType and x.__module__ == "torch")
                or
                # Another commonly used frozenset of types.
                x in torch.utils._pytree.BUILTIN_TYPES
            )
            for x in value
        ):
            # For the limited cases of frozenset here, we know the items won't
            # change across runs, so we can safely create sourceless VTs for
            # them and only guard on the frozenset id.
            # TODO support source for sets and remove the special logics here.
            items = [SourcelessBuilder.create(self.tx, v) for v in value]
            self.install_guards(GuardBuilder.ID_MATCH)
            return FrozensetVariable(items, source=self.source)
        elif isinstance(value, enum.Enum):
            self.install_guards(GuardBuilder.ID_MATCH)
            return EnumVariable(value=value, source=self.source)
        elif DebuggingVariable.is_reorderable_logging_function(value):
            # Put this above builtin_callable so that print() can be handled
            # along with other builtin debugging functions
            self.install_guards(GuardBuilder.BUILTIN_MATCH)
            return DebuggingVariable(value, source=self.source)
        elif isinstance(value, logging.Logger):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return LoggingLoggerVariable(value, source=self.source)
        elif is_utils_checkpoint(value):
            return build_checkpoint_variable(source=self.source)
        elif is_invoke_subgraph(value):
            return build_invoke_subgraph_variable(source=self.source)
        elif isinstance(value, functools.partial):
            func_src = AttrSource(self.get_source(), "func")
            func_obj = VariableBuilder(self.tx, func_src)(value.func)

            args = []
            args_source = AttrSource(self.get_source(), "args")
            for i, arg in enumerate(value.args):
                args.append(
                    VariableBuilder(self.tx, GetItemSource(args_source, i))(arg)
                )

            keywords = {}
            keywords_source = AttrSource(self.get_source(), "keywords")
            for k, v in value.keywords.items():
                if not ConstantVariable.is_literal(k):
                    unimplemented("functools.partial with non-literal keyword")
                keywords[k] = VariableBuilder(
                    self.tx, DictGetItemSource(keywords_source, k)
                )(v)

            install_guard(
                self.get_source().make_guard(GuardBuilder.TYPE_MATCH),
                keywords_source.make_guard(GuardBuilder.DICT_KEYS_MATCH),
                args_source.make_guard(GuardBuilder.SEQUENCE_LENGTH),
            )
            return FunctoolsPartialVariable(func_obj, args, keywords)
        elif is_typing(value):
            # typing.List, typing.Mapping, etc.
            self.install_guards(GuardBuilder.ID_MATCH)
            return TypingVariable(
                value,
                source=self.source,
            )
        elif np is not None and isinstance(value, np.generic):
            # numpy array scalars: convert to 0D arrays
            return self.wrap_numpy_ndarray(np.asarray(value))
        elif trace_rules.is_numpy(value):
            assert np
            self.install_guards(
                GuardBuilder.FUNCTION_MATCH
                if callable(value)
                else GuardBuilder.TYPE_MATCH
            )
            return NumpyVariable(value, source=self.source)
        elif trace_rules.is_numpy_dtype(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return NumpyDTypeVariable(value, source=self.source)
        elif trace_rules.is_numpy_type_info(value):
            if isinstance(value, np.iinfo):
                self.install_guards(GuardBuilder.TYPE_MATCH)
                dt_source = AttrSource(self.source, "dtype")
                install_guard(dt_source.make_guard(GuardBuilder.ID_MATCH))
            else:
                self.install_guards(GuardBuilder.ID_MATCH)
            return NumpyTypeInfoVariable(value, source=self.source)
        # NB: These can't be put in type_dispatch, they have to run later
        elif CollectiveFunctionRewriteVariable.can_rewrite(value):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return CollectiveFunctionRewriteVariable.create(
                self.tx,
                value,
                source=self.source,
            )
        elif istype(value, torch.autograd.function.FunctionMeta):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return AutogradFunctionVariable(
                value,
                source=self.source,
            )
        elif isinstance(value, torch.autograd.function.FunctionCtx):
            actual_saved_tensors = None
            try:
                actual_saved_tensors = value.saved_tensors
            except RuntimeError:
                pass

            saved_tensors = []
            guards = [self.source.make_guard(GuardBuilder.TYPE_MATCH)]
            if isinstance(actual_saved_tensors, tuple):
                saved_tensors_source = AttrSource(self.source, "saved_tensors")
                guards.append(
                    saved_tensors_source.make_guard(GuardBuilder.SEQUENCE_LENGTH)
                )
                for i, v in enumerate(actual_saved_tensors):
                    saved_tensors.append(
                        VariableBuilder(
                            self.tx, GetItemSource(saved_tensors_source, i)
                        )(v)
                    )
            install_guard(*guards)

            return self.tx.output.side_effects.track_object_existing(
                value,
                AutogradFunctionContextVariable(
                    value,
                    source=self.source,
                    saved_tensors=SavedTensorBox(saved_tensors),
                ),
            )
        elif (
            isinstance(value, types.MethodType)
            and istype(
                getattr(value, "__self__", None), torch.autograd.function.FunctionMeta
            )
            and getattr(value, "__name__", "") == "apply"
            and value == getattr(value.__self__, "apply", None)
        ):
            # handle aliased autograd function `apply` calls
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return GetAttrVariable(
                AutogradFunctionVariable(
                    value.__self__, source=AttrSource(self.source, member="__self__")
                ),
                "apply",
            )
        elif isinstance(value, torch._C._ImperativeEngine):
            self.install_guards(GuardBuilder.ID_MATCH)
            return AutogradEngineVariable(value, source=self.source)
        elif (
            value
            is torch._dynamo.external_utils.FakeCompiledAutogradEngine._exec_final_callbacks_stub
        ):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return LambdaVariable(
                lambda: UserFunctionVariable(
                    torch._dynamo.external_utils.FakeCompiledAutogradEngine.exec_final_callbacks,
                ).call_function(
                    self.tx,
                    (self.tx.output.side_effects.get_ca_final_callbacks_var(),),
                    {},
                )
            )
        elif callable(value) and trace_rules.lookup_callable(value) is not None:
            if trace_rules.is_callable_allowed(value):
                self.tx.output.has_user_defined_allowed_in_graph = True
            return trace_rules.lookup_callable(value).create_with_source(
                value, source=self.source
            )
        elif np and isinstance(value, np.number):
            return self.wrap_unspecialized_primitive(value)
        elif isinstance(value, HigherOrderOperator):
            if value is torch._higher_order_ops.invoke_subgraph:
                unimplemented(
                    "Directly using invoke_subgraph is not supported. Use mark_compile_region"
                )
            self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.NAME_MATCH)
            return TorchHigherOrderOperatorVariable.make(value, source=self.source)
        elif isinstance(value, torch.cuda.StreamContext):
            self.install_guards(GuardBuilder.ID_MATCH)
            stream_source = AttrSource(self.source, "stream")
            stream_var = VariableBuilder(self.tx, stream_source)(value.stream)
            return StreamContextVariable.create(self.tx, stream_var)
        elif isinstance(value, torch.Stream):
            self.install_guards(GuardBuilder.ID_MATCH)
            stream_proxy = self.tx.output.create_proxy(
                "call_function",
                type(value),
                (),
                {
                    "stream_id": value.stream_id,
                    "device_index": value.device_index,
                    "device_type": value.device_type,
                },
            )
            set_example_value(stream_proxy.node, value)
            return StreamVariable(
                stream_proxy,
                value,
                value.device,
                source=self.source,
            )
        elif isinstance(value, (torch._C._SDPAParams)):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return SDPAParamsVariable.create(self.tx, value, self.source)
        elif isinstance(value, torch.Event):
            self.install_guards(GuardBuilder.ID_MATCH)
            torch._dynamo.utils.store_user_object_weakref(value)
            event_proxy = self.tx.output.create_proxy(
                "call_function",
                torch._dynamo.utils.get_user_object_from_id,
                (id(value),),
                {},
            )
            set_example_value(event_proxy.node, value)
            return EventVariable(
                event_proxy,
                value,
                source=self.source,
            )
        elif (
            isinstance(value, torch._C._TensorMeta)
            and value in config.traceable_tensor_subclasses
        ):
            return TensorSubclassVariable(value, source=self.source)
        elif (
            istype(value, contextlib.nullcontext)
            and inspect.getattr_static(value, "enter_result", None) is None
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return NullContextVariable(source=self.source)
        elif KeyedJaggedTensorVariable.is_matching_object(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = KeyedJaggedTensorVariable(value, source=self.source)
            # TODO: this doing it manually is bad
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif isinstance(value, torch.optim.Optimizer):
            self.install_guards(GuardBuilder.ID_MATCH)
            self.source = OptimizerSource(self.source)
            return OptimizerVariable(value, source=self.source)
        elif WorldMetaClassVariable.is_group_member_type(value):
            return WorldMetaClassVariable(value, source=self.source)
        elif ProcessGroupVariable.is_process_group(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return ProcessGroupVariable(value, source=self.source)
        elif DeviceMeshVariable.is_device_mesh(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.EQUALS_MATCH)
            return DeviceMeshVariable(value, source=self.source)
        elif PlacementClassVariable.is_placement_type(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.ID_MATCH)
            return PlacementClassVariable(value, source=self.source)
        elif PlacementVariable.is_placement(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.EQUALS_MATCH)
            return PlacementVariable(
                value,
                source=self.source,
            )
        elif (
            id(value) in ITERTOOLS_TYPE_IDS
            and id(value) not in ITERTOOLS_POLYFILLED_TYPE_IDS
        ):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return ItertoolsVariable(value, source=self.source)
        elif is_torch_sym(value):
            # Note: this doesn't handle nested symints.
            # For SymBool input, we re-use the infra for SymInt by simulating SymBool with a SymInt in dynamo.

            # Concretely,
            # 1. We create a SymInt in dynamo's shape_env, whose source is constructed as ConvertIntSource(self.source).
            # so that guards on the SymInts can be effectively applied on the original SymBool in user program.
            # 2. We create a SymBool based on the SymInt in dynamo's ShapeEnv. Because the original user program
            # depends on the value being a SymBool. This allows dynamo to interpret the user's program correctly.
            source = (
                self.source
                if isinstance(value, torch.SymInt)
                else ConvertIntSource(self.source)
            )
            if value.node.has_hint():
                new_symint = (
                    self.tx.output.shape_env.create_unspecified_symint_and_symbol(
                        int(value.node.hint),
                        source,
                        dynamic_dim=DimDynamic.DYNAMIC,
                    )
                )
            else:
                if isinstance(value, torch.SymBool):
                    # We need to create an unbacked symint to replace the unbacked symbool.
                    new_symint = self.tx.output.shape_env.create_unbacked_symint()
                else:
                    # TODO (yidi): we need to figure out a way to propagate the guards
                    # we accumulated when tracing the subggraph to outer shape_env. For normal symints,
                    # this is automatically done by evaluating the guards once but this
                    # will cause data-dependent error when we evaluate the outer unbacked symints.
                    # The test case that triggers this graph break is test_cond_unbacked_symint_closure
                    unimplemented(
                        "unbacked symint input is not supported yet. If you need this feature, please file a github issue."
                    )

            sym_node_proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(new_symint),
                new_symint,
                source=source,
            )

            sym_node_proxy.node.meta["grapharg"] = GraphArg(
                source,
                new_symint,
                False,
                None,
                is_tensor=False,
                example_strong_ref=new_symint,
            )
            # We bind the new_symint to graph input.
            sym_expr = new_symint.node.expr
            assert isinstance(
                sym_expr, sympy.Symbol
            ), f"{sym_expr} is not a basic Symbol."
            self.tx.output.tracked_fakes.append(TrackedFake(new_symint, source, None))

            tracing_symint = (
                new_symint if isinstance(value, torch.SymInt) else new_symint == 1
            )  # cast it back to symbool for tracing
            return SymNodeVariable(sym_node_proxy, tracing_symint)

        elif isinstance(value, (JITFunction, Autotuner)):
            self.install_guards(GuardBuilder.ID_MATCH)
            return TritonKernelVariable(
                value,
                None,  # No kernel idx provided
                None,  # No grid provided
                source=self.source,
            )
        elif value is create_1d_tma_descriptor:
            return CreateTMADescriptorVariable(rank=1)
        elif value is create_2d_tma_descriptor:
            return CreateTMADescriptorVariable(rank=2)
        elif isinstance(value, torch.amp.autocast_mode.autocast):
            self.install_guards(GuardBuilder.ID_MATCH)
            return AutocastModeVariable(
                target_values=[
                    value.device,
                    value.fast_dtype,
                    value._enabled,
                    value._cache_enabled,
                ],
                source=self.source,
            )
        elif TorchCtxManagerClassVariable.is_matching_cls(value):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return TorchCtxManagerClassVariable(value, source=self.source)
        elif inspect.getattr_static(value, "__script_if_tracing_wrapper", False):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return WrapperUserFunctionVariable(
                value, "__original_fn", source=self.source
            )
        elif is_lru_cache_wrapped_function(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return WrapperUserFunctionVariable(value, "__wrapped__", source=self.source)
        elif is_function_or_wrapper(value) and inspect.getattr_static(
            value, "_torchdynamo_inline", False
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return WrapperUserFunctionVariable(
                value, "_torchdynamo_inline", source=self.source
            )
        elif is_function_or_wrapper(value):
            value, attr_name = unwrap_with_attr_name_if_wrapper(value)
            # For these wrappers, Dynamo points to the wrapped function,
            # so source needs to be updated as well.
            if attr_name is not None:
                self.source = AttrSource(self.source, attr_name)
            return trace_rules.lookup(value).create_with_source(
                value, source=self.source
            )
        elif value is random.Random:
            self.install_guards(GuardBuilder.ID_MATCH)
            return RandomClassVariable(source=self.source)
        elif istype(value, random.Random) and RandomVariable.is_supported_random_obj(
            value
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = RandomVariable(value, source=self.source)
            self.tx.output.side_effects.track_mutable(value, result)
            return result
        # Don't use istype, since some python modules are not subclasses of types.ModuleType directly.
        # E.g, type(torch.ops) -> <class 'torch._ops._Ops'>,
        # type(torch.backends.cudnn) -> <class 'torch.backends.cudnn.CudnnModule'>
        elif isinstance(value, (types.ModuleType, replay_record.DummyModule)):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            result = PythonModuleVariable(
                value,
                source=self.source,
            )
            self.tx.output.side_effects.track_object_existing(value, result)
            return result
        elif isinstance(value, types.MethodType) and isinstance(
            value.__self__, (torch.nn.Module, torch.utils._pytree.TreeSpec)
        ):
            # don't let MethodTypes fall through to UserDefinedObject,
            # which doesn't support 'CALL_FUNCTION'

            # TODO(whc): Why do we limit this to methods on NNModules?
            # I don't have a good reason for this, but it preserves the existing behavior
            # for MBartForConditionalGeneration, which generates many graph breaks and OOMs otherwise.
            # I suspect we probably want to relax this check and dig deeper there.

            # In order to construct a MethodVariable in Dynamo, we start with an actual method obj from python,
            # but need to separately wrap its underlying `__func__` and its `self` argument.  We wrap `self` here
            # and then `__func__` gets wrapped inside UserMethodVariable.
            self_obj = VariableBuilder(
                self.tx, source=AttrSource(self.source, "__self__")
            )(value.__self__)
            assert self_obj and isinstance(
                self_obj, VariableTracker
            ), "Failed to produce a valid self obj"
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return UserMethodVariable(
                value.__func__,
                self_obj,
                source=self.source,
            )
        elif isinstance(value, types.GetSetDescriptorType):
            # GetSet descriptors are C functions attached to an attribute lookup
            # using PyGetSetDef. Python, on attribute lookup, can decide to
            # create a new object on the fly, and therefore the `id` of the
            # descriptors is not guaranteed to be same for different attribute
            # accesses. Since these are unlikely to change during the program
            # execution, we can skip guarding on them.
            return GetSetDescriptorVariable(value)
        elif isinstance(value, types.MethodWrapperType):
            # Method-wrappers are written in C, and they are not guaranteed to
            # return the same object on attribute lookup. Therefore, we cannot
            # insert a FUNCTION_MATCH guard here. method-wrappers are very
            # unlikely to change, so its ok to skip the guard here.
            return MethodWrapperVariable(value)
        elif issubclass(type(value), type):
            if value in (
                torch.utils.hooks.BackwardHook,
                torch.nn.Parameter,
                torch.nn.Buffer,
            ):
                # TODO(jansel): combine this case with the one above
                return trace_rules.lookup(value).create_with_source(
                    value, source=self.source
                )
            if value is torch.autograd._unsafe_preserve_version_counter:
                self.install_guards(GuardBuilder.FUNCTION_MATCH)
                return PreserveVersionContextVariable.constructor(self.tx)
            # This is a userdefined class, so install an ID_MATCH even if its a
            # global variable.
            self.install_guards(GuardBuilder.ID_MATCH)
            return UserDefinedClassVariable(
                value,
                source=self.source,
            )
        elif RestrictedListSubclassVariable.is_matching_cls(type(value)):
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)
            return self.tx.output.side_effects.track_mutable(
                value,
                RestrictedListSubclassVariable(
                    [
                        LazyVariableTracker.create(
                            value=value[i], source=GetItemSource(self.source, i)
                        )
                        for i in range(len(value))
                    ],
                    user_cls=type(value),
                    user_cls_source=AttrSource(self.source, "__class__"),
                    source=self.source,
                ),
            )
        elif TorchScriptObjectVariable.is_matching_cls(type(value)):
            from ..source import (
                FlattenScriptObjectSource,
                ScriptObjectQualifiedNameSource,
            )

            if torch._library.fake_class_registry.tracing_with_real(value):
                proxy = self.tx.output.root_tracer.create_graph_input(
                    re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                    type(value),
                    value,
                    source=self.source,
                )

                # setting is_unspecialized=False to not insert a as_tensor call in reconstruct by default
                # seting example to be real value because these example values will be used
                # as example_inputs for user compiler.
                proxy.node.meta["grapharg"] = GraphArg(
                    self.source, value, False, None, False, value
                )
                return TorchScriptObjectVariable.create(
                    proxy,
                    value,
                    source=self.source,
                )

            # This exists to allow a smoother transition.
            # The implications are:
            # The script objects won't be tracked as proxies.
            # Methods on these objects won't show up in the graph.
            # The original script object might be mutated.
            if not hasattr(value, "__obj_flatten__"):
                return self.wrap_user_defined(value)

            # Install the guards on the fully qualified name of the script object
            LazyVariableTracker.realize_all(
                VariableBuilder(self.tx, ScriptObjectQualifiedNameSource(self.source))(
                    value._type().qualified_name()  # type: ignore[attr-defined]
                )
            )
            # Install the guards on the content of the script object by setting the source
            # to be FlattenScriptObjectSource, which calls __obj_flatten__() to get the contents.
            LazyVariableTracker.realize_all(
                VariableBuilder(self.tx, FlattenScriptObjectSource(self.source))(
                    value.__obj_flatten__()
                )
            )

            fake_script_obj = torch._library.fake_class_registry.maybe_to_fake_obj(
                self.tx.output.fake_mode, value
            )

            proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(value),
                fake_script_obj,
                source=self.source,
            )

            # setting is_unspecialized=False to not insert a as_tensor call in reconstruct by default
            # seting example to be real value because these example values will be used
            # as example_inputs for user compiler.
            proxy.node.meta["grapharg"] = GraphArg(
                self.source, value, False, None, False, fake_script_obj
            )
            return TorchScriptObjectVariable.create(
                proxy,
                fake_script_obj,
                source=self.source,
            )
        elif (
            isinstance(value, (dict, collections.OrderedDict))
            and type(value).__new__ is dict.__new__
        ):
            # Construct a dict_vt that will reside inside the UserDefinedDictVariable
            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

            # Guard on the key order
            self.tx.output.guard_on_key_order.add(self.source.name())

            # We need all the keys to be hashable. We do this within the
            # _HashableTracker class in dicts.py
            def build_key_value(i, k, v):
                source_key = ConstDictKeySource(self.get_source(), i)
                key = LazyVariableTracker.create(k, source_key)

                source_value = DictGetItemSource(self.get_source(), source_key)
                value = LazyVariableTracker.create(v, source_value)

                return key, value

            # Ensure that we call dict.keys and not value.keys (which can call
            # overridden keys method). In the C++ guards, we relied on
            # PyDict_Next to traverse the dictionary, which uses the internal
            # data structure and does not call the overridden keys method.
            result = dict(
                build_key_value(i, k, v)
                for i, (k, v) in enumerate(get_items_from_dict(value))
            )

            # NB: This is deliberately kept ValueMutationNew because dict_vt is
            # an internal representation. dict_vt tracks the mutation on the
            # dict side. side_effects infra uses the UserDefinedDictVariable to
            # apply side-effects of this dict_vt.
            dict_vt = ConstDictVariable(
                result,
                user_cls=collections.OrderedDict
                if isinstance(value, collections.OrderedDict)
                else dict,
                mutation_type=ValueMutationNew(),
            )

            result = UserDefinedDictVariable(value, dict_vt=dict_vt, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif issubclass(type(value), MutableMapping):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return MutableMappingVariable(value, source=self.source)
        elif is_frozen_dataclass(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = FrozenDataClassVariable.create(self.tx, value, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif isinstance(value, dict_keys):
            if all(ConstantVariable.is_literal(k) for k in value):
                # If the dict_keys object is passed from outside the compile region, it must either be passed along with
                # the corresponding dict object or treated as a set (when only the keys are passed into the compiled region).
                # - If it is passed along with the dict, the dict object itself is already guarded.
                # - If only the dict_keys object is passed, we add EQUALS_MATCH and SEQUENCE_LENGTH guards
                #   to ensure it remains unchanged across multiple runs.
                items = [SourcelessBuilder.create(self.tx, v) for v in value]
                install_guard(
                    self.get_source().make_guard(GuardBuilder.SEQUENCE_LENGTH),
                    self.get_source().make_guard(GuardBuilder.EQUALS_MATCH),
                )
                return DictKeySetVariable(items, source=self.source)
            else:
                unimplemented("dict_keys with non-constant keys are not supported")
        else:
            return self.wrap_user_defined(value)

    def wrap_user_defined(self, value: Any):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        result = UserDefinedObjectVariable(value, source=self.source)
        if not SideEffects.cls_supports_mutation_side_effects(type(value)):
            # don't allow STORE_ATTR mutation with custom __setattr__
            return result
        return self.tx.output.side_effects.track_object_existing(value, result)

    def wrap_listlike(self, value: Union[tuple, list, odict_values, NamedTuple]):
        if config.specialize_int and type(value) is torch.Size:
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return ConstantVariable.create(value=value)

        # One can index a tensor with a list/tuple. Therefore, we need to
        # have a stricter match.
        self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

        # Tuples are immutable objects, so we should mark its items static. This
        # avoids wrapping of tuple items as symints. This helps for nn module
        # attributes like conv2d strides, dilations.
        if (
            istype(value, tuple)
            and all(ConstantVariable.is_literal(item) for item in value)
            and self.source.guard_source().is_unspecialized_nn_module()
        ):
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return TupleVariable([ConstantVariable.create(item) for item in value])

        output = [
            LazyVariableTracker.create(
                item,
                source=GetItemSource(self.get_source(), i),
            )
            for i, item in enumerate(value)
        ]

        maybe_gm = self.tx.output.local_scope.get("self")
        if isinstance(
            self.source, LocalSource
        ) and self.source.local_name in get_locals_to_steal(maybe_gm):
            # The input tensor list to dynamo from compiled autograd may contain activations
            # which are freed as they are used in inductor. Dynamo's default behavior is to
            # lift all tensors to the graph inputs, but this will cause dynamo to hold an
            # extra reference to the activation tensors and increase peak memory usage.
            # To allow freeing ASAP, we keep the list as graph argument to the dynamo output
            # graph, and unpack it locally.
            # e.g. instead of `def forward(self, L_inputs_0_, L_inputs_1_, ...):`, we have
            # `def forward(self, L_inputs_):`
            source = self.source
            assert isinstance(value, list)
            tensor_list_proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(value),
                value,
                source=source,
            )
            tensor_list_proxy.node.meta["steal_arg"] = True

            list_variable = wrap_fx_proxy_cls(
                target_cls=TensorVariable,
                tx=self.tx,
                proxy=tensor_list_proxy,
                example_value=value,
                subclass_type=None,
                source=source,
            )

            guards = []
            for i, tensor_variable in enumerate(list_variable.items):
                source_i = GetItemSource(base=source, index=i, index_is_slice=False)
                # access unpacked tensor from this list instead of from a lifted arg
                self.tx.output.input_source_to_var[source_i] = tensor_variable
                tensor_variable.proxy.node.meta["tensor_dict"] = _extract_tensor_dict(
                    value[i]
                )

                guard = functools.partial(
                    GuardBuilder.TENSOR_MATCH, value=TensorWeakRef(value[i])
                )
                guards.append(source_i.make_guard(guard))

            install_guard(*guards, skip=1)

            grapharg = GraphArg(
                source,
                value,
                pass_arg_as_tensor=False,
                fake_tensor=None,
                is_tensor=False,
            )
            tensor_list_proxy.node.meta["grapharg"] = grapharg

        result = BaseListVariable.cls_for_instance(value)(output, source=self.source)
        if istype(value, (list, collections.deque)):
            return self.tx.output.side_effects.track_mutable(value, result)
        return result

    def wrap_tuple_iterator(self, value: tuple_iterator):
        self.install_guards(GuardBuilder.TUPLE_ITERATOR_LEN)
        output = [
            VariableBuilder(self.tx, TupleIteratorGetItemSource(self.get_source(), i))(
                tuple_iterator_getitem(value, i)
            )
            for i in range(tuple_iterator_len(value))
        ]
        result = TupleIteratorVariable(output, source=self.source)
        return self.tx.output.side_effects.track_mutable(value, result)

    def wrap_range_iterator(self, value: range_iterator):
        self.install_guards(GuardBuilder.RANGE_ITERATOR_MATCH)
        # Get all the values from the range iterator; no need to install guards
        # on items since `RANGE_ITERATOR_MATCH` guarantees the same items.
        items = [ConstantVariable.create(v) for v in copy.deepcopy(value)]
        result = ListIteratorVariable(items, source=self.source)
        return self.tx.output.side_effects.track_mutable(value, result)

    def wrap_slice_range(self, value: Union[slice, range]):
        items = [
            VariableBuilder(self.tx, AttrSource(self.get_source(), k))(
                getattr(value, k)
            )
            for k in ("start", "stop", "step")
        ]
        self.install_guards(GuardBuilder.TYPE_MATCH)
        if isinstance(value, slice):
            return SliceVariable(items, source=self.source)
        else:
            return RangeVariable(items, source=self.source)

    def mark_static_input(self, value: torch.Tensor, guard: bool):
        from ..decorators import mark_static_address

        static_inputs_log.debug(
            "Marking static input %s, id: %s)", self.source.name(), id(value)
        )
        mark_static_address(value, guard=guard)

        # Check if we've seen this tensor before and update graph metadata if needed
        # As long as this runs before AOT this is sound
        if value in self.tx.output.side_effects:
            var = self.tx.output.side_effects[value]
            var.proxy.node.meta["tensor_dict"][
                "_dynamo_static_input_type"
            ] = value._dynamo_static_input_type

    def wrap_module(self, value: torch.nn.Module):
        from ..eval_frame import OptimizedModule

        if len(value.__dict__) == 0:
            unimplemented(f"uninitialized nn.Module: {typestr(value)}")
        if istype(value, OptimizedModule):
            # Check if the optimized module was disabled
            if inspect.getattr_static(value.forward, "_torchdynamo_disable", False):
                # This bytecode is mostly of kind LOAD_ATTR or LOAD_METHOD. If
                # we graph break here, Dynamo does not know how to create
                # continuation functions for such bytecodes. So, we delay the
                # graph break to CALL_FUNCTION.
                return DelayGraphBreakVariable(source=self.source)

            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.source = AttrSource(self.source, "_orig_mod")
            return self.wrap_module(value._orig_mod)

        if (
            isinstance(value, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM))
            and not config.allow_rnn
        ):
            unimplemented("TorchDynamo purposely graph breaks on RNN, GRU, LSTMs")

        if getattr(value, "_is_fsdp_managed_module", False):
            # See note [Dynamo treats FSDP wrapped modules as UnspecializedNNModule]
            # in fully_sharded_data_parallel.py for more information

            # we can't do this assert inside FSDP constructor,
            # since we don't know yet whether dynamo will be used
            assert getattr(
                value, "_fsdp_use_orig_params", False
            ), "Dynamo only supports FSDP with use_orig_params=True"

            # Note on FSDP guarding
            # Eager FSDP already assumes (requires, but without enforcement)
            # that users don't mutate their model parameters/structure after
            # FSDP wrapping, because FSDP wouldn't notice or update its
            # FlatParams.
            #
            # Therefore, torch.compile can skip guarding on params or submodule
            # structure of fsdp_managed modules, by using FSDPNNModuleSource as
            # the guard source.  This behavior is gated on
            # config.skip_fsdp_guards.
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = FSDPManagedNNModuleVariable(value, source=self.get_source())
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif mutation_guard.is_dynamic_nn_module(value, self.tx.export):
            # created dynamically, don't specialize on it

            # Note [Tracing a torch.compiled function]
            # when make_fx tracing a compiled function, we need
            if isinstance(value, torch.fx.experimental.proxy_tensor._AttrProxy):
                value = value.get_base()
                self.source = AttrProxySource(self.source)

            self.install_guards(GuardBuilder.TYPE_MATCH)
            if torch._dynamo.config.inline_inbuilt_nn_modules:
                freezing = is_parameter_freezing()
                for p in value.parameters():
                    self.mark_static_input(p, guard=freezing)

                for b in value.buffers():
                    self.mark_static_input(b, guard=freezing)

                if freezing:
                    # we need to add the module to tracing context
                    # in order to allow its params to get invalidated
                    # this will get cleaned up once compile ends
                    self.tx.output.nn_modules[self.name] = value

            if value.__module__.startswith(("torch.nn.", "torch.ao.")) or getattr(
                value.__class__, "_dynamo_marked_static", False
            ):
                result = UnspecializedBuiltinNNModuleVariable(value, source=self.source)
            else:
                result = UnspecializedNNModuleVariable(value, source=self.source)

            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif issubclass(
            value.__class__, torch.nn.parallel.distributed.DistributedDataParallel
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return UnspecializedNNModuleVariable(value, source=self.get_source())
        else:
            return self.tx.output.register_attr_or_module(
                value,
                self.name,
                source=self.get_source(),
                # Guards are added inside register_attr_or_module
            )

    def wrap_literal(self, value):
        if not config.specialize_int and type(value) is int:
            # unspecializing int by default, but still
            # specialize for the following conditions
            if is_int_specialization_case(value, self.source):
                self.install_guards(GuardBuilder.CONSTANT_MATCH)
                return ConstantVariable.create(value=value, source=self.source)
            else:
                return self.wrap_symint(value)
        elif not config.specialize_float and type(value) is float:
            return self.wrap_symfloat(value)
        else:
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            result = ConstantVariable.create(value=value, source=self.source)
            if isinstance(value, (list, set)):
                return self.tx.output.side_effects.track_mutable(value, result)
            return result

    def assert_not_wrapped_by_this_graph(self, value: torch.Tensor):
        if is_fake(value) and maybe_get_fake_mode(value) is self.tx.fake_mode:
            raise InternalTorchDynamoError(
                "Cannot wrap a Tensor that has already been",
                "wrapped by this instance of Dynamo",
            )

    def wrap_tensor(self, value: torch.Tensor):
        source = self.get_source()

        # We cannot already be tracking the tensor, which implies
        # it would have already been wrapped
        assert value not in self.tx.output.side_effects

        is_static_input = get_static_address_type(value) is not None

        if (
            config.inline_inbuilt_nn_modules
            and not is_static_input
            and (
                isinstance(value, torch.nn.Parameter)
                # mark tensor attributes of nn modules static. This is done to keep inline_inbuilt_nn_modules behavior
                # compatible with previous behavior.
                or (source and source.guard_source().is_unspecialized_nn_module())
            )
        ):
            self.mark_static_input(value, guard=is_parameter_freezing())
            is_static_input = True

        make_graph_attribute = is_static_input and (
            not config.inline_inbuilt_nn_modules
            or is_parameter_freezing()
            or torch._dynamo.config.prepare_freezing
        )

        if (
            source.guard_source().is_specialized_nn_module() or make_graph_attribute
        ) and not source.guard_source().is_fsdp_module():
            self.assert_not_wrapped_by_this_graph(value)
            return self.tx.output.register_attr_or_module(
                value, self.name, source=source
            )

        if is_constant_source(source):
            self.assert_not_wrapped_by_this_graph(value)
            return self.tx.output.register_attr_or_module(
                value,
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                source=source,
                # Guards are added inside register_attr_or_module
            )

        if type(value) in config.traceable_tensor_subclasses:
            # Ordinarily, we would fakeify a tensor so that it can get dynamic
            # shapes and be computed on without triggering actual operations.
            # However, how can we fakeify a tensor subclass?  Ordinary
            # inheritance (nor multiple inheritance) won't work work.
            #
            # Instead, our plan is to *manually simulate* the tensor subclass
            # inheriting from a fake tensor with dynamo.  This means our
            # data representation for a tensor subclass will be a fake tensor
            # + tensor subclass type + any extra data the subclass may have
            # been storing on the tensor.  Because all Python accesses are
            # mediated through TensorWithTFOverrideVariable, we can ensure
            # that we dispatch differently, e.g., according to
            # __torch_function__
            #
            # To simplify things for now, the __dict__ tracking bits haven't
            # been implemented yet, but they can be added into this design at
            # a later point in time.
            subclass_type = type(value)
        else:
            assert type(value) in (
                torch.Tensor,
                torch.nn.Parameter,
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ) or is_traceable_wrapper_subclass(value), type(value)
            subclass_type = None

        # NB: this just says we accessed a tensor from the same source again
        # (e.g., a tensor lives in a global foo, and we LOAD_GLOBAL it twice).
        # This is distinct from two distinct sources mapping to the same
        # Tensor (per id())!  No guard is necessary here.  See below for the
        # other case.
        is_duplicate_tensor = source in self.tx.output.input_source_to_var
        if is_duplicate_tensor:
            return self.tx.output.input_source_to_var[source]

        if get_static_address_type(value) == "guarded":
            self.install_guards(GuardBuilder.ID_MATCH)

        # By this point, we should have deduplicated all tensors
        self.assert_not_wrapped_by_this_graph(value)

        options = {}
        if type(value) in config.traceable_tensor_subclasses:
            options["torch_function_fn"] = build_torch_function_fn(
                self.tx, value, self.source
            )
            self.install_guards(GuardBuilder.TYPE_MATCH)

        if (
            isinstance(value, torch.Tensor)
            and value.is_nested
            and not isinstance(value, torch.nested._internal.nested_tensor.NestedTensor)
        ):
            unimplemented("torch.compile does not support strided NestedTensor")

        # TODO(pearu,sparse-team) - Add the corresponding SPARSE_TENSOR_MATCH guards
        if (
            isinstance(value, torch.Tensor)
            and is_sparse_any(value)
            and (not self.tx.export or not config.capture_sparse_compute)
        ):
            # A hot fix for sparse tensors + torch.compile. Support for
            # export + sparsity is being added but we need to create
            # SPARSE_TENSOR_GUARDS for guards to work propertly.
            unimplemented("torch.compile does not support sparse Tensors")

        if (
            safe_has_grad(value)
            and safe_grad(value) is not None
            and value.dtype != safe_grad(value).dtype
        ):
            unimplemented(
                "Inconsistent dtype between tensor and its gradient. "
                "This can happen in FSDP and crashes meta tensor creation. "
                "This is potentially a workaround. Fixing it correctly "
                "requires some design around FSDP + torch.compile."
            )

        # tx.output has multiple tracers if we're introspecting HigherOrderOperator.
        # When we've discovered an untracked tensor, then we actually need
        # to get Dynamo to track the tensor (which is what this function does)
        # and put it as a graph input on the root tracer. Later on,
        # if the input is actually used in the body of the HigherOrderOperator,
        # then the relevant SubgraphTracer will lift it to being an input of
        # the subgraph.
        # See NOTE [HigherOrderOperator tracing design] for more details.

        example_value = wrap_to_fake_tensor_and_record(
            value, tx=self.tx, is_tensor=True, source=source
        )
        tensor_proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
            type(value),
            example_value,
            source=source,
        )
        cache_real_value_when_export(self.tx, tensor_proxy, value)

        tensor_variable = wrap_fx_proxy(
            tx=self.tx,
            proxy=tensor_proxy,
            example_value=example_value,
            subclass_type=subclass_type,
            source=source,
            **options,
        )

        if value._is_view():
            # If value is a view, add its base tensor to the tracked fakes list.
            # This is so we are able to access the correct source for its symbolic
            # shape values, in case we need them.
            wrap_to_fake_tensor_and_record(
                value._base,
                tx=self.tx,
                source=AttrSource(source, "_base"),
                is_tensor=True,
            )

        guard_type = GuardBuilder.TENSOR_MATCH

        if isinstance(source, GradSource) and is_from_optimizer_source(source):
            guard_type = GuardBuilder.NOT_NONE_MATCH

        self.install_guards(
            functools.partial(
                guard_type,
                value=(
                    value
                    if isinstance(source, NumpyTensorSource)
                    else TensorWeakRef(value)
                ),
            )
        )

        # We install TYPE_MATCH guards for traceable wrapper subclass object,
        # and recursively install corresponding guard for each inner attribute.
        if is_traceable_wrapper_subclass(value):
            self.install_guards(GuardBuilder.TENSOR_SUBCLASS_METADATA_MATCH)
            self.install_guards(GuardBuilder.TYPE_MATCH)
            install_guard(
                SubclassAttrListSource(source).make_guard(GuardBuilder.EQUALS_MATCH)
            )

            attrs, _ = value.__tensor_flatten__()
            for attr in attrs:
                inner_value = getattr(value, attr)
                inner_source = AttrSource(self.source, attr)
                LazyVariableTracker.realize_all(
                    VariableBuilder(self.tx, inner_source)(inner_value)
                )

        self.tx.output.input_source_to_var[source] = tensor_variable
        assert "tensor_dict" not in tensor_proxy.node.meta
        tensor_proxy.node.meta["tensor_dict"] = _extract_tensor_dict(value)

        # Note: this information is conveyed via subclass_type now
        fake_tensor_value = tensor_variable.proxy.node.meta["example_value"]
        if maybe_get_fake_mode(fake_tensor_value) is not self.tx.fake_mode:
            raise InternalTorchDynamoError("Wrapped Tensor must be this graph's fake")

        grapharg = GraphArg(source, value, False, fake_tensor_value)
        tensor_proxy.node.meta["grapharg"] = grapharg
        return tensor_variable

    def wrap_numpy_ndarray(self, value):
        assert np is not None
        assert isinstance(value, np.ndarray)

        source = NumpyTensorSource(self.get_source())

        from torch._numpy import _util

        readonly = not value.flags.writeable
        if readonly:
            try:
                value.flags.writeable = True
            except ValueError:
                # One can not easily make nditer elements writable,
                # but warning is not the end of the world
                assert isinstance(value.base, np.nditer)

        with torch_function_mode_stack_state_mgr.temp_restore_stack():
            try:
                tensor_value = _util._try_convert_to_tensor(value)
                if readonly:
                    from torch._prims_common import clone_preserve_strides

                    tensor_value = clone_preserve_strides(tensor_value)
            except NotImplementedError as e:
                # failed to convert to tensor, graph break
                unimplemented(str(e))

        # We do this because we want the full behavior of guarding the numpy ndarray as if it were
        # a tensor. It's a little annoying to make a VT to throw out, but there's so many side effects here
        # that there's not another great way to do this atm.
        # This creates the right graphargs, as well as registration for guards in tensor names and shape env.
        LazyVariableTracker.realize_all(VariableBuilder(self.tx, source)(tensor_value))
        example_value = wrap_to_fake_tensor_and_record(
            tensor_value,
            tx=self.tx,
            is_tensor=False,
            source=source,
        )
        proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
            type(tensor_value),
            example_value,
            source=source,
        )
        cache_real_value_when_export(self.tx, proxy, tensor_value)
        options = {"source": source}
        numpy_ndarray_variable = wrap_fx_proxy_cls(
            target_cls=NumpyNdarrayVariable,
            tx=self.tx,
            proxy=proxy,
            example_value=example_value,
            **options,
        )

        self.tx.output.input_source_to_var[source] = numpy_ndarray_variable
        example_value = numpy_ndarray_variable.proxy.node.meta["example_value"]

        # pass_arg_as_tensor should be true because we are wrapping a np.ndarray as argument input, and it needs to be
        # converted to a tensor.
        grapharg = GraphArg(
            source,
            tensor_value,
            pass_arg_as_tensor=True,
            fake_tensor=example_value,
            is_tensor=True,
            example_strong_ref=tensor_value,
        )
        proxy.node.meta["grapharg"] = grapharg

        return numpy_ndarray_variable

    def wrap_symint(self, value):
        assert type(value) is int

        if self.name in self.tx.output.unspec_variable_map:
            return self.tx.output.unspec_variable_map[self.name]

        shape_env = self.tx.output.shape_env
        if TracingContext.get().force_unspec_int_unbacked_size_like:
            wrapped_value = shape_env.create_unbacked_symint()
            _constrain_range_for_size(wrapped_value)
            self.tx.output.tracked_fakes.append(
                TrackedFake(wrapped_value, self.source, None)
            )

        # NB: We do not do float.  For motivation, see
        # https://docs.google.com/document/d/1INSCdYu1PxXcr43HrD82OudeEuS-qxQe1yZmLg2wy6A/edit
        # but the general idea is that we generate kernels that can
        # take unspecialized floats and use them in sizevar computation
        elif not is_constant_source(self.get_source()):
            if torch._dynamo.config.specialize_int:
                # If specialize_int is False, also return
                # a constant (but this should have been handled
                # in the caller, TBH)
                self.install_guards(GuardBuilder.CONSTANT_MATCH)
                return ConstantVariable.create(value=value, source=self.source)

            name = self.source.name()

            frame_state_entry = process_automatic_dynamic(
                self.tx,
                name,
                FrameStateSizeEntry.make_scalar(value),
                is_unspecialized_nn_module=self.source.guard_source().is_unspecialized_nn_module(),
            )

            # TODO: This should be dynamic, as we in general do not
            # know if bare integers are actually going to be sizevars
            # and it is inappropriate to eagerly duck size them with
            # real sizevars
            if (
                config.automatic_dynamic_shapes
                and frame_state_entry.scalar is auto_dynamic
            ):
                dynamic_dim = get_automatic_dynamic_shapes_mark_as()
            elif not config.assume_static_by_default:
                dynamic_dim = DimDynamic.DYNAMIC
            else:  # assume_static_by_default
                # TODO: dynamic_dim = DimDynamic.STATIC should work but
                # for some reason it doesn't
                self.install_guards(GuardBuilder.CONSTANT_MATCH)
                return ConstantVariable.create(value=value)

            wrapped_value = shape_env.create_unspecified_symint_and_symbol(
                value,
                source=self.source,
                dynamic_dim=dynamic_dim,
            )

            self.tx.output.tracked_fakes.append(
                TrackedFake(wrapped_value, self.source, None)
            )
        else:
            assert is_constant_source(self.get_source())
            # TODO: Do I actually need guard for constant source?
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return ConstantVariable.create(value=value, source=self.source)

        assert not isinstance(self.get_source(), RandomValueSource)
        install_guard(self.get_source().make_guard(GuardBuilder.TYPE_MATCH))

        options = {"source": self.get_source()}

        proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
            type(wrapped_value),
            wrapped_value,
            source=self.get_source(),
        )

        sym_expr = wrapped_value.node.expr
        assert isinstance(sym_expr, sympy.Symbol), f"{sym_expr} is not a basic Symbol."
        self.tx.output.root_tracer.bound_symbols[sym_expr] = proxy
        unspec_var = SymNodeVariable(proxy, wrapped_value, **options)
        self.tx.output.unspec_variable_map[self.name] = unspec_var

        if not is_constant_source(self.get_source()):
            if self.tx.export and not isinstance(self.get_source(), LocalSource):
                raise AssertionError(
                    f"Dynamo attempts to add additional input during export: value={wrapped_value}, source={self.get_source()}"
                )

            proxy.node.meta["grapharg"] = GraphArg(
                self.get_source(),
                wrapped_value,
                pass_arg_as_tensor=False,
                fake_tensor=None,
                is_tensor=False,
                example_strong_ref=wrapped_value,
            )

        return unspec_var

    def wrap_symfloat(self, value):
        # SymFloat wrapping is special.  We first wrap it in the same way we
        # do an unspecialized primitive, and then we item() it into a
        # SymFloat.  Removal of the item() call is left to a later FX pass,
        # mostly because that pass is more easily done after we have lowered
        # to ATen ops.  (Dynamo doesn't do decomposition right now).

        if self.name in self.tx.output.unspec_variable_map:
            return self.tx.output.unspec_variable_map[self.name]

        frame_state_entry = process_automatic_dynamic(
            self.tx,
            self.source.name(),
            FrameStateSizeEntry.make_scalar(value),
            is_unspecialized_nn_module=self.source.guard_source().is_unspecialized_nn_module(),
        )

        # NB: we specialize on nan input, because our guard modeling in
        # ShapeEnv cannot deal with nan
        if (
            torch._dynamo.config.specialize_float
            or is_constant_source(self.get_source())
            or math.isnan(value)
            or math.isinf(value)
            # We don't support cudagraphs for now. Without this cudagraphs
            # break because they expect all cuda inputs but our tensorified
            # float will be a f64[] cpu tensor. Fixes the following test
            # when specialize_float=False
            # python test/inductor/test_compiled_optimizers.py CompiledOptimizerTests.test_rmsprop_weight_decay_maximize_capturable_cuda # noqa: B950
            or torch._inductor.config.triton.cudagraphs
            or justknobs_check("pytorch/compiler:unspecialize_float_killswitch", False)
            or frame_state_entry.scalar is not auto_dynamic
        ):
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return ConstantVariable.create(value=value, source=self.source)

        # NB: At the point we've gotten here, we don't assume static by
        # default.  Since we have a guard mechanism, there isn't really any
        # downside to trying to be dynamic for float all the time.  Unlike
        # ints, this won't make codegen perf worse.  Modest cost to compile
        # time.

        wrapped_value = torch.tensor(value, dtype=torch.float64)

        # We don't support specializing floats for grad checking tensors
        # See https://github.com/pytorch/pytorch/pull/140828 for more
        # context.
        if torch._C._functorch.is_gradtrackingtensor(wrapped_value):
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return ConstantVariable.create(value=value, source=self.source)

        # TODO: Switch RandomValueSource over to use this, this is more
        # accurate
        assert not isinstance(self.get_source(), RandomValueSource)
        install_guard(self.get_source().make_guard(GuardBuilder.TYPE_MATCH))

        # The FloatTensorSource here is just for pedantic correctness: if you
        # guard against an UnspecializedPythonVariable, you need to guard
        # against the tensor-ified version of the local, otherwise it's not a
        # Tensor.  However, we never let the UnspecializedPythonVariable escape
        # here, so there should never actually be any guards against this
        # source.
        source = FloatTensorSource(self.get_source())
        options = {"source": source, "raw_value": value}

        # TODO: Maybe the tensor-ification should be built into the source,
        # rather than by special pattern match
        example_value = wrap_to_fake_tensor_and_record(
            wrapped_value, tx=self.tx, is_tensor=False, source=source
        )
        proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
            type(wrapped_value),
            example_value,
            source=source,
        )
        cache_real_value_when_export(self.tx, proxy, wrapped_value)

        unspec_var = wrap_fx_proxy_cls(
            UnspecializedPythonVariable,
            tx=self.tx,
            proxy=proxy,
            example_value=example_value,
            **options,
        )
        assert isinstance(unspec_var, UnspecializedPythonVariable)
        self.tx.output.unspec_variable_map[self.name] = unspec_var

        if self.tx.export and not isinstance(self.get_source(), LocalSource):
            raise AssertionError(
                f"Dynamo attempts to add additional input during export: value={wrapped_value}, source={self.get_source()}"
            )
        fake_tensor_value = None
        example_value = unspec_var.proxy.node.meta["example_value"]
        assert is_fake(example_value)

        fake_tensor_value = example_value
        assert fake_tensor_value.fake_mode is self.tx.fake_mode, (
            f"fake mode ({fake_tensor_value.fake_mode}) from fake tensor metadata doesn't match mode"
            "({self.tx.fake_mode}) from InstructionTranslator"
        )

        # There's something a bit incoherent about pass_arg_as_tensor,
        # specifically regarding sources.
        #
        # Specifically, suppose we have "x: float" local argument.  We
        # eventually end up with an UnspecializedPythonVariable denoting
        # torch.as_tensor(x)... but it's source is still L['x'] (which if you
        # accessed it directly is a float!)  So you gotta be careful when
        # setting up your guards, because it's still going to be a float at
        # this point, the conversion happens only precisely at the point we're
        # actually calling the FX graph.  This happens to be what we want for
        # shape guard generation, but it's kind of unintuitive.
        proxy.node.meta["grapharg"] = GraphArg(
            self.get_source(),
            wrapped_value,
            pass_arg_as_tensor=True,
            fake_tensor=fake_tensor_value,
            is_tensor=False,
            example_strong_ref=wrapped_value,
        )

        # Directly do item to bypass capture_scalar_outputs
        r = wrap_fx_proxy(
            self.tx,
            self.tx.output.create_proxy(
                "call_method",
                "item",
                *proxy_args_kwargs([unspec_var], {}),
            ),
        )
        self.tx.output.tracked_fakes.append(TrackedFake(r.sym_num, self.source, None))

        get_metrics_context().set("tensorify_float_attempt", True, overwrite=True)

        return r

    def wrap_unspecialized_primitive(self, value):
        if self.name in self.tx.output.unspec_variable_map:
            return self.tx.output.unspec_variable_map[self.name]

        wrapped_value = torch.tensor(value)
        if not isinstance(self.get_source(), RandomValueSource):
            install_guard(self.get_source().make_guard(GuardBuilder.TYPE_MATCH))

        options = {"source": self.get_source()}
        options.update({"raw_value": value})

        example_value = wrap_to_fake_tensor_and_record(
            wrapped_value, tx=self.tx, is_tensor=False, source=self.get_source()
        )
        proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
            type(wrapped_value),
            example_value,
            source=self.get_source(),
        )
        cache_real_value_when_export(self.tx, proxy, wrapped_value)

        unspec_var = wrap_fx_proxy_cls(
            UnspecializedPythonVariable,
            tx=self.tx,
            proxy=proxy,
            example_value=example_value,
            **options,
        )
        self.tx.output.unspec_variable_map[self.name] = unspec_var
        if not is_constant_source(self.get_source()):
            if self.tx.export and not isinstance(self.get_source(), LocalSource):
                raise AssertionError(
                    f"Dynamo attempts to add additional input during export: value={wrapped_value}, source={self.get_source()}"
                )
            fake_tensor_value = None
            if isinstance(unspec_var, ConstantVariable):
                # TODO: when can this happen?
                example_value = unspec_var.value
            else:
                example_value = unspec_var.proxy.node.meta["example_value"]
            assert is_fake(example_value)

            fake_tensor_value = example_value
            assert fake_tensor_value.fake_mode is self.tx.fake_mode, (
                f"fake mode ({fake_tensor_value.fake_mode}) from fake tensor metadata doesn't match mode"
                "({self.tx.fake_mode}) from InstructionTranslator"
            )

            proxy.node.meta["grapharg"] = GraphArg(
                self.get_source(),
                wrapped_value,
                pass_arg_as_tensor=True,
                fake_tensor=fake_tensor_value,
                is_tensor=False,
                example_strong_ref=wrapped_value,
            )
        return unspec_var


def _dataclasses_fields_lambda(obj):
    if isinstance(obj, UserDefinedObjectVariable):
        value = obj.value
    else:
        unimplemented(f"Dataclass fields handling fails for type {obj}")
    items = []
    for field in dataclasses.fields(value):
        source = None
        if obj.source:
            source = DictGetItemSource(
                AttrSource(obj.source, "__dataclass_fields__"), field.name
            )
        items.append(UserDefinedObjectVariable(field, source=source))
    return TupleVariable(items)


def _clone_input(value, fake_mode):
    if isinstance(value, torch.Tensor):
        # tensor subclasses will not be converted to FakeTensors and need to be cloned
        if not (
            isinstance(value, FakeTensor)
            or (
                # Is functional tensor fakeified by this instance of Dynamo
                torch._is_functional_tensor(value)
                and maybe_get_fake_mode(value) is fake_mode
            )
            or value.is_nested
        ):
            # NB: ensure strides are preserved
            value = clone_input(value)

    return value


def wrap_fx_proxy(
    tx, proxy, example_value=None, subclass_type=None, **options
) -> VariableTracker:
    kwargs = {
        "tx": tx,
        "proxy": proxy,
        "example_value": example_value,
        "subclass_type": subclass_type,
        **options,
    }
    if subclass_type is None:
        return wrap_fx_proxy_cls(target_cls=TensorVariable, **kwargs)
    else:
        result = wrap_fx_proxy_cls(target_cls=TensorWithTFOverrideVariable, **kwargs)
        result.install_global(tx)
        return result


def cache_real_value_when_export(tx, proxy, example_value):
    if tx.export:
        # The legacy behavior for real value cache with subclasses was
        # to perform a clone WITHOUT preserving the subclass.  It's
        # not entirely clear this is what you actually want though.
        with torch._C.DisableTorchFunctionSubclass():
            proxy.tracer.real_value_cache[proxy.node] = _clone_input(
                example_value, tx.fake_mode
            )


# Note: Unfortunate split due to some gross classes existing that subclass TensorVariable
# Should be compositional instead
#
# This is a horribly complicated function that does too many things, to
# explain what it does, let's first talk about the classic usage wrap_fx_proxy
# for a TensorVariable.  There are two primary modes of use:
#
#   1. Wrapping a pre-existing Tensor.  In this case, example_value is set
#      to the pre-existing Tensor.  (Note that this example_value will NOT
#      be the final example_value we put into node.meta['example_value'],
#      instead it is converted into a fake tensor using
#      wrap_to_fake_tensor_and_record and registered as a graph input.)
#
#   2. "Wrapping" the result of some Tensor operation Dynamo traced over. In
#      this case, example_value is None (and we are going to figure it out
#      ourselves using FakeTensors, via get_fake_value, which will run
#      the operation represented by the (singular!) FX node referenced by
#      the passed in proxy.)
#
# The expectation is you end up with a Tensor output, and everything is
# straightforwardly traced into the graph.
#
# In all cases, the returned `TensorVariable` subclass will have an `example_value`
# and that `example_value` must be a `FakeTensor` produced by the currently running
# instance of Dynamo.
#
# Upon closer inspection, you may notice that there are a slurry of non-Tensor
# output cases in handle_traced_output.  What gives?  Well, we sometimes trace operations into the
# graph that don't involve tensors.
#
#   * Some operators return tuples; we need to recursively handle their
#     contents
#
#   * Some operators have side effects that will affect subsequent AOTAutograd
#     tracing but don't otherwise return anything.
#
#   * Some operators return symbolic ints/floats/bools which can go in the
#     graph and be traced (but only if they're actually symbolic!  If they're
#     static you don't want to put them in the graph, which means you
#     shouldn't call this function.)
#
# The common theme is that you only use this function WHEN YOU ARE TRACING
# SOMETHING INTO THE GRAPH.  This is sort of obvious, because you can't call
# this function without a proxy.
def wrap_fx_proxy_cls(
    target_cls, tx, proxy, example_value=None, subclass_type=None, **options
):
    if example_value is None:
        return _wrap_fx_proxy(
            target_cls, tx, proxy, example_value, subclass_type, **options
        )
    elif isinstance(example_value, torch.Tensor):
        return _wrap_fx_preexisting_tensor(
            target_cls, tx, proxy, example_value, subclass_type, **options
        )
    else:
        # This will skip tracing an op and recursively reinvoke wrap_fx_proxy_cls on supported
        # data structures. In essence this just handles tracing some other value which may
        # contain Fake Tensors or is otherwise proxyable.
        return handle_traced_output(
            example_value, tx, proxy, options, subclass_type, target_cls
        )


# This is 1 above (wrapping a preexisting tensor)
def _wrap_fx_preexisting_tensor(
    target_cls, tx, proxy, tensor, subclass_type=None, **options
):
    from ..symbolic_convert import InstructionTranslatorBase

    assert isinstance(
        tensor, torch.Tensor
    ), f"_wrap_fx_preexisting_tensor expected tensor, got {type(tensor)}"

    assert isinstance(tx, InstructionTranslatorBase)
    if "guards" in options and options["guards"] is not None:
        tx.output.guards.update(options["guards"])

    # Placeholders always carry example_value in node.meta.
    # non-placeholders always have no example_value in node.meta
    if proxy.node.op == "placeholder":
        assert (
            "example_value" in proxy.node.meta
        ), f"placeholder {proxy} doesn't have 'example_value' in node.meta"
    else:
        assert (
            "example_value" not in proxy.node.meta
        ), f"{proxy.node.meta['example_value']}"

    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    with torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
        # Handle recursive calls here
        if maybe_get_fake_mode(tensor) is tx.fake_mode:
            pass
        else:
            cache_real_value_when_export(tx, proxy, tensor)
            if tx.export:
                # The legacy behavior for real value cache with subclasses was
                # to perform a clone WITHOUT preserving the subclass.  It's
                # not entirely clear this is what you actually want though.
                with torch._C.DisableTorchFunctionSubclass():
                    proxy.tracer.real_value_cache[proxy.node] = _clone_input(
                        tensor, tx.fake_mode
                    )
            # NB: If we're ignoring subclass, then the expectation is you will
            # take the returned TensorVariable and wrap it into a more
            # accurate TensorVariable that is able to track subclass-ness;
            # otherwise this is wrong!
            kwargs = {
                "is_tensor": target_cls
                in (TensorVariable, TensorWithTFOverrideVariable),
            }
            assert "source" in options and options["source"] is not None
            kwargs["source"] = options["source"]
            tensor = wrap_to_fake_tensor_and_record(tensor, tx=tx, **kwargs)

        if tensor.device.type != "meta" and (
            maybe_get_fake_mode(tensor) is not tx.fake_mode
        ):
            raise InternalTorchDynamoError(
                "`tensor` needs to be a `FakeTensor`"
                f"wrapped by this instance of Dynamo. Found: {tensor}"
            )

    return handle_traced_output(tensor, tx, proxy, options, subclass_type, target_cls)


# This is 2 in the above comment (wrapping the output of a traced op)
def _wrap_fx_proxy(
    target_cls, tx, proxy, example_value=None, subclass_type=None, **options
):
    from ..symbolic_convert import InstructionTranslatorBase

    assert isinstance(tx, InstructionTranslatorBase)
    if "guards" in options and options["guards"] is not None:
        tx.output.guards.update(options["guards"])

    assert "example_value" not in proxy.node.meta, f"{proxy.node.meta['example_value']}"

    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    with torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
        # with preserve_rng_state():
        # only allow_non_graph_fake in this instance because we handle the non-fake
        # cases properly below.
        example_value = get_fake_value(proxy.node, tx, allow_non_graph_fake=True)

    return handle_traced_output(
        example_value, tx, proxy, options, subclass_type, target_cls
    )


# This handles wrapping of the output of an op traced into the graph
def handle_traced_output(example_value, tx, proxy, options, subclass_type, target_cls):
    import torch._functorch.vmap
    import torch._subclasses.fake_tensor
    import torch._utils

    if isinstance(example_value, torch.Tensor):
        is_parameter = isinstance(example_value, torch.nn.Parameter)
        is_buffer = isinstance(example_value, torch.nn.Buffer)

        # NB: In most (all?) cases, this does not actually do a clone.
        # (WARNING: this means that if we mutate metadata on the fake
        # tensor, the stored example value will update too!)
        example_value = _clone_input(example_value, tx.fake_mode)
        set_example_value(proxy.node, example_value)
        # We bind the unbacked symints in sizes/trdies of tensor lazily.
        # So that subgraphs can access the unbacked symbol's proxy in parent graph
        # when lifting unbacked symbols of input tensors to subgraph inputs.
        # We do it lazily because the tensor may not be used in subgraphs.
        tx.output.current_tracer.track_unbacked_symbols(example_value, proxy)
        specialized_props = target_cls.specialize(example_value)
        # TODO: not sure about this fake mode test
        if (
            isinstance(example_value, torch._subclasses.fake_tensor.FakeTensor)
            and example_value.fake_mode is tx.fake_mode
        ):
            tensor_type = subclass_type if subclass_type else torch.Tensor
            specialized_props["class_type"] = (
                torch.nn.Parameter
                if is_parameter
                else torch.nn.Buffer
                if is_buffer
                else tensor_type
            )

        options.update(specialized_props)
        return target_cls(proxy, **options)
    elif (
        hasattr(proxy.node.target, "__name__")
        and proxy.node.target.__name__ == "set_state"
        and isinstance(proxy.node.target.__self__, torch._C.Generator)
        or proxy.node.target == torch.random.set_rng_state
    ):
        return TorchInGraphFunctionVariable(proxy.node.target)
    elif (
        proxy.node.target == torch._C._DisableFuncTorch
        or proxy.node.target == torch.cuda._is_in_bad_fork
    ):
        return UserDefinedObjectVariable(example_value)
    elif istype(example_value, torch.Size) and all(
        isinstance(x, int) for x in example_value
    ):
        sizes = [ConstantVariable.create(x) for x in example_value]
        return SizeVariable(sizes, **options)
    elif isinstance(example_value, (tuple, list)):
        set_example_value(proxy.node, example_value)
        unpacked = []
        for i, val in enumerate(example_value):
            if val is None:
                # nn.MultiheadAttention() can return None, see issue #175
                unpacked.append(
                    ConstantVariable.create(None, **options),
                )
            else:
                proxy_i = proxy.tracer.create_proxy(
                    kind="call_function",
                    target=operator.getitem,
                    args=(proxy, i),
                    kwargs={},
                )

                if "source" in options:
                    # This path should only trigger for list stealing, so it's
                    # safe to use `GetItemSource`.
                    assert isinstance(example_value, list)
                    source = options["source"]
                    options_i = options.copy()
                    options_i["source"] = GetItemSource(
                        base=source, index=i, index_is_slice=False
                    )
                else:
                    # use the same options object as parent
                    options_i = options

                # WARNING: this assumes the same target_cls as this tuple/list call
                unpacked.append(
                    wrap_fx_proxy_cls(
                        target_cls=target_cls,
                        tx=tx,
                        proxy=proxy_i,
                        example_value=val,
                        **options_i,
                    )
                )
        if isinstance(example_value, torch.Size):
            # NB: Keep the old proxy around.  See SizeVariable for an
            # explanation why
            return SizeVariable(unpacked, proxy, **options)
        elif istype(example_value, tuple):
            return TupleVariable(unpacked, **options)
        elif istype(example_value, (list, immutable_list)):
            return ListVariable(unpacked, **options)
        else:
            assert example_value.__class__.__module__ == "torch.return_types" or hasattr(
                example_value, "_fields"
            ), f"expected {example_value.__class__.__module__} == torch.return_types or named tuple but got {type(example_value)}"
            return NamedTupleVariable(unpacked, example_value.__class__, **options)
    elif example_value is None or proxy.node.target is torch.manual_seed:
        return ConstantVariable.create(None, **options)
    elif isinstance(example_value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        tx.output.current_tracer.track_unbacked_symbols(example_value, proxy)
        set_example_value(proxy.node, example_value)
        return SymNodeVariable(proxy, example_value, **options)
    elif (
        inspect.isclass(proxy.node.target)
        and issubclass(proxy.node.target, torch.Stream)
    ) or proxy.node.target in [
        device_interface.current_stream
        for _, device_interface in get_registered_device_interfaces()
    ]:
        set_example_value(proxy.node, example_value)
        return StreamVariable(proxy, example_value, example_value.device, **options)
    elif (
        inspect.isclass(proxy.node.target)
        and issubclass(proxy.node.target, torch.Event)
    ) or proxy.node.target in [
        device_interface.Event
        for _, device_interface in get_registered_device_interfaces()
    ]:
        set_example_value(proxy.node, example_value)
        return EventVariable(proxy, example_value, **options)
    elif proxy.node.target == "query" and proxy.node.op == "call_method":
        set_example_value(proxy.node, example_value)
        return ConstantVariable(example_value, **options)
    elif (
        example_value is not None
        and isinstance(example_value, torch.Event)
        and proxy.node.target == "record_event"
        and proxy.node.op == "call_method"
    ):
        set_example_value(proxy.node, example_value)
        return EventVariable(proxy, example_value, **options)
    elif isinstance(example_value, int) and (
        proxy.node.target
        in [
            torch.sym_int,
            getattr,
            operator.getitem,
            torch._utils._element_size,
            torch.seed,
            operator.mod,
            torch._functorch.vmap._validate_and_get_batch_size,
            # some mac builds are missing torch.distributed.get_rank()
            getattr(torch.distributed, "get_rank", _missing),
            getattr(torch.distributed, "get_world_size", _missing),
            # This always wants to be in the graph, even if the constraint
            # results in a constant int
            torch._constrain_as_size,
        ]
        or (
            # TODO: this is a little sus, because we didn't check what the self is
            proxy.node.op == "call_method"
            and proxy.node.target in ["bit_length"]
        )
    ):
        set_example_value(proxy.node, example_value)
        return ConstantVariable.create(example_value, **options)
    elif isinstance(example_value, torch.backends.cuda.SDPAParams):
        from .sdpa import SDPAParamsVariable

        set_example_value(proxy.node, example_value)
        return SDPAParamsVariable(proxy, **options)
    elif isinstance(example_value, bool) and (
        proxy.node.target
        in [
            torch._C._are_functorch_transforms_active,
            torch.backends.cuda.is_flash_attention_available,
            torch.backends.cuda.can_use_flash_attention,
            torch.backends.cuda.can_use_efficient_attention,
            "is_integer",
        ]
        + list(supported_const_comparison_op_values.keys())
    ):
        set_example_value(proxy.node, example_value)
        return ConstantVariable.create(example_value, **options)
    elif (
        isinstance(example_value, (int, float, bool))
        and proxy.node.target is call_torchbind
    ):
        set_example_value(proxy.node, example_value)
        return ConstantVariable.create(example_value, **options)
    elif isinstance(example_value, float) or proxy.node.target in ["hex", "__round__"]:
        set_example_value(proxy.node, example_value)
        return ConstantVariable.create(example_value, **options)
    else:
        unimplemented(
            "torch.* op returned non-Tensor "
            + f"{typestr(example_value)} {proxy.node.op} {proxy.node.target}",
            case_name="unsupported_operator",
        )


def get_automatic_dynamic_shapes_mark_as():
    if config.automatic_dynamic_shapes_mark_as == "dynamic":
        return DimDynamic.DYNAMIC
    elif config.automatic_dynamic_shapes_mark_as == "unbacked":
        return DimDynamic.SIZE_LIKE_UNBACKED
    elif config.automatic_dynamic_shapes_mark_as == "oblivious":
        return DimDynamic.OBLIVIOUS_SIZE
    else:
        raise ValueError(
            f"invalid automatic_dynamic_shapes_mark_as = {config.automatic_dynamic_shapes_mark_as}"
        )


# Tracks the sources of all fake tensors we wrap in Dynamo.
# Used by shape guard computation.
@dataclasses.dataclass
class TrackedFake:
    fake: Union[FakeTensor, SymInt]
    source: Source
    # Is None when fake is SymInt
    symbolic_context: Optional[SymbolicContext]

    def __hash__(self) -> int:
        return hash((self.fake, self.source.name()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrackedFake):
            return self.fake is other.fake and self.source.name() == other.source.name()
        return False


# Performs automatic dynamic dim determination.
# Returns a SymbolicContext
def _automatic_dynamic(
    e, tx, source, static_shapes, outer_only=False
) -> SymbolicContext:
    # strided NT not supported
    if e.is_nested and not isinstance(
        e, torch.nested._internal.nested_tensor.NestedTensor
    ):
        unimplemented("torch.compile does not support strided NestedTensor")

    name = source.name()
    prior_policy = tx.output.tracing_context.tensor_to_context.get(e, None)
    shape_env_to_source_to_symbol_cache = (
        prior_policy.shape_env_to_source_to_symbol_cache if prior_policy else None
    )

    # Get base context if the tensor is a view
    view_base_context: Optional[SymbolicContext] = None
    if e._is_view():
        base_source = AttrSource(source, "_base")
        view_base_context = _automatic_dynamic(e._base, tx, base_source, static_shapes)

    if is_traceable_wrapper_subclass(e) and not outer_only:
        # Get symbolic context for outer tensor
        outer_context = _automatic_dynamic(
            e, tx, source, static_shapes, outer_only=True
        )

        # Get symbolic contexts for inner tensors
        inner_contexts = {}  # mapping from attr -> symbolic context
        attrs, _ = type(e).__tensor_flatten__(e)
        for attr in attrs:
            inner_tensor = getattr(e, attr)
            inner_source = AttrSource(source, attr)
            inner_contexts[attr] = _automatic_dynamic(
                inner_tensor, tx, inner_source, static_shapes
            )

        return SubclassSymbolicContext(
            dynamic_sizes=outer_context.dynamic_sizes,
            dynamic_strides=outer_context.dynamic_strides,
            constraint_sizes=outer_context.constraint_sizes,
            constraint_strides=outer_context.constraint_strides,
            view_base_context=view_base_context,
            tensor_source=outer_context.tensor_source,
            shape_env_to_source_to_symbol_cache=outer_context.shape_env_to_source_to_symbol_cache,
            inner_contexts=inner_contexts,
        )

    if static_shapes:
        return StatefulSymbolicContext(
            dynamic_sizes=[DimDynamic.STATIC] * e.dim(),
            dynamic_strides=[DimDynamic.INFER_STRIDE] * e.dim(),
            constraint_sizes=[None] * e.dim(),
            constraint_strides=[None] * e.dim(),
            view_base_context=view_base_context,
            tensor_source=source,
            shape_env_to_source_to_symbol_cache=shape_env_to_source_to_symbol_cache,
        )

    # We preserve the dynamism of inputs. For example, when users call
    # make_fx(torch.cond, tracing_mode="symbolic")(*args), inputs have SymInt sizes.
    from torch.fx.experimental.symbolic_shapes import is_nested_int

    if any(isinstance(s, SymInt) and not is_nested_int(s) for s in e.size()):
        return StatefulSymbolicContext(
            dynamic_sizes=[
                DimDynamic.DYNAMIC if isinstance(s, SymInt) else DimDynamic.STATIC
                for s in e.size()
            ],
            dynamic_strides=[DimDynamic.INFER_STRIDE] * e.dim(),
            constraint_sizes=[None] * e.dim(),
            constraint_strides=[None] * e.dim(),
            view_base_context=view_base_context,
            tensor_source=source,
            shape_env_to_source_to_symbol_cache=shape_env_to_source_to_symbol_cache,
        )

    # Prep for automatic dynamic

    # This mimics stride inference algorithm in _create_symbolic_sizes_strides_storage_offset
    ex_size = e.size()
    if not is_sparse_any(e):
        ex_stride = e.stride()
        dim = e.dim()

        stride = [None] * dim
        pending = [(ex_stride[i], -i) for i in range(dim)]
        pending.sort(key=_nested_int_aware_sort)
        candidates = {}
        for i_stride, neg_i in pending:
            i = -neg_i
            stride[i] = candidates.get(i_stride, i_stride)
            candidates.setdefault(i_stride * ex_size[i], InferStride(i))
    else:
        stride = []

    frame_state_entry = process_automatic_dynamic(
        tx, name, FrameStateSizeEntry.make_tensor(tuple(ex_size), tuple(stride))
    )

    # TODO: index export_constraints ahead of time so we don't have to
    # do a linear scan every time here
    t_id = id(e)
    dim2constraint = {}

    def update_dim2constraint(dim, constraint_range, name):
        if dim in dim2constraint:
            from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint

            old_constraint_range, old_name = dim2constraint[dim]
            new_constraint_range = StrictMinMaxConstraint(
                vr=constraint_range.vr & old_constraint_range.vr,
                warn_only=False,
            )
            # It is possible for (non-None) old_name and name to be different
            # but this will only happen the corresponding Dims can be derived equal.
            new_name = old_name or name
            dim2constraint[dim] = new_constraint_range, new_name
        else:
            dim2constraint[dim] = constraint_range, name

    from torch.export.dynamic_shapes import _RelaxedConstraint

    if tx.output.export_constraints:
        for constraint in tx.output.export_constraints:
            if isinstance(constraint, _RelaxedConstraint):
                continue
            if constraint.t_id == t_id:
                update_dim2constraint(
                    constraint.dim, constraint.constraint_range, constraint.name
                )

    dynamic_sizes = []
    dynamic_strides = []
    constraint_sizes = []
    constraint_strides = []
    for i in range(e.dim()):
        # NB: mark dynamic has precedence over static
        marked_unbacked = i in getattr(e, "_dynamo_unbacked_indices", set())
        marked_dynamic = i in getattr(e, "_dynamo_dynamic_indices", set())
        marked_weak_dynamic = i in getattr(e, "_dynamo_weak_dynamic_indices", set())
        marked_static = i in getattr(e, "_dynamo_static_indices", set())

        # Reflect the user directive in the frame_state
        # For dynamic, apply None always
        if marked_dynamic:
            # TODO: This can be batched
            # TODO: Doing this here is kind of sus, maybe better to set this
            # up when we initially created the FrameStateSizeEntry to bong
            # into the mutable state
            log.debug("automatic dynamic %s marked dynamic", name)
            mark_size = [auto_unset] * e.dim()
            mark_size[i] = auto_dynamic
            frame_state_entry |= FrameStateSizeEntry.make_size(size=mark_size)

        # NB: both static and dynamic have precedence over
        automatic_dynamic_size = (
            config.automatic_dynamic_shapes and frame_state_entry.is_size_dynamic(i)
        )
        # NB: previously, if size was dynamic, we wouldn't make its stride
        # dynamic.  But now, because of InferStride concept, we will properly
        # not make stride dynamic even if it's wobbling
        automatic_dynamic_stride = (
            config.automatic_dynamic_shapes and frame_state_entry.is_stride_dynamic(i)
        )

        automatic_dynamic = automatic_dynamic_size or automatic_dynamic_stride

        # We will process constraints first, as they will imply that we
        # have a dynamic dimension
        # Precedence: export constraints > eager constraints
        constraint = dim2constraint.get(i)
        if constraint is None:
            constraint_size = None
            constraint_stride = None
            if marked_dynamic and not config.allow_ignore_mark_dynamic:
                # constraint_stride is deliberaly kept None because no easy way to provide value ranges for mark dynamic
                constraint_stride = None
                if hasattr(e, "_dynamo_dynamic_range"):
                    dim_range = [
                        dr for dr in e._dynamo_dynamic_range if dr.dim == i
                    ].pop()
                    if dim_range.min is None and dim_range.max is None:
                        constraint_size = RelaxedUnspecConstraint(warn_only=False)
                    else:
                        from torch.fx.experimental.symbolic_shapes import (
                            StrictMinMaxConstraint,
                        )

                        constraint_size = StrictMinMaxConstraint(
                            vr=ValueRanges(lower=dim_range.min, upper=dim_range.max),
                            warn_only=False,
                        )
                else:
                    constraint_size = RelaxedUnspecConstraint(warn_only=False)
            elif not marked_static and automatic_dynamic:
                if automatic_dynamic_size:
                    constraint_size = RelaxedUnspecConstraint(warn_only=True)
                if automatic_dynamic_stride:
                    constraint_stride = RelaxedUnspecConstraint(warn_only=True)
            else:
                constraint_size = None
                constraint_stride = None
        else:
            constraint_size, name_ = constraint
            constraint_stride = None
            dim_name = f"{name}.size()[{i}]"
            tx.output.shape_env.source_name_to_debug_name[dim_name] = name_
        constraint_sizes.append(constraint_size)
        constraint_strides.append(constraint_stride)

        if marked_unbacked:
            dynamic_size = DimDynamic.SIZE_LIKE_UNBACKED
        elif (
            constraint_size is not None
            or marked_dynamic
            or marked_weak_dynamic
            or is_nested_int(e.size()[i])
        ):
            # NB: We could assert static_shapes is False here, but it
            # seems better to allow the user to override symbolic_context in this
            # case
            if automatic_dynamic:
                dynamic_size = get_automatic_dynamic_shapes_mark_as()
            else:
                dynamic_size = DimDynamic.DYNAMIC
        elif static_shapes or config.assume_static_by_default or marked_static:
            dynamic_size = DimDynamic.STATIC
        else:
            # TODO: When does this show up?
            dynamic_size = DimDynamic.DUCK

        if constraint_stride is not None:
            dynamic_stride = DimDynamic.DYNAMIC
        else:
            dynamic_stride = DimDynamic.INFER_STRIDE

        dynamic_sizes.append(dynamic_size)
        dynamic_strides.append(dynamic_stride)

    return StatefulSymbolicContext(
        dynamic_sizes=dynamic_sizes,
        dynamic_strides=dynamic_strides,
        constraint_sizes=constraint_sizes,
        constraint_strides=constraint_strides,
        view_base_context=view_base_context,
        tensor_source=source,
        shape_env_to_source_to_symbol_cache=shape_env_to_source_to_symbol_cache,
    )


# See note [Tensor Fakification and Symbol Caching]
def wrap_to_fake_tensor_and_record(
    e, tx, *, source: Optional[Source], is_tensor: bool, parent_context=None
):
    if (
        type(e) in (torch.Tensor, torch.nn.Parameter, FakeTensor)
        or isinstance(e, torch.Tensor)
        or is_traceable_wrapper_subclass(e)
    ):
        assert source is not None
        static_shapes, _reason = tensor_always_has_static_shape(
            e,
            is_tensor,
            tensor_source=source,
        )

        if not parent_context:
            symbolic_context = _automatic_dynamic(e, tx, source, static_shapes)
        else:
            # Parent contexts are passed in when we are recursively creating
            # fake tensors for subclasses. A better design would be not to create a
            # parent/child relationship, but to recursively call _automatic_dynamic
            # as we recursively call wrap_to_fake_tensor_and_record. This runs
            # into bugs around how meta_utils knows and works to create fake tensors
            # with tensor subclasses. Ideally, dynamo would drive both the recursive
            # wrap_to_fake_tensor_and_record and _automatic_dynamic policy creation.
            assert isinstance(source, AttrSource)
            inner_context_name = source.member
            symbolic_context = parent_context.inner_contexts[inner_context_name]

        log.debug(
            "wrap_to_fake %s %s %s %s",
            source.name(),
            tuple(e.shape),
            symbolic_context,
            type(e),
        )
        fake_e = wrap_fake_exception(
            lambda: tx.fake_mode.from_tensor(
                e,
                source=source,
                symbolic_context=symbolic_context,
            )
        )
        if (
            source is not None
            and isinstance(fake_e, FakeTensor)
            and (sym_val := fake_e.item_memo) is not None
        ):
            tx.output.tracked_fakes.append(
                TrackedFake(sym_val, CallMethodItemSource(source), symbolic_context)
            )

        if is_traceable_wrapper_subclass(fake_e):
            attrs, _ = fake_e.__tensor_flatten__()
            for attr in attrs:
                fake_inner = getattr(fake_e, attr)
                inner = getattr(e, attr)
                inner_source = AttrSource(source, attr)
                wrap_to_fake_tensor_and_record(
                    inner,
                    tx,
                    source=inner_source,
                    is_tensor=isinstance(fake_inner, torch.Tensor),
                    parent_context=symbolic_context,
                )

        tx.output.tracing_context.tensor_to_context[e] = symbolic_context
        if is_sparse_any(fake_e):
            # TODO: for TensorGuards, this eventually may need more
            #       fields for the size/stride of any other constituents
            values = fake_e._values() if fake_e.is_sparse else fake_e.values()
            tx.output.input_source_to_sizes_strides[source] = {
                "size": fake_e.size(),
                # TODO: revise this, but for now this stride instead of ()
                #       avoids SegFault with PYTORCH_TEST_WITH_DYNAMO=1
                "stride": (1,) * fake_e.ndim,
                "values_size": values.size(),
                "values_stride": values.stride(),
            }
        else:
            tx.output.input_source_to_sizes_strides[source] = {
                "size": fake_e.size(),
                "stride": fake_e.stride(),
            }

        if (
            is_tensor
            and not (static_shapes and source.is_specialized_nn_module())
            and not is_constant_source(source)
        ):
            tx.output.tracked_fakes.append(
                TrackedFake(fake_e, source, symbolic_context)
            )
            tx.output.tracked_fakes_id_to_source[id(e)].append(source)

        return fake_e
    else:
        return e


class SourcelessBuilder:
    """
    Like builder, but stateless and does not require a source. Useful for simple type->VT objects, or objects
    that are being created/evaporated during inlining (ex: consider a locally made list of tensors we then iterate over
    .), such a list should not show up as an artifact from inputs, nor in reconstruction, nor in the graph. However,
    there may be reasons to represent it as a ListVariable internally.

    NOTE - Objects produced here are born UNGUARDED due to the nature of sources!

    NOTE - This class is very new! It will have some rough edges, but it was created to stem the bleeding of giant
    if/else type->VariableTracker trees that were cropping up all over dynamo.
    """

    def __init__(self) -> None:
        raise AssertionError("Use SourcelessBuilder.create()")

    @staticmethod
    def create(tx: "InstructionTranslator", value) -> VariableTracker:
        value_type = type(value)
        fast_handler = SourcelessBuilder._type_handlers.get(value_type)
        if fast_handler:
            return fast_handler(tx, value)

        if isinstance(value, VariableTracker):
            # This is always valid to call, and useful for recursive calls.
            return value
        elif isinstance(value, dataclasses._HAS_DEFAULT_FACTORY_CLASS):
            return UserDefinedObjectVariable(value)
        elif ConstantVariable.is_literal(value):
            return ConstantVariable.create(value)
        elif callable(value) and trace_rules.lookup_callable(value) is not None:
            if trace_rules.is_callable_allowed(value):
                tx.output.has_user_defined_allowed_in_graph = True
            return trace_rules.lookup_callable(value)(value)
        elif is_function_or_wrapper(value):
            return trace_rules.lookup(value)(value)
        elif isinstance(value, enum.Enum):
            return EnumVariable(value)
        elif isinstance(value, (type, abc.ABCMeta)):
            return UserDefinedClassVariable(value)
        elif isinstance(value, types.MethodWrapperType):
            return MethodWrapperVariable(value)
        elif isinstance(value, torch.fx.graph_module.GraphModule):
            return SourcelessGraphModuleVariable(value)
        elif isinstance(
            value, (torch.utils._pytree.TreeSpec, torch.utils._pytree.LeafSpec)
        ):
            return UserDefinedObjectVariable(value)
        elif PlacementVariable.is_placement(value):
            return PlacementVariable(value)
        elif DeviceMeshVariable.is_device_mesh(value):
            return DeviceMeshVariable(value)
        elif isinstance(value, re.Pattern):
            return RegexPatternVariable(value)
        elif isinstance(value, torch._dynamo.variables.lazy.LazySymNodeFormatString):
            return ConstantVariable.create(str(value))
        unimplemented(
            f"Unexpected type in sourceless builder {value_type.__module__}.{value_type.__qualname__}"
        )

    @staticmethod
    def wrap_constant_literal(value):
        assert ConstantVariable.is_literal(value)
        return ConstantVariable.create(value=value)

    @staticmethod
    def make_type_handlers():
        create = SourcelessBuilder.create
        handlers = {}
        for t in common_constant_types:
            handlers[t] = lambda tx, value: ConstantVariable(value)
        handlers[set] = lambda tx, value: SetVariable(
            [create(tx, x) for x in value], mutation_type=ValueMutationNew()
        )
        handlers[dict] = lambda tx, value: ConstDictVariable(
            {create(tx, k): create(tx, v) for k, v in value.items()},
            type(value),
            mutation_type=ValueMutationNew(),
        )
        handlers[list] = lambda tx, value: ListVariable(
            [create(tx, x) for x in value], mutation_type=ValueMutationNew()
        )
        handlers[tuple] = lambda tx, value: TupleVariable(
            [create(tx, x) for x in value]
        )
        handlers[torch.Size] = lambda tx, value: SizeVariable(
            [create(tx, x) for x in value]
        )
        handlers[collections.OrderedDict] = handlers[dict]
        handlers[immutable_dict] = handlers[dict]
        handlers[immutable_list] = handlers[list]
        handlers[random.Random] = lambda tx, value: RandomClassVariable()
        handlers[types.ModuleType] = lambda tx, value: PythonModuleVariable(value)

        handlers[
            torch.distributions.constraints._Real
        ] = lambda tx, value: UserDefinedObjectVariable(
            value, mutation_type=ValueMutationNew()
        )
        handlers[
            torch.distributions.constraints._Interval
        ] = lambda tx, value: UserDefinedObjectVariable(
            value, mutation_type=ValueMutationNew()
        )
        handlers[
            torch.distributions.constraints.Constraint
        ] = lambda tx, value: UserDefinedObjectVariable(
            value, mutation_type=ValueMutationNew()
        )

        def passthrough(tx: "InstructionTranslator", value):
            return value

        for cls in VariableTrackerMeta.all_subclasses:
            handlers[cls] = passthrough
        return handlers


SourcelessBuilder._type_handlers = SourcelessBuilder.make_type_handlers()


class SourcelessUserDefinedObjectBuilder:
    """
    SourceLessBuilder does not return a UserDefinedObjectVariable, but in some
    cases it might be ok to return UserDefinedObjects. In such case, use this
    builder.
    """

    def __init__(self) -> None:
        raise AssertionError("Use SourcelessUserDefinedObjectBuilder.create()")

    @staticmethod
    def create(tx: "InstructionTranslator", value) -> VariableTracker:
        value_type = type(value)
        if issubclass(value_type, MutableMapping):
            return MutableMappingVariable(value, mutation_type=ValueMutationNew())
        elif isinstance(value, torch.nn.Module):
            return UnspecializedNNModuleVariable(
                value, mutation_type=ValueMutationNew()
            )
        else:
            return UserDefinedObjectVariable(value, mutation_type=ValueMutationNew())
