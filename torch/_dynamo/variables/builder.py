import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import operator
import re
import sys
import types
from typing import List, NamedTuple, Optional, Union

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import torch

from torch import SymInt
from torch._guards import GuardSource, TracingContext
from torch._ops import HigherOrderOperator
from torch._streambase import _EventBase, _StreamBase
from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    DimConstraint,
    DimDynamic,
    RelaxedUnspecConstraint,
    StatefulSymbolicContext,
    SymbolicContext,
)
from torch.fx.immutable_collections import immutable_list
from torch.nested._internal.nested_tensor import NestedTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import TensorWeakRef
from .. import config, mutation_guard, replay_record, skipfiles, trace_rules
from ..allowed_functions import (
    is_allowed,
    is_builtin_callable,
    is_numpy,
    is_user_defined_allowed,
)

from ..device_interface import get_registered_device_interfaces
from ..exc import InternalTorchDynamoError, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..side_effects import SideEffects
from ..source import (
    AttrSource,
    ConstantSource,
    ConvertIntSource,
    GetItemSource,
    GlobalWeakRefSource,
    is_constant_source,
    LocalSource,
    NumpyTensorSource,
    RandomValueSource,
    Source,
    TupleIteratorGetItemSource,
)
from ..utils import (
    build_checkpoint_variable,
    clone_input,
    get_fake_value,
    get_static_address_type,
    global_key_name,
    is_namedtuple,
    is_typing,
    is_utils_checkpoint,
    istype,
    odict_values,
    preserve_rng_state,
    tensor_always_has_static_shape,
    tuple_iterator,
    tuple_iterator_getitem,
    tuple_iterator_len,
    wrap_fake_exception,
)

from .base import MutableLocal, typestr, VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
    AutocastModeVariable,
    EventVariable,
    NullContextVariable,
    StreamVariable,
)
from .dicts import (
    ConstDictVariable,
    DataClassVariable,
    DefaultDictVariable,
    HFPretrainedConfigVariable,
    PythonSysModulesVariable,
    SetVariable,
)
from .distributed import (
    DeviceMeshVariable,
    PlacementClassVariable,
    PlacementVariable,
    ProcessGroupVariable,
)
from .functions import (
    CollectiveFunctionRewriteVariable,
    FunctoolsPartialVariable,
    TritonKernelVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lazy import LazyVariableTracker
from .lists import (
    BaseListVariable,
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
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    ComptimeVariable,
    GetAttrVariable,
    GetSetDescriptorVariable,
    InspectSignatureVariable,
    LambdaVariable,
    MethodWrapperVariable,
    NumpyVariable,
    PythonModuleVariable,
    SavedTensorBox,
    SkipFilesVariable,
    TypingVariable,
)

from .nn_module import FSDPManagedNNModuleVariable, UnspecializedNNModuleVariable
from .optimizer import OptimizerVariable
from .tensor import (
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorSubclassVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .torch import torch_special_class_types, TorchVariable
from .torch_function import build_torch_function_fn, TensorWithTFOverrideVariable
from .user_defined import (
    KeyedJaggedTensorVariable,
    UserDefinedClassVariable,
    UserDefinedObjectVariable,
)


log = logging.getLogger(__name__)


DimList = List


class _missing:
    pass


@dataclasses.dataclass
class GraphArg:
    source: Source
    # TODO: storing a SymInt here but not a FakeTensor is a pretty strange
    # thing to do.  Probably should have example (which stores an int) and
    # fake_example
    _example: Union[TensorWeakRef, torch.SymInt]
    is_unspecialized: bool
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

    def load(self, tx):
        return self.source.reconstruct(tx)

    def erase(self):
        self._example = None
        self.example_strong_ref = None

    def __eq__(self, other):
        return self.source.name() == other.source.name()


@dataclasses.dataclass
class FrameStateSizeEntry:
    scalar: Optional[int]
    size: Optional[List[int]]


class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""

    def __init__(
        self,
        tx,
        source: Source,
    ):
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
        vt = self._wrap(value).clone(**self.options())
        if self._can_lift_attrs_to_inputs(vt):
            vt = self.tx.output.side_effects.track_object_existing(
                self.source, value, vt
            )
        return vt

    def _can_lift_attrs_to_inputs(self, vt):
        if type(vt) in [
            TensorVariable,
            TensorWithTFOverrideVariable,
            UserDefinedObjectVariable,
            NumpyNdarrayVariable,
        ]:
            return True
        return False

    @staticmethod
    @functools.lru_cache(None)
    def _common_constants():
        return {
            # We zero-one specialize shapes, so specialize these constants
            # too
            0,
            1,
            # NB: There used to be more constants here, but honestly it was
            # pretty confusing.  Note we specialize floats by default, and
            # DON'T specialize ints by default.  This all only matters with
            # dynamic_shapes
        }

    def get_source(self):
        return self.source

    def options(self):
        return {"source": self.get_source()}

    def install_guards(self, *guards):
        source = self.get_source()
        if (
            isinstance(source, ConstantSource)
            or source.guard_source() == GuardSource.CONSTANT
        ):
            return None
        install_guard(*[source.make_guard(guard) for guard in guards], skip=1)
        return {}

    @classmethod
    @functools.lru_cache(None)
    def _type_dispatch(cls):
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
            ((tuple, list, odict_values, collections.deque), cls.wrap_listlike),
            (tuple_iterator, cls.wrap_tuple_iterator),
            ((slice, range), cls.wrap_slice_range),
            (
                (
                    int,
                    float,
                    bool,
                    type(None),
                    str,
                    torch.Size,
                    torch.device,
                    torch.dtype,
                ),
                cls.wrap_literal,
            ),
        ]

        if config.trace_numpy and np:
            entries.append((np.ndarray, cls.wrap_numpy_ndarray))

        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, tuple) else (ts,):
                assert t not in result
                result[t] = fn

        return result

    @classmethod
    @functools.lru_cache(None)
    def _id_dispatch(cls):
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
        ]

        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, (tuple, list)) else (ts,):
                assert t not in result
                result[id(t)] = fn

        return result

    def _wrap(self, value):
        # import here to avoid circular dependencies
        from torch.utils._triton import has_triton

        if has_triton():
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction
        else:

            class JITFunction:
                pass

            class Autotuner:
                pass

        # Handle exact type() match
        type_dispatch = self._type_dispatch().get(type(value))
        if type_dispatch is not None:
            return type_dispatch(self, value)

        # Handle exact id() match
        id_dispatch = self._id_dispatch().get(id(value))
        if id_dispatch is not None:
            return id_dispatch(self, value)

        # Note - There are some nested values where types mismatch!
        # We want to get those out and wrap those.
        value = inspect.getattr_static(value, "_torchdynamo_inline", value)

        # Everything else (NB: order matters!)
        if is_traceable_wrapper_subclass(value) or istype(
            value, config.traceable_tensor_subclasses
        ):
            return self.wrap_tensor(value)
        elif is_namedtuple(value):
            return self.wrap_listlike(value)

        elif value is torch.utils._pytree.SUPPORTED_NODES:
            # For SUPPORTED_NODES, we guard on the dictionary version (PEP509)
            # under the assumption that the values themselves don't change.
            self.install_guards(GuardBuilder.DICT_VERSION)
            result = {
                k: UserDefinedObjectVariable(
                    value[k],
                    source=GetItemSource(self.get_source(), k),
                )
                for k in value.keys()
            }
            return ConstDictVariable(result, type(value))
        elif value is sys.modules:
            return PythonSysModulesVariable(source=self.source)
        elif istype(
            value, (dict, collections.defaultdict, collections.OrderedDict)
        ) and all(
            ConstantVariable.is_literal(k)
            or self.tensor_can_be_dict_key(k)
            or isinstance(k, enum.Enum)
            for k in value.keys()
        ):
            if not value and self.get_source().is_nn_module():
                # It is faster to guard on 'false' property than to guard
                # on actual dict keys, but we can't do this fast guard in general because
                # it omits a crucial type check that ensures the value is actually still a dict at runtime.

                # Why is this OK for (specialized) nnmodules? We set up a setattr hook
                # to check for module property mutations, which does a reasonable,
                # but not completely secure job ensuring a property wasn't changed.
                self.install_guards(GuardBuilder.BOOL_FALSE)
            else:
                self.install_guards(GuardBuilder.DICT_KEYS)

            # store key variables in global location for reconstruction
            for key in value.keys():
                if self.tensor_can_be_dict_key(key):
                    self.tx.store_global_weakref(global_key_name(key), key)

            def index_source(key):
                if self.tensor_can_be_dict_key(key):
                    return GlobalWeakRefSource(global_key_name(key))
                else:
                    return key

            result = {
                k: LazyVariableTracker.create(
                    value[k],
                    source=GetItemSource(self.get_source(), index_source(k)),
                )
                for k in value.keys()
            }

            if istype(value, collections.defaultdict):
                result = DefaultDictVariable(
                    result,
                    type(value),
                    self._wrap(value.default_factory),
                )
            else:
                result = ConstDictVariable(result, type(value))

            return self.tx.output.side_effects.track_dict(self.source, value, result)
        elif isinstance(value, torch.nn.Module):
            return self.wrap_module(value)
        elif ConstantVariable.is_literal(value):  # non-atomic literals
            return self.wrap_literal(value)
        elif istype(value, frozenset) and (
            all(is_allowed(x) or ConstantVariable.is_literal(x) for x in value)
        ):
            # For frozenset, we can guard by object ID instead of value
            # equality, this allows us to handle non-literal values
            self.install_guards(GuardBuilder.ID_MATCH)
            return ConstantVariable.create(value=value, source=self.source)
        elif isinstance(value, enum.Enum):
            self.install_guards(GuardBuilder.ID_MATCH)
            return EnumVariable(value=value, source=self.source)
        elif is_builtin_callable(value):
            self.install_guards(GuardBuilder.BUILTIN_MATCH)
            return BuiltinVariable(value, source=self.source)
        elif is_utils_checkpoint(value):
            return build_checkpoint_variable(source=self.source)
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
                keywords[k] = VariableBuilder(
                    self.tx, GetItemSource(keywords_source, k)
                )(v)

            install_guard(
                self.get_source().make_guard(GuardBuilder.TYPE_MATCH),
                keywords_source.make_guard(GuardBuilder.DICT_KEYS),
                args_source.make_guard(GuardBuilder.LIST_LENGTH),
            )
            return FunctoolsPartialVariable(func_obj, args, keywords, original=value)
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
        elif is_numpy(value):
            assert np
            self.install_guards(
                GuardBuilder.FUNCTION_MATCH
                if callable(value)
                else GuardBuilder.TYPE_MATCH
            )
            return NumpyVariable(value, source=self.source)
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
            saved_tensors_source = AttrSource(self.source, "saved_tensors")
            install_guard(
                self.source.make_guard(GuardBuilder.TYPE_MATCH),
                saved_tensors_source.make_guard(GuardBuilder.LIST_LENGTH),
            )
            saved_tensors = [
                VariableBuilder(self.tx, GetItemSource(saved_tensors_source, n))(v)
                for n, v in enumerate(value.saved_tensors)
            ]
            return self.tx.output.side_effects.track_object_existing(
                self.source,
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
                AutogradFunctionVariable(value.__self__, source=self.source),
                "apply",
            )
        elif np and isinstance(value, np.number):
            return self.wrap_unspecialized_primitive(value)
        elif DataClassVariable.is_matching_object(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return DataClassVariable.wrap(self, value)
        elif HFPretrainedConfigVariable.is_matching_object(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return HFPretrainedConfigVariable(value)
        elif isinstance(value, HigherOrderOperator):
            self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.NAME_MATCH)
            return TorchHigherOrderOperatorVariable.make(value, source=self.source)
        elif type(value).__name__ == "builtin_function_or_method" and isinstance(
            value.__self__, torch_special_class_types
        ):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return TorchVariable(
                value,
            )
        elif isinstance(value, _StreamBase):
            self.install_guards(GuardBuilder.ID_MATCH)
            return StreamVariable(
                None,
                value,
                value.device.type,
                source=self.source,
            )
        elif isinstance(value, _EventBase):
            self.install_guards(GuardBuilder.ID_MATCH)
            return EventVariable(
                None,
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
            return self.tx.output.side_effects.track_object_existing(
                self.source, value, result
            )
        elif isinstance(value, torch.optim.Optimizer):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return OptimizerVariable(value, source=self.source)
        elif ProcessGroupVariable.is_process_group(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return ProcessGroupVariable(value, source=self.source)
        elif DeviceMeshVariable.is_device_mesh(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.ID_MATCH)
            return DeviceMeshVariable(value, source=self.source)
        elif PlacementClassVariable.is_placement_type(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.ID_MATCH)
            return PlacementClassVariable(value, source=self.source)
        elif PlacementVariable.is_placement(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.ID_MATCH)
            return PlacementVariable(
                value,
                source=self.source,
            )
        elif isinstance(value, torch.SymBool):
            # Note: the idea here is to re-use the infra we've built for SymInt by simulating the
            # user provided SymBool with a SymInt in dynamo.

            # Concretely,
            # 1. We create a SymInt in dynamo's shape_env, whose source is constructed as ConvertIntSource(self.source).
            # so that guards on the SymInts can be effectively applied on the original SymBool in user program.
            # 2. We create a SymBool based on the SymInt in dynamo's ShapeEnv. Because the original user program
            # depends on the value being a SymBool. This allows dynamo to interpret the user's program correctly.

            value_hint = value.node.require_hint()
            new_source = ConvertIntSource(self.source)

            new_symint = self.tx.output.shape_env.create_unspecified_symint_and_symbol(
                int(value_hint),
                new_source,
                dynamic_dim=DimDynamic.DYNAMIC,
            )

            sym_node_proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(new_symint),
                source=new_source,
            )

            sym_node_proxy.node.meta["grapharg"] = GraphArg(
                new_source,
                new_symint,
                False,
                None,
                is_tensor=False,
                example_strong_ref=new_symint,
            )
            self.tx.output.bound_symbols.add(new_symint.node.expr)
            self.tx.output.tracked_fakes.append(
                TrackedFake(new_symint, new_source, None)
            )
            return SymNodeVariable(
                sym_node_proxy,
                new_symint == 1,
            )
        elif isinstance(value, (JITFunction, Autotuner)):
            self.install_guards(GuardBuilder.ID_MATCH)
            return TritonKernelVariable(
                value,
                None,  # No kernel idx provided
                None,  # No grid provided
                source=self.source,
            )
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
        elif trace_rules.lookup(value) is not None:
            if is_user_defined_allowed(value):
                self.tx.output.has_user_defined_allowed_in_graph = True
            return trace_rules.lookup(value).create_with_source(
                value, source=self.source
            )
        elif is_allowed(value):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return TorchVariable(
                value,
                source=self.source,
            )
        elif (
            istype(value, (type, types.FunctionType))
            and skipfiles.check(value, is_inlined_call=True)
            and not inspect.getattr_static(value, "_torchdynamo_inline", False)
            and not inspect.getattr_static(value, "__script_if_tracing_wrapper", False)
        ):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return SkipFilesVariable(
                value,
                skipfiles.check_verbose(value, is_inlined_call=True).reason,
                source=self.source,
            )
        elif istype(value, (types.FunctionType, torch.jit.ScriptFunction)):
            self.install_guards(GuardBuilder.CLOSURE_MATCH)
            return UserFunctionVariable(
                value,
                source=self.source,
            )
        elif isinstance(value, types.MethodType) and isinstance(
            value.__self__, torch.nn.Module
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
        elif istype(value, (types.ModuleType, replay_record.DummyModule)):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return PythonModuleVariable(
                value,
                source=self.source,
            )
        elif isinstance(value, types.GetSetDescriptorType):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return GetSetDescriptorVariable(value)
        elif isinstance(value, types.MethodWrapperType):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return MethodWrapperVariable(value, source=self.source)
        elif issubclass(type(value), type):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return UserDefinedClassVariable(
                value,
                source=self.source,
            )
        elif RestrictedListSubclassVariable.is_matching_cls(type(value)):
            self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.LIST_LENGTH)
            return self.tx.output.side_effects.track_list(
                self.source,
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
                ),
            )
        else:
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = UserDefinedObjectVariable(value, source=self.source)
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(
                self.source, value, result
            )

    def tensor_can_be_dict_key(self, value):
        # only allow Parameter and another specific Tensor can be used as dict key
        return (
            isinstance(value, torch.nn.Parameter)
            or isinstance(self.source, AttrSource)
            and self.source.member == "state"
            and isinstance(self.source.base, LocalSource)
        )

    def tensor_should_specialize(self):
        return (
            self.source
            and isinstance(self.source, GetItemSource)
            and isinstance(self.source.base, GetItemSource)
            and self.source.base.index == "params"
            and isinstance(self.source.base.base, GetItemSource)
            and isinstance(self.source.base.base.base, AttrSource)
            and self.source.base.base.base.member == "param_groups"
            and isinstance(self.source.base.base.base.base, LocalSource)
            and (
                isinstance(
                    self.tx.f_locals[self.source.base.base.base.base.local_name],
                    torch.optim.Optimizer,
                )
                if self.source.base.base.base.base.local_name in self.tx.f_locals.keys()
                else True
            )
        )

    def wrap_listlike(self, value: Union[tuple, list, odict_values, NamedTuple]):
        # One can index a tensor with a list/tuple. Therefore, we need to
        # have a stricter match.
        self.install_guards(GuardBuilder.LIST_LENGTH)

        for item in value:
            if item is value:
                unimplemented("list elements are pointing to the list itself")

        output = [
            VariableBuilder(self.tx, GetItemSource(self.get_source(), i))(item)
            for i, item in enumerate(value)
        ]
        result = BaseListVariable.cls_for_instance(value)(
            output, mutable_local=MutableLocal()
        )
        if istype(value, list):
            return self.tx.output.side_effects.track_list(self.source, value, result)
        return result

    def wrap_tuple_iterator(self, value: tuple_iterator):
        self.install_guards(GuardBuilder.TUPLE_ITERATOR_LEN)
        output = [
            VariableBuilder(self.tx, TupleIteratorGetItemSource(self.get_source(), i))(
                tuple_iterator_getitem(value, i)
            )
            for i in range(tuple_iterator_len(value))
        ]
        return TupleIteratorVariable(output, mutable_local=MutableLocal())

    def wrap_slice_range(self, value: Union[slice, range]):
        items = [
            VariableBuilder(self.tx, AttrSource(self.get_source(), k))(
                getattr(value, k)
            )
            for k in ("start", "stop", "step")
        ]
        self.install_guards(GuardBuilder.TYPE_MATCH)
        if isinstance(value, slice):
            return SliceVariable(items)
        else:
            return RangeVariable(items)

    def wrap_module(self, value: torch.nn.Module):
        from ..eval_frame import OptimizedModule

        if istype(value, OptimizedModule):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.source = AttrSource(self.source, "_orig_mod")
            return self.wrap_module(value._orig_mod)

        if (
            isinstance(value, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM))
            and not config.allow_rnn
        ):
            unimplemented("TorchDynamo purposely graph breaks on RNN, GRU, LSTMs")
        if mutation_guard.is_dynamic_nn_module(value):
            # created dynamically, don't specialize on it
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = UnspecializedNNModuleVariable(value)
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(
                self.source, value, result
            )
        elif issubclass(
            value.__class__, torch.nn.parallel.distributed.DistributedDataParallel
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return UnspecializedNNModuleVariable(value)
        elif getattr(value, "_is_fsdp_managed_module", False):
            # See note [Dynamo treats FSDP wrapped modules as UnspecializedNNModule]
            # in fully_sharded_data_parallel.py for more information

            # we can't do this assert inside FSDP constructor,
            # since we don't know yet whether dynamo will be used
            assert getattr(
                value, "_fsdp_use_orig_params", False
            ), "Dynamo only supports FSDP with use_orig_params=True"

            # Note on FSDP guarding
            # 1. We expect FSDP wrapping mutates an nn module irreversably (no way to de-wrap).
            # 2. Eager FSDP already assumes (requires, but without enforcement) that users don't mutate their
            #    model parameters/structure after FSDP wrapping, because FSDP wouldn't notice or update its FlatParams.
            #
            # Due to (1), once we enter this path we expect not to go back nor have to guard on type
            # or _is_fsdp_managed_module.
            #
            # TODO(whc) We could add a guard on the opposite case, where a user compiled/ran
            # pre-FSDP-wrapped model, then wrapped, to ensure that we recompile with the FSDP handling.
            #
            # Due to (2), we skip guards on inner contents of fsdp_managed modules, by using FSDPNNModuleSource as the
            # guard source.  This behavior is gated on config.skip_fsdp_guards.
            #
            # ID_MATCH is required to disambiguate cases as simple as a unit test that constructs 2 models and wraps
            # them differently with different FSDP configs.  (test_dynamo_distributed.py -k test_fsdp_aot_eager)
            self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.ID_MATCH)
            return FSDPManagedNNModuleVariable(value, source=self.get_source())
        else:
            return self.tx.output.register_attr_or_module(
                value,
                self.name,
                source=self.get_source(),
                # Guards are added inside register_attr_or_module
            )

    def wrap_literal(self, value):
        unspec = not config.specialize_int
        if unspec and type(value) is torch.Size:
            self.install_guards(GuardBuilder.LIST_LENGTH)
            return SizeVariable(
                [
                    VariableBuilder(self.tx, GetItemSource(self.get_source(), i))(v)
                    for i, v in enumerate(value)
                ]
            )
        elif unspec and type(value) is int:
            # unspecializing int by default, but still
            # specialize for the following conditions
            if not TracingContext.get().force_unspec_int_unbacked_size_like and (
                value in self._common_constants()
                # Assume integers from global variables want to be specialized
                or not self.source.guard_source().is_local()
                # Assume that integers that came from NN modules want to be
                # specialized (as we don't expect users to be changing the
                # NN modules on the fly)
                or self.source.guard_source().is_nn_module()
            ):
                self.install_guards(GuardBuilder.CONSTANT_MATCH)
                return ConstantVariable.create(value=value)
            else:
                return self.wrap_unspecialized_primitive(value)
        else:
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return ConstantVariable.create(value=value)

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

        if (
            source.guard_source().is_nn_module()
            or get_static_address_type(value) is not None
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

        # By this point, we should have deduplicated all tensors
        self.assert_not_wrapped_by_this_graph(value)

        # tx.output has multiple tracers if we're introspecting HigherOrderOperator.
        # When we've discovered an untracked tensor, then we actually need
        # to get Dynamo to track the tensor (which is what this function does)
        # and put it as a graph input on the root tracer. Later on,
        # if the input is actually used in the body of the HigherOrderOperator,
        # then the relevant SubgraphTracer will lift it to being an input of
        # the subgraph.
        # See NOTE [HigherOrderOperator tracing design] for more details.

        tensor_proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name), type(value), source=source
        )
        options = {}
        if type(value) in config.traceable_tensor_subclasses:
            options["torch_function_fn"] = build_torch_function_fn(
                self.tx, value, self.source
            )
            self.install_guards(GuardBuilder.TYPE_MATCH)

        if (
            isinstance(value, torch.Tensor)
            and value.is_nested
            and not isinstance(value, NestedTensor)
        ):
            unimplemented("torch.compile does not support strided NestedTensor")

        tensor_variable = wrap_fx_proxy(
            tx=self.tx,
            proxy=tensor_proxy,
            example_value=value,
            should_specialize=self.tensor_should_specialize(),
            subclass_type=subclass_type,
            source=source,
            **options,
        )

        self.install_guards(
            functools.partial(
                GuardBuilder.TENSOR_MATCH,
                value=value
                if isinstance(source, NumpyTensorSource)
                else TensorWeakRef(value),
            )
        )

        self.tx.output.input_source_to_var[source] = tensor_variable
        assert "tensor_dict" not in tensor_proxy.node.meta
        tensor_proxy.node.meta["tensor_dict"] = value.__dict__.copy()

        # Note: this information is conveyed via subclass_type now
        fake_tensor_value = tensor_variable.proxy.node.meta["example_value"]
        if maybe_get_fake_mode(fake_tensor_value) is not self.tx.fake_mode:
            raise InternalTorchDynamoError("Wrapped Tensor must be this graph's fake")

        grapharg = GraphArg(source, value, False, fake_tensor_value)
        tensor_proxy.node.meta["grapharg"] = grapharg
        self.tx.output.add_symbol_bindings(grapharg)
        return tensor_variable

    def wrap_numpy_ndarray(self, value):
        assert np is not None
        assert isinstance(value, np.ndarray)

        source = NumpyTensorSource(self.get_source())

        from torch._numpy import _util

        readonly = not value.flags.writeable
        if readonly:
            value.flags.writeable = True

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
        VariableBuilder(self.tx, source)(tensor_value).recursive_realize()
        proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name), type(tensor_value), source=source
        )
        options = {"source": source}
        numpy_ndarray_variable = wrap_fx_proxy_cls(
            target_cls=NumpyNdarrayVariable,
            tx=self.tx,
            proxy=proxy,
            example_value=tensor_value,
            **options,
        )

        self.tx.output.input_source_to_var[source] = numpy_ndarray_variable
        example_value = numpy_ndarray_variable.proxy.node.meta["example_value"]

        # is_unspecialized should be true because we are wrapping a np.ndarray as argument input, and it needs to be
        # converted to a tensor.
        grapharg = GraphArg(
            source,
            tensor_value,
            is_unspecialized=True,
            fake_tensor=example_value,
            is_tensor=True,
            example_strong_ref=tensor_value,
        )
        proxy.node.meta["grapharg"] = grapharg

        return numpy_ndarray_variable

    def wrap_unspecialized_primitive(self, value):
        if self.name in self.tx.output.unspec_variable_map:
            return self.tx.output.unspec_variable_map[self.name]
        else:
            shape_env = self.tx.output.shape_env
            if TracingContext.get().force_unspec_int_unbacked_size_like and isinstance(
                value, int
            ):
                wrapped_value = shape_env.create_unbacked_symint()
                _constrain_range_for_size(wrapped_value)
                self.tx.output.bound_symbols.add(wrapped_value.node.expr)
                self.tx.output.tracked_fakes.append(
                    TrackedFake(wrapped_value, self.source, None)
                )

            # NB: We do not do float.  For motivation, see
            # https://docs.google.com/document/d/1INSCdYu1PxXcr43HrD82OudeEuS-qxQe1yZmLg2wy6A/edit
            # but the general idea is that we generate kernels that can
            # take unspecialized floats and use them in sizevar computation
            elif (
                isinstance(value, int)
                and not is_constant_source(self.get_source())
                and not isinstance(self.get_source(), RandomValueSource)
            ):
                if torch._dynamo.config.specialize_int:
                    # If specialize_int is False, also return
                    # a constant (but this should have been handled
                    # in the caller, TBH)
                    self.install_guards(GuardBuilder.CONSTANT_MATCH)
                    return ConstantVariable.create(value=value)

                name = self.source.name()
                if name not in self.tx.output.frame_state:
                    # Note - this essentially means that if this name gets reused as a tensor,
                    # it will start fully dynamic. That should always be a safe option, and not awfully inefficient.
                    # Alternatively, if we want to improve pef here, we can add a third state of unset, but I am not
                    # sure that is necessary for now.
                    frame_state_entry = FrameStateSizeEntry(scalar=value, size=None)
                else:
                    frame_state_entry = self.tx.output.frame_state[name]
                    if frame_state_entry.scalar != value:
                        log.debug(
                            "automatic dynamic int %s val %s != %s",
                            name,
                            value,
                            frame_state_entry.scalar,
                        )
                        frame_state_entry.scalar = None
                self.tx.output.frame_state[name] = frame_state_entry

                # TODO: This should be dynamic, as we in general do not
                # know if bare integers are actually going to be sizevars
                # and it is inappropriate to eagerly duck size them with
                # real sizevars
                if (
                    config.automatic_dynamic_shapes and frame_state_entry.scalar is None
                ) or not config.assume_static_by_default:
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
                self.tx.output.bound_symbols.add(wrapped_value.node.expr)

                self.tx.output.tracked_fakes.append(
                    TrackedFake(wrapped_value, self.source, None)
                )
            else:
                wrapped_value = torch.tensor(value)
            if not isinstance(self.get_source(), RandomValueSource):
                install_guard(self.get_source().make_guard(GuardBuilder.TYPE_MATCH))
            options = {"source": self.get_source()}
            if isinstance(wrapped_value, torch.Tensor):
                options.update({"raw_value": value})

            proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(wrapped_value),
                source=self.get_source(),
            )

            unspec_var = wrap_fx_proxy_cls(
                UnspecializedPythonVariable,
                tx=self.tx,
                proxy=proxy,
                example_value=wrapped_value,
                **options,
            )
            self.tx.output.unspec_variable_map[self.name] = unspec_var
            if not is_constant_source(self.get_source()):
                if self.tx.export and not isinstance(self.get_source(), LocalSource):
                    raise AssertionError(
                        "Dynamo attempts to add additional input during export: value={}, source={}".format(
                            wrapped_value, self.get_source()
                        )
                    )
                fake_tensor_value = None
                if isinstance(unspec_var, ConstantVariable):
                    example_value = unspec_var.value
                else:
                    example_value = unspec_var.proxy.node.meta["example_value"]
                if is_fake(example_value):
                    fake_tensor_value = example_value
                    assert fake_tensor_value.fake_mode is self.tx.fake_mode, (
                        f"fake mode ({fake_tensor_value.fake_mode}) from fake tensor metadata doesn't match mode"
                        "({self.tx.fake_mode}) from InstructionTranslator"
                    )

                proxy.node.meta["grapharg"] = GraphArg(
                    self.get_source(),
                    wrapped_value,
                    isinstance(wrapped_value, torch.Tensor),
                    fake_tensor_value,
                    is_tensor=False,
                    example_strong_ref=wrapped_value,
                )
            return unspec_var


def _dataclasses_fields_lambda(obj):
    if isinstance(obj, UserDefinedObjectVariable):
        value = obj.value
    elif isinstance(obj, DataClassVariable):
        value = obj.user_cls
    else:
        unimplemented(f"Dataclass fields handling fails for type {obj}")
    items = []
    for field in dataclasses.fields(value):
        source = None
        if obj.source:
            source = GetItemSource(
                AttrSource(obj.source, "__dataclass_fields__"), field.name
            )
        items.append(UserDefinedObjectVariable(field, source=source))
    return TupleVariable(items)


def wrap_fx_proxy(tx, proxy, example_value=None, subclass_type=None, **options):
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
# output cases.  What gives?  Well, we sometimes trace operations into the
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
    from ..symbolic_convert import InstructionTranslatorBase

    assert isinstance(tx, InstructionTranslatorBase)
    if "guards" in options and options["guards"] is not None:
        tx.output.guards.update(options["guards"])

    assert "example_value" not in proxy.node.meta, f"{proxy.node.meta['example_value']}"

    initial_example_value = example_value

    def _clone_input(value):
        if isinstance(value, torch.Tensor):
            # tensor subclasses will not be converted to FakeTensors and need to be cloned
            if not (
                isinstance(value, FakeTensor)
                or (
                    # Is functional tensor fakeified by this instance of Dynamo
                    torch._is_functional_tensor(value)
                    and maybe_get_fake_mode(value) is tx.fake_mode
                )
                or value.is_nested
            ):
                # NB: ensure strides are preserved
                value = clone_input(value)

        return value

    with preserve_rng_state():
        if example_value is None:
            # only allow_non_graph_fake in this instance because we handle the non-fake
            # cases properly below.
            example_value = get_fake_value(proxy.node, tx, allow_non_graph_fake=True)

        # Handle recursive calls here
        elif maybe_get_fake_mode(example_value) is tx.fake_mode:
            pass

        elif isinstance(example_value, torch.Tensor):
            if tx.export:
                # The legacy behavior for real value cache with subclasses was
                # to perform a clone WITHOUT preserving the subclass.  It's
                # not entirely clear this is what you actually want though.
                with torch._C.DisableTorchFunctionSubclass():
                    proxy.tracer.real_value_cache[proxy.node] = _clone_input(
                        example_value
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
            example_value = wrap_to_fake_tensor_and_record(
                example_value, tx=tx, **kwargs
            )
        if isinstance(example_value, torch.Tensor) and (
            maybe_get_fake_mode(example_value) is not tx.fake_mode
        ):
            raise InternalTorchDynamoError(
                "`example_value` needs to be a `FakeTensor`"
                f"wrapped by this instance of Dynamo. Found: {example_value}"
            )

    if isinstance(example_value, torch.Tensor):
        is_parameter = isinstance(example_value, torch.nn.Parameter)
        should_specialize = options.pop("should_specialize", False)
        if is_parameter or should_specialize:
            specialized_value = initial_example_value
        else:
            specialized_value = None

        # NB: In most (all?) cases, this does not actually do a clone.
        # (WARNING: this means that if we mutate metadata on the fake
        # tensor, the stored example value will update too!)
        example_value = _clone_input(example_value)
        proxy.node.meta["example_value"] = example_value
        specialized_props = target_cls.specialize(example_value)
        # TODO: not sure about this fake mode test
        if (
            isinstance(example_value, torch._subclasses.fake_tensor.FakeTensor)
            and example_value.fake_mode is tx.fake_mode
        ):
            tensor_type = subclass_type if subclass_type else torch.Tensor
            specialized_props["class_type"] = (
                torch.nn.Parameter if is_parameter else tensor_type
            )

        specialized_props["specialized_value"] = specialized_value

        options.update(specialized_props)
        return target_cls(proxy, **options)
    elif (
        hasattr(proxy.node.target, "__name__")
        and proxy.node.target.__name__ == "set_state"
        and isinstance(proxy.node.target.__self__, torch._C.Generator)
        or proxy.node.target == torch.random.set_rng_state
    ):
        return TorchVariable(proxy.node.target)
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
    elif isinstance(example_value, (tuple, list, set)):
        proxy.node.meta["example_value"] = example_value
        unpacked = []
        for i, val in enumerate(example_value):
            if val is None:
                # nn.MultiheadAttention() can return None, see issue #175
                unpacked.append(
                    ConstantVariable.create(None, **options),
                )
            else:
                unpacked.append(
                    wrap_fx_proxy_cls(
                        target_cls,
                        tx,
                        proxy.tracer.create_proxy(
                            "call_function", operator.getitem, (proxy, i), {}
                        ),
                        example_value=val,
                        **options,
                    )
                )
        if isinstance(example_value, torch.Size):
            # NB: Keep the old proxy around.  See SizeVariable for an
            # explanation why
            return SizeVariable(unpacked, proxy, **options)
        elif istype(example_value, tuple):
            return TupleVariable(unpacked, **options)
        elif istype(example_value, (list, immutable_list)):
            return ListVariable(unpacked, mutable_local=MutableLocal(), **options)
        elif istype(example_value, set):
            return SetVariable(unpacked, mutable_local=MutableLocal(), **options)
        else:
            assert example_value.__class__.__module__ == "torch.return_types" or hasattr(
                example_value, "_fields"
            ), f"expected {example_value.__class__.__module__} == torch.return_types or named tuple but got {type(example_value)}"
            return NamedTupleVariable(unpacked, example_value.__class__, **options)
    elif example_value is None or proxy.node.target is torch.manual_seed:
        return ConstantVariable.create(None, **options)
    elif isinstance(example_value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        proxy.node.meta["example_value"] = example_value
        return SymNodeVariable(proxy, example_value, **options)
    elif (
        inspect.isclass(proxy.node.target)
        and issubclass(proxy.node.target, _StreamBase)
    ) or proxy.node.target in [
        device_interface.current_stream
        for _, device_interface in get_registered_device_interfaces()
    ]:
        proxy.node.meta["example_value"] = example_value
        return StreamVariable(
            proxy, example_value, example_value.device.type, **options
        )
    elif (
        inspect.isclass(proxy.node.target) and issubclass(proxy.node.target, _EventBase)
    ) or proxy.node.target in [
        device_interface.Event
        for _, device_interface in get_registered_device_interfaces()
    ]:
        proxy.node.meta["example_value"] = example_value
        return EventVariable(proxy, example_value, **options)
    elif proxy.node.target == "query" and proxy.node.op == "call_method":
        proxy.node.meta["example_value"] = example_value
        return ConstantVariable(example_value, **options)
    elif (
        example_value is not None
        and isinstance(example_value, _EventBase)
        and proxy.node.target == "record_event"
        and proxy.node.op == "call_method"
    ):
        proxy.node.meta["example_value"] = example_value
        return EventVariable(proxy, example_value, **options)
    elif isinstance(example_value, int) and proxy.node.target in [
        torch.sym_int,
        getattr,
        operator.getitem,
        torch._utils._element_size,
        torch.seed,
        operator.mod,
        # some mac builds are missing torch.distributed.get_rank()
        getattr(torch.distributed, "get_rank", _missing),
        getattr(torch.distributed, "get_world_size", _missing),
        # This always wants to be in the graph, even if the constraint
        # results in a constant int
        torch._constrain_as_value,
        torch._constrain_as_size,
    ]:
        proxy.node.meta["example_value"] = example_value
        return ConstantVariable.create(example_value, **options)
    else:
        unimplemented(
            "torch.* op returned non-Tensor "
            + f"{typestr(example_value)} {proxy.node.op} {proxy.node.target}"
        )


# Tracks the sources of all fake tensors we wrap in Dynamo.
# Used by shape guard computation.
@dataclasses.dataclass
class TrackedFake:
    fake: Union[FakeTensor, SymInt]
    source: Source
    # Is None when fake is SymInt
    constraint_dims: Optional[DimList[DimConstraint]]

    def __hash__(self) -> int:
        return hash((self.fake, self.source.name()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrackedFake):
            return self.fake is other.fake and self.source.name() == other.source.name()
        return False


# Performs automatic dynamic dim determination.
# Returns a SymbolicContext
def _automatic_dynamic(e, tx, source, static_shapes) -> SymbolicContext:
    name = source.name()
    prior_policy = tx.output.tracing_context.tensor_to_context.get(e, None)
    source_to_symint_node_cache = (
        prior_policy.source_to_symint_node_cache if prior_policy else None
    )

    if static_shapes:
        return StatefulSymbolicContext(
            dynamic_sizes=[DimDynamic.STATIC] * e.dim(),
            constraint_sizes=[None] * e.dim(),
            tensor_source=source,
            source_to_symint_node_cache=source_to_symint_node_cache,
        )

    # We preserve the dynamism of inputs. For example, when users call
    # make_fx(torch.cond, tracing_mode="symbolic")(*args), inputs have SymInt sizes.
    if any(isinstance(s, SymInt) for s in e.size()):
        return StatefulSymbolicContext(
            dynamic_sizes=[
                DimDynamic.DYNAMIC if isinstance(s, SymInt) else DimDynamic.STATIC
                for s in e.size()
            ],
            constraint_sizes=[None] * e.dim(),
            tensor_source=source,
            source_to_symint_node_cache=source_to_symint_node_cache,
        )

    # Prep for automatic dynamic
    frame_state_entry = None
    if name not in tx.output.frame_state:
        # If there is no entry for this source, add the tensor to frame state with its current static size.
        # E.g., {} -> {"x": [2, 4]}
        frame_state_entry = FrameStateSizeEntry(None, None)
        frame_state_entry.size = list(e.size())
    else:
        frame_state_entry = tx.output.frame_state[name]
        if frame_state_entry.size is not None:
            if e.ndim != len(frame_state_entry.size):
                # If there is already an entry, and the dim mismatches, replace the frame state entry with None.
                # E.g. {"x": [2, 3, 4]} -> {"x": None}
                log.debug(
                    "automatic dynamic %s dim %s != %s",
                    name,
                    e.ndim,
                    frame_state_entry.size,
                )
                frame_state_entry.size = None
            else:
                # If there is already an entry, and the dim matches, for every size in the frame state which
                # disagrees with the current static size, replace it with None. E.g., {"x": [2, 3]} -> {"x": [2, None]}
                for i, dim in enumerate(frame_state_entry.size):
                    if dim is not None and e.size()[i] != dim:
                        log.debug(
                            "automatic dynamic %s size(%s) %s != %s",
                            name,
                            i,
                            e.size(i),
                            dim,
                        )
                        frame_state_entry.size[i] = None

    # TODO: index export_constraints ahead of time so we don't have to
    # do a linear scan every time here
    t_id = id(e)
    dim2constraint = {}

    def update_dim2constraint(dim, constraint_range, debug_name):
        if dim in dim2constraint:
            from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint

            old_constraint_range, old_debug_name = dim2constraint[dim]
            new_constraint_range = StrictMinMaxConstraint(
                vr=constraint_range.vr & old_constraint_range.vr,
                warn_only=False,
            )
            if old_debug_name is not None:
                assert debug_name is None or debug_name == old_debug_name
                new_debug_name = old_debug_name
            else:
                new_debug_name = debug_name
            dim2constraint[dim] = new_constraint_range, new_debug_name
        else:
            dim2constraint[dim] = constraint_range, debug_name

    if tx.output.export_constraints:
        for constraint in tx.output.export_constraints:
            if constraint.t_id == t_id:
                update_dim2constraint(
                    constraint.dim, constraint.constraint_range, constraint.debug_name
                )
            if constraint.shared is not None and constraint.shared.t_id == t_id:
                # We process constraint ranges for each shared dimension separately
                # so that we can directly check range constraint violations on them
                # without looking up which other shared dimensions have this info.
                # In other words, for this t_id, we will have processed all of its
                # constraint ranges, no matter where / how they were specified, by
                # by the end of this loop.
                update_dim2constraint(
                    constraint.shared.dim,
                    constraint.constraint_range,
                    constraint.debug_name,
                )

    dynamic_dims = []
    constraint_dims = []
    for i in range(e.dim()):
        # NB: mark dynamic has precedence over static
        marked_dynamic = i in getattr(e, "_dynamo_dynamic_indices", set())
        marked_weak_dynamic = i in getattr(e, "_dynamo_weak_dynamic_indices", set())
        marked_static = i in getattr(e, "_dynamo_static_indices", set())

        # NB: both static and dynamic have precedence over
        automatic_dynamic = config.automatic_dynamic_shapes and (
            frame_state_entry.size is None or frame_state_entry.size[i] is None
        )

        # Reflect the user directive in the frame_state
        # For dynamic, apply None always
        if frame_state_entry.size and marked_dynamic:
            log.debug("automatic dynamic %s marked dynamic", name)
            frame_state_entry.size[i] = None

        # We will process constraints first, as they will imply that we
        # have a dynamic dimension
        # Precedence: export constraints > eager constraints
        constraint = dim2constraint.get(i)
        if constraint is None:
            if marked_dynamic and not config.allow_ignore_mark_dynamic:
                constraint_dim = RelaxedUnspecConstraint(warn_only=False)
            elif not marked_static and automatic_dynamic:
                constraint_dim = RelaxedUnspecConstraint(warn_only=True)
            else:
                constraint_dim = None
        else:
            constraint_dim, debug_name = constraint
            if debug_name is not None:
                dim_name = f"{name}.size()[{i}]"
                tx.output.shape_env.source_name_to_debug_name[dim_name] = debug_name
        constraint_dims.append(constraint_dim)

        # Now, figure out if the dim is dynamic/duck/static
        if constraint_dim is not None or marked_dynamic or marked_weak_dynamic:
            # NB: We could assert static_shapes is False here, but it
            # seems better to allow the user to override symbolic_context in this
            # case
            dynamic = DimDynamic.DYNAMIC
        elif static_shapes or config.assume_static_by_default or marked_static:
            dynamic = DimDynamic.STATIC
        else:
            dynamic = DimDynamic.DUCK

        dynamic_dims.append(dynamic)

    tx.output.frame_state[name] = frame_state_entry

    return StatefulSymbolicContext(
        dynamic_sizes=dynamic_dims,
        constraint_sizes=constraint_dims,
        tensor_source=source,
        source_to_symint_node_cache=source_to_symint_node_cache,
    )


# See note [Tensor Fakification and Symbol Caching]
def wrap_to_fake_tensor_and_record(e, tx, *, source: Optional[Source], is_tensor: bool):
    if (
        type(e) in (torch.Tensor, torch.nn.Parameter, FakeTensor)
        or isinstance(e, torch.Tensor)
        or is_traceable_wrapper_subclass(e)
    ):
        assert source is not None
        static_shapes, reason = tensor_always_has_static_shape(
            e, is_tensor, guard_source=source.guard_source()
        )

        symbolic_context = None
        if not e.is_nested:
            # TODO: We should probably support this for nested tensors too
            symbolic_context = _automatic_dynamic(e, tx, source, static_shapes)

        if symbolic_context:
            tx.output.tracing_context.tensor_to_context[e] = symbolic_context

        log.debug(
            "wrap_to_fake %s %s %s %s",
            source.name(),
            tuple(e.shape),
            symbolic_context.dynamic_sizes if symbolic_context is not None else None,
            symbolic_context.constraint_sizes if symbolic_context is not None else None,
        )
        fake_e = wrap_fake_exception(
            lambda: tx.fake_mode.from_tensor(
                e,
                source=source,
                symbolic_context=symbolic_context,
            )
        )
        if is_tensor and not (static_shapes and source.is_nn_module()):
            # TODO: just store the whole symbolic_context here
            tx.output.tracked_fakes.append(
                TrackedFake(
                    fake_e,
                    source,
                    symbolic_context.constraint_sizes
                    if symbolic_context is not None
                    else None,
                )
            )
            tx.output.tracked_fakes_id_to_source[id(e)].append(source)
        tx.output.tensor_weakref_to_sizes_strides[e] = {
            "size": fake_e.size(),
            "stride": fake_e.stride(),
        }
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

    def __call__(self, tx, value) -> VariableTracker:
        if isinstance(value, VariableTracker):
            # This is always valid to call, and useful for recursive calls.
            return value
        if isinstance(value, dataclasses._HAS_DEFAULT_FACTORY_CLASS):
            return UserDefinedObjectVariable(value)
        if ConstantVariable.is_literal(value):
            return SourcelessBuilder.wrap_constant_literal(value)
        elif is_builtin_callable(value):
            return BuiltinVariable(value)
        elif is_allowed(value):
            if is_user_defined_allowed(value):
                self.tx.output.has_user_defined_allowed_in_graph = True
            return TorchVariable(value)
        elif isinstance(value, types.FunctionType):
            return UserFunctionVariable(value)
        elif isinstance(value, enum.Enum):
            return EnumVariable(value)
        elif isinstance(value, (type, abc.ABCMeta)):
            return UserDefinedClassVariable(value)
        elif isinstance(value, dict):
            return ConstDictVariable(
                {k: self(tx, v) for k, v in value.items()},
                dict,
                mutable_local=MutableLocal(),
            )
        elif isinstance(value, set):
            return SetVariable(
                [self(tx, x) for x in value], mutable_local=MutableLocal()
            )
        elif isinstance(value, (tuple, list)):
            cls = BaseListVariable.cls_for(type(value))
            return cls([self(tx, x) for x in value], mutable_local=MutableLocal())
        elif isinstance(value, types.MethodWrapperType):
            return MethodWrapperVariable(value)
        unimplemented(f"Unexpected type in sourceless builder {type(value)}")

    @staticmethod
    def wrap_constant_literal(value):
        assert ConstantVariable.is_literal(value)
        return ConstantVariable.create(value=value)
