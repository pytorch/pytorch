"""
Core graph building functionality for PyTorch's Dynamo system. This module contains
the essential components for constructing and managing FX graphs during compilation:

- OutputGraph: Manages the overall graph construction and compilation process. It owns
  a SubgraphTracer and handles graph compilation, execution, and state management.
  OutputGraph also manages features like graph deduplication, symbolic shape handling,
  and tracking of side effects.

- SubgraphTracer: Handles the actual FX graph construction by tracing Python code.
  It supports advanced features like higher-order operators through nested tracers,
  lifting of free variables, and handling of symbolic shapes.

The module supports key Dynamo features including:
- Higher-order operators through nested SubgraphTracers
- Graph deduplication for optimization
- Symbolic shape handling and propagation
- Side effect tracking and management
- Guard insertion and management
"""

import collections
import contextlib
import copy
import functools
import inspect
import itertools
import logging
import operator
import re
import sys
import traceback
import warnings
import weakref
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field as dc_field
from types import CodeType
from typing import Any, Callable, cast, Optional, TYPE_CHECKING, Union
from typing_extensions import ParamSpec, TypeVar

import sympy

import torch._guards
import torch._logging
import torch.distributed as dist
import torch.nn
import torch.utils._pytree as pytree
from torch import fx, Tensor
from torch._C._dynamo import guards
from torch._dynamo.exc import ShortenTraceback, TensorifyScalarRestartAnalysis
from torch._guards import (
    CompileContext,
    CompileId,
    GlobalContextCheckpointState,
    Source,
    tracing,
    TracingContext,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch._utils_internal import signpost_event
from torch.export.dynamic_shapes import _ConstraintTarget
from torch.fx._lazy_graph_module import _make_graph_module  # type: ignore[attr-defined]
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    guard_scalar,
    is_symbolic,
    ShapeEnv,
    Specialization,
)
from torch.fx.node import Target
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from . import config, exc, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
    create_binary_slice,
    create_binary_subscr,
    create_build_tuple,
    create_call_function,
    create_dup_top,
    create_instruction,
    create_load_const,
    create_rot_n,
    create_swap,
    Instruction,
    unique_id,
)
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import enter_new_scope
from .device_interface import get_interface_for_device
from .exc import (
    BackendCompilerFailed,
    exceptions_allowed_to_be_fallback,
    SkipFrame,
    unimplemented_v2,
    unimplemented_v2_with_warning,
)
from .graph_deduplication import apply_graph_deduplication
from .graph_region_tracker import GraphRegionTracker
from .guards import GuardBuilder, install_guard
from .mutation_guard import is_dynamic_nn_module
from .side_effects import AttributeMutationExisting, SideEffects, ValueMutationExisting
from .source import (
    _get_source_debug_name,
    AttrSource,
    BackwardStateSource,
    ConstantSource,
    GetItemSource,
    GlobalStateSource,
    is_constant_source,
    is_from_local_source,
    LocalSource,
    NumpyTensorSource,
    ParamBufferSource,
    ShapeEnvSource,
    SyntheticLocalSource,
    TensorProperty,
    TensorPropertySource,
)
from .utils import (
    _extract_tensor_dict,
    checkpoint_params,
    CleanupHook,
    clone_inputs,
    count_calls,
    counters,
    dynamo_timed,
    get_instruction_source_311,
    get_locals_to_steal,
    get_static_address_type,
    get_unique_name_wrt,
    graph_break_reasons,
    increment_op_count,
    istype,
    lazy_format_graph_code,
    LazyString,
    nn_module_proxy,
    same,
    set_example_value,
)
from .variables.base import VariableTracker
from .variables.builder import (
    BackwardStateGraphArg,
    GraphArg,
    TrackedFake,
    wrap_fx_proxy,
)
from .variables.ctx_manager import ContextWrappingVariable
from .variables.lists import BaseListVariable
from .variables.misc import NullVariable
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .variables.torch_function import TensorWithTFOverrideVariable
from .variables.user_defined import UserDefinedDictVariable


if TYPE_CHECKING:
    from torch._dynamo.package import CompilePackage
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

log = logging.getLogger(__name__)
graph_tabular_log = torch._logging.getArtifactLogger(__name__, "graph")
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")
graph_sizes_log = torch._logging.getArtifactLogger(__name__, "graph_sizes")
trace_call_log = torch._logging.getArtifactLogger(__name__, "trace_call")

RootGuardManager = guards.RootGuardManager


# Capture fn pointer at import time
# This is to guard against trying to mark the iterated tensors
# as static in case user overrides fn ptr
og_module_named_buffers_fn_ptr = torch.nn.Module.named_buffers
og_module_named_parameters_fn_ptr = torch.nn.Module.named_parameters


@dataclass(frozen=True)
class VariableTrackerCacheKey:
    vt_id: int
    # Two different source can point to the same object. However, Dynamo handles
    # globals and local source differently when it comes to guards and possibly
    # some other parts as well. So, cache also relies on the source.
    source: Source


@dataclass(frozen=True)
class AliasingInfo:
    has_aliasing: bool
    msg: str


@dataclass(frozen=True)
class MutationInfo:
    has_mutation: bool
    msg: str


class VariableTrackerCache:
    def __init__(self) -> None:
        self.cache: dict[VariableTrackerCacheKey, VariableTracker] = {}

    def lookup(self, value: Any, source: Source) -> Optional[VariableTracker]:
        key = VariableTrackerCacheKey(id(value), source)
        if key not in self.cache:
            return None
        return self.cache[key]

    def add(self, value: Any, source: Source, vt: VariableTracker) -> None:
        key = VariableTrackerCacheKey(id(value), source)
        self.cache[key] = vt

    def clone(self) -> "VariableTrackerCache":
        # Needed for copy and restore graph state
        new_cache = VariableTrackerCache()
        new_cache.cache.update(self.cache)
        return new_cache

    def clear(self) -> None:
        self.cache.clear()


@functools.cache
def _step_logger() -> Any:
    return torchdynamo_logging.get_step_logger(log)


@dataclass
class GraphCompileReason:
    """Stores why a given output graph was compiled; i.e. what caused the graph break."""

    reason: str
    user_stack: list[traceback.FrameSummary]

    # Indicates if this was a graph break reason due to graph break.
    graph_break: bool = True

    def __post_init__(self) -> None:
        if self.graph_break:
            graph_break_reasons.append(self)


def _get_gen_rand_values_fn(random_calls: Any) -> Callable[[], list[Any]]:
    def _gen_rand_values() -> list[Any]:
        return [fn(*args, **kwargs) for fn, args, kwargs in random_calls]

    return _gen_rand_values


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: dict[str, torch.nn.Module]):
        super().__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return "FakeRootModule(...)"

    def add_nn_modules(self, nn_modules: dict[str, torch.nn.Module]) -> None:
        for k, v in nn_modules.items():
            setattr(self, k, v)


class WrapperBackend:
    def __init__(self, backend: CompilerFn) -> None:
        self.backend: CompilerFn = backend

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn:
        self.restore = checkpoint_params(gm)
        self.gm = gm
        copy_gm = copy.deepcopy(self.gm)
        self.candidate = self.backend(copy_gm, example_inputs)

        if self.candidate is None or self.candidate is self.gm.forward:
            return self.gm.forward

        if not config.verify_correctness:
            return self.candidate

        # if verify_correctness=True
        try:
            correct = self.gm.forward(*clone_inputs(example_inputs))
            result = self.candidate(*clone_inputs(example_inputs))

            # TODO: replace `same` function with the one in testing
            if same(correct, result):
                return self.candidate

            raise RuntimeError(f"incorrect results of backend {self}")

        except Exception:
            log.exception("error in verify_correctness")
            raise
        finally:
            self.restore()


Scope = dict[str, object]


@dataclass
class OutputGraphGuardsState:
    """
    A base class containing fields that are considered "persistent" when we
    want to save all the important state for reconstrucing guards in a different
    process. Normally we don't need to add states here, but we may have to when
    the information is needed to serialize the guards, so the fields here are
    supposed to be serializable as a requirement.
    """

    local_scope: Scope
    global_scope: Scope
    # This records the initial torch function mode stack for guarding
    torch_function_mode_stack: list[torch.overrides.TorchFunctionMode]
    guard_on_key_order: set[Source]
    # Map from graph input's `Source` to sizes / strides metadata
    input_source_to_sizes_strides: dict[Source, dict[str, Any]]
    dual_level: int
    functorch_layers: list[torch._functorch.pyfunctorch.FuncTorchInterpreter]
    current_device: Optional[torch.device]
    global_state_guard: torch._C._dynamo.guards.GlobalStateGuard
    _guards: torch._guards.GuardsSet
    _aotautograd_guards: list[torch._guards.GuardEnvExpr]

    # Whether or not the guards should be checked for correctness

    export: bool = False
    skip_guards_check: bool = False
    export_constraints: bool = False
    name_of_builtins_dict_key_in_fglobals: Optional[str] = None

    @property
    def shape_env(self) -> ShapeEnv:
        raise AssertionError(f"shape_env shouldn't be accessed from {type(self)}")

    @property
    def guards(self) -> torch._guards.GuardsSet:
        return self._guards

    @property
    def aotautograd_guards(self) -> list[torch._guards.GuardEnvExpr]:
        return self._aotautograd_guards

    def dump_guards_state(self) -> "OutputGraphGuardsState":
        # Dump a serializable version of self without extras
        return OutputGraphGuardsState(
            local_scope=self.local_scope,
            global_scope=self.global_scope,
            torch_function_mode_stack=self.torch_function_mode_stack,
            guard_on_key_order=self.guard_on_key_order,
            input_source_to_sizes_strides=self.input_source_to_sizes_strides,
            dual_level=self.dual_level,
            functorch_layers=self.functorch_layers,
            current_device=self.current_device,
            global_state_guard=self.global_state_guard,
            name_of_builtins_dict_key_in_fglobals=self.name_of_builtins_dict_key_in_fglobals,
            export=self.export,
            export_constraints=self.export_constraints,
            _guards=self.guards,
            _aotautograd_guards=self.aotautograd_guards,
            skip_guards_check=self.skip_guards_check,
        )


@dataclass
class StackLocalsMetadata:
    """
    Stores metadata for a frame's stack and locals for the purposes of building resume functions
    """

    num_stack: int = 0  # number of stack elements, minus removed NULLs
    locals_names: dict[str, int] = dc_field(
        default_factory=dict
    )  # order of locals codegen'd to the stack
    stack_null_idxes: list[int] = dc_field(default_factory=list)
    locals_null_keys: list[str] = dc_field(default_factory=list)
    stack_ctx_args: list[tuple[int, tuple[Any, ...]]] = dc_field(default_factory=list)
    stack_ctx_idxes_orig: list[int] = dc_field(default_factory=list)
    locals_ctx_args: list[tuple[str, tuple[Any, ...]]] = dc_field(default_factory=list)


# TODO we should expand this to make it work for atribtrary in/out
@dataclass
class ExportMetaData:
    # maps graph input index to its' source which is later
    # used in export to map to correct user input. In its' flat form,
    # just looks like GetItem(base=LocalSource("foo", idx=0))
    graph_input_idx_to_local_source: dict[int, Source] = dc_field(default_factory=dict)
    # maps user output idx to what type of output it is. There are 3 options:
    # 1) graph out
    # 2) user input
    # 3) constants
    output_return_type: dict[int, tuple[str, Any]] = dc_field(default_factory=dict)
    # output spec of the traced function
    out_spec: Union[torch.utils._pytree.TreeSpec, torch.utils._pytree.LeafSpec] = (
        torch.utils._pytree._LEAF_SPEC
    )
    module_call_spec: dict[
        str,
        dict[str, Union[torch.utils._pytree.TreeSpec, torch.utils._pytree.LeafSpec]],
    ] = dc_field(default_factory=dict)


def get_builtins_dict(global_scope: Scope) -> dict[str, Any]:
    # f_globals["__builtins__"] can be a dict or a module. This is an
    # implementation detail -
    # https://docs.python.org/3/library/builtins.html.

    # This makes guarding on any builtin messy because the guard check_fn
    # has to check if the __builtins__ is a module or dict, and then access
    # by either using getattr or getitem respectively.

    # To solve this problem, we insert a new entry in f_globals which points
    # to the builtins __dict__ and then we guard any builtin on this dict.
    # To avoid any collision with the pre-existing keys, we use the
    # install_global to give us a unique dict key.

    f_builtins = global_scope["__builtins__"]
    if not isinstance(f_builtins, dict):
        f_builtins = f_builtins.__dict__
    return f_builtins


class OutputGraphCommon(OutputGraphGuardsState):
    """
    A minimal interface for full graph capture. It is intended to be
    the target of any tracer that feeds into backends.

    Currently dynamo's OutputGraph is the only known implementation
    of this interface, used by (aot) precompile and (strict) export.
    Importantly, that implementation also contains many other fields
    that are using during tracing but not included in this interface
    because they are not used once tracing is complete.

    It should be safe to assume that (caching) precompile also uses
    this interface.

    In the future, we want make_fx, used by (non-strict) export, to
    also implement this interface.

    The serializable part of this interface is OutputGraphGuardsState.
    We do not need to serialize other parts; however it will pay to
    be disciplined about what those other parts are, especially since
    we want other tracers to be able to meaningfully implement them,
    and we should generally try to cut them down when possible.
    """

    def __init__(
        self,
        output_graph_guards_state: OutputGraphGuardsState,
        shape_env: Optional[ShapeEnv] = None,
        export_metadata: Optional[ExportMetaData] = None,
        tracked_fakes_id_to_source: Optional[dict[int, list[Source]]] = None,
    ):
        super().__init__(
            output_graph_guards_state.local_scope,
            output_graph_guards_state.global_scope,
            output_graph_guards_state.torch_function_mode_stack,
            output_graph_guards_state.guard_on_key_order,
            output_graph_guards_state.input_source_to_sizes_strides,
            output_graph_guards_state.dual_level,
            output_graph_guards_state.functorch_layers,
            output_graph_guards_state.current_device,
            output_graph_guards_state.global_state_guard,
            output_graph_guards_state._guards,
            output_graph_guards_state._aotautograd_guards,
            output_graph_guards_state.export,
            output_graph_guards_state.skip_guards_check,
            output_graph_guards_state.export_constraints,
            output_graph_guards_state.name_of_builtins_dict_key_in_fglobals,
        )

        # The following fields are currently known to be used by clients.
        # In particular, we need:
        # - shape_env, for building guards
        # - export_metadata, for un/flattening inputs and outputs
        # - tracked_fakes_id_to_source, for processing tensor dim constraints
        self._shape_env = shape_env or ShapeEnv()  # private for inheritance
        self.export_metadata = export_metadata or ExportMetaData()
        self.tracked_fakes_id_to_source: dict[int, list[Source]] = (
            tracked_fakes_id_to_source or {}
        )

    @property
    def shape_env(self) -> ShapeEnv:
        return self._shape_env

    def bypass_package(self, reason: str = "", **kwargs: Any) -> None:
        # NOTE: currently there are no tests for this but it is reachable
        # when building guards, so technically necessary to include here.
        # It is unclear whether we should include packaging altogether.
        raise NotImplementedError


class OutputGraph(OutputGraphCommon):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.

    OutputGraph is 1:1 with a frame being processed. Each frame is associated
    with some root InstructionTranslator. When user code calls a function,
    we construct a InliningInstructionTranslator that continues to write into
    the root InstructionTranslator's OutputGraph.
    """

    side_effects: SideEffects

    def __init__(
        self,
        code_options: dict[str, Any],
        compiler_fn: Optional[CompilerFn],
        root_tx: "InstructionTranslatorBase",
        export: bool,
        export_constraints: Sequence[_ConstraintTarget],
        frame_state: Any,
        local_scope: Scope,
        global_scope: Scope,
        f_code: CodeType,
        torch_function_mode_stack: list[torch.overrides.TorchFunctionMode],
        package: Optional["CompilePackage"],
        one_graph: bool = False,
    ) -> None:
        OutputGraphGuardsState.__init__(
            self,
            local_scope,
            global_scope,
            torch_function_mode_stack,
            guard_on_key_order=set(),
            input_source_to_sizes_strides={},
            dual_level=torch.autograd.forward_ad._current_level,
            functorch_layers=torch._functorch.pyfunctorch.retrieve_all_functorch_interpreters(),
            current_device=torch.utils._device.CURRENT_DEVICE,
            # initial_global_state is only None during NopTest.
            global_state_guard=torch._dynamo.convert_frame.initial_global_state
            or torch._C._dynamo.guards.GlobalStateGuard(),
            # These are set by @property instead, just initialize them as blank
            _guards=torch._guards.GuardsSet(),
            _aotautograd_guards=[],
        )
        self.tracers = [SubgraphTracer(self, is_export=export)]
        # Map from graph input's `Source` to its `VariableTracker` to
        # de-duplicate graph inputs by source and reuse the tracker
        self.input_source_to_var: dict[Source, VariableTracker] = {}
        self.export = export
        self.export_constraints = export_constraints  # type: ignore[assignment]
        self.frame_state = frame_state
        self.cleanup_hooks: list[Callable[[], Any]] = []
        # compile_id is an id number for the current torch.compile
        self.compile_id: int = next(_compile_id_counter)
        # Set of globals installed via install_global* APIs
        self.installed_globals: set[str] = set()

        # TODO: maybe should just pass the entire f_code in here?  Not
        # sure...
        self.co_fields = {
            "co_name": f_code.co_name,
            "co_filename": f_code.co_filename,
            "co_firstlineno": f_code.co_firstlineno,
        }

        self.region_tracker = GraphRegionTracker()

        # tracked_fakes says where any tensor that was wrapped to fake came
        # from.  It is similar to GraphArg, in that all GraphArgs will get
        # will get added to TrackedFakes, but TrackedFakes also contains
        # GraphArgs that got pruned, and things like Tensor attributes which
        # aren't explicit graph inputs.  Used by shape guard
        self.tracked_fakes: list[TrackedFake] = []

        shape_env = ShapeEnv(
            # Reference Cycle!
            # Share a reference to the list of TrackedFake.
            #
            # ShapeEnv needs this in order to be able to reproduce the call
            # to produce_guards at an arbitrary time point. That is because
            # TrackedFake instances may have its metadata changed throughout
            # the program execution.
            tracked_fakes=self.tracked_fakes,
            # We want to allow capture scalar outputs and allow_dynamic_output_shape_ops when fullgraph=True
            allow_scalar_outputs=one_graph or config.capture_scalar_outputs,
            allow_dynamic_output_shape_ops=one_graph
            or config.capture_dynamic_output_shape_ops,
            prefer_deferred_runtime_asserts_over_guards=config.prefer_deferred_runtime_asserts_over_guards,
            co_fields=self.co_fields,
        )

        # In export mode, we force the shape_env to strictly disallow any constraining
        # of the user marked dynamic dims
        import torch._functorch.config as _config

        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            fake_mode = torch._subclasses.FakeTensorMode(
                shape_env=shape_env,
                # TODO (tmanlaibaatar) Remove this once we always lift params and buffers
                allow_non_fake_inputs=bool(self.export),
                export=self.export,
            )
        self.tracing_context: TracingContext = TracingContext(fake_mode)
        self.tracing_context.traced_code.append(f_code)
        self.traced_code = self.tracing_context.traced_code
        self.dynamo_compile_id: Optional[CompileId] = (
            CompileContext.current_compile_id()
        )
        self.init_ambient_guards()

        # Map each tensor id to a list of sources. This is necessary because
        # tensor ids cannot be recovered from tracked fakes (in general).
        # We use this map to interpret (i.e., check for violations of) constraints,
        # specifically equality constraints, which have shared tensor ids in them.
        # This map should also be generally useful, e.g., for (de)serialization.
        self.tracked_fakes_id_to_source: dict[int, list[Source]] = (
            collections.defaultdict(list)
        )
        # Stores the full fqn of a param or buffer to the relevant source.
        self.param_name_to_source: Optional[dict[str, Source]] = {}
        self.side_effects = SideEffects(self)
        # Cached variable trackers. This makes symbolic analysis of LOAD_GLOBAL
        # and LOAD_ATTR for same python objects free.
        self.variable_tracker_cache = VariableTrackerCache()
        self.unique_var_id = itertools.count()
        self.code_options: dict[str, Any] = dict(code_options)
        self.output_instructions: list[Instruction] = []
        # used to track nodes that are added between calls of copy_graphstate
        # and restore_graphstate
        self.timestamp = 0

        # A list of register_finalizer_fns to apply to the output graph module
        self.register_finalizer_fns: list[Callable[[fx.GraphModule], None]] = []

        # Not checkpointed
        self.compiler_fn: Optional[CompilerFn] = compiler_fn
        self.root_tx = root_tx

        self.package = package
        # Given a source, what are the user stacks of all locations that
        # accessed it?
        #
        # For efficiency, we only populate this:
        #   - During export, and
        #   - If the source could potentially lead to a spurious export input
        #
        # Feel free to populate this more frequently if other use-cases arise,
        # but be aware that we have to generate full stacks for each
        # recording!
        self.source_to_user_stacks: dict[Source, list[traceback.StackSummary]] = {}

        self._current_tx: list[InstructionTranslatorBase] = []
        self.cleanups: list[CleanupHook] = []
        self.should_exit = False
        self.unspec_variable_map: dict[str, UnspecializedPythonVariable] = {}

        # This returns false if TF Overall (both mode and subclass) is disabled OR that TF Mode stack is empty
        self.torch_function_mode_enabled = torch._C._is_torch_function_mode_enabled()

        # Tracks if the output graph has a user defined allowed function in the
        # graph. This is used later to determine if we should fallback to eager
        # for certain exceptions. THe idea is that if the user has applied
        # allow_in_graph, they would like to see the error instead of falling
        # back for backend errors.
        self.has_user_defined_allowed_in_graph = False

        # Tracks a list of called ops that were not tagged with "pt2_compliant_tag".
        # This information is useful for logging.
        self.non_compliant_ops: set[torch._ops.OpOverload] = set({})

        # Tracks a list of called custom ops that were tagged with "pt2_compliant_tag".
        # This information is useful for logging.
        self.compliant_custom_ops: set[torch._ops.OpOverload] = set({})

        # We save the global torch state here to be restored in case of graph
        # breaks. The relevant issue is seen here
        # https://github.com/pytorch/pytorch/pull/100570#issuecomment-1543427086
        # where inlining of a function changes the global state (because of the
        # presence of torch.no_grad) and there is a graph break.
        self.save_global_state()

        # Tracks the original FQNs of the constant tensors from the original graph,
        # i.e. buffers and parameters.
        self.dynamo_flat_name_to_original_fqn: dict[str, str] = {}

        # All calls to random() are replaced with a single call to __gen_rand_values
        # functions that returns a tuple of random values for each original call.
        # random_calls tracks calls to random() and random_values_var stores the name of
        # the variable that stores __gen_rand_values results.
        self.random_calls: list[
            tuple[Callable[..., object], tuple[object, ...], dict[str, object]]
        ] = []
        self.random_values_var: Any = None

        # Bytecode to insert right before we call the graph
        self.pregraph_bytecode: list[Instruction] = []

        # Use to pass values to backward hooks when using compiled autograd
        self.backward_state: dict[str, VariableTracker] = {}
        self.backward_state_proxy: Optional[torch.fx.Proxy] = None
        self.backward_state_var: Optional[str] = None

        # pyrefly: ignore  # bad-override
        self.name_of_builtins_dict_key_in_fglobals: str = (
            self.install_builtins_dict_in_fglobals()
        )

        self.compiler_trace_stack = contextlib.ExitStack()

        # These are the ambient, currently-global saved_tensor_hooks stashed in autograd,
        # that are set for the entire duration of the compiled region.
        # This is an invariant today because we graph break on the saved_tensor_hook
        # context manager inside a compiled region
        self.saved_tensors_hooks_subgraph_names: Optional[list[str]] = (
            self.maybe_install_saved_tensors_hooks_subgraphs()
        )

        # mangled alias -> module fqn name
        self.import_sources: dict[str, str] = {}

        self.export_metadata = ExportMetaData()

        # Set of inlined unspecialized modules names to generate the
        # dynamo_flat_name_to_original_fqn mapping.
        self.used_inlined_inbuilt_modules_names: OrderedSet[str] = OrderedSet()

    def mark_bytecode_tracing_start(self) -> None:
        self.compiler_trace_stack.enter_context(
            dynamo_timed(
                "bytecode_tracing",
                log_pt2_compile_event=True,
            )
        )

    def mark_bytecode_tracing_stop(self) -> None:
        self.compiler_trace_stack.close()

    def install_builtins_dict_in_fglobals(self) -> str:
        f_builtins = get_builtins_dict(self.global_scope)
        return self.install_global("__builtins_dict__", f_builtins)

    def add_backward_state_hook(
        self, hook: VariableTracker, prefix: str = "hook"
    ) -> tuple[str, torch.fx.Proxy]:
        name = f"{prefix}{len(self.backward_state)}"
        assert name not in self.backward_state
        self.backward_state[name] = hook
        return name, self.get_backward_state_proxy()

    def get_backward_state_proxy(self) -> torch.fx.Proxy:
        if self.backward_state_proxy is None:
            if self.export:
                unimplemented_v2(
                    gb_type="backward_state does not support export",
                    context="",
                    explanation="Compiled autograd doesn't work with `torch.export`.",
                    hints=[],
                )
            example_value = BackwardState()
            self.backward_state_proxy = self.root_tracer.create_graph_input(
                "dynamo_backward_state",
                type(example_value),
                example_value,
                source=BackwardStateSource(),
            )
            self.backward_state_proxy.node.meta["grapharg"] = BackwardStateGraphArg()
            self.backward_state_var = self.new_var()
        return self.backward_state_proxy

    # This gets its own helper function so guards DEBUG logs are more informative
    def init_ambient_guards(self) -> None:
        # Register a SHAPE_ENV guard to make sure we setup shape guards
        # that show up in ShapeEnv
        self.guards.add(ShapeEnvSource().make_guard(GuardBuilder.SHAPE_ENV))

        self.guards.add(
            GlobalStateSource().make_guard(GuardBuilder.DETERMINISTIC_ALGORITHMS)
        )

        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.GRAD_MODE))

        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.DEFAULT_DEVICE))

        self.guards.add(
            GlobalStateSource().make_guard(GuardBuilder.TORCH_FUNCTION_STATE)
        )

        ci = torch._C._functorch.peek_interpreter_stack()
        if ci is not None:
            self.guards.add(
                GlobalStateSource().make_guard(GuardBuilder.FUNCTORCH_STACK_MATCH)
            )
        if not torch._dynamo.compiled_autograd.in_compiled_autograd_region:
            self.guards.add(
                GlobalStateSource().make_guard(
                    GuardBuilder.AUTOGRAD_SAVED_TENSORS_HOOKS
                )
            )

    def maybe_install_saved_tensors_hooks_subgraphs(self) -> Optional[list[str]]:
        if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
            return None

        get_hooks = torch._functorch._aot_autograd.utils.top_saved_tensors_hooks
        are_inline_hooks = (
            torch._functorch._aot_autograd.utils.saved_tensors_hooks_are_inlineable
        )
        hooks = get_hooks()
        if not are_inline_hooks(hooks):
            return None

        # If GraphModule provided by user contains fx.wrap,
        # We can only rely on user provided cache hash in this case.
        # If user did not provide cache hash - then we always bypass cache.

        pack_gm, unpack_gm = hooks
        pack_subgraph_name = self.install_subgraph(
            "saved_tensors_hooks_pack",
            torch.fx.GraphModule(self.nn_modules, pack_gm.graph),
        )
        unpack_subgraph_name = self.install_subgraph(
            "saved_tensors_hooks_unpack",
            torch.fx.GraphModule(self.nn_modules, unpack_gm.graph),
        )
        assert pack_subgraph_name == "saved_tensors_hooks_pack_0"
        assert unpack_subgraph_name == "saved_tensors_hooks_unpack_0"
        return [pack_subgraph_name, unpack_subgraph_name]

    def synthetic_graph_input(
        self, fn: Callable[..., Any], args: tuple[Any, ...]
    ) -> VariableTracker:
        """
        call fn(*args) before the graph runs and turn the result into a fake input.
        """
        example_value = fn(*args)
        varname = self.new_var()
        cg = PyCodegen(self.root_tx)
        cg.add_push_null(
            lambda: cg.load_import_from(
                fn.__module__,
                fn.__name__,
            )
        )
        cg.foreach(map(variables.ConstantVariable.create, args))
        cg.call_function(len(args), False)
        cg.store(varname)
        self.pregraph_bytecode.extend(cg.get_instructions())
        source = SyntheticLocalSource(varname)
        result = VariableTracker.build(self.root_tx, example_value, source)
        # Realize the VT because we will delete the guards on it in the next line.
        result = result.realize()
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )
        return result

    def add_cleanup_hook(self, fn: Callable[[], Any]) -> None:
        self.cleanup_hooks.append(fn)

    def call_cleanup_hooks(self) -> None:
        for hook in reversed(self.cleanup_hooks):
            hook()
        self.cleanup_hooks.clear()

    @property
    def root_tracer(self) -> "SubgraphTracer":
        return self.tracers[0]

    @property
    def current_tracer(self) -> "SubgraphTracer":
        return self.tracers[-1]

    def is_root_tracer(self) -> bool:
        # Helper to tell if we are inside the higher order operator tracing.
        return len(self.tracers) == 1

    @property
    def graph(self) -> torch.fx.Graph:
        return self.current_tracer.graph

    # TODO(rzou): can delete after we refactor speculate_subgraph to use nested GraphTracer.
    @graph.setter
    def graph(self, value: torch.fx.Graph) -> None:
        self.current_tracer.graph = value

    @property
    def input_name_to_proxy(self) -> dict[str, fx.Proxy]:
        return self.current_tracer.input_name_to_proxy

    @property
    def real_value_cache(self) -> dict[fx.Node, torch.Tensor]:
        return self.current_tracer.real_value_cache

    @property
    def bound_symbols(self) -> dict[sympy.Symbol, Union[torch.fx.Proxy, "LazyProxy"]]:
        return self.current_tracer.bound_symbols

    # If you are here, and you're looking for create_graph_input,
    # to avoid ambiguity, please call one of the following:
    # - self.current_tracer.create_graph_input
    # - self.root_tracer.create_graph_input
    # See NOTE [HigherOrderOperator tracing design] for more context.

    def create_proxy(self, *args: Any, **kwargs: Any) -> torch.fx.Proxy:
        return self.current_tracer.create_proxy(*args, **kwargs)

    def create_node(self, *args: Any, **kwargs: Any) -> torch.fx.Node:
        return self.current_tracer.create_node(*args, **kwargs)

    def remove_node(self, *args: Any, **kwargs: Any) -> None:
        return self.current_tracer.remove_node(*args, **kwargs)

    @contextlib.contextmanager
    def subtracer(
        self, source_target: Optional[Target], prior_tracer: "SubgraphTracer"
    ) -> Generator[fx.Tracer, None, None]:
        new_scope_ctx = enter_new_scope()
        try:
            if prior_tracer:
                # Lineage MUST stay preserved
                assert prior_tracer.parent is self.current_tracer
            new_scope_ctx.__enter__()
            tracer = (
                prior_tracer
                if prior_tracer
                else SubgraphTracer(
                    self,
                    parent=self.current_tracer,
                    source_target=source_target,
                    is_export=self.current_tracer.is_export,
                )
            )
            self.tracers.append(tracer)
            yield tracer
        finally:
            new_scope_ctx.__exit__(None, None, None)
            self.tracers.pop()

    @property
    def output(self) -> "OutputGraph":
        return self

    @property
    def fake_mode(self) -> torch._subclasses.FakeTensorMode:
        assert self.tracing_context.fake_mode is not None
        return self.tracing_context.fake_mode

    @property
    def shape_env(self) -> ShapeEnv:
        assert self.tracing_context.fake_mode is not None
        assert self.tracing_context.fake_mode.shape_env is not None
        return self.tracing_context.fake_mode.shape_env

    @property
    def guards(self) -> torch._guards.GuardsSet:
        return self.tracing_context.guards_context.dynamo_guards

    @property
    def nn_modules(self) -> dict[str, Any]:
        return self.tracing_context.module_context.nn_modules

    @property
    def aotautograd_guards(self) -> list[torch._guards.GuardEnvExpr]:
        return self.tracing_context.guards_context.aotautograd_guards

    def save_global_state(
        self, out: Optional[dict[str, tuple[Callable[..., Any], bool]]] = None
    ) -> None:
        """
        Saves to out if it is provided. Else saves to the tracing context's global_state.
        """
        global_state = cast(
            dict[str, tuple[Callable[..., Any], bool]],
            (
                out
                if out is not None
                else self.tracing_context.global_context.global_state
            ),
        )

        global_state["grad_enabled"] = (torch.set_grad_enabled, torch.is_grad_enabled())

        global_state["autocast_enabled"] = (
            functools.partial(torch.set_autocast_enabled, "cuda"),
            torch.is_autocast_enabled("cuda"),
        )
        global_state["autocast_cpu_enabled"] = (
            functools.partial(torch.set_autocast_enabled, "cpu"),
            torch.is_autocast_enabled("cpu"),
        )
        global_state["autocast_gpu_dtype"] = (  # type:ignore[assignment]
            functools.partial(torch.set_autocast_dtype, "cuda"),
            torch.get_autocast_dtype("cuda"),
        )
        global_state["autocast_cpu_dtype"] = (  # type:ignore[assignment]
            functools.partial(torch.set_autocast_dtype, "cpu"),
            torch.get_autocast_dtype("cpu"),
        )
        global_state["autocast_cache_enabled"] = (
            torch.set_autocast_cache_enabled,
            torch.is_autocast_cache_enabled(),
        )

    def push_tx(self, tx: "InstructionTranslatorBase") -> None:
        self._current_tx.append(tx)

    def pop_tx(self) -> "InstructionTranslatorBase":
        return self._current_tx.pop()

    @property
    def current_tx(self) -> "InstructionTranslatorBase":
        return self.root_tx if not self._current_tx else self._current_tx[-1]

    def count_calls(self) -> int:
        return count_calls(self.graph)

    def is_empty_graph(self) -> bool:
        return len(list(self.graph.nodes)) == 0

    def has_outputs(self) -> bool:
        return len([x for x in self.graph.nodes if x.op == "output"]) > 0

    def get_submodule(self, keys: str) -> Union[torch.nn.Module, Any]:
        assert keys
        obj: Union[torch.nn.Module, dict[str, torch.nn.Module]] = self.nn_modules
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

    def new_var(self, name: str = "tmp") -> str:
        existing = set(self.code_options["co_varnames"])
        # In common case, this will be O(1)
        while True:
            var = f"{name}_{next(self.unique_var_id)}"
            if var not in existing:
                self.code_options["co_varnames"] += (var,)
                return var

    def update_co_names(self, name: str) -> None:
        """Ensure self.code_options.co_names contains name"""
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)

    @staticmethod
    def module_key_name(*names: Any) -> str:
        # create a new unique name
        name = "_".join(map(str, names))
        # Strip _buffers[..]/_parmeters[..]/_modules[..] names
        name = re.sub(
            r"\._(?:modules|parameters|buffers)\[(['\"])([^'\"\]]+)\1\]", r".\2", name
        )
        # Replace getattr(a, b) with a.b
        name = re.sub(
            r"getattr\(\s*([^,]+?)\s*,\s*(['\"])([^'\"]+)\2\s*\)", r"\1.\3", name
        )
        # Strip the guard lookup L/G access
        name = re.sub(r"^[GL]\['?(.*?)'?\]$", r"\1", name)
        # e.g. replace abc.xyz[123].qkv with abc.xyz_123.qkv
        name = re.sub(r"\[(\d+)\]", r"_\g<1>", name)
        # e.g. replace abc.xyz_123.qkv with abc_xyz_123_qkv
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)

        if not name or not name[0].isalpha():
            name = "sub" + name

        return name

    def register_static_attr_and_return_proxy(
        self, attr_prefix: str, attr_value: Any
    ) -> fx.Proxy:
        attr_name = get_unique_name_wrt(attr_prefix, self.nn_modules)
        # TODO `nn_modules` has been historically overloaded to store a lot more
        # than just nn module objects, fix that.
        self.nn_modules[attr_name] = attr_value
        proxy = self.create_proxy("get_attr", attr_name, (), {})
        set_example_value(proxy.node, attr_value)
        return proxy

    def register_attr_or_module(
        self,
        target: Union[torch.nn.Module, torch.Tensor, Any],
        *names: Any,
        **options: Any,
    ) -> VariableTracker:
        if is_dynamic_nn_module(target, self.export):
            # Instead of returning UnspecializedNNModuleVariable, call
            # VariableTracker.build so that it is tracked for mutation.
            return VariableTracker.build(self.current_tx, target, **options)

        options = dict(options)
        assert "source" in options
        source = options["source"]
        assert not isinstance(source, ParamBufferSource)

        if isinstance(target, torch.Tensor):
            tracer = self.current_tracer
            if not self.is_root_tracer():
                # For higher order ops, we don't want to insert the get_attr in
                # innermost graph. Instead, we want to raise the params/buffers
                # as inputs to the higher-order graph, and register them as
                # get_attrs in the root tracer.

                # Note that Dynamo will still call lift_tracked_freevar_to_input
                # when these inputs are encountered for the inner graph. The
                # only difference is what happens at the root tracer for
                # nn.Parameters vs free inputs. The free inputs are registered
                # as placeholders in the root graph, whereas the nn.Parameters
                # are registered as get_attr nodes in the root graph.
                tracer = self.root_tracer

            def wrap_name(module_key: str) -> VariableTracker:
                assert self.param_name_to_source is not None
                self.param_name_to_source[module_key] = source

                # Check if the attr has already been registered. This can happen
                # when two different sources point to the same tensor.
                assert self.root_tx is not None
                if target in self.root_tx.output.side_effects:
                    return self.root_tx.output.side_effects[target]

                if get_static_address_type(target) == "guarded" and not isinstance(
                    source, NumpyTensorSource
                ):
                    install_guard(source.make_guard(GuardBuilder.ID_MATCH))
                elif not is_constant_source(source):
                    install_guard(source.make_guard(GuardBuilder.TENSOR_MATCH))

                vt = wrap_fx_proxy(
                    self.root_tx,
                    tracer.create_proxy("get_attr", module_key, (), {}),
                    example_value=target,
                    **options,
                )

                # Track the object so to avoid duplicate registration in case of
                # different sources pointing to the same tensor object.
                vt = self.root_tx.output.side_effects.track_object_existing(target, vt)

                assert "tensor_dict" not in vt.as_proxy().node.meta
                # pyrefly: ignore  # bad-argument-type
                vt.as_proxy().node.meta["tensor_dict"] = _extract_tensor_dict(target)

                return vt

        elif isinstance(target, torch.nn.Module):
            assert isinstance(target, torch.nn.Module)

            if source:
                install_guard(source.make_guard(GuardBuilder.NN_MODULE))

                def wrap_name(module_key: str) -> VariableTracker:
                    # pyrefly: ignore  # bad-argument-type
                    return NNModuleVariable(type(target), module_key, target, **options)

            else:
                # This is Dynamo created graph module, e.g., graph module coming
                # from higher order ops. NNModuleVariable tracker can't be
                # sourceless, so let's return a unspecializedNNModule variable
                # tracker.
                def wrap_name(module_key: str) -> VariableTracker:
                    return variables.UnspecializedNNModuleVariable(target, **options)

        elif isinstance(target, (torch.SymInt, torch.SymFloat)):
            # HACKY CODE REGION BEGIN
            # WE ARE PIGGYBACKING ON EXISTING INFRA TO REGISTER ATTRS
            # This ultimately gets written to self.nn_modules, which is unfortunate
            # Attrs that are tenors and symints and such need to be migrated to have their
            # own storage
            # alas, this is like this for now

            def wrap_name(module_key: str) -> VariableTracker:
                return SymNodeVariable.create(
                    self,
                    self.create_proxy("get_attr", module_key, (), {}),
                    sym_num=target,
                    **options,
                )

            # HACKY CODE REGION END
        else:

            def wrap_name(module_key: str) -> VariableTracker:
                self.output.update_co_names(module_key)
                self.global_scope[module_key] = target
                return VariableTracker.build(
                    self,  # type: ignore[arg-type]
                    target,
                    ConstantSource(source_name=module_key),
                )

        for k, v in self.nn_modules.items():
            if v is target:
                # it already exists
                return wrap_name(k)

        name = OutputGraph.module_key_name(*names)
        name = get_unique_name_wrt(name, self.nn_modules, self.global_scope)
        self.nn_modules[name] = target
        if isinstance(target, torch.nn.Module):

            def register_leaf_name(leaf_name: str) -> None:
                assert self.param_name_to_source is not None
                new_source = ParamBufferSource(source, leaf_name)
                new_name = f"{name}.{leaf_name}"
                self.param_name_to_source[new_name] = new_source
                if isinstance(source, LocalSource):
                    self.dynamo_flat_name_to_original_fqn[
                        OutputGraph.module_key_name(new_source.name())
                    ] = leaf_name

            # annoying, but there are cases when we do not have parameters
            # see test_nn_moduledict_contains
            if hasattr(target, "_parameters"):
                for leaf_name, _ in target.named_parameters():
                    register_leaf_name(leaf_name)
            if hasattr(target, "_buffers"):
                for leaf_name, _ in target.named_buffers():
                    register_leaf_name(leaf_name)

        return wrap_name(name)

    def handle_aliases_for_stolen_lists(
        self, tx: "InstructionTranslatorBase"
    ) -> tuple[list[Instruction], dict[Source, Source]]:
        # If list inputs are stolen, but still needed after the function call, create aliases to keep them alive
        maybe_gm = self.local_scope.get("self")
        stolen_list_names = get_locals_to_steal(maybe_gm)
        if not stolen_list_names:
            return [], {}

        alias_insts = []
        needs_alias: dict[str, list[VariableTracker]] = {}

        queue = [
            *tx.stack,
            *tx.symbolic_locals.values(),
            *self.side_effects.store_attr_mutations.keys(),
        ]

        while queue:
            x = queue.pop()
            if isinstance(x, BaseListVariable):
                assert isinstance(x.items, list)
                queue += x.items
                continue

            if not (
                (
                    x not in self.side_effects.store_attr_mutations
                    or isinstance(x.mutation_type, AttributeMutationExisting)
                )
                and isinstance(x.source, GetItemSource)
                and isinstance(x.source.base, LocalSource)
                and x.source.base.local_name in stolen_list_names
            ):
                continue

            stolen_name = x.source.base.local_name
            if stolen_name not in needs_alias:
                needs_alias[stolen_name] = []
            needs_alias[stolen_name].append(x)

        visited = {}
        overridden_sources: dict[Source, Source] = {}
        for arg in self.graphargs:
            if not (
                isinstance(arg._example, list)
                and isinstance(arg.source, LocalSource)
                and arg.source.local_name in needs_alias
            ):
                continue

            # arg is a list that will be cleared by the compiled function
            list_name = arg.source.local_name
            assert list_name in self.code_options["co_varnames"]
            for x in needs_alias[list_name]:
                # Skip if already handled.
                if x.source in overridden_sources:
                    continue

                # A small codegen optimization because we might have different
                # VariableTrackers that share the same source.
                list_idx = x.source.index  # type: ignore[attr-defined]
                if list_idx not in visited:
                    alias_name = self.new_var(
                        f"{list_name}_ref"
                    )  # self.new_var already adds unique id suffix

                    visited[list_idx] = alias_name
                    # bytecode of `alias_name = list_name[list_idx]`
                    alias_insts.extend(
                        [
                            create_instruction("LOAD_FAST", argval=list_name),
                            create_load_const(list_idx),
                            create_binary_subscr(),
                            create_instruction("STORE_FAST", argval=alias_name),
                        ]
                    )

                # operate on alias, handled by suffix codegen
                old_source = x.source
                overridden_sources[old_source] = LocalSource(visited[list_idx])

        # NOTE: we need `overridden_sources` because (1) we want to codegen for
        # these list items to use the new local source, but (2) we want to avoid
        # updating `source` in place because that might break invariants in
        # other parts of Dynamo like guards.
        return alias_insts, overridden_sources

    def _get_stack_values_to_restore(
        self, tx: "InstructionTranslatorBase", stack_pops: int
    ) -> tuple[list[VariableTracker], StackLocalsMetadata]:
        """
        Gets the stack + locals values belonging to tx that need to be restored.

        Also prunes dead tx locals and realizes all VTs in the tx's stack.

        NullVariables in stack/locals will NOT be restored, unless they are the top `stack_pops`
        elements of the stack - it is expected that the next instruction to run will pop the top
        `stack_pops` elements of the stack, so we should codegen NULLs.

        Returns:
            - stack_values: stack and locals values that need to be restored
            - meta: locations of NULLs and ContextWrappingVariables in the stack/locals
                (ignores the top `stack_pops` values on the stack)
        """
        tx.prune_dead_locals()

        stack_values = []
        meta = StackLocalsMetadata()

        # realize any unrealized tensor VTs in case they
        # need to be added to self.nn_modules as attributes
        for i, value in enumerate(tx.stack):
            variables.LazyVariableTracker.realize_all(value)
            # ignore top `stack_pops` values on the stack
            if len(tx.stack) - i <= stack_pops:
                stack_values.append(value)
                continue
            if isinstance(value, NullVariable):
                meta.stack_null_idxes.append(i)
            else:
                stack_values.append(value)
            if isinstance(value, ContextWrappingVariable):
                target_values = (
                    () if value.target_values is None else tuple(value.target_values)
                )
                # NOTE: track index in stack after NULLs have been removed
                meta.stack_ctx_args.append((len(stack_values) - 1, target_values))
                meta.stack_ctx_idxes_orig.append(i)

        meta.num_stack = len(stack_values)

        cell_and_freevars = set(tx.cellvars() + tx.freevars())

        # NB: Typically (i.e., for graph compile from RETURN_VALUE),
        # symbolic_locals will be empty at this point, as prune_dead_locals
        # will clear out all of symbolic_locals because RETURN_VALUE is the
        # last instruction and no more locals are used.  The fanciness here
        # is only needed for partial graphs.
        # NOTE: All cell and free variables are represented as CellVariable,
        # so checks for NULLs and context managers in the case of codegen'ing resume
        # functions will not be performed on them. This is expected behavior.
        for k, v in tx.symbolic_locals.items():
            # Note! this explicitly uses .local_name for matching
            # Failure to do so will cause spurious registrations in val_to_names.
            # This will in turn result in spurious variables showing up in the graph.
            # This was very tricky to debug. For an example, dump the graph at call_user_compiler
            # while running test_subgraphs.py
            # Do not include top-frame unmodified locals here - otherwise, the compiled graph may
            # erroneously include them as part of the return. We manually codegen them afterward.
            if (
                isinstance(v.source, LocalSource)
                and v.source.local_name == k
                and tx is self.root_tx
            ):
                continue
            # Do not load cell/free vars
            if k in cell_and_freevars:
                continue
            # Do not load variable if it is NULL.
            if sys.version_info >= (3, 12):
                # NOTE: do not use isinstance, since it realizes lazy VT's
                # Continuation function will load the NULL for v.
                if type.__instancecheck__(NullVariable, v):
                    meta.locals_null_keys.append(k)
                    continue
            else:
                # A variable should never be NULL in < 3.12
                assert not type.__instancecheck__(NullVariable, v)
            meta.locals_names[k] = len(meta.locals_names)
            if isinstance(v, ContextWrappingVariable):
                target_values = (
                    () if v.target_values is None else tuple(v.target_values)
                )
                meta.locals_ctx_args.append((k, target_values))
            stack_values.append(v)

        return stack_values, meta

    def compile_subgraph(
        self,
        tx: "InstructionTranslatorBase",
        reason: GraphCompileReason,
        partial_convert: bool = False,
        stack_pops: int = 0,
    ) -> list[StackLocalsMetadata]:
        """
        Compiles the current subgraph, with inputs w.r.t. self.root_tx, and codegens:
            - Call the compiled subgraph
            - Apply side effects
            - Codegen stack and locals
            - Store the locals

        Python does not allow NULL to be an arg to a function, so we do not codegen NULLs on the stack,
        unless the value is one of the top `stack_pops` values on the stack (these values are expected to be
        popped immediately after this generated code. The prologue of the resume function is expected to restore
        any dropped NULLs.

        Returns stack indices and locals keys where we dropped NULLs, and where we found inactive context manager objects.
        """

        assert self.root_tx is not None

        if not config.nested_graph_breaks:
            # expect to only compile 1 frame
            assert self.root_tx is tx

        # bytecode tracing has finished. Pop the context manager for dynamo_timed
        self.mark_bytecode_tracing_stop()

        self.partial_convert = partial_convert
        self.compile_subgraph_reason = reason
        self.should_exit = True

        log.debug("COMPILING GRAPH due to %s", reason)

        # prefix instructions (Python 3.11+)
        prefix_insts: list[Instruction] = []
        if sys.version_info >= (3, 11):
            for inst in self.root_tx.prefix_insts:
                if inst.opname == "COPY_FREE_VARS":
                    prefix_insts.append(
                        create_instruction(
                            "COPY_FREE_VARS",
                            arg=len(self.root_tx.code_options["co_freevars"]),
                        )
                    )
                else:
                    prefix_insts.append(copy.copy(inst))

        # stack values and restore vars for each frame are pushed in reverse order
        # i.e. last element corresponds to root frame (1),
        # first element corresponds to current frame (N)
        all_stack_values = []
        all_stack_locals_metas = []
        cur_tx: Optional[InstructionTranslatorBase] = tx
        while cur_tx is not None:
            # this should have been checked by the caller
            assert all(block.can_restore() for block in cur_tx.block_stack)

            stack_values, meta = self._get_stack_values_to_restore(
                cur_tx, stack_pops if cur_tx is tx else 0
            )
            all_stack_values.append(stack_values)
            all_stack_locals_metas.append(meta)

            # Exit from all context manager variables to make sure global state is restored
            for block in reversed(cur_tx.block_stack):
                block.exit(cur_tx, is_graph_break=reason.graph_break)

            cur_tx = cur_tx.parent

        # "Garbage collect the heap".
        self.side_effects.prune_dead_object_new(tx)

        self.add_output_instructions(prefix_insts)

        assert not (self.pregraph_bytecode and self.export), (
            "export does not support pregraph_bytecode"
        )
        self.add_output_instructions(self.pregraph_bytecode)

        alias_insts, overridden_sources = self.handle_aliases_for_stolen_lists(
            self.root_tx
        )
        self.add_output_instructions(alias_insts)

        self.cleanup_graph()

        # Use nn.Module "proxies" in the constructed GraphModule so that
        # the resulting GM does not hold additional strong references to the original modules.
        # This prevents a strong ref cycle where Dynamo created code holds on to references
        # to modules that also have Dynamo code cache invalidation checks.
        # When cache invalidation runs, the generated GM will be invalidated, which also deletes
        # the proxies.
        nn_modules_proxies = {
            name: nn_module_proxy(mod) for name, mod in self.nn_modules.items()
        }
        root = FakeRootModule(nn_modules_proxies)

        from .decorators import disable

        # to handle random calls
        if len(self.random_calls) > 0:
            random_calls_instructions = []
            self.random_values_var = self.new_var("random_values")
            rand_fn = disable(
                _get_gen_rand_values_fn(self.random_calls),
                reason="do not trace into Dynamo rng recovery function",
            )
            rand_fn_name = self.install_global("__gen_rand_values", rand_fn)
            codegen = PyCodegen(
                self.root_tx, root, overridden_sources=overridden_sources
            )
            random_calls_instructions.extend(
                codegen.load_function_name(rand_fn_name, True)
            )
            random_calls_instructions.extend(create_call_function(0, False))
            random_calls_instructions.append(
                codegen.create_store(self.random_values_var),
            )
            self.add_output_instructions(random_calls_instructions)

        # Codegen stack convention before the unsupported instruction
        # NOTE: in these comment blocks, "locals" EXCLUDE free and cell vars.
        # NOTE: stack/locals/cells must be codegen'd BEFORE the unsupported instruction, since the latter
        # can arbitrarily mutate the former.
        # [frame N cells, .., frame 1 cells],
        # [
        #   frame N locals,
        #   frame N-1 stack + locals,
        #   ...,
        #   frame 1 stack + locals,
        # ], frame N stack

        # see symbolic_convert.py for
        # codegen stack convention after the unsupported instruction
        # NOTE: cells will be loaded into continuation functions directly by symbolic_convert

        # this determines the order that values are codegen'd to the stack
        stack_values_flat = [val for vals in all_stack_values for val in vals]
        stored_graph_output_var = False
        graph_output_var = None

        # call compiled fx graph and codegen all values - stack and locals
        if (
            self.root_tx is tx  # single frame
            and stack_values_flat
            and all(
                not isinstance(
                    v,
                    (
                        UnspecializedPythonVariable,
                        NumpyNdarrayVariable,
                        TensorWithTFOverrideVariable,
                    ),
                )
                and not (isinstance(v, SymNodeVariable) and v.python_type() is float)
                for v in stack_values_flat
            )
            and all(isinstance(x, TensorVariable) for x in stack_values_flat)
            and len(set(stack_values_flat)) == len(stack_values_flat)
            and self.side_effects.is_empty()
            and not tx.debug_locals
            and not self.backward_state
            and not all_stack_locals_metas[-1].stack_null_idxes
            and not all_stack_locals_metas[-1].locals_null_keys
        ):
            # optimization to generate better code in a common case

            # codegen cells
            # no side effects, so no new cells created - no need to call side_effects.codegen_save_tempvars
            cell_cg = PyCodegen(self.root_tx)
            self.codegen_cells(tx, cell_cg)
            self.add_output_instructions(
                [
                    # load in reverse since UNPACK_SEQUENCE will reverse
                    *self.compile_and_call_fx_graph(
                        tx, list(reversed(stack_values_flat)), root
                    ),
                    *cell_cg.get_instructions(),
                    *create_swap(2),
                    create_instruction("UNPACK_SEQUENCE", arg=len(stack_values_flat)),
                ]
            )
            # function output will be moved to the correct places below
        else:
            graph_output_var = self.new_var("graph_out")
            # load stack values in a flat manner - we will codegen bytecode to place them correctly
            # according to our convention above
            pass1 = PyCodegen(
                self.root_tx,
                root,
                graph_output_var,
                overridden_sources=overridden_sources,
            )
            self.codegen_suffix(tx, stack_values_flat, pass1)

            # Use `pass1.uses` to selectively cache multi-user variables into a
            # temporary local source. This (a). speeds up loading VTs with long
            # chained source, and (b). avoids redundantly saving single-user VT
            # into a temporary local.
            tempvars = {}  # type: ignore[var-annotated]
            for val, count in pass1.uses.items():
                # If it's already a local source, no need to cache it
                if count > 1 and not istype(val, (SyntheticLocalSource, LocalSource)):
                    tempvars[val] = None
            pass2 = PyCodegen(
                self.root_tx,
                root,
                graph_output_var,
                tempvars=tempvars,
                overridden_sources=overridden_sources,
            )
            self.codegen_suffix(tx, stack_values_flat, pass2)

            if (
                torch._dynamo.config.log_graph_in_out_metadata
                and stack_values_flat
                and len(stack_values_flat) == 1
            ):
                vt = stack_values_flat[0]
                if (
                    isinstance(vt, torch._dynamo.variables.NamedTupleVariable)
                    and vt.tuple_cls
                    is torch._dynamo.functional_export.ExportTracerOutput
                ):
                    flat_returns = vt.items[0]
                    out_spec = vt.items[1]
                    assert isinstance(
                        flat_returns, torch._dynamo.variables.ListVariable
                    )

                    vt_to_graph_out_idx: dict[VariableTracker, int] = {}
                    for value in pass2.graph_outputs.values():
                        assert isinstance(value, torch._dynamo.codegen.GraphOutputEntry)
                        variable: VariableTracker = value.variable
                        vt_to_graph_out_idx[variable] = value.index

                    for idx, vt in enumerate(flat_returns.items):
                        if vt in vt_to_graph_out_idx:
                            self.export_metadata.output_return_type[idx] = (
                                "graph_out",
                                vt_to_graph_out_idx[vt],
                            )
                        elif (
                            vt.source is not None
                            and (source := getattr(vt.source, "base", None))
                            and source.is_input
                        ):
                            self.export_metadata.output_return_type[idx] = (
                                "input",
                                vt.source,
                            )
                        elif isinstance(vt, torch._dynamo.variables.ConstantVariable):
                            self.export_metadata.output_return_type[idx] = (
                                "constant",
                                vt.as_python_constant(),
                            )
                        else:
                            assert f"Encountered unrecognized type {vt} at output {idx}"  # noqa: PLW0129

                    self.export_metadata.out_spec = out_spec.as_python_constant()

            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(
                    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
                )

                if len(pass2.graph_outputs) != 0:
                    output.append(pass2.create_store(graph_output_var))
                    stored_graph_output_var = True
                else:
                    output.append(create_instruction("POP_TOP"))
            else:
                # NB: Important to run compiler collective even when there is
                # a graph break
                self.run_compiler_collective()
            self.add_output_instructions(output + pass2.get_instructions())

        # store all stack and locals for each frame
        # current state of the stack:
        # all cells,
        # *(frame N stack), *(frame N locals),
        # ...,
        # *(frame 1 stack), *(frame 1 locals)

        self.add_output_instructions(
            [
                create_instruction(
                    "BUILD_LIST",
                    arg=len(stack_values_flat) - all_stack_locals_metas[0].num_stack,
                ),
            ]
        )

        # current state of the stack:
        # all cells,
        # *(frame N stack), [
        #     *(frame N locals),
        #     *(frame N-1 stack), *(frame N-1 locals),
        #     ...
        #     *(frame 1 stack), *(frame 1 locals),
        # ]
        # iterate current frame (N) to root frame (1)
        # sliding window over frame stack/locals
        start_idx = 0
        end_idx = 0
        for i, meta in enumerate(all_stack_locals_metas):
            # do not pack frame N's stack into the value list
            n_vals = len(meta.locals_names)
            if i != 0:
                n_vals += meta.num_stack
            if n_vals == 0:
                self.add_output_instructions(
                    [
                        create_instruction("BUILD_LIST", arg=0),
                        *create_swap(2),
                    ]
                )
                # [], stack_values_flat
            else:
                end_idx += n_vals
                self.add_output_instructions(
                    [
                        create_dup_top(),
                        *create_binary_slice(start_idx, end_idx),
                        *create_swap(2),
                    ]
                )
                start_idx += n_vals
                # stack_values_flat[x:y], stack_values_flat

            # add root frame's unmodified locals here
            if i == len(all_stack_locals_metas) - 1:
                root_cg = PyCodegen(self.root_tx)
                unmodified_locals_names: dict[str, int] = {}
                for k, v in self.root_tx.symbolic_locals.items():
                    if isinstance(v.source, LocalSource) and v.source.local_name == k:
                        root_cg.append_output(root_cg.create_load(k))
                        unmodified_locals_names[k] = len(meta.locals_names) + len(
                            unmodified_locals_names
                        )
                self.add_output_instructions(
                    root_cg.get_instructions()
                    + [
                        create_instruction(
                            "BUILD_LIST", arg=len(unmodified_locals_names)
                        ),
                        # arg=2 because we already swapped the locals list back
                        create_instruction("LIST_EXTEND", arg=2),
                    ]
                )
                meta.locals_names.update(unmodified_locals_names)

            # *(frame N stack), metas[0] stack + locals, ..., metas[i] stack + locals, stack_values_flat

        # current state of the stack:
        # all cells,
        # *(frame N stack),
        # frame N locals,
        # frame N-1 stack, frame N-1 locals,
        # ...
        # frame 1 stack, frame 1 locals,
        # stack_values_flat
        #

        self.add_output_instructions(
            [
                create_instruction("POP_TOP"),
                create_instruction("BUILD_LIST", arg=len(all_stack_locals_metas)),
                *create_rot_n(all_stack_locals_metas[0].num_stack + 1),
            ]
        )

        # final state of the stack before running the unsupported bytecode:
        # all cells,
        # [
        #   [frame N locals],
        #   [frame N-1 stack + locals],
        #   ...,
        #   [frame 1 stack + locals],
        # ], *(frame N stack)

        if graph_output_var and stored_graph_output_var:
            self.add_output_instructions(
                [create_instruction("DELETE_FAST", argval=graph_output_var)]
            )

        if self.export:
            from torch.export._trace import _ExportModuleSpecTrackerDict

            potential_side_effects = []
            for var in self.side_effects._get_modified_vars():
                if hasattr(var, "mutation_type"):
                    mut_type = var.mutation_type
                    # Make sure to skip codegen specific mutations
                    if isinstance(
                        mut_type, (AttributeMutationExisting, ValueMutationExisting)
                    ):
                        if isinstance(var, UserDefinedDictVariable) and isinstance(
                            var.value, _ExportModuleSpecTrackerDict
                        ):
                            for k, v in var.items.items():
                                specs = {}
                                for k_spec, val in v.items.items():
                                    specs[k_spec.vt.as_python_constant()] = (
                                        val.as_python_constant()
                                    )
                                assert ["in_spec", "out_spec"] == list(specs.keys())
                                self.export_metadata.module_call_spec[
                                    k.vt.as_python_constant()
                                ] = specs
                        # export uses tracepoint pass to dump submodule inp/out spec
                        # into global state, so we filter it here
                        if not (
                            isinstance(var, UserDefinedDictVariable)
                            and isinstance(var.value, _ExportModuleSpecTrackerDict)
                        ):
                            potential_side_effects.append(var)

            side_effect_refs = [
                _get_source_debug_name(var.source) for var in potential_side_effects
            ]

            if len(side_effect_refs):
                warnings.warn(
                    f"While exporting, we found certain side effects happened in the model.forward. "
                    f"Here are the list of potential sources you can double check: {side_effect_refs}"
                )

        return all_stack_locals_metas

    def codegen_cells(self, tx: "InstructionTranslatorBase", cg: PyCodegen) -> None:
        # no need to codegen if reason.graph_break is False (since we won't resume)
        if self.compile_subgraph_reason.graph_break:
            tx_cnt = 0
            cur_tx: Optional[InstructionTranslatorBase] = tx
            while cur_tx is not None:
                # NOTE: we generate cells in the same order as resume_execution.py: sorted freevars + cellvars
                # Emitting `LOAD_FAST/LOAD_CLOSURE` with names in `co_freevars`
                # requires that in the generated bytecode, these cells would keep
                # their original local names, which we ensure via
                # `CellVariable.local_name`.
                freevars = tuple(sorted(cur_tx.cell_and_freevars()))
                for cell in freevars:
                    if cur_tx is self.root_tx:  # root frame
                        cg.append_output(cg.create_load_closure(cell))
                    else:  # nested frame
                        assert cur_tx.post_prune_cell_and_freevars
                        cg(cur_tx.post_prune_cell_and_freevars[cell])
                cg.append_output(create_build_tuple(len(freevars)))
                cur_tx = cur_tx.parent
                tx_cnt += 1
            cg.append_output(create_instruction("BUILD_LIST", arg=tx_cnt))
        else:
            cg.append_output(create_instruction("BUILD_LIST", arg=0))

    def codegen_suffix(
        self,
        tx: "InstructionTranslatorBase",
        stack_values: list[VariableTracker],
        cg: PyCodegen,
    ) -> None:
        # NOTE: `codegen_save_tempvars` must run first to update `source` fields
        # for variables with `AttributeMutationNew`, as they don't implement
        # `reconstruct` themselves.
        self.side_effects.codegen_save_tempvars(cg)
        if self.backward_state:
            assert not self.export
            for name, val in self.backward_state.items():
                cg(val)
                assert self.backward_state_var is not None
                cg.append_output(cg.create_load(self.backward_state_var))
                cg.store_attr(name)
        self.side_effects.codegen_hooks(cg)

        # TODO get debug_locals working for nested graph breaks
        # Return variables used for logging at the end
        for debug_var, args in tx.debug_locals:
            cg.add_push_null(lambda: cg(debug_var))
            for arg in args:
                cg(arg)
            cg.extend_output(create_call_function(len(args), False))
            cg.extend_output([create_instruction("POP_TOP")])

        # codegen cells before we apply side effects
        self.codegen_cells(tx, cg)

        cg.restore_stack(stack_values, value_from_source=not tx.export)
        self.side_effects.codegen_update_mutated(cg)

    def cleanup_graph(self) -> None:
        """
        Remove "creation_timestamp" from node meta

        Remove this pattern from the graph:
            torch._C._set_grad_enabled(False)
            torch._C._set_grad_enabled(True)
        """
        assert self.should_exit
        nodes = list(self.graph.nodes)
        for node in nodes:
            node.meta.pop("creation_timestamp", None)

        grad_enabled = torch.is_grad_enabled()
        for node1, node2 in itertools.pairwise(nodes):
            if (
                node1.target is torch._C._set_grad_enabled
                and tuple(node1.args) == (not grad_enabled,)
                and not node1._erased
            ):
                grad_enabled = node1.args[0]
                if (
                    node2.target is torch._C._set_grad_enabled
                    and tuple(node2.args) == (not grad_enabled,)
                    and not node2._erased
                ):
                    grad_enabled = node2.args[0]
                    self.graph.erase_node(node1)
                    self.graph.erase_node(node2)

    def bypass_package(self, reason: str = "", **kwargs: Any) -> None:
        """
        Do not save this output graph to the CompilePackage
        """
        if not self.package:
            return
        if torch._dynamo.config.strict_precompile:
            raise torch._dynamo.exc.PackageError(
                "Detected a package bypass: %s", reason
            )
        log.warning("Detected a package bypass: %s", reason)
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "precompile_cache_bypass",
                "encoding": "json",
            },
            payload_fn=lambda: {
                # precede with underscore so it always appear first in JSON in tlparse
                "_reason": reason,
                **kwargs,
            },
        )
        self.package.bypass_current_entry()
        self.package = None

    def get_graph_sizes_structured(self) -> dict[str, list[Union[int, str]]]:
        ret: dict[str, list[Union[int, str]]] = {}
        for node in self.graph.nodes:
            example_value = node.meta.get("example_value", None)
            if isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()
                ret[node.name] = [s if isinstance(s, int) else repr(s) for s in size]
        return ret

    def get_graph_sizes(self, name: str) -> str:
        graph_sizes_str = "TRACED GRAPH TENSOR SIZES\n"
        graph_sizes_str += f"===== {name} =====\n"
        for node in self.graph.nodes:
            example_value = node.meta.get("example_value", None)
            if isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()
                graph_sizes_str += f"{node.name}: {tuple(size)}\n"
                concrete_size = []
                has_symint = False
                for sz in size:
                    if isinstance(sz, int):
                        concrete_size.append(sz)
                    elif isinstance(sz, torch.SymInt):
                        has_symint = True
                        concrete_size.append(sz.node.hint)
                    else:
                        break
                else:
                    if has_symint:
                        graph_sizes_str += (
                            f"{node.name} (concrete): {tuple(concrete_size)}\n"
                        )
        return graph_sizes_str

    @contextlib.contextmanager
    def restore_global_state(self) -> Any:
        """
        Momentarily restores the global state to what it was prior to tracing the current output
        """
        prior_global_state = self.tracing_context.global_context.copy_graphstate()
        current_global_state: dict[str, tuple[Any, bool]] = {}
        self.save_global_state(out=current_global_state)
        try:
            # Set to state prior to tracing the graph
            self.tracing_context.global_context.restore_graphstate(prior_global_state)
            yield
        finally:
            # Reset to state at the current time (e.g. before calling the user compiler)
            self.tracing_context.global_context.restore_graphstate(
                GlobalContextCheckpointState(current_global_state)
            )

    def run_compiler_collective(self) -> None:
        tx = self.root_tx
        assert tx is not None
        if (ds := tx.distributed_state) is not None and ds.all_states is None:
            # pyrefly: ignore  # unbound-name
            compile_pg = ds.compile_pg
            # pyrefly: ignore  # unbound-name
            log.info("compiler_collective %s", ds.local_state)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "compiler_collective",
                    "encoding": "string",
                },
                # pyrefly: ignore  # unbound-name
                payload_fn=lambda: ds.local_state.render(),
            )
            device_types = compile_pg._device_types
            assert len(device_types) == 1, (
                "Expect only one device type but got {}".format("+".join(device_types))
            )
            with (
                get_interface_for_device(device_types.pop()).device(  # type: ignore[attr-defined]
                    compile_pg.rank() % torch.accelerator.device_count()
                ),
                dynamo_timed("compiler_collective", log_pt2_compile_event=True),
            ):
                all_states: list[Any] = [None] * compile_pg.size()
                # pyrefly: ignore  # unbound-name
                dist.all_gather_object(all_states, ds.local_state, group=compile_pg)
                # pyrefly: ignore  # unbound-name
                ds.all_states = all_states
            # Clear speculation log, because are tracing may diverge due to
            # this information from the compiler collective
            tx.speculation_log.clear()
            raise exc.CompileCollectiveRestartAnalysis

    def compile_and_call_fx_graph(
        self,
        tx: "InstructionTranslatorBase",
        rv: list[VariableTracker],
        root: FakeRootModule,
    ) -> list[Instruction]:
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.

        Code is generated w.r.t. self.root_tx.
        tx is only used for preserving GraphModule metadata
        """
        with torch._guards.TracingContext.clear_frame():
            from .decorators import disable

            assert self.should_exit

            self.run_compiler_collective()
            if count_calls(self.graph) == 0 and len(rv) == 0:
                return []

            name = unique_id("__compiled_fn", with_uuid=True)

            assert isinstance(rv, list)
            assert isinstance(root, FakeRootModule)

            output_node = self.create_node(
                "output",
                "output",
                (self.current_tracer.create_arg(tuple(x.as_proxy() for x in rv)),),
                {},
            )
            sub_gms = self.dedup_pass()
            root.add_nn_modules(sub_gms)  # type: ignore[arg-type]

            self.current_tracer._maybe_preserve_original_meta(tx, output_node)
            if not config.do_not_emit_runtime_asserts:
                # There is a rare scenario where codegen_suffix adds a new entry
                # to self.nn_modules while `root` knows only about the
                # nn_modules at the time of its creation. This causes failures
                # while creating the graph module because self.graph and root
                # are out of sync. This only happens for `get_attr` nodes, so
                # here we clean up the get_attr nodes that are unused.
                self.remove_unused_get_attr_nodes()
                insert_deferred_runtime_asserts(
                    fx.GraphModule(root, self.graph),
                    self.shape_env,
                    name,
                    export=self.export,
                )
            # NB: deferred runtime asserts can keep graphargs live, so make sure
            # those are inserted before pruning
            self.remove_unused_graphargs()
            ncalls = count_calls(self.graph)
            counters["stats"]["calls_captured"] += ncalls

            self.remove_tensorify_specialized_graphargs()

            # free a bit of memory
            self.real_value_cache.clear()

            gm = _make_graph_module(root, self.graph)

            # Saved tensors hooks are not used by the graph.
            # GraphModule by default only copies used in the graph submodules.
            # Copying them into the result graph manually.
            if self.saved_tensors_hooks_subgraph_names:
                for subgraph_name in self.saved_tensors_hooks_subgraph_names:
                    setattr(gm, subgraph_name, getattr(root, subgraph_name))

            for register_finalizer in self.register_finalizer_fns:
                register_finalizer(gm)

            if next(gm.parameters(), None) is not None:
                # If dynamo produces a graph with parameters, skip package stuff
                # Bypass output graph
                self.bypass_package(
                    "Graph contains named parameters: either inline_inbuilt_nn_modules=False or there are static addresses.",
                    inline_builtin_nn_modules=torch._dynamo.config.inline_inbuilt_nn_modules,
                    gm=gm.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    ),
                )

            if self.package is not None:
                gm._backend_id = name

            gm.compile_subgraph_reason = self.compile_subgraph_reason
            gm.meta["dynamo_flat_name_to_original_fqn"] = (
                self.dynamo_flat_name_to_original_fqn.copy()
            )
            gm.meta["dynamo_compile_id"] = self.dynamo_compile_id
            gm.meta["backend_id"] = name

            graph_code_log.debug(
                "%s",
                lazy_format_graph_code(
                    name, gm, include_stride=True, include_device=True, colored=True
                ),
            )
            torch._logging.trace_structured(
                "dynamo_output_graph",
                lambda: {"sizes": self.get_graph_sizes_structured()},
                payload_fn=lambda: gm.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            self.call_cleanup_hooks()
            old_fake_mode = self.tracing_context.fake_mode
            assert old_fake_mode is not None
            if not self.export:
                import torch._functorch.config as _config

                with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                    # TODO(voz): The way export uses gm, and fake tensors, is not supported with us resetting
                    backend_fake_mode = torch._subclasses.FakeTensorMode(
                        shape_env=old_fake_mode.shape_env,
                    )
                # TODO(voz): Ostensibily, this should be scoped and
                # restore back to old_fake_mode, but doing so currently violates
                # a lot of fake_tensor ownership assumptions and runs afoul of detect_fake_mode
                self.tracing_context.fake_mode = backend_fake_mode

            with self.restore_global_state():
                compiled_fn = self.call_user_compiler(gm, self.example_inputs())

            from torch.fx._lazy_graph_module import _LazyGraphModule

            if isinstance(compiled_fn, _LazyGraphModule) or (
                isinstance(getattr(compiled_fn, "__self__", None), _LazyGraphModule)
                and compiled_fn.__name__ == "_lazy_forward"  # type: ignore[attr-defined]
            ):
                # Since dynamo will run the forward method for the GraphModule shortly
                # anyways, it does not hurt to do the real recompilation here if
                # this is a _LazyGraphModule. This makes it easier for dynamo to
                # optimize a _LazyGraphModule.

                lazy_gm = (
                    compiled_fn
                    if isinstance(compiled_fn, _LazyGraphModule)
                    else compiled_fn.__self__  # type: ignore[attr-defined]
                )

                _LazyGraphModule.force_recompile(lazy_gm)

                if not isinstance(compiled_fn, _LazyGraphModule):
                    # replace compiled_fn with the real forward method
                    compiled_fn = lazy_gm.forward

            if self.package is not None:
                self.package.add_backend_id(name, compiled_fn)

            compiled_fn = disable(
                compiled_fn, reason="do not trace Dynamo-compiled graph"
            )

            counters["stats"]["unique_graphs"] += 1
            assert old_fake_mode.shape_env is not None
            if specializations := old_fake_mode.shape_env.specializations:
                specialization_guards = []
                specialization_cache: dict[Specialization, Callable[[Any], Any]] = {}
                sources = [a.source for a in self.graphargs]
                for specialization in specializations:
                    source_index = sources.index(specialization.source)
                    check_fn_source = inspect.getsource(specialization.check_fn).strip()
                    # Required because the LABDA_GUARD API requires a root guard manager
                    unused_root_guard_manager = RootGuardManager()
                    check_fn = guards.LAMBDA_GUARD(  # type: ignore[attr-defined]
                        unused_root_guard_manager,
                        specialization.check_fn,
                        [check_fn_source],
                    )

                    log.debug(
                        "Compiling backend specialized graph with specialization=%s",
                        check_fn_source,
                    )

                    specialization_guards.append(
                        (
                            functools.partial(
                                lambda idx, args, check_fn=check_fn: check_fn(
                                    args[idx]
                                ),
                                source_index,
                            ),
                            specialization,
                        )
                    )

                @torch._dynamo.disable(reason="do not trace Dynamo-compiled graph")  # type: ignore[misc]
                def specialized_dispatch(*args: Any, **kwargs: Any) -> Any:
                    for check_fn, specialization in specialization_guards:
                        if check_fn(args):
                            if specialization in specialization_cache:
                                return specialization_cache[specialization](
                                    *args, **kwargs
                                )

                            with self.shape_env.patch_source_specialization(
                                specialization.source, specialization.check_fn
                            ):
                                # Modify gm so AOTAutogradCache key changes per specialization
                                gm.meta["specialization"] = specialization
                                example_inputs: list[Tensor] = list(args)
                                with tracing(self.tracing_context):
                                    specialization_cache[specialization] = (
                                        self.call_user_compiler(gm, example_inputs)
                                    )

                            return specialization_cache[specialization](*args, **kwargs)
                    return compiled_fn(*args, **kwargs)

                # This is safe because we pre-process name to be unique
                self.install_global_unsafe(name, specialized_dispatch)
            else:
                # This is safe because we pre-process name to be unique
                self.install_global_unsafe(name, compiled_fn)

            assert self.root_tx is not None
            cg = PyCodegen(self.root_tx)

            for idx, arg in enumerate(self.graphargs):
                self.export_metadata.graph_input_idx_to_local_source[idx] = arg.source

            cg.make_call_generated_code(name)
            return cg.get_instructions()

    @property
    def placeholders(self) -> list[fx.Node]:
        return self.graph.find_nodes(op="placeholder")

    @property
    def graphargs(self) -> list[GraphArg]:
        return [node.meta["grapharg"] for node in self.placeholders]

    def call_user_compiler(
        self, gm: fx.GraphModule, example_inputs: list[Tensor]
    ) -> CompiledFn:
        with dynamo_timed(
            "OutputGraph.call_user_compiler",
            phase_name="backend_compile",
            log_pt2_compile_event=True,
            log_waitcounter=True,
            waitcounter_name_override="compile_aot_autograd",
            dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
        ):
            return self._call_user_compiler(gm, example_inputs)

    def _call_user_compiler(
        self, gm: fx.GraphModule, example_inputs: list[Tensor]
    ) -> CompiledFn:
        assert self.compiler_fn is not None
        tot = 0
        placeholders = []
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                tot += 1
            if node.op == "placeholder":
                placeholders.append(node)
        increment_op_count(tot)
        for pl in placeholders:
            if not hasattr(pl, "_dynamo_source"):
                arg = pl.meta["grapharg"]
                # TODO: Why isn't this stored in meta :think:
                # NOTE: can't move these into meta: https://github.com/pytorch/pytorch/issues/141640
                pl._dynamo_source = arg.source

        # NOTE: can't move these into meta: https://github.com/pytorch/pytorch/issues/141640
        gm._param_name_to_source = self.param_name_to_source  # type: ignore[assignment]
        gm._source_to_user_stacks = self.source_to_user_stacks  # type: ignore[assignment]

        name = (
            self.compiler_fn.__name__
            if hasattr(self.compiler_fn, "__name__")
            else "<unknown compiler_fn>"
        )
        try:
            _step_logger()(logging.INFO, f"calling compiler function {name}")
            compiler_fn = self.compiler_fn
            if config.verify_correctness:
                compiler_fn = WrapperBackend(compiler_fn)
            compiled_fn = compiler_fn(gm, example_inputs)
            _step_logger()(logging.INFO, f"done compiler function {name}")
            assert callable(compiled_fn), "compiler_fn did not return callable"
        except (TensorifyScalarRestartAnalysis, ShortenTraceback):
            raise
        except exceptions_allowed_to_be_fallback as e:
            if self.has_user_defined_allowed_in_graph:
                raise BackendCompilerFailed(
                    self.compiler_fn, e, inspect.currentframe()
                ).with_traceback(e.__traceback__) from None
            unimplemented_v2_with_warning(
                e,
                self.root_tx.f_code,
                gb_type="Backend compiler exception",
                context=f"Backend: {name}\nException:{str(e)}\nTraceback:\n{self.root_tx.format_frame_summary()}",
                explanation=f"Backend compiler `{name}` failed with {str(e)}. Adding a graph break.",
                hints=[
                    "Report an issue to the backend compiler repo.",
                ],
            )
        except SkipFrame as e:
            # The backend compiler has requested that we skip the frame, instead of
            # aborting execution.
            raise e
        except Exception as e:
            raise BackendCompilerFailed(
                self.compiler_fn, e, inspect.currentframe()
            ).with_traceback(e.__traceback__) from None

        signpost_event(
            "dynamo",
            "OutputGraph.call_user_compiler",
            {
                **self.co_fields,
                "op_count": tot,
                "node_count": len(gm.graph.nodes),
                "input_count": len(placeholders),
            },
        )

        # pyrefly: ignore  # unbound-name
        return compiled_fn

    def dedup_pass(self) -> dict[str, torch.fx.GraphModule]:
        if torch._dynamo.config.use_graph_deduplication:
            return apply_graph_deduplication(self)
        else:
            return {}

    def install_subgraph(self, name: str, sub_gm: torch.fx.GraphModule) -> str:
        next_name = get_unique_name_wrt(name, self.nn_modules, requires_suffix=True)
        sub_gm.__name__ = next_name  # type: ignore[assignment]
        sub_gm.torchdynamo_force_dynamic = False  # type: ignore[assignment]
        # This graph module is not present in the user space, so it can't be
        # accessed by a source. Set source=None.
        self.register_attr_or_module(sub_gm, next_name, source=None)
        return next_name

    def example_inputs(self) -> list[torch.Tensor]:
        result = [arg.example for arg in self.graphargs]
        return result

    def remove_unused_get_attr_nodes(self) -> None:
        for node in sorted(self.graph.find_nodes(op="get_attr"), reverse=True):
            if len(list(node.users)) == 0:
                self.remove_node(node)

    def remove_unused_graphargs(self) -> None:
        # NB: It's OK to drop GraphArg for symbols that ended up being
        # specialized iff they are not used in runtime assertions.  You don't
        # even have to make a guard for it, because ShapeEnv produce_guards
        # operates on tracked_fakes, which never gets pruned.
        # That being said, you'll get marginally better generated
        # guard code if you promote the guard into a Dynamo guard (since that
        # allows for the guard to be done using C++ guards.)  If we get
        # ShapeEnv guards to go into C++ guards, this will stop being a thing
        # though!

        assert self.should_exit

        # Miniature DCE pass, but only for obviously trivial operations
        def is_static_true(b_node: fx.node.Argument) -> bool:
            if b_node is True:
                return True
            if not isinstance(b_node, fx.Node):
                return False
            b = b_node.meta.get("example_value")
            if b is None:
                return False
            if b is True:
                return True
            if (
                isinstance(b, torch.SymBool)
                and (r := b.node.maybe_as_bool()) is not None
            ):
                # pyrefly: ignore  # unbound-name
                return r
            # TODO: We can also technically remove all cases when the input
            # doesn't have unbacked inputs, since it's all in the ShapeEnv
            return False

        def is_symnode_arg(a: fx.node.Argument) -> bool:
            from torch.fx.experimental.sym_node import SymTypes

            if isinstance(a, (int, float, bool)):
                return True
            if isinstance(a, fx.Node):
                return isinstance(a.meta.get("example_value"), SymTypes)
            return False

        # NB: We assume that you cannot do mutations on int/float/bool,
        # because they are immutable types, and therefore is always safe to
        # DCE.
        def is_symnode_compute_node(node: fx.Node) -> bool:
            from torch.fx.experimental.sym_node import SymTypes

            if node.op != "call_function":
                return False
            # TODO: I don't think it's possible to have a bare int/float here?
            if not isinstance(node.meta.get("example_value"), SymTypes):
                return False
            # TODO: This will bail here if you ever end up with a more complicated
            # computation function, like sum(list_of_ints), even though it
            # should be DCE'able
            if not all(is_symnode_arg(a) for a in node.args):
                return False
            if not all(is_symnode_arg(a) for a in node.kwargs.values()):
                return False
            return True

        from torch.fx.experimental.symbolic_shapes import is_accessor_node

        for node in reversed(list(self.graph.nodes)):
            if len(list(node.users)) == 0:
                if (
                    node.op == "get_attr"
                    or (node.op == "call_function" and node.target is operator.getitem)
                    or (
                        node.op == "call_function"
                        and node.target is torch._check
                        and is_static_true(node.args[0])
                    )
                    or is_symnode_compute_node(node)
                    or is_accessor_node(node)
                ):
                    self.remove_node(node)

        def placeholder_binds_symbol(node: fx.Node) -> Optional[sympy.Symbol]:
            arg = node.meta["grapharg"]
            example = arg.example
            if isinstance(example, torch.SymInt) and isinstance(
                example.node.expr, sympy.Symbol
            ):
                return example.node.expr
            return None

        def remove_unused(node: fx.Node) -> None:
            log.debug("REMOVE UNUSED GRAPHARG %s", node.meta["grapharg"].source.name())
            # I'm not really sure why you need to delete these from the
            # node since the node is going to get removed
            del node.meta["grapharg"]
            self.remove_node(node)
            self.real_value_cache.pop(node, None)

        used_symbols: set[sympy.Symbol] = set()

        def update_used_symbols(
            used_symbols: set[sympy.Symbol], fake: Union[torch.SymInt, torch.Tensor]
        ) -> None:
            used_symbols |= free_symbols(fake)

        recheck_placeholders = []
        for node in self.placeholders:
            binds_symbol = placeholder_binds_symbol(node) is not None
            # Don't delete symbol bindings yet
            if binds_symbol:
                if not node.users:
                    recheck_placeholders.append(node)
            else:
                if not node.users and not isinstance(
                    node.meta["grapharg"], BackwardStateGraphArg
                ):
                    remove_unused(node)
                else:
                    # Register the free symbols as uses
                    arg = node.meta["grapharg"]
                    if isinstance(arg, BackwardStateGraphArg):
                        continue
                    if isinstance(node.meta["grapharg"].example, torch.ScriptObject):
                        real_script_obj = node.meta["grapharg"].example
                        fake_script_obj = node.meta["grapharg"].example_strong_ref
                        if not torch._library.fake_class_registry.tracing_with_real(
                            real_script_obj
                        ):
                            flat_dict = dict(real_script_obj.__obj_flatten__())  # type: ignore[attr-defined]
                            for attr in flat_dict.keys():
                                fake_attr_val = getattr(
                                    fake_script_obj.wrapped_obj, attr
                                )
                                pytree.tree_map_only(
                                    (torch.SymInt, torch.Tensor),
                                    lambda t: update_used_symbols(used_symbols, t),
                                    fake_attr_val,
                                )
                        continue
                    fake = (
                        arg.fake_tensor if arg.fake_tensor is not None else arg.example
                    )
                    update_used_symbols(used_symbols, fake)

        # After removing unused graphargs, prune unused binds_symbol
        for node in recheck_placeholders:
            symbol = placeholder_binds_symbol(node)
            if symbol is not None:
                if symbol not in used_symbols:
                    remove_unused(node)
                else:
                    # Make sure we delete later occurrences of the same symbol
                    used_symbols.remove(symbol)

    def remove_tensorify_specialized_graphargs(self) -> None:
        # This is a pretty interesting function. Basically we have this problem
        # where our compiler tends to choke when we have unused inputs. The way
        # we support dynamic float arguments is by doing a joint fx pass and
        # tensorifying away as many symfloats as we can. For the remaining symfloats
        # we have no choice but to specialize... HOWEVER at that point in time
        # we can no longer remove graph inputs. So our sledgehammer solution is to
        # save the state of what inputs we should have specialized in dynamo and
        # restart analysis. This function incorporates this "view from the future"
        # state and specializes inputs that we know we won't be able to tensorify
        # away in the joint pass. In principle we shouldn't choke on unused inputs
        # and so this shouldn't be necessary. In practice CUDA graphs choke on
        # unused inputs so we need this for now.

        # Import here to prevent circular import
        from torch._dynamo.symbolic_convert import TensorifyState

        for node in self.graph.nodes:
            example_value = node.meta.get("example_value")
            if (
                isinstance(example_value, FakeTensor)
                and example_value.item_memo is not None
                and hasattr(example_value.item_memo.node._expr, "name")
                and all(u.target == "item" for u in node.users)
                and TensorifyState.should_specialize(
                    # We use _expr instead of expr b/c we want the symbol not the replacement
                    example_value.item_memo.node._expr.name
                )
            ):
                for u in list(node.users):
                    u.replace_all_uses_with(guard_scalar(example_value.item_memo))
                    self.remove_node(u)
                self.remove_node(node)

    def add_output_instructions(self, prefix: list[Instruction]) -> None:
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.
        """
        self.output_instructions.extend(prefix)
        self.should_exit = True

    def install_global_unsafe(self, name: str, value: Any) -> None:
        """
        WARNING: prefer the safer `install_global_by_id/install_global`.
        torch.compile instances should be independent of each other;
        one footgun is to have one instance depend on the existence of
        a global installed by another instance. This can happen if we mangle
        a global the same way across both instances.
        """
        assert name not in self.installed_globals
        self.installed_globals.add(name)
        self.cleanups.append(CleanupHook.create(self.global_scope, name, value))

    def install_global_by_id(self, prefix: str, value: Any) -> str:
        """
        Installs a global if it hasn't been installed already.
        This is determined by (prefix, id(value)) pair.

        Returns the name of the newly installed global.
        """
        # NB: need self.compile_id to distinguish this global
        # from another global created in a different torch.compile instance
        name = f"{prefix}_{id(value)}_c{self.compile_id}"
        if name in self.installed_globals:
            return name
        self.install_global_unsafe(name, value)
        return name

    def install_global(self, prefix: str, value: Any) -> str:
        """
        Installs a global, generating a unique name for it.

        Returns the name of the newly installed global.
        """
        # NB: unique_id is unique, even across torch.compile instances
        name = unique_id(prefix)
        self.install_global_unsafe(name, value)
        return name

    def cleanup(self) -> None:
        # There is a reference cycle between tracer and OutputGraph, causing
        # some of the tensor objects to be held alive for longer than necessary.
        self.root_tx = None  # type: ignore[assignment]
        self.nn_modules.clear()
        self.used_inlined_inbuilt_modules_names.clear()
        self.param_name_to_source = None

        for node in self.graph.nodes:
            if "grapharg" in node.meta:
                del node.meta["grapharg"]
        self.real_value_cache.clear()
        self.input_name_to_proxy.clear()
        self.side_effects.clear()
        self.variable_tracker_cache.clear()
        self.register_finalizer_fns.clear()
        self.dynamo_flat_name_to_original_fqn.clear()
        self.tracing_context.clear()
        self.input_source_to_var.clear()
        self.unspec_variable_map.clear()
        self.backward_state.clear()

    def add_graph_finalizer(
        self, register_finalizer: Callable[[fx.GraphModule], None]
    ) -> None:
        self.register_finalizer_fns.append(register_finalizer)

    def example_value_from_input_node(self, node: torch.fx.Node) -> Any:
        """Extract the non-fake example tensor"""
        if node.op == "placeholder":
            return node.meta["grapharg"].example
        assert node.op == "get_attr"
        return self.nn_modules[node.target]  # type: ignore[index]

    def add_fqn_info_for_inlined_modules(
        self, inlined_module: torch.nn.Module, source: Source
    ) -> None:
        name = OutputGraph.module_key_name(source.name())
        name = get_unique_name_wrt(
            name, self.used_inlined_inbuilt_modules_names, self.global_scope
        )
        self.used_inlined_inbuilt_modules_names.add(name)

        def register_leaf_name(leaf_name: str) -> None:
            assert self.param_name_to_source is not None
            new_source = ParamBufferSource(source, leaf_name)
            new_name = f"{name}.{leaf_name}"
            self.param_name_to_source[new_name] = new_source
            if isinstance(source, LocalSource):
                self.dynamo_flat_name_to_original_fqn[
                    OutputGraph.module_key_name(new_source.name())
                ] = leaf_name

        # annoying, but there are cases when we do not have parameters
        # see test_nn_moduledict_contains
        if hasattr(inlined_module, "_parameters"):
            if (
                callable(inlined_module.named_parameters)
                and inlined_module.named_parameters.__func__  # type: ignore[attr-defined]
                is og_module_named_parameters_fn_ptr
            ):
                for leaf_name, _ in inlined_module.named_parameters():
                    register_leaf_name(leaf_name)
        if hasattr(inlined_module, "_buffers"):
            if (
                callable(inlined_module.named_buffers)
                and inlined_module.named_buffers.__func__  # type: ignore[attr-defined]
                is og_module_named_buffers_fn_ptr
            ):
                for leaf_name, _ in inlined_module.named_buffers():
                    register_leaf_name(leaf_name)


class DynamoTracerOutput:
    error_on_graph_break: bool
    is_tracing_resume_prologue: bool
    output_graph: Optional[OutputGraph]

    def __init__(
        self, tracer: "InstructionTranslatorBase", error: Optional[Any] = None
    ) -> None:
        self.error_on_graph_break = tracer.error_on_graph_break
        self.is_tracing_resume_prologue = tracer.is_tracing_resume_prologue
        if error:
            self.output_graph = None
        else:
            self.output_graph = tracer.output


err_epilogue = (
    "With the current config, we will graph break "
    "(and fall back to eager-mode PyTorch) on all ops "
    "that have do not have the 'pt2_compliant_tag'. "
    "Please see the following doc for how to mark this op as PT2 compliant "
    "https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html"
)


def check_pt2_compliant_op(
    output_graph: OutputGraph, kind: str, target: Any, args: Any, kwargs: Any
) -> None:
    if kind != "call_function":
        return

    def encountered_compliant_op(target: torch._ops.OpOverload) -> None:
        if target.namespace in {"prim", "prims", "aten"}:
            return
        output_graph.compliant_custom_ops.add(target)

    def encountered_non_compliant_op(target: torch._ops.OpOverload, msg: str) -> None:
        output_graph.non_compliant_ops.add(target)
        if config.only_allow_pt2_compliant_ops:
            unimplemented_v2(
                gb_type="Encountered non-PT2-compliant op",
                context="",
                explanation=msg + " " + err_epilogue,
                hints=[],
            )

    if isinstance(target, torch._ops.OpOverload):
        if torch.Tag.pt2_compliant_tag in target.tags:
            encountered_compliant_op(target)
            return
        encountered_non_compliant_op(
            target,
            f"Encountered the torch.ops.OpOverload {target} that is not PT2 compliant.",
        )
        return

    if isinstance(target, torch._ops.OpOverloadPacket):
        overloads = tuple(target.overloads())
        # Optimization: Overload resolution is expensive.
        # If there's only one overload, we know what it will resolve to.
        if len(overloads) == 1:
            op = getattr(target, overloads[0])
            if torch.Tag.pt2_compliant_tag in op.tags:
                encountered_compliant_op(op)
                return
            encountered_non_compliant_op(
                op,
                f"Encountered the non-overloaded "
                f"torch.ops.OpOverloadPacket {target} "
                f"that is not PT2 compliant. ",
            )
            return

        args, kwargs = torch._dynamo.utils.get_fake_values_from_nodes(
            output_graph.current_tx, (args, kwargs), False
        )
        try:
            overload = torch._C._jit_resolve_packet(
                target._qualified_op_name, *args, **kwargs
            )
        except RuntimeError as e:
            unimplemented_v2(
                gb_type="Error when attempting to resolve op packet",
                context="",
                explanation=str(e),
                hints=[],
            )

        # pyrefly: ignore  # unbound-name
        op = getattr(target, overload)
        if torch.Tag.pt2_compliant_tag in op.tags:
            encountered_compliant_op(op)
        else:
            encountered_non_compliant_op(
                op,
                f"Encountered the torch.ops.OpOverloadPacket {target} "
                # pyrefly: ignore  # unbound-name
                f"which resolves to the overload ({overload}) that is "
                f"not PT2 compliant.",
            )


_compile_id_counter = itertools.count()

P = ParamSpec("P")
R = TypeVar("R")


class LazyProxy:
    def __init__(
        self,
        tracer: "SubgraphTracer",
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        self.tracer = tracer
        # pyrefly: ignore  # invalid-type-var
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.fn(*self.args, **self.kwargs)


class SubgraphTracer(fx.Tracer):
    """
    Holds an FX graph that is being traced. OutputGraph owns a SubgraphTracer
    and the separation of responsibilities is that SubgraphTracer is
    responsible for building the graph while OutputGraph is responsible for
    compiling and executing the graph.
    """

    def __init__(
        self,
        output_graph: "OutputGraph",
        parent: Optional["SubgraphTracer"] = None,
        is_export: bool = False,
        source_target: Optional[Target] = None,
    ) -> None:
        super().__init__()
        self.output_graph = weakref.proxy(output_graph)
        self.graph = torch.fx.Graph()

        # See note [Export inputs must be explicitly passed in]
        self.is_export = is_export
        # Map from graph input name to its placeholder proxy object, where the
        # map's keys give all current placeholder node names and can be used to
        # create unique node names
        self.input_name_to_proxy: dict[str, fx.Proxy] = {}
        # Node => computed real value (see utils.get_real_value)
        self.real_value_cache: dict[fx.Node, torch.Tensor] = {}

        # SubgraphTracers can be nested. See NOTE [HigherOrderOperator tracing design]
        self.parent = parent
        self.source_target = source_target
        # A dict mapping previously free variables (Proxy objects)
        # to new Proxy objects that wrap inputs to this subgraph.
        #
        # This dict maps proxies in outer graphs to placeholders in current graph.
        # It serves two purposes:
        # - Proxies are associated with VariableTrackers. If we see
        # the same VariableTracker twice (and it is a free variable),
        # then we want to use the same Proxy in the current subgraph to
        # record the tracing.
        # - If we are tracing a HigherOrderOperator's body_fn, then we
        # need to keep track of what free variables were lifted so we can
        # rewrite the HigherOrderOperator call using the traced body_fn.
        # Dicts maintain the order of args for the HigherOrderOperator call.
        self.lifted_freevars: dict[fx.Proxy, fx.Proxy] = {}

        # map basic symbols (unbacked and unbacked) to their bound proxies.
        # There are only two cases where bound_symbols will be recorded:
        # 1. when we create_graph_input for a backed SymInt that's basic symbol
        # 2. when we track_produced_symints for intermediate results
        # bound_symbols always map the symbol to the proxy whose
        # tracer is the current tracer that's readily accessible in current tracer's graph.
        self.bound_symbols: dict[sympy.Symbol, Union[torch.fx.Proxy, LazyProxy]] = {}

        # Maps _DynamicScalar object ids to allocated SymInt nodes, for symbol reuse
        self.dynamic_scalar_nodes: dict[int, torch.SymInt] = {}

        self.prev_inst = None
        # True if this tracer is currently tracing into torch.utils.checkpoint
        # as part of speculate_subgraph.
        self.under_activation_checkpoint = False
        # True if we want to allow externally visible side-effects (doesn't throw error on their existence)
        # during this tracer's tracing of torch.utils.checkpoint (via speculate_subgraph).
        # Only safe if we know for sure that *NOT* replaying these side-effects during
        # backward recomputation of the checkpoint region doesn't affect its correctness.
        self.allow_side_effects_under_checkpoint = False
        # True if we want to allow externally visible side-effects (doesn't throw error on their existence)
        # during this tracer's tracing. This is currently only used by experimental AC out-of-tree
        # via torch._dynamo.utils._disable_side_effect_safety_checks_for_current_subtracer.
        # Note: Externally visible side-effects are allowed if this flag OR the above flag is True.
        self.unsafe_allow_externally_visible_side_effects = False

        # True if this tracer is currently tracing (reconstructing) into a Python generator
        self.is_reconstructing_generator = False

        self.debug_level: int = parent.debug_level + 1 if parent is not None else 0

        self._cur_code = None
        self._orig_gm_meta: Optional[list[Any]] = None
        self._orig_gm_lineno_map: Optional[dict[int, Optional[int]]] = None
        self._orig_gm_firstlineno: Optional[int] = None
        # Each SubgraphTracer is associated with a source target, which indicates
        # which operator this subgraph is attached to. We compute a source_fn_stack
        # based on the source target. For the root tracer, it's set to [].
        # This is useful for debugging and transforming the exported graph.
        if self.parent is None:
            self.source_fn_stack: list[Any] = []
        else:
            self.source_fn_stack = self.parent.source_fn_stack + [
                (self.graph._target_to_str(source_target), source_target)
            ]

        # This is used to create a unique name for the placeholder
        self._used_names: OrderedSet[str] = OrderedSet()
        # Stores the versions of the input tensors at the time they are inserted
        # as placeholders in the graph. This is used to track input mutation.
        self._input_versions_at_beginning: list[int] = []
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Inference mode is supposed to be disabled during compilation. Please open an issue."
            )

    # preserve original meta if it is available
    def _maybe_preserve_original_meta(
        self, tx: "InstructionTranslatorBase", node: fx.Node
    ) -> None:
        if (
            self._orig_gm_meta
            and self._orig_gm_lineno_map
            and self._orig_gm_firstlineno
        ):
            lineno = tx.current_instruction.starts_line
            node_idx = None
            if lineno is not None:
                node_idx = self._orig_gm_lineno_map.get(
                    lineno - self._orig_gm_firstlineno, None
                )
            if node_idx is not None:
                meta = self._orig_gm_meta[node_idx]
                for field in fx.proxy._COPY_META_FIELDS:
                    if field in meta:
                        node.meta[field] = meta[field]
                if "stack_trace" in meta:
                    node.meta["stack_trace"] = meta["stack_trace"]

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: Any,
        kwargs: Any,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Optional[Callable[[fx.Node], fx.Proxy]] = None,
    ) -> fx.Proxy:
        # NOTE: [Nested SubgraphTracer and free_variable handling]
        # --------------------------------------------------------
        # Read NOTE [HigherOrderOperator tracing design] first.
        #
        # Let's say we're in the middle of introspecting the body of a possibly
        # nested HigherOrderOperator, and we see a free variable.
        #
        # There are two cases:
        # 1. We see a free variable that is already tracked by Dynamo.
        # 2. We see a free variable that has not been tracked by Dynamo
        #
        # In case 1, we call `maybe_lift_tracked_freevar_to_input` (below)
        # which will lift the freevar to be an input of this subgraph
        # and also recursively lift it to be an input on the parent(s).
        #
        # In case 2, before the call to `create_proxy`, the InstructionTranslator
        # will see the freevar when it gets loaded by Python bytecode.
        # E.g. for Python 3.11 the bytecodes that may do this are LOAD_DEREF or
        # LOAD_GLOBAL.
        # There, the InstructionTranslator asks Dynamo to begin tracking the
        # freevar by building a new Variable.
        # Building a new Variable automatically lifts the freevar to be an
        # input of the root SubgraphTracer.
        #
        # The implications for the code below are:
        # - We will always be in Case 1 when we get to this code.
        # - Any "free variable" we encounter here is guaranteed to already be
        #   bound, that is, it is either a graph input of the root graph, or
        #   some local variable of the root graph or a subgraph.
        # - The additional work we need to do here is *only* that we need to
        #   lift this free variable into inputs (recursively) of each nested
        #   higher-order-op subgraph until we hit the subgraph where the free
        #   variable is bound
        if self.parent is not None:
            flat_args, tree_spec = pytree.tree_flatten((args, kwargs))
            new_flat_args = []
            for arg in flat_args:
                maybe_new_arg = self.maybe_lift_tracked_freevar_to_input(arg)
                new_flat_args.append(maybe_new_arg)

            args, kwargs = pytree.tree_unflatten(new_flat_args, tree_spec)

        rv = super().create_proxy(
            kind,
            target,
            args,
            kwargs,
            name,
            type_expr,
            proxy_factory_fn,  # type: ignore[arg-type]
        )

        # append stack trace to fx node
        tx = self.output_graph.current_tx

        # log detailed location of line of code in 3.11
        if sys.version_info >= (3, 11) and kind in (
            "call_function",
            "call_method",
            "call_module",
        ):
            cur_inst = tx.current_instruction
            if (
                cur_inst is not self.prev_inst
                and cur_inst.positions is not None
                and cur_inst.positions.lineno is not None
            ):
                tx_code = tx.f_code
                header = tx.get_line_of_code_header(lineno=cur_inst.positions.lineno)

                def get_trace_call_log_str() -> str:
                    line = get_instruction_source_311(tx_code, cur_inst).rstrip()
                    return f"TRACE FX call {rv.node.name} from {header}\n{line}"

                trace_call_log.debug("%s", LazyString(get_trace_call_log_str))
                self.prev_inst = cur_inst

        # update reference to original meta if we're tracing a new code object
        is_retracing = False
        if tx.f_code is not self._cur_code:
            orig_graphmodule_maybe = code_context.get_context(tx.f_code).get(
                "orig_graphmodule", lambda: None
            )()
            if isinstance(orig_graphmodule_maybe, torch.fx.GraphModule):
                is_retracing = True
                self._orig_gm_meta = [
                    nd.meta for nd in orig_graphmodule_maybe.graph.nodes
                ]
                self._orig_gm_lineno_map = orig_graphmodule_maybe._lineno_map
                self._orig_gm_firstlineno = (
                    orig_graphmodule_maybe.forward.__code__.co_firstlineno
                )
            else:
                self._orig_gm_meta = None
                self._orig_gm_lineno_map = None
                self._orig_gm_firstlineno = None
        nn_module_stack = tx.nn_module_stack
        if nn_module_stack:
            rv.node.meta["nn_module_stack"] = nn_module_stack.copy()

        if kind in {"call_function", "call_method"}:
            stack = (rv.node.name, target)
            if nn_module_stack:
                # Current codebase assumes that the nn_module_stack has the
                # builtin modules in the stack.
                current_nn_module = list(rv.node.meta["nn_module_stack"].values())[-1][
                    1
                ]
                if current_nn_module.__module__.startswith(
                    ("torch.nn.modules", "torch.ao.")
                ) and not current_nn_module.__module__.startswith(
                    "torch.nn.modules.container"
                ):
                    stack = (rv.node.name, current_nn_module)

            rv.node.meta["source_fn_stack"] = self.source_fn_stack + [stack]
        elif kind == "call_module":
            if self.parent is not None:
                # TODO can remove once inline_inbuilt_nn_modules is always True
                unimplemented_v2(
                    gb_type="Invoking an nn.Module inside a higher order operator",
                    context=f"Higher order op name: {self.source_target}",
                    explanation="This is not supported.",
                    hints=[],
                )
            # For modules we store the class
            rv.node.meta["source_fn_stack"] = self.source_fn_stack + [
                (
                    rv.node.name,
                    next(
                        ty
                        for k, (_, ty) in rv.node.meta["nn_module_stack"].items()
                        if k.split("@")[0] == target
                    ),
                )
            ]

        self._maybe_preserve_original_meta(tx, rv.node)

        if not is_retracing:
            if "nn_module_stack" not in rv.node.meta:
                nn_module_stack = tx.nn_module_stack
                if nn_module_stack:
                    rv.node.meta["nn_module_stack"] = nn_module_stack.copy()

            if "source_fn_stack" not in rv.node.meta:
                if kind in {"call_function", "call_method"}:
                    rv.node.meta["source_fn_stack"] = self.source_fn_stack + [
                        (rv.node.name, target)
                    ]
                elif kind == "call_module":
                    if self.parent is not None:
                        # TODO can remove once inline_inbuilt_nn_modules is always True
                        unimplemented_v2(
                            gb_type="Invoking an nn.Module inside a HigherOrderOperator",
                            context="",
                            explanation="This is not supported.",
                            hints=[],
                        )
                    # For modules we store the class
                    rv.node.meta["source_fn_stack"] = self.source_fn_stack + [
                        (
                            rv.node.name,
                            rv.node.meta["nn_module_stack"][target][1],
                        )
                    ]

        if "stack_trace" not in rv.node.meta:
            frame_summaries: list[traceback.FrameSummary] = []
            while tx:
                # Avoid frame summaries from inside the torch/nn/modules. This ensures that we keep the stack trace of
                # the user code.
                if not tx.is_co_filename_from_nn_modules():
                    frame_summaries.append(tx.frame_summary())
                tx = getattr(tx, "parent", None)
            # Reverse the frame_summaries, such that the innermost frame is at the last
            frame_summaries.reverse()

            # official from_list stub doesn't have new-style type
            msgs = traceback.StackSummary.from_list(frame_summaries).format()
            rv.node.stack_trace = "".join(msgs)

        if (
            torch._dynamo.config.use_graph_deduplication
            or torch._dynamo.config.track_nodes_for_deduplication
        ):
            self.output_graph.region_tracker.track_node(
                self.output_graph.current_tx, rv.node
            )
        return rv

    def create_node(
        self,
        op: str,
        target: Target,
        args: Any = None,
        kwargs: Any = None,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> fx.Node:
        check_pt2_compliant_op(self.output_graph, op, target, args, kwargs)
        if self.parent is not None:
            flat_args = pytree.arg_tree_leaves(*args, **kwargs)
            for arg in flat_args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                assert arg.graph == self.graph, (
                    "create_node using arg not from this SubgraphTracer"
                )

        node = super().create_node(op, target, args, kwargs, name, type_expr)
        node.meta["creation_timestamp"] = self.output_graph.timestamp
        self._used_names.add(node.name)
        return node

    # Note: we did not override erase_node since
    # we call self.graph.erase_node elsewhere
    def remove_node(self, node: fx.Node) -> None:
        if len(node.users) > 0:
            user_graph_nodes: list[torch.fx.Node] = []
            for user in node.users.keys():
                # For the case where user.graph == self.graph, that is a real bug and will raise
                # properly.
                if user.graph != self.graph:
                    # This is a nested graph, which needs to be deleted.
                    # If we do not do this, we will raise on attempting to remove this.
                    # As we only get here during restoration cleanup, this is sound.
                    user_graph_nodes.extend(reversed(list(user.graph.nodes)))
            for other_graph_node in user_graph_nodes:
                other_graph_node.graph.erase_node(other_graph_node)
        self.graph.erase_node(node)
        self.input_name_to_proxy.pop(node.name, None)

    # when before=True, we will insert this input before the most recent
    # inserted proxy.  This is a hack to get around an ordering problem,
    # where we first insert a tensor argument, and then insert bindings
    # for SymInts that may occur in the tensor argument.
    # Remove this if https://github.com/pytorch/pytorch/issues/99007 gets
    # fixed.
    def create_graph_input(
        self,
        name: str,
        type_expr: Any,
        example_value: Any,
        before: bool = False,
        source: Optional[Source] = None,
    ) -> fx.Proxy:
        if isinstance(example_value, torch.Tensor):
            self._input_versions_at_beginning.append(example_value._version)
        log.debug(
            "create_graph_input %s %s %s at debug_level %s before=%s",
            name,
            source.name() if source is not None else "(none)",
            example_value,
            self.debug_level,
            before,
        )
        if source is None:
            assert self.parent is not None, (
                f"you are required to provide a source for inputs {name} example_val {example_value} on the root tracer"
            )

        # Note [Export inputs must be explicitly passed in]
        # In eager, we are generally OK with adding graph inputs whenever we
        # want, because we take care of writing the bytecode that knows how
        # to source all the inputs.
        #
        # In export, this is bad, because you want a self-contained export
        # object which only depends on the inputs you explicitly passed to it.
        # So we are a bit more strict about what sources can become inputs
        # in export
        if self.is_export and self.parent is None:
            assert source is not None
            if not is_from_local_source(source, only_allow_input=True):
                self.output_graph.source_to_user_stacks.setdefault(source, []).append(
                    TracingContext.extract_stack()
                )

        # _used_names contains the names of all the nodes in the graph,
        # including intermediates. This ensures that we do not have a name
        # collision.
        name = get_unique_name_wrt(name, self._used_names)
        if self.input_name_to_proxy:
            prev_name = next(reversed(self.input_name_to_proxy))
            node = self.input_name_to_proxy[prev_name].node
            if before:
                ctx = self.graph.inserting_before(node)
            else:
                ctx = self.graph.inserting_after(node)
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            proxy = self.create_proxy("placeholder", name, (), {}, type_expr=type_expr)
            set_example_value(proxy.node, example_value)
            if self.input_name_to_proxy and before:
                k, v = self.input_name_to_proxy.popitem()
                self.input_name_to_proxy[name] = proxy
                self.input_name_to_proxy[k] = v
            else:
                self.input_name_to_proxy[name] = proxy

            # For placeholder nodes, `name` is passed as a str to the target,
            # and then torch.fx decides the node.name. So, record the `target`
            # name as well in the _used_names to prevent any collision.
            self._used_names.add(name)

            # NOTE: [Auto lift basic free symbols when create_graph_input]
            # There are two sources of basic symbols:
            #
            # - They can come from inputs, e.g. when an input tensor is specified as dynamic. We handle
            # this case by intercepting at create_graph_input. Whenever we call create_graph_input, we
            # try to also lift the basic symbols in example values as graph input.
            #
            #  1. When create_graph_input for a tensor that has symbolic shapes,
            #     we look for basic symbols in its size and stride, we check if the symbol is bound
            #     in current graph (i.e. bound_symbols), it it's not bound, we'll create a placeholder
            #     for it then recursively check its parent, creates ph if not bound at parent until.
            #     reachting the top-level, where we require a source is attached to the proxy.
            #
            #  2. When create_graph_input for a tensor that contains compound exprs,
            #     for example, if an input to subgraph takes size [s1+s2//8], we'll look for the
            #     the free basic symbols in the sizes and lift all of them following 1.
            #
            #  3. When create_graph_input for a symint. The following invariants hold:
            #     a. if symint's expr is a basic symbol, we only lift it once.
            #     b. if symint's expr is compuned, we lift the expr as a single input. We won't lift The basic symbols
            #       in the compuned expr are NOT lifted. Because if the basic symbols are used inside the subgraph
            #       they will be lifted according to 3.a
            #
            # - They can come from intermediate results:
            # For example, data-dependent operators such as t.item(), t.nonzero(), where basic symbols
            # might be created. For this purpose, we track the basic symbols of intermediate results
            # immediately after they're created at wrap_fx_proxy with track_produced_symints. Notice
            # that for basic symbols that're already tracked by create_graph_input, we won't track it again.
            #
            # Also see NOTE: [Export inputs must be explicitly passed in]
            is_strict_export = self.is_export
            is_non_strict_export = torch.compiler.is_compiling()
            if not is_strict_export and not is_non_strict_export:
                if isinstance(example_value, torch.Tensor):
                    self._lift_basic_symbols(example_value, source)
                elif isinstance(example_value, (list, tuple)):
                    for i, e in enumerate(example_value):
                        if not isinstance(e, torch.Tensor):
                            continue

                        e_source = None
                        if source:
                            e_source = GetItemSource(
                                base=source, index=i, index_is_slice=False
                            )

                        self._lift_basic_symbols(e, e_source)

            # Bound the symbol to ph if example_value is a SymInt with basic symbol.
            if isinstance(example_value, torch.SymInt) and isinstance(
                example_value.node.expr, sympy.Symbol
            ):
                self.bound_symbols[example_value.node.expr] = proxy
            return proxy

    # See NOTE: [Nested SubgraphTracer and free_variable handling] for more details
    def lift_tracked_freevar_to_input(
        self, proxy: fx.Proxy
    ) -> Union[LazyProxy, fx.Proxy]:
        # You're doing something wrong if we are the root SubgraphTracer because
        # Dynamo adds tensors to graph inputs before creating a proxy for them.
        assert self.parent is not None, (
            "lift_tracked_freevar_to_input should not be called on root SubgraphTracer"
        )

        example_value = proxy.node.meta["example_value"]

        # To avoid lifting the same symbol twice, we check whether basic symbols has been tracked.
        # For example, the basic symbols may have already been lifted for current subgraph when
        # we automatically lift basic symbols in the sizes/strides of a tensor t.
        # Suppose parent graph calls sz = t.size()[0], it creates
        # a proxy in parent and the subgraph accesses sz via closure. sz's proxy is not tracked
        # in current sub-tracer so we may lift the same symbol twice.
        if (
            isinstance(example_value, torch.SymInt)
            and example_value.node.expr in self.bound_symbols
        ):
            return self.bound_symbols[example_value.node.expr]

        # Proxies are associated with VariableTracker.
        # It is possible that we've already lifted the Proxy to be an input.
        # If that is the case, just return the already lifted Proxy.
        if proxy in self.lifted_freevars:
            return self.lifted_freevars[proxy]

        # We first lift proxy to parent's graph then lift to current grpah's input
        # so that when we bind symints of the sizes in current graph, those symints
        # would already be lifted as inputs to parent graph.
        if proxy.tracer != self.parent:
            self.parent.lift_tracked_freevar_to_input(proxy)

        example_value = proxy.node.meta["example_value"]
        new_proxy = self.create_graph_input(
            proxy.node.name, type(example_value), example_value
        )
        self.lifted_freevars[proxy] = new_proxy
        return new_proxy

    def maybe_lift_tracked_freevar_to_input(self, arg: Any) -> Any:
        """
        If arg is a free variable, then lift it to be an input.
        Returns the new lifted arg (if arg was a freevar), else the
        original arg.
        """
        if not isinstance(arg, torch.fx.Proxy):
            # Note: arg can be a python built-in slice type e.g.
            # x[:max_seq] is represented as get_item(t, (slice(None, max_seq, None)))
            # we need to also look into the slice variable itself to lift the
            # proxies there.
            if isinstance(arg, slice):
                return slice(
                    *(
                        self.maybe_lift_tracked_freevar_to_input(sub_arg)
                        for sub_arg in (arg.start, arg.stop, arg.step)
                    )
                )
            else:
                return arg
        elif arg.tracer == self:
            return arg
        return self.lift_tracked_freevar_to_input(arg)

    # See NOTE: [Auto lift basic free symbols when create_graph_input] for overall design
    # You MUST call this API every time when creating a proxy in wrap_fx_proxy for a call
    # that produced symints or tensors with unbacked symint shapes.
    # This function is used to track the symints with its proxies created during
    # dynamo tracing so that subgraph knows how to bind a symbol input with parent's proxy.
    # LazyProxy are created for tensor shapes that're unbacked so that we don't create proxies
    # for symbols that're not going to be used, the LazyProxy will be turned into a proxy
    # when it's lifted as input to subgraph.
    def track_produced_symints(
        self, example_value: Any, e_proxy: Union[LazyProxy, torch.fx.Proxy]
    ) -> None:
        # When binding the symbols in an exmaple_value, we bind the symbols
        # to the proxy's associated Tracer instead of current tracer.
        # This is because:
        # 1. We may be calling wrap_tensors during speculate_subgraph because
        # the variables are lazily realized. The proxy are top-level phs but
        # current tracer is a subtracer.
        # 2. For autograd.Function, we trace the backward graph with a new tracer
        # whose parent is the forward tracer, but we're using all the proxies created
        # in forward tracer to trace the backward.
        # For example, forward calls save_for_backward for a input tensor t.
        # Backward calls t.tolist(). In this case, all the proxies that backward tracer
        # sees are from parent tracer (i.e. the forward tracer). (e.g. t[0].item())
        # See test_validate_outputs_unbacked for repro on 2.
        tracer = e_proxy.tracer
        assert isinstance(tracer, SubgraphTracer)

        def need_bind(s: Any) -> bool:
            from torch.fx.experimental.symbolic_shapes import is_symbolic

            return (
                is_symbolic(s)
                and isinstance(s.node.expr, sympy.Symbol)
                and s.node.expr not in self.bound_symbols
            )

        def _proxy_with_example_value(
            example_value: Any, *args: Any, **kwargs: Any
        ) -> fx.Proxy:
            # We need to insert proxy for creating sym_size/sym_stride/sym_storage right after e_proxy
            nonlocal e_proxy
            e_proxy = e_proxy() if isinstance(e_proxy, LazyProxy) else e_proxy
            assert isinstance(e_proxy, torch.fx.Proxy)
            with tracer.graph.inserting_after(e_proxy.node):
                proxy = tracer.create_proxy(*args, **kwargs)
                set_example_value(proxy.node, example_value)
                return proxy

        if isinstance(example_value, torch.Tensor):
            for i, s in enumerate(example_value.size()):
                if need_bind(s):
                    log.debug(
                        "track_produced_symints %s for %s.size()[%s] at debug_level %s",
                        s,
                        e_proxy,
                        i,
                        tracer.debug_level,
                    )
                    lazy_proxy = LazyProxy(
                        tracer,
                        _proxy_with_example_value,
                        s,
                        "call_function",
                        torch.ops.aten.sym_size.int,
                        (e_proxy, i),
                        {},
                        type_expr=type(s),
                    )
                    self.track_produced_symints(s, lazy_proxy)

            storage_offset = example_value.storage_offset()
            if need_bind(storage_offset):
                log.debug(
                    "track_produced_symints %s for %s.storage_offset() at debug_level %s",
                    storage_offset,
                    e_proxy,
                    tracer.debug_level,
                )
                lazy_proxy = LazyProxy(
                    tracer,
                    _proxy_with_example_value,
                    storage_offset,
                    "call_function",
                    torch.ops.aten.sym_storage_offset,
                    (e_proxy,),
                    {},
                    type_expr=type(storage_offset),
                )
                self.track_produced_symints(storage_offset, lazy_proxy)

            if example_value.layout is torch.strided:
                for i, s in enumerate(example_value.stride()):
                    if need_bind(s):
                        log.debug(
                            "track_produced_symints %s for %s.stride()[%s] at debug_level %s",
                            s,
                            e_proxy,
                            i,
                            tracer.debug_level,
                        )
                        lazy_proxy = LazyProxy(
                            tracer,
                            _proxy_with_example_value,
                            s,
                            "call_function",
                            torch.ops.aten.sym_stride.int,
                            (e_proxy, i),
                            {},
                            type_expr=type(s),
                        )
                        self.track_produced_symints(s, lazy_proxy)

            elif example_value.layout is torch.sparse_coo:
                self.track_produced_symints(example_value._indices(), e_proxy)
                self.track_produced_symints(example_value._values(), e_proxy)
            elif example_value.layout in {torch.sparse_csr, torch.sparse_bsr}:
                self.track_produced_symints(example_value.crow_indices(), e_proxy)
                self.track_produced_symints(example_value.col_indices(), e_proxy)
            elif example_value.layout in {torch.sparse_csc, torch.sparse_bsc}:
                self.track_produced_symints(example_value.ccol_indices(), e_proxy)
                self.track_produced_symints(example_value.row_indices(), e_proxy)
            if is_traceable_wrapper_subclass(example_value):
                attrs, ctx = example_value.__tensor_flatten__()
                for attr in attrs:
                    inner_t = getattr(example_value, attr)
                    self.track_produced_symints(inner_t, getattr(e_proxy, attr))
        elif isinstance(example_value, torch.SymInt):
            if need_bind(example_value):
                expr = example_value.node.expr
                tracer.bound_symbols[expr] = e_proxy

    # See Note [Auto lift basic free symbols when create_graph_input]
    def _lift_basic_symbols(
        self, example_value: Union[torch.SymInt, torch.Tensor], src: Optional[Source]
    ) -> None:
        # The before arg is for inserting symints in the sizes/strides of a tensor
        # before the tensor. This ordering ensures that when we look at the tensor's
        # symbols, they're already lifted/tracked. E.g. this assumption is used
        # in insert_deferred_runtime_asserts.
        def _lift_symbols_in_symint(
            s: Union[int, torch.SymInt],
            source: Optional[Source],
            before: bool = False,
        ) -> None:
            if not is_symbolic(s):
                return

            assert isinstance(s, torch.SymInt)
            self_to_be_bound = self.lookup_unbound_symbols(s)
            if len(self_to_be_bound) == 0:
                return

            # For subgraph
            if self.parent is not None:
                # Recursively lift symbols in symint until top-level.
                self.parent._lift_basic_symbols(s, source)
                for s0 in self_to_be_bound:
                    parent_proxy = self.parent.bound_symbols[s0]
                    example_val = parent_proxy.node.meta["example_value"]  # type: ignore[union-attr]
                    assert isinstance(example_val, torch.SymInt)
                    ph = self.create_graph_input(
                        str(s0),
                        type(example_val),
                        example_val,
                        before=before,
                        source=source,
                    )
                    log.debug(
                        "_lift_symbols_in_symint %s from %s at debug_level %s",
                        s0,
                        source.name() if source is not None else "subgraph inputs",
                        self.debug_level,
                    )
                    self.lifted_freevars[parent_proxy] = ph  # type: ignore[index]
            # For root_tracer:
            else:
                assert len(self_to_be_bound) == 1, (
                    f"For root tracer, we only expect to bind basic symbols (compound symbols "
                    f"should be cached before) but got unbound symbols {self_to_be_bound} in {s}"
                )
                assert source is not None, (
                    f"Source of '{s}' is None when lifting it to input of top-level. If it's an unbacked symbol, "
                    "this could be because it's not tracked with lazy_bind_unbacked_symbols. "
                    f"Otherwise, should provide a source when create_graph_input for `{s}` at root tracer."
                )
                s0 = next(iter(self_to_be_bound))
                ph = self.create_graph_input(
                    str(s0),
                    type(s),
                    s,
                    before=before,
                    source=source,
                )
                log.debug(
                    "_lift_symbols_in_symint %s from %s at debug_level %s",
                    s,
                    source.name() if source is not None else "subgraph inputs",
                    self.debug_level,
                )
                ph.node.meta["grapharg"] = GraphArg(
                    source,
                    s,
                    pass_arg_as_tensor=False,
                    fake_tensor=None,
                    is_tensor=False,
                )

        if isinstance(example_value, torch.Tensor):
            for i, s in enumerate(example_value.size()):
                _lift_symbols_in_symint(
                    s,
                    (
                        TensorPropertySource(src, TensorProperty.SIZE, i)
                        if src is not None
                        else None
                    ),
                    before=True,
                )
            if example_value.layout is torch.strided:
                for i, s in enumerate(example_value.stride()):
                    _lift_symbols_in_symint(
                        s,
                        (
                            TensorPropertySource(src, TensorProperty.STRIDE, i)
                            if src is not None
                            else None
                        ),
                        before=True,
                    )
                _lift_symbols_in_symint(
                    example_value.storage_offset(),
                    (
                        TensorPropertySource(src, TensorProperty.STORAGE_OFFSET)
                        if src is not None
                        else None
                    ),
                    before=True,
                )
            elif example_value.layout is torch.sparse_coo:
                self._lift_basic_symbols(example_value._indices(), src)
                self._lift_basic_symbols(example_value._values(), src)
            elif example_value.layout in {torch.sparse_csr, torch.sparse_bsr}:
                self._lift_basic_symbols(example_value.crow_indices(), src)
                self._lift_basic_symbols(example_value.col_indices(), src)
            elif example_value.layout in {torch.sparse_csc, torch.sparse_bsc}:
                self._lift_basic_symbols(example_value.ccol_indices(), src)
                self._lift_basic_symbols(example_value.row_indices(), src)
            if is_traceable_wrapper_subclass(example_value):
                attrs, ctx = example_value.__tensor_flatten__()
                for attr in attrs:
                    inner_t = getattr(example_value, attr)
                    self._lift_basic_symbols(
                        inner_t, AttrSource(src, attr) if src is not None else None
                    )
        elif isinstance(example_value, torch.SymInt):
            _lift_symbols_in_symint(
                example_value,
                src,
            )

    # Lookup the proxy in current tracer for each symbol in expressions of s,
    # See Note [Auto lift basic free symbols when create_graph_input]
    def lookup_unbound_symbols(self, s: torch.SymInt) -> list[sympy.Symbol]:
        free_symbols = s.node.expr.free_symbols
        if len(free_symbols) == 0:
            return []

        to_be_bound = []
        for s0 in free_symbols:
            if s0 not in self.bound_symbols:
                to_be_bound.append(s0)
                continue

            proxy = self.bound_symbols[s0]
            if isinstance(proxy, LazyProxy):
                proxy = proxy()
                self.bound_symbols[s0] = proxy
            assert isinstance(proxy, torch.fx.Proxy) and proxy.tracer is self, (
                f"The proxy of symbol {s0} doesn't belong to current tracer."
            )
        # Sort the symbols so that we can have a deterministic lifting order
        return sorted(to_be_bound, key=lambda s: s.name)

    def has_input_mutation(self) -> MutationInfo:
        input_versions_at_beginning = self._input_versions_at_beginning
        input_nodes = []

        input_versions_at_end = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                example_value = node.meta["example_value"]
                if isinstance(example_value, torch.Tensor):
                    input_versions_at_end.append(example_value._version)
                    input_nodes.append(node)
            else:
                break

        mutated_inputs = [
            i
            for i, (v1, v2) in enumerate(
                zip(input_versions_at_beginning, input_versions_at_end)
            )
            if v1 != v2
        ]

        if len(mutated_inputs):
            mutated_nodes = [input_nodes[i] for i in mutated_inputs]
            msg = f"Input mutation detected at {mutated_nodes}"
            return MutationInfo(True, msg)

        return MutationInfo(False, "")

    def has_aliasing(self) -> AliasingInfo:
        from torch._higher_order_ops.utils import _collect_fake_inputs

        input_storages: dict[StorageWeakRef, torch.fx.Node] = dict()

        for node in self.graph.nodes:
            if node.op == "placeholder":
                example_value = _collect_fake_inputs([node])[0]
                if isinstance(example_value, torch.Tensor):
                    storage = StorageWeakRef(example_value._typed_storage())
                    if storage in input_storages:
                        # input-input aliasing
                        msg = f"Input-to-input aliasing detected at nodes {input_storages[storage]} and {node}"
                        return AliasingInfo(True, msg)
                    input_storages[storage] = node
            else:
                break

        output_storages: dict[StorageWeakRef, torch.fx.Node] = dict()
        out_nodes = self.graph.find_nodes(op="output")[0]
        for out_node in pytree.tree_leaves(out_nodes.args[0]):
            if out_node:
                example_value = _collect_fake_inputs([out_node])[0]
                assert not isinstance(example_value, list)
                if isinstance(example_value, torch.Tensor):
                    storage = StorageWeakRef(example_value._typed_storage())
                    if storage in output_storages:
                        # output-output aliasing
                        msg = f"Output-to-output aliasing detected at nodes {output_storages[storage]} and {out_node}"
                        return AliasingInfo(True, msg)
                    output_storages[storage] = out_node

        intersected_storages = input_storages.keys() & output_storages.keys()
        if len(intersected_storages) > 0:
            # input-output aliasing
            aliased = [
                (input_storages[s], output_storages[s]) for s in intersected_storages
            ]
            aliased = ", ".join([f"{i} and {o}" for i, o in aliased])
            msg = f"Input-to-output aliasing detected at nodes {aliased}"
            return AliasingInfo(True, msg)

        return AliasingInfo(False, "")


# NOTE: [HigherOrderOperator tracing design]
# Ignoring HigherOrderOperators for a moment,
# OutputGraph represents the graph being built by Dynamo that may be compiled
# and executed. It holds a root SubgraphTracer where the FX graph is built.
#
# HigherOrderOperators are operators that take functions as their arguments.
# When Dynamo encounters a HigherOrderOperator, then it attempts to introspect
# the function passed to it (call this the "body function"), capture it into a
# GraphModule, and rewrite the call to the HigherOrderOperator to use the
# GraphModule.
#
# The way we handle the capture of body functions is through having
# (possibly nested) SubgraphTracers, one per body function.
#
# Mechanically, we do the introspection by:
# - Creating a new SubgraphTracer via OutputGraph.subtracer
# - Executing the body function.
# This constructs the graph of the body function in the new SubgraphTracer
# while modifying the state of the OutputGraph. For example:
# - the OutputGraph can receive new GraphArgs (if we discover any new
#   untracked Tensors)
# - side effects from the body function get accumulated into
#   OutputGraph.side_effects
# - guards produced by the body function get accumulated into OutputGraph.guards
#
# The traced function has some special properties that make it easier for us
# to transform later down the line:
# - we lift all free variables to being inputs.
#
# If the introspection fails (due to the existence of graph breaks), then
# we roll back the current OutputGraph state and graph break on the
# HigherOrderOperator.
