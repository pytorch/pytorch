import collections
import contextlib
import copy
import functools
import itertools
import logging
import operator
import re
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, OrderedDict, Set, Union

import sympy

import torch._guards

import torch._logging

import torch.nn
import torch.utils._pytree as pytree
from torch import fx
from torch._guards import (
    Checkpointable,
    Guard,
    GuardsCheckpointState,
    Source,
    TracingContext,
)
from torch.fx.experimental.symbolic_shapes import free_symbols, ShapeEnv
from torch.utils.weak import WeakIdKeyDictionary

from . import config, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
    create_call_function,
    create_instruction,
    Instruction,
    unique_id,
)
from .codegen import PyCodegen
from .exc import BackendCompilerFailed, unimplemented
from .guards import GuardBuilder
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import (
    ConstantSource,
    DefaultDeviceSource,
    DeterministicAlgorithmsSource,
    is_constant_source,
    LocalSource,
    ParamBufferSource,
    ShapeEnvSource,
    TensorProperty,
    TensorPropertySource,
)
from .utils import (
    assert_no_fake_params_or_buffers,
    checkpoint_params,
    CleanupHook,
    clone_inputs,
    count_calls,
    counters,
    dynamo_timed,
    graph_break_reasons,
    lazy_format_graph_code,
    lazy_format_graph_tabular,
    nnmodule_doc_url_msg,
    nnmodule_has_hooks,
    same,
)
from .variables.base import VariableTracker
from .variables.builder import GraphArg, TrackedFake, VariableBuilder, wrap_fx_proxy
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)

log = logging.getLogger(__name__)
graph_tabular_log = torch._logging.getArtifactLogger(__name__, "graph")
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")


class OutputGraphState(NamedTuple):
    input_source_to_var: Dict[Source, VariableTracker]
    tracked_fakes: List[TrackedFake]
    guard_state: GuardsCheckpointState
    nn_modules: Optional[Dict[str, torch.nn.Module]]
    global_state: Optional[Dict[str, bool]]
    param_name_to_source: Optional[Dict[str, Source]]
    side_effects: SideEffects
    timestamp: int
    tensor_weakref_to_sizes_strides: WeakIdKeyDictionary

    def diff(self, other: "OutputGraphState", *, prefix: str = "") -> Optional[str]:
        for k in self._fields:
            if k == "guard_state":
                r = self.guard_state.diff(other.guard_state)
                if r is not None:
                    return r
                continue
            elif k == "side_effects":
                r = self.side_effects.diff(other.side_effects)
                if r is not None:
                    return r
                continue

            sv = getattr(self, k)
            ov = getattr(other, k)
            if sv != ov:
                return f"{prefix}{k} mismatch: {sv} != {ov}"
        return None

    # Back compat .guards api
    @property
    def guards(self):
        return self.guard_state.dynamo_guards


@functools.lru_cache(None)
def _step_logger():
    return torchdynamo_logging.get_step_logger(log)


@dataclass
class GraphCompileReason:
    """Stores why a given output graph was compiled; i.e. what caused the graph break."""

    reason: str
    user_stack: List[traceback.FrameSummary]

    # Indicates if this was a graph compile reason due to graph break.
    graph_break: bool = True

    def __post_init__(self):
        if self.graph_break:
            graph_break_reasons.append(self)


def _get_gen_rand_values_fn(random_calls):
    def _gen_rand_values():
        return [fn(*args, **kwargs) for fn, args, kwargs in random_calls]

    return _gen_rand_values


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: Dict[str, torch.nn.Module]):
        super().__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)

    def __repr__(self):
        return "FakeRootModule(...)"


class WrapperBackend:
    def __init__(self, backend: CompilerFn):
        self.backend: CompilerFn = backend

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
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
            return self.gm.forward

        except Exception:
            log.exception("error in verify_correctness")
            raise
        finally:
            self.restore()


class OutputGraph(Checkpointable[OutputGraphState]):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.

    OutputGraph is 1:1 with a frame being processed. Each frame is associated
    with some root InstructionTranslator. When user code calls a function,
    we construct a InliningInstructionTranslator that continues to write into
    the root InstructionTranslator's OutputGraph.
    """

    def __init__(
        self,
        f_globals: Dict[str, Any],
        code_options: Dict[str, Any],
        compiler_fn: CompilerFn,
        root_tx,
        export: bool,
        export_constraints,
        frame_state,
    ):
        super().__init__()
        self.tracers = [SubgraphTracer(self)]
        # Map from graph input's `Source` to its `VariableTracker` to
        # de-duplicate graph inputs by source and reuse the tracker
        self.input_source_to_var: Dict[Source, VariableTracker] = {}
        self.export = export
        self.export_constraints = export_constraints
        self.frame_state = frame_state
        self.tensor_weakref_to_sizes_strides: WeakIdKeyDictionary = {}
        # In export mode, we force the shape_env to strictly disallow any constraining
        # of the user marked dynamic dims
        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(
                allow_scalar_outputs=config.capture_scalar_outputs,
                allow_dynamic_output_shape_ops=config.capture_dynamic_output_shape_ops,
                frame_id=frame_state["_id"],
            )
            if config.dynamic_shapes
            else None,
            # TODO (tmanlaibaatar) Remove this once we always lift params and buffers
            allow_non_fake_inputs=True if self.export else False,
        )
        self.tracing_context: TracingContext = TracingContext(fake_mode)
        if config.dynamic_shapes:
            # Register a SHAPE_ENV guard to make sure we setup shape guards
            # that show up in ShapeEnv
            self.guards.add(ShapeEnvSource().make_guard(GuardBuilder.SHAPE_ENV))

        self.guards.add(
            DeterministicAlgorithmsSource().make_guard(
                GuardBuilder.DETERMINISTIC_ALGORITHMS
            )
        )

        self.guards.add(DefaultDeviceSource().make_guard(GuardBuilder.DEFAULT_DEVICE))

        # tracked_fakes says where any tensor that was wrapped to fake came
        # from.  It is similar to GraphArg, in that all GraphArgs will get
        # will get added to TrackedFakes, but TrackedFakes also contains
        # GraphArgs that got pruned, and things like Tensor attributes which
        # aren't explicit graph inputs.  Used by shape guard
        self.tracked_fakes: List[TrackedFake] = []
        # Map each tensor id to a list of sources. This is necessary because
        # tensor ids cannot be recovered from tracked fakes (in general).
        # We use this map to interpret (i.e., check for violations of) constraints,
        # specifically equality constraints, which have shared tensor ids in them.
        # This map should also be generally useful, e.g., for (de)serialization.
        self.tracked_fakes_id_to_source: Dict[
            int, List[Source]
        ] = collections.defaultdict(list)
        # Stores the full fqn of a param or buffer to the relevant source.
        self.param_name_to_source: Optional[Dict[str, Source]] = dict()
        self.side_effects = SideEffects()
        self.code_options = dict(code_options)
        self.output_instructions: List[Instruction] = []
        # used to track nodes that are added between calls of copy_graphstate
        # and restore_graphstate
        self.timestamp = 0

        # Not checkpointed
        self.compiler_fn: CompilerFn = compiler_fn
        self.root_globals = f_globals
        self.root_tx = root_tx
        from torch._dynamo.symbolic_convert import InstructionTranslatorBase

        self._current_tx: List[InstructionTranslatorBase] = []
        self.cleanups: List[CleanupHook] = []
        self.should_exit = False
        self.random_values_var = None
        self.unspec_variable_map: Dict[str, UnspecializedPythonVariable] = {}

        # We save the global torch state here to be restored in case of graph
        # breaks. The relevant issue is seen here
        # https://github.com/pytorch/pytorch/pull/100570#issuecomment-1543427086
        # where inlining of a function changes the global state (because of the
        # presence of torch.no_grad) and there is a graph break.
        self.save_global_state()

    @property
    def root_tracer(self):
        return self.tracers[0]

    @property
    def current_tracer(self):
        return self.tracers[-1]

    @property
    def graph(self):
        return self.current_tracer.graph

    # TODO(rzou): can delete after we refactor speculate_subgraph to use nested GraphTracer.
    @graph.setter
    def graph(self, value):
        self.current_tracer.graph = value

    @property
    def input_name_to_proxy(self):
        return self.current_tracer.input_name_to_proxy

    @property
    def real_value_cache(self):
        return self.current_tracer.real_value_cache

    # If you are here, and you're looking for create_graph_input,
    # to avoid ambiguity, please call one of the following:
    # - self.current_tracer.create_graph_input
    # - self.root_tracer.create_graph_input
    # See NOTE [HigherOrderOperator tracing design] for more context.

    def create_proxy(self, *args, **kwargs):
        return self.current_tracer.create_proxy(*args, **kwargs)

    def create_node(self, *args, **kwargs):
        return self.current_tracer.create_node(*args, **kwargs)

    def remove_node(self, *args, **kwargs):
        return self.current_tracer.remove_node(*args, **kwargs)

    @contextlib.contextmanager
    def new_subtracer(self):
        try:
            tracer = SubgraphTracer(self, parent=self.current_tracer)
            self.tracers.append(tracer)
            yield tracer
        finally:
            self.tracers.pop()

    @property
    def output(self):
        return self

    @property
    def fake_mode(self):
        return self.root_tx.fake_mode

    @property
    def shape_env(self):
        return self.tracing_context.fake_mode.shape_env

    @property
    def guards(self) -> Set[Guard]:
        return self.tracing_context.guards_context.dynamo_guards

    @property
    def nn_modules(self) -> Dict[str, torch.nn.Module]:
        return self.tracing_context.module_context.nn_modules

    def save_global_state(self):
        global_state = self.tracing_context.global_context.global_state
        global_state["is_grad_enabled"] = torch.is_grad_enabled()

    def push_tx(self, tx):
        self._current_tx.append(tx)

    def pop_tx(self):
        return self._current_tx.pop()

    @property
    def current_tx(self):
        return self.root_tx if not self._current_tx else self._current_tx[-1]

    def copy_graphstate(self) -> OutputGraphState:
        """Create a checkpoint of the current state by copying everything"""
        assert self.param_name_to_source is not None
        guards_graph_state = self.tracing_context.guards_context.copy_graphstate()
        module_state = self.tracing_context.module_context.copy_graphstate()
        global_state = self.tracing_context.global_context.copy_graphstate()
        state = OutputGraphState(
            dict(self.input_source_to_var),
            list(self.tracked_fakes),
            guards_graph_state,
            module_state,
            global_state,
            dict(self.param_name_to_source),
            self.side_effects.clone(),
            self.timestamp,
            dict(self.tensor_weakref_to_sizes_strides),
        )
        self.timestamp += 1
        return state

    def restore_graphstate(self, state: OutputGraphState):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            self.input_source_to_var,
            self.tracked_fakes,
            guards_state,
            module_state,
            global_state,
            self.param_name_to_source,
            self.side_effects,
            self.timestamp,
            self.tensor_weakref_to_sizes_strides,
        ) = state
        self.tracing_context.guards_context.restore_graphstate(guards_state)
        self.tracing_context.module_context.restore_graphstate(module_state)
        self.tracing_context.global_context.restore_graphstate(global_state)

        # FX deepcopy doesn't work for a partially created graph, so just remove new nodes
        removed_nodes = 0
        for node in reversed(list(self.graph.nodes)):
            if node.meta["creation_timestamp"] > self.timestamp:
                # Erasing node alone does not remove the meta information
                # So, remove the help tensor explicitly
                if "example_value" in node.meta:
                    del node.meta["example_value"]
                self.remove_node(node)
                self.real_value_cache.pop(node, None)
                removed_nodes += 1
        log.debug("restore_graphstate: removed %s nodes", removed_nodes)

    def add_symbol_bindings(self, arg: GraphArg):
        # Insert implicit size vars as necessary.  With dynamic shapes, we
        # maintain the invariant that every sizevar gets a direct SymInt input
        # into the graph.  This means downstream graph transforms can assume
        # every size variable is explicitly bound and accessible, instead of
        # having to pull it out implicitly from tensors.

        if self.export:
            return

        assert arg.fake_tensor is not None

        def bind_symint(s, prop):
            if not (
                isinstance(s, torch.SymInt) and isinstance(s.node.expr, sympy.Symbol)
            ):
                return
            # TODO: don't readd symint if we already have it in graph
            # (this is harmless because we do remove the unused ones later)
            proxy = self.root_tracer.create_graph_input(
                str(s.node.expr), torch.SymInt, before=True
            )
            proxy.node.meta["grapharg"] = GraphArg(
                prop(arg.source),
                s,
                is_unspecialized=False,
                fake_tensor=None,
                is_tensor=False,
            )

        for i, s in enumerate(arg.fake_tensor.size()):
            bind_symint(
                s, lambda src: TensorPropertySource(src, TensorProperty.SIZE, i)
            )
        for i, s in enumerate(arg.fake_tensor.stride()):
            bind_symint(
                s, lambda src: TensorPropertySource(src, TensorProperty.STRIDE, i)
            )
        bind_symint(
            arg.fake_tensor.storage_offset(),
            lambda src: TensorPropertySource(src, TensorProperty.STORAGE_OFFSET),
        )

    def count_calls(self):
        return count_calls(self.graph)

    def get_submodule(self, keys):
        assert keys
        obj = self.nn_modules
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

    def new_var(self, name="tmp"):
        existing = set(self.code_options["co_varnames"])
        for i in itertools.count():
            var = f"___{name}_{i}"
            if var not in existing:
                self.code_options["co_varnames"] += (var,)
                return var

    def update_co_names(self, name):
        """Ensure self.code_options.co_names contains name"""
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)

    def register_attr_or_module(
        self,
        target: Union[torch.nn.Module, torch.Tensor, Any],
        *names,
        **options,
    ):
        if is_dynamic_nn_module(target):
            return variables.UnspecializedNNModuleVariable(target, **options)

        options = dict(options)
        options["guards"] = set(options.get("guards", []))
        assert "source" in options
        source = options["source"]
        assert not isinstance(source, ParamBufferSource)

        if isinstance(target, torch.Tensor):
            if len(self.tracers) > 1:
                unimplemented(
                    "accessing attribute of nn.Module inside HigherOrderOperator"
                )
            if not is_constant_source(source):
                options["guards"].add(source.make_guard(GuardBuilder.TENSOR_MATCH))

            def wrap_name(module_key):
                assert self.param_name_to_source is not None
                self.param_name_to_source[module_key] = source
                return wrap_fx_proxy(
                    self.root_tx,
                    self.create_proxy("get_attr", module_key, tuple(), {}),
                    example_value=target,
                    **options,
                )

        elif isinstance(target, torch.nn.Module):
            assert isinstance(target, torch.nn.Module)
            if nnmodule_has_hooks(target, check_forward_hooks=True):
                torch._logging.warning_once(
                    log,
                    "nn.Module forward/_pre hooks are only partially supported, and were detected in your model. "
                    "In particular, if you do not change/remove hooks after calling .compile(), you can disregard this "
                    "warning, and otherwise you may need to set torch._dynamo.config.skip_nnmodule_hook_guards=False "
                    "to ensure recompiling after changing hooks."
                    f"{nnmodule_doc_url_msg} ",
                )
            if nnmodule_has_hooks(
                target, check_backward_hooks=True, check_state_dict_hooks=True
            ):
                torch._logging.warning_once(
                    log,
                    "nn.Module state_dict and backward hooks are not yet supported by torch.compile, "
                    f"but were detected in your model and will be silently ignored. {nnmodule_doc_url_msg}",
                )

            options["guards"].add(source.make_guard(GuardBuilder.NN_MODULE))

            def wrap_name(module_key):
                return NNModuleVariable(type(target), module_key, **options)

        elif isinstance(target, (torch.SymInt, torch.SymFloat)):
            # HACKY CODE REGION BEGIN
            # WE ARE PIGGYBACKING ON EXISTING INFRA TO REGISTER ATTRS
            # This ultimately gets written to self.nn_modules, which is unfortunate
            # Attrs that are tenors and symints and such need to be migrated to have their
            # own storage
            # alas, this is like this for now

            def wrap_name(module_key):
                return SymNodeVariable.create(
                    self,
                    self.create_proxy("get_attr", module_key, tuple(), {}),
                    sym_num=target,
                    **options,
                )

            # HACKY CODE REGION END
        else:

            def wrap_name(module_key):
                self.output.update_co_names(module_key)
                self.root_globals[module_key] = target
                return VariableBuilder(self, ConstantSource(source_name=module_key))(
                    target
                )

        for k, v in self.nn_modules.items():
            if v is target:
                # it already exists
                return wrap_name(k)
        # create a new unique name
        name = "_".join(map(str, names))
        # Strip the guard lookup L/G access
        name = re.sub(r"^[GL]\['?(.*?)'?\]$", r"\1", name)
        # e.g. replace abc.xyz[123].qkv with abc.xyz_123.qkv
        name = re.sub(r"\[(\d+)\]", r"_\g<1>", name)
        # e.g. replace abc.xyz_123.qkv with abc_xyz_123_qkv
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)

        if not name or not name[0].isalpha():
            name = "sub" + name
        base = name
        for i in itertools.count():
            if name not in self.nn_modules:
                self.nn_modules[name] = target
                if isinstance(target, torch.nn.Module):

                    def register_leaf_name(leaf_name):
                        assert self.param_name_to_source is not None
                        new_source = ParamBufferSource(source, leaf_name)
                        new_name = f"{name}.{leaf_name}"
                        self.param_name_to_source[new_name] = new_source

                    # annoying, but there are cases when we do not have parameters
                    # see test_nn_moduledict_contains
                    if hasattr(target, "_parameters"):
                        for leaf_name, _ in target.named_parameters(
                            remove_duplicate=False
                        ):
                            register_leaf_name(leaf_name)
                    if hasattr(target, "_buffers"):
                        for leaf_name, _ in target.named_buffers(
                            remove_duplicate=False
                        ):
                            register_leaf_name(leaf_name)

                return wrap_name(name)
            name = f"{base}_{i}"

        raise AssertionError("unreachable")

    def compile_subgraph(
        self, tx, partial_convert=False, reason: Optional[GraphCompileReason] = None
    ):
        """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
        assert reason is not None

        from .eval_frame import disable

        self.partial_convert = partial_convert
        self.compile_subgraph_reason = reason

        log.debug("COMPILING GRAPH due to %s", reason)

        if not all(block.can_restore() for block in tx.block_stack):
            unimplemented("compile_subgraph with block_depth != 0")

        prefix_insts: List[Instruction] = []
        if sys.version_info >= (3, 11):
            # prefix instructions (Python 3.11+)
            for inst in tx.prefix_insts:
                if inst.opname == "MAKE_CELL":
                    prefix_insts.append(
                        create_instruction("MAKE_CELL", argval=inst.argval)
                    )
                elif inst.opname == "COPY_FREE_VARS":
                    prefix_insts.append(
                        create_instruction(
                            "COPY_FREE_VARS", arg=len(tx.code_options["co_freevars"])
                        )
                    )
                else:
                    prefix_insts.append(copy.copy(inst))

        def append_prefix_insts():
            self.add_output_instructions(prefix_insts)
            prefix_insts.clear()

        for block in reversed(tx.block_stack):
            block.exit(tx)

        self.cleanup_graph()
        tx.prune_dead_locals()
        stack_values = list(tx.stack)
        root = FakeRootModule(self.nn_modules)

        # Add all the local vars to the "stack" so restore at the end
        restore_vars = []
        val_to_names: OrderedDict[
            VariableTracker, List[str]
        ] = collections.OrderedDict()
        if stack_values:
            val_to_names[stack_values[-1]] = list()
        for k, v in tx.symbolic_locals.items():
            # Note! this explicitly uses .local_name for matching
            # Failure to do so will cause spurious registrations in val_to_names.
            # This will in turn result in spurious variables showing up in the graph.
            # This was very tricky to debug. For an example, dump the graph at call_user_compiler
            # while running test_subgraphs.py
            if isinstance(v.source, LocalSource) and v.source.local_name == k:
                continue  # no need to restore initial state
            if v not in val_to_names:
                val_to_names[v] = list()
            val_to_names[v].append(k)
        for v in val_to_names.keys():
            restore_vars.extend(val_to_names[v])
            stack_values.extend([v] * len(val_to_names[v]))

        # to handle random calls
        if len(tx.random_calls) > 0:
            append_prefix_insts()
            random_calls_instructions = []
            self.random_values_var = self.new_var("random_values")
            rand_fn_name = unique_id("__gen_rand_values")
            rand_fn = disable(_get_gen_rand_values_fn(tx.random_calls))
            self.install_global(rand_fn_name, rand_fn)
            codegen = PyCodegen(tx, root)
            random_calls_instructions.extend(
                codegen.load_function_name(rand_fn_name, True)
            )
            random_calls_instructions.extend(create_call_function(0, False))
            random_calls_instructions.append(
                codegen.create_store(tx.output.random_values_var),
            )
            self.add_output_instructions(random_calls_instructions)

        if (
            stack_values
            and all(
                not isinstance(v, UnspecializedPythonVariable) for v in stack_values
            )
            and all(isinstance(x, TensorVariable) for x in stack_values)
            and len(set(stack_values)) == len(stack_values)
            and self.side_effects.is_empty()
        ):
            append_prefix_insts()
            # optimization to generate better code in a common case
            self.add_output_instructions(
                self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root)
                + [create_instruction("UNPACK_SEQUENCE", arg=len(stack_values))]
            )
            codegen = PyCodegen(tx)
        else:
            graph_output_var = self.new_var("graph_out")
            pass1 = PyCodegen(tx, root, graph_output_var)
            self.side_effects.codegen_save_tempvars(pass1)
            pass1.foreach(stack_values)
            self.side_effects.codegen_update_mutated(pass1)

            # one more time now that we have established tempvars
            pass2 = PyCodegen(
                tx,
                root,
                graph_output_var,
                tempvars={val: None for val, count in pass1.uses.items() if count > 1},
            )
            self.side_effects.codegen_save_tempvars(pass2)
            pass2.foreach(stack_values)
            self.side_effects.codegen_update_mutated(pass2)

            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(
                    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
                )

                if len(pass2.graph_outputs) != 0:
                    output.append(pass2.create_store(graph_output_var))
                else:
                    output.append(create_instruction("POP_TOP"))
            append_prefix_insts()
            self.add_output_instructions(output + pass2.get_instructions())
            codegen = pass2

        if config.numpy_ndarray_as_tensor:
            from .variables.tensor import NumpyNdarrayVariable

            self.add_output_instructions(
                NumpyNdarrayVariable.reconstruct_ndarray_before_return(codegen, self)
            )
        # restore all the live local vars
        self.add_output_instructions(
            [PyCodegen(tx).create_store(var) for var in reversed(restore_vars)]
        )

    def cleanup_graph(self):
        """
        Remove this pattern from the graph:
            torch._C._set_grad_enabled(False)
            torch._C._set_grad_enabled(True)
        """
        nodes = list(self.graph.nodes)
        grad_enabled = torch.is_grad_enabled()
        for node1, node2 in zip(nodes, nodes[1:]):
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

    @torch._guards.TracingContext.clear_frame()
    def compile_and_call_fx_graph(self, tx, rv, root):
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.
        """
        from .eval_frame import disable

        assert isinstance(rv, list)
        assert isinstance(root, FakeRootModule)
        for output in rv:
            self.guards.update(output.guards)

        self.create_node(
            "output",
            "output",
            (self.current_tracer.create_arg(tuple(x.as_proxy() for x in rv)),),
            {},
        )
        self.remove_unused_graphargs()
        ncalls = count_calls(self.graph)
        counters["stats"]["calls_captured"] += ncalls

        # free a bit of memory
        self.real_value_cache.clear()

        gm = fx.GraphModule(root, self.graph)
        gm.compile_subgraph_reason = self.compile_subgraph_reason
        name = unique_id("__compiled_fn")

        assert_no_fake_params_or_buffers(gm)
        compiled_fn = self.call_user_compiler(gm)
        compiled_fn = disable(compiled_fn)

        counters["stats"]["unique_graphs"] += 1
        self.install_global(name, compiled_fn)

        graph_code_log.debug("%s", lazy_format_graph_code(name, gm))
        graph_tabular_log.debug("%s", lazy_format_graph_tabular(name, gm))

        cg = PyCodegen(tx)
        cg.make_call_generated_code(name)
        return cg.get_instructions()

    @property
    def placeholders(self) -> List[fx.Node]:
        r = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                r.append(node)
                continue
            break
        return r

    @property
    def graphargs(self) -> List[GraphArg]:
        return [node.meta["grapharg"] for node in self.placeholders]

    @dynamo_timed(phase_name="backend_compile")
    def call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:
        tot = 0
        placeholders = []
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                tot += 1
            if node.op == "placeholder":
                placeholders.append(node)
        torch._dynamo.utils.increment_op_count(tot)
        for pl in placeholders:
            arg = pl.meta["grapharg"]
            # TODO: Why isn't this stored in meta :think:
            pl._dynamo_source = arg.source

        gm._param_name_to_source = self.param_name_to_source

        try:
            name = (
                self.compiler_fn.__name__
                if hasattr(self.compiler_fn, "__name__")
                else ""
            )
            _step_logger()(logging.INFO, f"calling compiler function {name}")
            compiler_fn = self.compiler_fn
            if config.verify_correctness:
                compiler_fn = WrapperBackend(compiler_fn)

            compiled_fn = compiler_fn(gm, self.example_inputs())
            _step_logger()(logging.INFO, f"done compiler function {name}")
            assert callable(compiled_fn), "compiler_fn did not return callable"
        except Exception as e:
            raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
                e.__traceback__
            ) from None
        return compiled_fn

    def example_inputs(self) -> List[torch.Tensor]:
        result = []
        for arg in self.graphargs:
            result.append(arg.example)
        return result

    def remove_unused_graphargs(self) -> None:
        # Miniature DCE pass, but only for obviously trivial operations
        for node in reversed(list(self.graph.nodes)):
            if len(list(node.users)) == 0:
                if node.op == "get_attr":
                    self.remove_node(node)
                elif node.op == "call_function" and node.target is operator.getitem:
                    self.remove_node(node)

        def placeholder_binds_symbol(node):
            arg = node.meta["grapharg"]
            example = arg.example
            if isinstance(example, torch.SymInt) and isinstance(
                example.node.expr, sympy.Symbol
            ):
                return example.node.expr
            return None

        def remove_unused(node):
            log.debug("REMOVE UNUSED GRAPHARG %s", node.meta["grapharg"].source.name())
            # I'm not really sure why you need to delete these from the
            # node since the node is going to get removed
            del node.meta["grapharg"]
            self.remove_node(node)
            self.real_value_cache.pop(node, None)

        used_symbols = set()
        recheck_placeholders = []
        for node in self.placeholders:
            binds_symbol = placeholder_binds_symbol(node) is not None
            # Don't delete symbol bindings yet
            if binds_symbol:
                if not node.users:
                    recheck_placeholders.append(node)
            else:
                if not node.users:
                    remove_unused(node)
                else:
                    # Register the free symbols as uses
                    arg = node.meta["grapharg"]
                    fake = (
                        arg.fake_tensor if arg.fake_tensor is not None else arg.example
                    )
                    used_symbols |= free_symbols(fake)

        # After removing unused graphargs, prune unused binds_symbol
        for node in recheck_placeholders:
            symbol = placeholder_binds_symbol(node)
            if symbol is not None:
                if symbol not in used_symbols:
                    remove_unused(node)
                else:
                    # Make sure we delete later occurrences of the same symbol
                    used_symbols.remove(symbol)

    def add_output_instructions(self, prefix: List[Instruction]) -> None:
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.
        """
        self.output_instructions.extend(prefix)
        self.should_exit = True

    def install_global(self, name, value) -> None:
        self.cleanups.append(CleanupHook.create(self.root_globals, name, value))

    def cleanup(self) -> None:
        # There is a reference cycle between tracer and OutputGraph, causing
        # some of the tensor objects to be held alive for longer than necessary.

        self.root_tx = None
        self.nn_modules.clear()
        self.param_name_to_source = None

        for node in self.graph.nodes:
            if "grapharg" in node.meta:
                del node.meta["grapharg"]
        self.real_value_cache.clear()
        self.input_name_to_proxy.clear()
        self.side_effects.clear()


class SubgraphTracer(fx.Tracer):
    """
    Holds an FX graph that is being traced. OutputGraph owns a SubgraphTracer
    and the separation of responsibilities is that SubgraphTracer is
    responsible for building the graph while OutputGraph is responsible for
    compiling and executing the graph.
    """

    def __init__(self, output_graph, parent=None):
        super(SubgraphTracer, self).__init__()
        self.output_graph = weakref.proxy(output_graph)
        self.graph = torch.fx.Graph()
        # Map from graph input name to its placeholder proxy object, where the
        # map's keys give all current placeholder node names and can be used to
        # create unique node names
        self.input_name_to_proxy: OrderedDict[str, fx.Proxy] = collections.OrderedDict()
        # Node => computed real value (see utils.get_real_value)
        self.real_value_cache: Dict[fx.Node, torch.Tensor] = {}

        # SubgraphTracers can be nested. See NOTE [HigherOrderOperator tracing design]
        self.parent = parent
        # A list of proxies that exist in the graph being traced. We use this
        # list to determine that, when tracing the body function of a HigherOrderOperator,
        # if a new proxy is actually a free variable.
        self.seen_proxies = set({})
        # A list of previously free variables that we lifted to being inputs
        # of the graph. If we are tracing a HigherOrderOperator's body_fn,
        # then we need to keep track of this so we can rewrite the
        # HigherOrderOperator call using the traced body_fn.
        self.lifted_freevars = set({})

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
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
        # In case 1, we call `lift_tracked_freevar_to_input` (below)
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
            flat_args, _ = pytree.tree_flatten(args)
            for arg in flat_args:
                if not isinstance(arg, torch.fx.Proxy):
                    # Is a constant
                    continue
                if arg in self.seen_proxies:
                    continue
                if not hasattr(arg, "node"):
                    continue
                if arg.node.name in self.input_name_to_proxy:
                    continue
                if "saved_tensor_marked" in arg.node.meta:
                    continue

                self.lift_tracked_freevar_to_input(arg)

        rv = super().create_proxy(
            kind, target, args, kwargs, name, type_expr, proxy_factory_fn
        )

        # append stack trace to fx node
        tx = self.output_graph.current_tx

        nn_module_stack = tx.nn_module_stack
        if nn_module_stack:
            rv.node.meta["nn_module_stack"] = nn_module_stack.copy()

        if kind in {"call_function", "call_method"}:
            rv.node.meta["source_fn"] = (rv.node.name, target)
        elif kind == "call_module":
            if self.parent is not None:
                unimplemented("Invoking an nn.Module inside HigherOrderOperator")
            # For modules we store the class
            rv.node.meta["source_fn"] = (
                rv.node.name,
                rv.node.meta["nn_module_stack"][target][1],
            )

        frame_summaries: List[traceback.FrameSummary] = []
        while tx:
            frame_summaries.append(tx.frame_summary())
            tx = getattr(tx, "parent", None)
        # Reverse the frame_summaries, such that the innermost frame is at the last
        frame_summaries.reverse()

        # official from_list stub doesn't have new-style type
        msgs = traceback.StackSummary.from_list(frame_summaries).format()  # type: ignore[arg-type]
        rv.node.stack_trace = "".join(msgs)

        self.seen_proxies.add(rv)
        return rv

    def create_node(self, *args, **kwargs):
        node = super().create_node(*args, **kwargs)
        node.meta["creation_timestamp"] = self.output_graph.timestamp
        return node

    # Note: we did not override erase_node since
    # we call self.graph.erase_node elsewhere
    def remove_node(self, node):
        if len(node.users) > 0:
            user_graph_nodes: List[torch.fx.Node] = []
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
    def create_graph_input(self, name, type_expr=None, before=False):
        # unique
        if name in self.input_name_to_proxy:
            for i in itertools.count():
                candidate_name = f"{name}_{i}"
                if candidate_name not in self.input_name_to_proxy:
                    name = candidate_name
                    break

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
            if self.input_name_to_proxy and before:
                k, v = self.input_name_to_proxy.popitem()
                self.input_name_to_proxy[name] = proxy
                self.input_name_to_proxy[k] = v
            else:
                self.input_name_to_proxy[name] = proxy
            return proxy

    def is_name_bound(self, name):
        if name in self.input_name_to_proxy:
            return True
        for proxy in self.seen_proxies:
            if proxy.node.name == name:
                return True
        return False

    # See NOTE: [Nested SubgraphTracer and free_variable handling] for more details
    def lift_tracked_freevar_to_input(self, proxy):
        # You're doing something wrong if we are the root SubgraphTracer because
        # Dynamo adds tensors to graph inputs before creating a proxy for them.
        assert (
            self.parent is not None
        ), "lift_tracked_freevar_to_input on root SubgraphTracer"
        self.create_graph_input(proxy.node.name)
        self.lifted_freevars.add(proxy)
        if self.parent is not None and not self.parent.is_name_bound(proxy.node.name):
            self.parent.lift_tracked_freevar_to_input(proxy)


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
# - Creating a new SubgraphTracer via OutputGraph.new_subtracer
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
