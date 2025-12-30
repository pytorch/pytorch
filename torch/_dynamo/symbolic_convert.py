"""
Core module responsible for converting Python bytecode into TorchDynamo's symbolic execution format.

This module implements the bytecode-level tracing system that allows TorchDynamo to analyze
and transform Python code. It converts Python bytecode instructions into a symbolic format
that tracks the flow of tensors and other values through the program.

Key components:
- InstructionTranslatorBase: Base class for converting bytecode to symbolic execution
- InstructionTranslator: Main translator for function bytecode
- InliningInstructionTranslator: Handles inlining of called functions
- SpeculationLog: Manages state for speculative execution and rollback

The symbolic conversion process handles:
- Control flow (loops, conditionals, etc.)
- Function inlining and call stack management
- Tracking of program values and side effects
- Graph breaks and resumption points
- Exception handling and stack frame management

This is a core part of TorchDynamo's tracing system that enables ahead-of-time
optimization of PyTorch programs.
"""

from __future__ import annotations

import collections
import collections.abc
import contextlib
import copy
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import operator
import re
import sys
import threading
import traceback
import types
import weakref
from collections import deque
from typing import Any, cast, NoReturn, Optional, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import TypeIs

import torch
import torch._logging
from torch._dynamo.exc import ObservedException, TensorifyScalarRestartAnalysis
from torch._guards import tracing, TracingContext
from torch._logging.structured import dump_file
from torch.fx.experimental.symbolic_shapes import guard_bool
from torch.utils._functools import cache_method

from . import (
    config,
    exc,
    graph_break_hints,
    logging as torchdynamo_logging,
    trace_rules,
    variables,
)
from .bytecode_analysis import (
    get_indexof,
    JUMP_OPNAMES,
    livevars_analysis,
    propagate_line_nums,
)
from .bytecode_transformation import (
    cleaned_instructions,
    create_binary_slice,
    create_call_function,
    create_call_function_ex,
    create_copy,
    create_dup_top,
    create_instruction,
    create_jump_absolute,
    create_rot_n,
    create_swap,
    get_code_keys,
    Instruction,
    is_generator,
    is_jump_absolute,
    unique_id,
)
from .code_context import code_context
from .codegen import PyCodegen
from .exc import (
    augment_exc_message_with_hop_name,
    BackendCompilerFailed,
    collapse_resume_frames,
    format_frame_info,
    get_stack_above_dynamo,
    ResumePrologueTracingError,
    StepUnsupported,
    unimplemented,
    Unsupported,
)
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph, StackLocalsMetadata
from .polyfills import impl_CONTAINS_OP_fallback
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import (
    ContinueExecutionCache,
    IS_TRACING_RESUME_PROLOGUE_VARNAME,
    ReenterWith,
)
from .source import (
    AttrSource,
    DictGetItemSource,
    GlobalSource,
    GlobalWeakRefSource,
    LocalCellSource,
    LocalSource,
    SkipGuardSource,
    Source,
)
from .trace_rules import is_builtin_constant, is_forbidden
from .utils import (
    _get_error_on_graph_break,
    counters,
    get_fake_value,
    get_instruction_source_311,
    get_metrics_context,
    graph_break_dup_warning_checker,
    istype,
    LazyString,
    proxy_args_kwargs,
)
from .variables.base import typestr, ValueMutationNew, VariableTracker
from .variables.builder import FrameStateSizeEntry, VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable
from .variables.ctx_manager import (
    ContextWrappingVariable,
    GenericContextWrappingVariable,
    WithEnterFunctionVariable,
    WithExitFunctionVariable,
)
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import (
    BaseUserFunctionVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .variables.iter import MAX_ITERATOR_LIMIT
from .variables.lazy import LazyVariableTracker
from .variables.lists import (
    BaseListVariable,
    IteratorVariable,
    ListIteratorVariable,
    ListVariable,
    SliceVariable,
    TupleVariable,
)
from .variables.misc import (
    CellVariable,
    ExceptionVariable,
    GetAttrVariable,
    NullVariable,
    PythonModuleVariable,
    TracebackVariable,
    UnknownVariable,
)
from .variables.nn_module import NNModuleVariable, UnspecializedNNModuleVariable
from .variables.streams import SymbolicStreamState
from .variables.tensor import supported_comparison_ops, SymNodeVariable
from .variables.torch_function import (
    SymbolicTorchFunctionState,
    TorchFunctionModeVariable,
)
from .variables.user_defined import (
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
    UserDefinedObjectVariable,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch._subclasses.fake_tensor import FakeTensorMode

    from .package import CompilePackage

log = logging.getLogger(__name__)
graph_break_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")
trace_call_log = torch._logging.getArtifactLogger(__name__, "trace_call")
trace_source_log = torch._logging.getArtifactLogger(__name__, "trace_source")
trace_bytecode_log = torch._logging.getArtifactLogger(__name__, "trace_bytecode")
tls = threading.local()
compare_op_handlers: dict[str, Any] = {
    k: BuiltinVariable(v).call_function for k, v in supported_comparison_ops.items()
}
handle_contains = BuiltinVariable(operator.contains).call_function
handle_not = BuiltinVariable(operator.not_).call_function
compare_op_handlers["in"] = lambda tx, args, _: handle_contains(
    tx, [*reversed(args)], {}
)
compare_op_handlers["not in"] = lambda tx, args, _: handle_not(
    tx, [handle_contains(tx, [*reversed(args)], {})], {}
)

ExceptionVals: TypeAlias = Union[
    variables.ExceptionVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
]


@functools.cache
def _import_module(name: str) -> types.ModuleType:
    """
    Import the named module and cache the result. importlib.import_module()
    seems to do some filesystem checking to validate the name so not caching
    this can be slow.
    """
    return importlib.import_module(name)


@dataclasses.dataclass
class SpeculationEntry:
    filename: str
    lineno: int
    instruction_pointer: int
    inst: Instruction  # for debugging only
    _failed: bool = False
    error_on_graph_break: Optional[bool] = None
    reason: Optional[GraphCompileReason] = None

    def fail_and_restart_analysis(self, error_on_graph_break: bool) -> None:
        """
        Start tracing of the current frame over again, and don't take this branch.
        """
        self._failed = True
        self.error_on_graph_break = error_on_graph_break
        if self.reason is not None:
            restart_reason = self.reason.reason
        else:
            restart_reason = "Unknown fail_and_restart_analysis"
        raise exc.SpeculationRestartAnalysis(restart_reason=restart_reason)

    def failed(self, tx: InstructionTranslatorBase) -> bool:
        if self._failed:
            assert self.error_on_graph_break is not None
            tx.error_on_graph_break = self.error_on_graph_break
            return True
        return False


@dataclasses.dataclass
class SpeculationLog:
    """
    SpeculationLog replaces the prior copy_graphstate/restore_graphstate
    checkpointing.  Rather than saving/restoring state, we restart the
    dynamo conversion process over from the beginning -- but when we
    hit the start of the speculation that failed, we instead generate
    a graph break.
    """

    entries: list[SpeculationEntry] = dataclasses.field(default_factory=list)
    index: int = 0

    def restart(self) -> None:
        self.index = 0

    def clear(self) -> None:
        self.entries.clear()
        self.index = 0

    def next(
        self, filename: str, lineno: int, instruction_pointer: int, inst: Instruction
    ) -> SpeculationEntry:
        """
        Lookup or create a SpeculationEntry() that is shared across
        RestartAnalysis calls.  Args are used only for debug checks.
        """
        if len(self.entries) == self.index:
            self.entries.append(
                SpeculationEntry(filename, lineno, instruction_pointer, inst)
            )
        entry = self.entries[self.index]
        prev_entry_msg = ""
        if self.index != 0:
            prev_entry = self.entries[self.index - 1]
            prev_entry_msg = (
                f"Previous instruction: {prev_entry.filename}:{prev_entry.lineno}"
                f"({prev_entry.inst.opname} @ {prev_entry.instruction_pointer})\n"
            )
        if not (
            entry.instruction_pointer == instruction_pointer
            and entry.filename == filename
            and entry.lineno == lineno
        ):
            raise SpeculationLogDivergence(
                f"""
SpeculationLog diverged at index {self.index} (log had {len(self.entries)} entries):
- Expected: {entry.filename}:{entry.lineno} ({entry.inst.opname} at ip={entry.instruction_pointer})
- Actual: {filename}:{lineno} ({inst.opname} at ip={instruction_pointer})
{prev_entry_msg}
There are two usual reasons why this may have occurred:
- When Dynamo analysis restarted, the second run took a different path than
  the first.  If this occurred, the previous instruction is the critical instruction that
  behaved differently.
- Speculation entries are only added under certain conditions (as seen in
  step()), e.g., there must exist operators in the graph; those conditions may
  have changed on restart.

If this divergence was intentional, clear the speculation log before restarting (do NOT
do this for graph breaks, you will infinite loop).

Otherwise, please submit a bug report, ideally including the contents of TORCH_LOGS=+dynamo
"""
            )
        self.index += 1
        return entry


@dataclasses.dataclass
class LocalState:
    automatic_dynamic: dict[str, FrameStateSizeEntry] = dataclasses.field(
        default_factory=dict
    )

    def render(self) -> str:
        return "\n".join(
            f"{k}: {v.render()}" for k, v in self.automatic_dynamic.items()
        )


# Mutable box that is shared across restarts
@dataclasses.dataclass
class DistributedState:
    compile_pg: Any
    local_state: LocalState
    all_states: Optional[list[LocalState]] = None


class TensorifyState:
    # These are the set of string symfloats names (eg. "zf0") that we collect
    # from the tensorify_python_scalars.py joint fx pass to inform us about
    # which float inputs we should specialize when we restart analysis.
    force_specializations: set[str] = set()

    @classmethod
    def specialize(cls, index: str) -> None:
        cls.force_specializations.add(index)

    @classmethod
    def should_specialize(cls, index: str) -> bool:
        return index in cls.force_specializations

    @classmethod
    def clear(cls) -> None:
        cls.force_specializations.clear()

    @classmethod
    def empty(cls) -> bool:
        return len(cls.force_specializations) == 0


@functools.cache
def _step_logger() -> Callable[..., None]:
    return torchdynamo_logging.get_step_logger(log)


@contextlib.contextmanager
def save_and_restart_speculation_log(
    tx: InstructionTranslatorBase,
) -> Generator[None, None, None]:
    # When reconstructing a generator after a graph break, we advance it until
    # it is fully exhausted. This process adds new entries to the speculation
    # log that were not previously observed. Without temporarily clearing the
    # speculation log, this could lead to a divergence error.

    entries = tx.speculation_log.entries
    index = tx.speculation_log.index
    try:
        tx.speculation_log.entries = []
        tx.speculation_log.index = 0
        yield
    finally:
        tx.speculation_log.entries = entries
        tx.speculation_log.index = index


@contextlib.contextmanager
def temporarely_allow_writes_to_output_graph(
    tx: InstructionTranslatorBase,
) -> Generator[None, None, None]:
    try:
        tmp = tx.output.should_exit
        tx.output.should_exit = False
        yield
    finally:
        tx.output.should_exit = tmp


@dataclasses.dataclass
class BlockStackEntry:
    # Current instruction that pushes something to block_stack
    inst: Instruction
    target: Instruction | None
    stack_index: int
    with_context: Optional[
        Union[ContextWrappingVariable, GenericContextWrappingVariable]
    ] = None

    def can_restore(self) -> bool:
        return self.with_context is not None

    def resume_fn(self) -> ReenterWith:
        assert self.stack_index is not None
        if (
            self.with_context
            and hasattr(self.with_context, "target_values")
            and self.with_context.target_values
        ):
            return ReenterWith(
                self.stack_index - 1, tuple(self.with_context.target_values)
            )
        else:
            return ReenterWith(self.stack_index - 1)

    def exit(
        self, tx: InstructionTranslatorBase, is_graph_break: bool
    ) -> VariableTracker | None:
        assert self.with_context is not None
        if (
            is_graph_break and self.with_context.exit_on_graph_break()
        ) or not is_graph_break:
            return self.with_context.exit(tx)  # type: ignore[arg-type]
        return None


class SpeculationLogDivergence(AssertionError):
    pass


class ReturnValueOp(Exception):
    pass


class YieldValueOp(Exception):
    """
    Signal to the symbolic tracer to stop and return control flow to the
    caller
    """


def stack_op(fn: Callable[..., object]) -> Callable[..., Any]:
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self: InstructionTranslator, inst: Instruction) -> None:
        self.push(fn_var.call_function(self, self.popn(nargs), {}))

    return impl


def is_stdlib(mod: object) -> bool:
    if not isinstance(mod, types.ModuleType):
        return False
    return mod.__name__.split(".")[0] in sys.stdlib_module_names


@functools.cache
def get_assert_bytecode_sequence(with_msg: bool) -> list[str]:
    if with_msg:

        def fn(x: Any) -> None:
            assert x, "msg"
    else:

        def fn(x: Any) -> None:
            assert x

    insts = [inst.opname for inst in dis.get_instructions(fn)]

    # expect to find POP_JUMP_[FORWARD_]IF_TRUE
    begin_idx = next(i for i, inst in enumerate(insts) if inst.startswith("POP_JUMP"))
    end_idx = insts.index("RAISE_VARARGS")

    return insts[begin_idx + 1 : end_idx + 1]


def _detect_and_normalize_assert_statement(
    self: InstructionTranslatorBase,
    truth_fn: Callable[[object], bool],
    push: bool,
) -> bool:
    # Detect if this jump instruction is assert and normalize the assert
    # by pushing dummy error message when nothing is given.
    #
    # Python 3.9-3.13 assertion is in following format (minus small differences)
    # 18 POP_JUMP_IF_TRUE       28
    # 20 LOAD_ASSERTION_ERROR
    # 22 LOAD_CONST               3 ('Assert message') -> optional instruction
    # 24 CALL_FUNCTION            1                    -> optional instruction
    # 26 RAISE_VARARGS

    if (truth_fn is not operator.truth) or push:
        return False

    assert isinstance(self.instruction_pointer, int)
    current_instruction_pointer = self.instruction_pointer

    for with_msg in (False, True):
        assert_insts = get_assert_bytecode_sequence(with_msg)
        cur_insts = self.instructions[
            current_instruction_pointer : current_instruction_pointer
            + len(assert_insts)
        ]
        cur_insts = [inst.opname for inst in cur_insts]
        if cur_insts == assert_insts:
            if with_msg:
                load_const_idx = assert_insts.index("LOAD_CONST")
                error_msg = self.instructions[
                    current_instruction_pointer + load_const_idx
                ].argval
            else:
                error_msg = "assertion error"
            self.push(ConstantVariable.create(error_msg))
            return True

    return False


explain = False


# [NOTE] graph break handling in symbolic_convert
# There are 4 possible graph break cases that InstructionTranslatorBase handles:
#   1. Regular graph breaks from CALL, BINARY_SUBSCR, etc. (implemented by break_graph_if_unsupported)
#   2. Data-dependent condition graph breaks (implemented by generic_jump)
#   4. All other unhandled graph breaks - unsupported step graph breaks (implemented in InstructionTranslatorBase.step)
#
# Graph breaks are handled in the following manner:
#   1. The Unsupported exception is caught. If we cannot compile a partial graph (should_compile_partial_graph() is False),
#      then propagate the exception upward. For unsupported step graph breaks, the condition to abort partial compilation is
#      more restrictive (see InstructionTranslatorBase.step).
#   2. If the Unsupported exception escapes symbolic_convert.py, then we are done.
#      Otherwise, we want to attempt partial compilation.
#      Log the graph break via log_graph_break. If we're handling a data-dependent graph break (type 2.), then we can immediately
#      codegen the compiled graph and resume function and we're done. This is because the jump instruction we graph break on is
#      limited in how it can manipulate Python state (say, in comparison, to CALL, which can modify Python state arbitrarily).
#      Otherwise, we need to restart compilation. We need to restart because by processing the unsupported instruction,
#      we may have modified the VariableTrackers, and we need all of our VariableTrackers to be in the state BEFORE tracing the
#      unsupported instruction.
#   3. During the first compilation, we updated a speculation log, indicating points in the code that we can resume from.
#      On the second compilation, we will stop tracing at the first speculation log that fails. Then we compile the partial
#      graph and resume function.
#
# Logging invariants:
#   1. No logs need to be made if Unsupported escapes symbolic_convert.py. Python's default exception printing will
#      print out all of the necessary information and no partial compilation will be attempted.
#   2. log_graph_break should be called as soon as Unsupported is caught and we determined we want to partial compile.
#      This always happens on the first compilation, NOT the restart handling this graph
#   3. Any compile_subgraph call should be preceded immediately by a log in the form of "... triggered compile".


def generic_jump(
    truth_fn: Callable[[object], bool], push: bool
) -> Callable[[InstructionTranslatorBase, Instruction], None]:
    def raise_jump_graph_break(value: VariableTracker) -> NoReturn:
        unimplemented(
            gb_type="Data-dependent branching",
            context=f"attempted to jump with {value}",
            explanation="Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). "
            "Dynamo does not support tracing dynamic control flow.",
            hints=[
                *graph_break_hints.FUNDAMENTAL,
                "Use `torch.cond` to express dynamic control flow.",
            ],
        )

    def jump_graph_break(
        self: InstructionTranslatorBase,
        inst: Instruction,
        value: VariableTracker,
        extra_msg: str = "",
    ) -> None:
        assert self.should_compile_partial_graph()

        exc = None
        try:
            raise_jump_graph_break(value)
        except Unsupported as e:
            exc = e

        assert exc is not None

        # compile a partial subgraph prefix then skip the rest of user code
        if self.maybe_has_backedge():
            self.raise_loop_graph_break(self.f_code, exc)

        self.log_graph_break(
            self.code_options,
            reason=str(exc),
            exc=exc,
        )

        self.push(value)
        log.debug("generic_jump triggered compile")
        all_stack_locals_metadata = self.output.compile_subgraph(
            self,
            reason=GraphCompileReason(
                f"generic_jump {typestr(value)}{extra_msg}", [self.frame_summary()]
            ),
            stack_pops=1,
        )
        self.pop()

        if_next = self.create_call_resume_at(
            self.next_instruction,
            all_stack_locals_metadata,
        )
        if push:
            self.push(value)
        assert inst.target is not None
        if_jump = self.create_call_resume_at(
            inst.target,
            all_stack_locals_metadata,
        )

        if sys.version_info >= (3, 13):
            # 3.13 requires stack[-1] to be bool type
            self.output.add_output_instructions([create_instruction("TO_BOOL")])

        jump_inst = create_instruction(inst.opname, target=if_jump[0])
        jump_inst.copy_positions(inst)
        self.output.add_output_instructions([jump_inst] + if_next + if_jump)

    def inner(self: InstructionTranslatorBase, inst: Instruction) -> None:
        value: VariableTracker = self.pop()
        if (
            config.rewrite_assert_with_torch_assert
            and _detect_and_normalize_assert_statement(self, truth_fn, push)
        ):
            error_msg: VariableTracker = self.pop()
            # Skip over things like `assert True`
            if value.is_python_constant():
                if bool(value.as_python_constant()):
                    return self.jump(inst)
                elif self.should_compile_partial_graph():
                    jump_graph_break(self, inst, value)
                else:
                    unimplemented(
                        gb_type="Data-dependent assertion failed (cannot compile partial graph)",
                        context=f"value: {value}",
                        explanation="Dynamo has determined when encountering a data-dependent assert failure "
                        "that it should not compile the partial graph.",
                        hints=[
                            *graph_break_hints.FUNDAMENTAL,
                            "Use `torch._assert()` to raise a hard AssertionError when the check fails. "
                            "This error will propagate back the user code "
                            "that called the compiled function (i.e. Dynamo will not trace any exception handling).",
                            "Remove the assert statement.",
                            "Move the assert statement outside of any context managers in order to graph break with "
                            "partial graph compilation (if fullgraph=False).",
                        ],
                    )

            # TODO maybe should respect DtoH sync intention of users later??
            # Manually insert torch._assert_async instead of python assert and jump over
            # assert related instructions as we don't need them anymore.

            # if we see Tensor as assert statement, no need to call scalar_tensor
            if value.is_tensor():
                self.output.create_proxy(
                    "call_function",
                    torch._assert_async,
                    *proxy_args_kwargs((value, error_msg), {}),
                )
                self.jump(inst)
                return

            if isinstance(value, SymNodeVariable):
                # if the assertion is normal shape expression.
                # just install guard and bail out.
                sym_expr = value.sym_num
                if not isinstance(sym_expr, torch.SymBool):
                    sym_expr = sym_expr != 0

                result = torch.fx.experimental.symbolic_shapes.expect_true(sym_expr)
                if not result:
                    unimplemented(
                        gb_type="Assertion failed on symbolic shapes",
                        context=str(sym_expr),
                        explanation="",
                        hints=[*graph_break_hints.USER_ERROR],
                    )
                self.jump(inst)
                return

            scalar_to_tensor_proxy = self.output.create_proxy(
                "call_function", torch.scalar_tensor, *proxy_args_kwargs((value,), {})
            )

            scalar_to_tensor = wrap_fx_proxy(
                self,
                scalar_to_tensor_proxy,
                example_value=get_fake_value(scalar_to_tensor_proxy.node, self),
            )

            self.output.create_proxy(
                "call_function",
                torch._assert_async,
                *proxy_args_kwargs((scalar_to_tensor, error_msg), {}),
            )
            self.jump(inst)
            return

        if value.is_python_constant():
            # ConstDictVariable is optimized to be very lazy about insertion of
            # guards, so we have to manually insert a SEQUENCE_LENGTH guard
            # here.
            if isinstance(value, ConstDictVariable) and value.source:
                install_guard(value.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            if truth_fn(value.as_python_constant()):
                if push:
                    self.push(value)
                self.jump(inst)
        elif value.is_tensor() and self.should_compile_partial_graph():
            jump_graph_break(self, inst, value)
        elif isinstance(value, NNModuleVariable):
            # Equivalent of "self.nn_module is not None"
            mod = self.output.get_submodule(value.module_key)
            if truth_fn(mod):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, UserDefinedObjectVariable):
            try:
                x = value.var_getattr(self, "__bool__")  # type: ignore[arg-type]
            except exc.ObservedAttributeError:
                exc.handle_observed_exception(self)
                # if __bool__ is missing, trying __len__ to infer a truth value.
                try:
                    x = value.var_getattr(self, "__len__")  # type: ignore[arg-type]
                except exc.ObservedAttributeError:
                    exc.handle_observed_exception(self)
                    x = None

            # __bool__ or __len__ is function
            if isinstance(x, UserMethodVariable):
                result = x.call_function(self, [], {})  # type: ignore[arg-type, assignment]
                method_name = getattr(getattr(x, "fn", None), "__name__", None)
                if result.is_python_constant():
                    result_value = result.as_python_constant()
                    if method_name == "__bool__" and not isinstance(result_value, bool):
                        msg = variables.ConstantVariable.create(
                            f"__bool__ should return bool, returned {type(result_value).__name__}"
                        )
                        exc.raise_observed_exception(TypeError, self, args=[msg])
                    if isinstance(result_value, (bool, int)) and truth_fn(result_value):
                        if push:
                            self.push(value)
                        self.jump(inst)
                elif isinstance(result, SymNodeVariable):
                    if result.evaluate_expr():
                        if push:
                            self.push(value)
                        self.jump(inst)
                else:
                    unimplemented(
                        gb_type="Data-dependent branching with non-constant __bool__",
                        context=f"method: {x}, result: {result}",
                        explanation="Attempted to perform data-dependent branching on a user-defined "
                        "object with a __bool__ method that did not return a constant.",
                        hints=[],
                    )
            # __bool__ or __len__ is non-function or not existed in the user defined object
            else:
                if truth_fn(True):
                    if push:
                        self.push(value)
                    self.jump(inst)
        elif not value.is_tensor() and value.has_unpack_var_sequence(self):
            if truth_fn(len(value.unpack_var_sequence(self))):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, SymNodeVariable):
            try:
                # if the user is branching on a SymBool, guard on it
                # if the user has code like:
                #    if size:
                #        ...
                # then they are just testing truthiness: guard that the expr != 0
                if isinstance(value.sym_num, torch.SymBool):
                    eval_result = value.evaluate_expr(self.output)
                else:
                    eval_result = guard_bool(value.sym_num != 0)
            except exc.UserError as e:
                if self.should_compile_partial_graph():
                    return jump_graph_break(self, inst, value, extra_msg=f"\n{e}")
                raise
            if truth_fn(eval_result):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, variables.BackwardHookVariable):
            if truth_fn(True):
                if push:
                    self.push(value)
                self.jump(inst)
        else:
            from .source import is_constant_source

            if value.source is not None and is_constant_source(value.source):
                if truth_fn(value.get_real_value()):  # type: ignore[attr-defined]
                    if push:
                        self.push(value)
                    self.jump(inst)
            else:
                raise_jump_graph_break(value)

    return inner


# NOTE: for the purposes of nested graph breaks, break_graph_if_unsupported only works on instructions
# with 0 or 1 outputs. If you wish to support bytecodes with 2+ outputs, either rewrite the instruction
# into a sequence of simpler instructions, or file an issue for consultation.
# There is an additional requirement that if the instruction causes a function call, e.g. STORE_ATTR,
# nothing should happen to the result of the function call.
def break_graph_if_unsupported(
    *, push: bool, msg_prefix: str
) -> Callable[
    [Callable[..., None]], Callable[[InstructionTranslatorBase, Instruction], None]
]:
    def decorator(
        inner_fn: Callable[..., None],
    ) -> Callable[[InstructionTranslatorBase, Instruction], None]:
        @functools.wraps(inner_fn)
        def wrapper(self: InstructionTranslatorBase, inst: Instruction) -> None:
            prev_push = self.current_instruction_push
            self.current_instruction_push = push
            speculation = self.speculate()
            if speculation.failed(self):
                # no need to restore current_instruction_push if speculation failed
                assert speculation.reason is not None
                return handle_graph_break(self, inst, speculation.reason)
            try:
                return inner_fn(self, inst)
            except Unsupported as excp:
                if self.active_generic_context_managers:
                    # raise original graph break if fullgraph/error_on_graph_break=True
                    if self.one_graph or self.error_on_graph_break:
                        raise

                    # We don't support graph break under GenericContextWrappingVariable,
                    # If there is, we roll back to the checkpoint and fall back.
                    excp.remove_from_stats()
                    unimplemented(
                        gb_type="Graph break under GenericContextWrappingVariable",
                        context=f"Active generic context managers: {self.active_generic_context_managers}",
                        explanation="Attempted to graph break in an active context manager(s) that doesn't support graph breaking.",
                        hints=[
                            "Move the offending context manager(s) to outside the compiled region.",
                            *graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK,
                        ],
                        from_exc=excp,
                    )

                if excp.skip_frame:
                    raise

                if not self.should_compile_partial_graph():
                    raise

                if self.maybe_has_backedge():
                    self.raise_loop_graph_break(self.f_code, excp)

                self.log_graph_break(
                    self.code_options,
                    reason=f"{msg_prefix}:\n\n{str(excp)}",
                    exc=excp,
                )

                excp.remove_from_stats()
                excp.add_to_stats("graph_break")
                speculation.reason = GraphCompileReason(excp.msg, excp.real_stack)
            finally:
                self.current_instruction_push = prev_push
            speculation.fail_and_restart_analysis(self.error_on_graph_break)

        def handle_graph_break(
            self: InstructionTranslatorBase,
            inst: Instruction,
            reason: GraphCompileReason,
        ) -> None:
            if (
                sys.version_info >= (3, 11)
                and sys.version_info < (3, 12)
                and inst.opname == "CALL"
            ):
                # stack effect for PRECALL + CALL is split between the two instructions
                stack_effect = dis.stack_effect(
                    dis.opmap["PRECALL"], inst.arg
                ) + dis.stack_effect(dis.opmap["CALL"], inst.arg)
            else:
                stack_effect = dis.stack_effect(inst.opcode, inst.arg)

            log.debug("%s triggered compile", inst.opname)
            all_stack_locals_metadata = self.output.compile_subgraph(
                self, reason=reason, stack_pops=int(push) - stack_effect
            )
            cg = PyCodegen(self.output.root_tx)
            cleanup: list[Instruction] = []
            # Reconstruct the context variable CLASS in the block stack
            for b in self.block_stack:
                # Don't exit any modes we have entered,
                # output bytecode will mutate the tf mode stack accordingly
                if isinstance(b.with_context, TorchFunctionModeVariable):
                    cg.extend_output(
                        b.resume_fn().try_except_torch_function_mode(
                            cg.code_options, cleanup
                        )
                    )
                    continue
                assert b.with_context is not None
                assert isinstance(b.with_context, (ContextWrappingVariable))
                b.with_context.reconstruct_type(cg)
                cg.extend_output(b.resume_fn().try_finally(cg.code_options, cleanup))
            self.output.add_output_instructions(cg.get_instructions())
            del cg

            if sys.version_info >= (3, 11) and inst.opname == "CALL":
                kw_names = (
                    self.kw_names.as_python_constant()
                    if self.kw_names is not None
                    else ()
                )
                if len(kw_names) > 0:
                    # KW_NAMES no longer used in 3.13
                    assert sys.version_info < (3, 13)
                    self.output.add_output_instructions(
                        [create_instruction("KW_NAMES", argval=kw_names)]
                    )
                assert inst.arg is not None
                call_insts = create_call_function(inst.arg, False)
                call_insts[-1].copy_positions(inst)
                self.output.add_output_instructions(call_insts)
            else:
                # copy instruction, but without exception table data
                assert inst.target is None
                inst_copy = copy.copy(inst)
                inst_copy.exn_tab_entry = None
                self.output.add_output_instructions([inst_copy])

            self.output.add_output_instructions(cleanup)

            self.popn(int(push) - stack_effect)
            if push:
                self.push(UnknownVariable())
            self.output.add_output_instructions(
                self.create_call_resume_at(
                    self.next_instruction,
                    all_stack_locals_metadata,
                )
            )

        return wrapper

    return decorator


class BytecodeDispatchTableMeta(type):
    """Installs a `cls.dispatch_table` on every subclass to speed up calls to self.OPCODE()"""

    def __init__(cls: type, name: str, bases: Any, dct: Any) -> None:
        super().__init__(name, bases, dct)  # type: ignore[misc]

        def _missing(opname: str, *args: Any) -> None:
            unimplemented(
                gb_type="Missing bytecode handler",
                context=f"{opname} with args {args}",
                explanation=f"Dynamo does not know how to handle the bytecode instruction `{opname}`.",
                hints=[
                    f"Do not trace code that produces the `{opname}` bytecode instruction "
                    "(see https://docs.python.org/3/library/dis.html for bytecode semantics).",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        dispatch_table = {
            op: getattr(cls, opname, functools.partial(_missing, opname))
            for opname, op in dis.opmap.items()
        }
        # pyrefly: ignore [missing-attribute]
        cls.dispatch_table = [dispatch_table.get(i) for i in range(2**8)]


@dataclasses.dataclass
class ExceptionStack:
    """
    Exception stack that it is shared among all InstructionTranslator instances
    """

    # Exception handling in CPython is a bit confusing and some of the bytecode
    # have a slightly different behavior than what is documented. While reading
    # the documentation, is important to notice that the terms "current exception"
    # and "stack" sometimes refers to a C variable with the same name and the
    # exception stack, respectively.
    #
    # The lifetime of an exception is (Python 3.11+):
    #  + tx._raise_exception_variable(...) := sets the current_exception variable
    #  + PUSH_EXC_INFO := pushes the current_exception to the *exception stack*
    #  + POP_EXCEPT := pops TOS from the *exception stack*

    _exc_stack: list[ExceptionVals] = dataclasses.field(default_factory=list)
    _current_exception: Optional[ExceptionVals] = dataclasses.field(default=None)

    def clear_current_exception(self) -> None:
        self._current_exception = None

    def set_current_exception(self, val: ExceptionVals) -> None:
        self._set_context_and_break_context_reference_cycle(val)
        self._current_exception = val

    def move_current_exception_to_stack(self) -> None:
        assert self._current_exception is not None
        self.append(self._current_exception)
        self.clear_current_exception()

    def get_current_exception(self) -> ExceptionVals:
        assert self._current_exception is not None
        return self._current_exception

    def _set_context_recursive(
        self, val: ExceptionVals, prev_idx: int
    ) -> ExceptionVals:
        if (ctx := val.__context__) and type(ctx) is not ConstantVariable:  # type: ignore[union-attr]
            return val
        if len(self._exc_stack) + prev_idx > 0:
            prev = self._exc_stack[prev_idx]
            self._set_context_recursive(prev, prev_idx - 1)
            val.set_context(prev)  # type: ignore[union-attr, arg-type]
        return val

    def _break_context_reference_cycle(self, val: ExceptionVals) -> None:
        # See test_exceptions::test_raise_does_not_create_context_chain_cycle
        # Based on https://github.com/python/cpython/blob/e635bf2e49797ecb976ce45a67fce2201a25ca68/Python/errors.c#L207-L228
        # As noted on CPython, this is O(chain length) but the context chains
        # are usually very small
        o = slow_o = val
        slow_update_toggle = False  # floyd's algorithm for detecting cycle
        while True:
            context = o.__context__  # type: ignore[union-attr]
            if type(context) is ConstantVariable:  # context not set
                break

            if context is val:
                o.set_context(ConstantVariable(None))  # type: ignore[union-attr, arg-type]
                break

            o = context  # type: ignore[assignment]
            if o is slow_o:
                # pre-existing cycle - all exceptions on the path were
                # visited and checked
                break

            if slow_update_toggle:
                # visited all exceptions
                slow_o = slow_o.__context__  # type: ignore[union-attr, assignment]
            slow_update_toggle = not slow_update_toggle

    def _set_context_and_break_context_reference_cycle(
        self, val: ExceptionVals
    ) -> None:
        # set Exception.__context__
        self._set_context_recursive(val, len(self._exc_stack) - 1)
        self._break_context_reference_cycle(val)

    def pop(self) -> ExceptionVals:
        return self._exc_stack.pop()

    def append(self, val: ExceptionVals) -> None:
        self._exc_stack.append(val)

    def __len__(self) -> int:
        return len(self._exc_stack)

    def __getitem__(self, index: int) -> ExceptionVals:
        return self._exc_stack[index]

    def __str__(self) -> str:
        return f"{self._exc_stack=} - {self._current_exception=}"

    __repr__ = __str__


class InstructionTranslatorBase(
    metaclass=BytecodeDispatchTableMeta,
):
    output: OutputGraph
    symbolic_locals: dict[str, VariableTracker]
    symbolic_globals: dict[str, VariableTracker]
    symbolic_torch_function_state: SymbolicTorchFunctionState
    symbolic_stream_state: SymbolicStreamState
    post_prune_cell_and_freevars: Optional[dict[str, VariableTracker]]
    stack: list[VariableTracker]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    current_instruction_push: bool
    block_stack: list[BlockStackEntry]
    lineno: int
    kw_names: Optional[ConstantVariable]
    accept_prefix_inst: bool
    prefix_insts: list[Instruction]
    inline_depth: int
    inconsistent_side_effects: bool
    current_speculation: Optional[SpeculationEntry]
    dispatch_table: list[Any]
    exn_vt_stack: ExceptionStack
    exec_recorder: Optional[ExecutionRecorder]
    strict_checks_fn: Optional[Callable[[VariableTracker], bool]]
    start_point: Optional[int]
    is_leaf_tracer: bool
    parent: Optional[InstructionTranslatorBase]
    debug_locals: list[tuple[VariableTracker, list[VariableTracker]]]
    package: Optional[CompilePackage]
    latest_bytecode_queue: deque[str]
    # Store the latest bytecode before graph_break() call by user

    def mark_inconsistent_side_effects(self) -> None:
        """
        InstructionTranslator has encountered instructions which may cause
        dynamo to see a different version of history from eager
        See: https://github.com/pytorch/pytorch/issues/110765
        """
        self.inconsistent_side_effects = True

    def maybe_has_backedge(self) -> bool:
        # This function employs a heuristic. It does not reliably detect a backedge.
        # The heuristic is straightforward: starting from the current instruction and
        # continuing to the end, if any jump instruction targets an instruction before
        # the current one, there might be a backedge.

        # Python 3.12 introduced changes to bytecode that group common paths in
        # blockstacks (with or try...else) and allow for early returns. Consequently,
        # there can be multiple RETURN_VALUE instructions. Another heuristic is to
        # halt detection upon encountering the first RETURN_VALUE or RETURN_CONST.

        # These heuristics can result in both false positives and negatives, but
        # in either case, the Dynamo code remains valid. For false positives
        # (where an edge is incorrectly marked as a backedge), Dynamo will
        # graph break with a frame skip instead of potentially applying optimizations. For
        # false negatives (where an edge that should be marked as a backedge
        # isn't), multiple graphs may be generated if there's a break in the
        # graph during a for loop. In general, its better to have fewer false
        # negatives so that Dynamo does not skip the whole frame.

        # If any parent tx has a backedge, then return True
        cur_tx: Optional[InstructionTranslatorBase] = self
        while cur_tx is not None:
            cur_offset = cur_tx.current_instruction.offset
            assert cur_tx.instruction_pointer is not None
            for inst in cur_tx.instructions[cur_tx.instruction_pointer :]:
                if inst.opname in ("RETURN_VALUE", "RETURN_CONST"):
                    break
                if inst.opname in JUMP_OPNAMES:
                    jump_offset = inst.argval
                    if jump_offset < cur_offset:
                        return True
            cur_tx = cur_tx.parent
        return False

    def cellvars(self) -> list[str]:
        return self.code_options["co_cellvars"]

    def freevars(self) -> list[str]:
        return self.code_options["co_freevars"]

    def cell_and_freevars(self) -> list[str]:
        if not hasattr(self, "_cell_and_freevars"):
            self._cell_and_freevars = self.cellvars() + self.freevars()
        return self._cell_and_freevars

    def prune_dead_locals(self) -> None:
        # keep cell and freevar references alive
        self.post_prune_cell_and_freevars = {
            k: v
            for k, v in self.symbolic_locals.items()
            if k in self.cell_and_freevars()
        }
        # Only keep the locals that must remain on the stack.
        reads = livevars_analysis(self.instructions, self.current_instruction)
        self.symbolic_locals = {
            k: v for k, v in self.symbolic_locals.items() if k in reads
        }

    def call_function(
        self,
        fn: VariableTracker,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> None:
        assert isinstance(fn, VariableTracker)
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        assert all(
            isinstance(x, VariableTracker)
            for x in itertools.chain(args, kwargs.values())
        )
        inner_fn = None
        if hasattr(fn, "value"):
            inner_fn = fn.value
        if hasattr(fn, "fn"):
            inner_fn = fn.fn
        if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
            raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
        self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]

    def inline_generator_function(
        self, fn: VariableTracker, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Any:
        """
        Redirect the call to the generator "call_function"
        """
        if not isinstance(fn, LocalGeneratorFunctionVariable):
            fn = LocalGeneratorFunctionVariable(fn)  # type: ignore[arg-type]
        return fn.call_function(self, args, kwargs)  # type: ignore[arg-type]

    def inline_user_function_return(
        self, fn: VariableTracker, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Any:
        """
        A call to some user defined function by inlining it.
        """
        self.is_leaf_tracer = False
        if config.enable_faithful_generator_behavior and is_generator(fn.get_code()):  # type: ignore[attr-defined]
            return self.inline_generator_function(fn, args, kwargs)
        else:
            return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

    def get_line_of_code_header(self, lineno: Optional[int] = None) -> str:
        if lineno is None:
            lineno = self.lineno
        inline_depth_str = (
            f" (inline depth: {self.inline_depth})" if self.inline_depth > 0 else ""
        )
        funcname = get_funcname(self.f_code.co_filename, lineno)
        funcname_str = "" if funcname is None else f" ({funcname})"
        return f"{self.f_code.co_filename}:{lineno} in {self.f_code.co_name}{funcname_str}{inline_depth_str}"

    def get_log_starts_line_log_str(self) -> str:
        log_str = f"TRACE starts_line {self.get_line_of_code_header()}\n"
        line = linecache.getline(self.f_code.co_filename, self.lineno).rstrip()
        log_str += f"    {line}"
        return log_str

    def starts_line(self, lineno: int) -> None:
        if self.lineno == lineno:
            return
        self.lineno = lineno
        TracingContext.set_current_loc(
            self.f_code.co_filename, lineno, self.f_code.co_name
        )

        if self.is_trace_source_log_enabled:
            trace_source_log.debug("%s", LazyString(self.get_log_starts_line_log_str))

    def step(self) -> bool:
        """Process exactly one instruction, return False we should exit"""
        self.error_on_graph_break = _get_error_on_graph_break()

        ip = self.instruction_pointer
        if ip is None:
            return False
        self.current_instruction = inst = self.instructions[ip]
        self.instruction_pointer = ip + 1

        if inst.starts_line:
            self.starts_line(inst.starts_line)

        if (
            not self.stack
            and self.should_compile_partial_graph()
            and self.is_non_empty_graph()
        ):
            self.current_speculation = self.speculate()
            if self.current_speculation.failed(self):
                self.step_graph_break(inst)
                return False

        if self.is_trace_bytecode_log_enabled:
            trace_bytecode_log.debug(
                "TRACE %s %s %s", inst.opname, inst.argval, repr(self.stack)
            )

        # Store the latest 20 bytecode execution for the process,
        # Used repr for byte processing and limiting the length to 2048
        if config.verbose:
            try:
                stack_repr = repr(self.stack)
            except ValueError:
                # Handle large integers that exceed sys.int_info.str_digits_check_threshold
                stack_repr = "<self.stack repr truncated due to large integer>"
            self.latest_bytecode_queue.append(
                f"TRACE {inst.opname} {repr(inst.argval)} {stack_repr}"
            )

        self.update_block_stack(inst)

        try:
            self.dispatch_table[inst.opcode](self, inst)
            return not self.output.should_exit
        except TensorifyScalarRestartAnalysis:
            raise
        except exc.ObservedException as e:
            self.exception_handler(e)
            return True
        except (ReturnValueOp, YieldValueOp):
            return False
        except (Unsupported, StepUnsupported) as e:
            # More restrictive condition than should_compile_partial_graph:
            # if this condition is true, then we SHOULD NOT attempt to find
            # a previous checkpoint to resume from and try to resume - we should
            # immediately error out.
            # The condition is more restrictive because, it may be possible to resume significantly earlier
            # in the code (the most recent speculation point). This happens, for example, in the case
            # of a graph break in a try block.
            if (
                self.one_graph
                or self.error_on_graph_break
                or self.is_tracing_resume_prologue
                or (isinstance(e, Unsupported) and e.skip_frame)
            ):
                if isinstance(e, StepUnsupported):
                    unimplemented(
                        gb_type="cannot resume from torch._dynamo.step_unsupported()",
                        context="",
                        explanation="traced torch._dynamo.step_unsupported(), but Dynamo is instructed "
                        "to error on graph break. This graph break is used for debugging only.",
                        hints=[
                            "Remove the torch._dynamo.step_unsupported() call.",
                            "Make sure fullgraph=False and error_on_graph_break=False.",
                            *graph_break_hints.DYNAMO_BUG,
                        ],
                    )
                raise
            if self.current_speculation is None:
                log.debug("empty checkpoint - cannot resume from graph break")
                if isinstance(e, StepUnsupported):
                    unimplemented(
                        gb_type="torch._dynamo.step_unsupported() with empty checkpoint",
                        context="",
                        explanation="traced torch._dynamo.step_unsupported(), but there is no checkpoint "
                        "to step_graph_break from. This graph break is used for debugging only.",
                        hints=[
                            "Remove the torch._dynamo.step_unsupported() call.",
                            "Include at least one checkpoint: (1) include at least 2 ops and (2) make sure there is some "
                            "line of code that is not in a try/with block, and has an empty Python stack.",
                            *graph_break_hints.DYNAMO_BUG,
                        ],
                        skip_frame=True,
                    )
                assert isinstance(e, Unsupported)
                e.skip_frame = True
                raise
            reason = (
                "Encountered graph break that we cannot resume from. "
                "Compiling up to the previous resumable state, "
                "then skipping the rest of the function. "
                f"Graph break encountered:\n\n{str(e)}"
            )
            self.log_graph_break(
                self.code_options,
                reason=reason,
                exc=e,
            )

        self.current_speculation.fail_and_restart_analysis(self.error_on_graph_break)
        return False

    if sys.version_info >= (3, 11):

        def update_block_stack(self, inst: Instruction) -> None:
            # 3.11+ no longer uses a block stack, but we still keep track of one
            # so that we know which contexts are currently active.
            # For our purposes, all exception table entries with the same target
            # are considered to be part of the same "block".
            # NOTE: we only keep track of with blocks that are not contained in try blocks.
            # This is because we will not create continuation functions on graph breaks in try blocks,
            # but we may for with blocks. We do not push blocks here since
            # with blocks are pushed when handling BEFORE_WITH.
            entry = inst.exn_tab_entry
            if entry:
                # Detect when we have exited the top with block.
                # The with blocks on the block stack are not enclosed in try
                # blocks, so a with block's cleanup code should be in the
                # previous with block (if any).
                if (
                    len(self.block_stack) >= 2
                    and entry.target is not self.block_stack[-1].target
                    and entry.target is self.block_stack[-2].target
                ):
                    # exit the current block
                    self.block_stack.pop()
            else:
                # no longer in any block
                # It is possible for NOPs to be between two instructions
                # in the same block, but the NOPs are not covered by an
                # exception table entry. In this case, assume that we
                # are still in the same block.
                # In 3.12+, JUMP_BACKWARD might also not be covered by
                # an exception table entry, so we also assume that we
                # are still in the same block. It is probably safe to do
                # this in 3.11, even though we haven't encountered this case before.
                # In 3.14+, NOT_TAKEN might also not be covered by an exn table entry.
                if self.block_stack and inst.opname not in (
                    "NOP",
                    "JUMP_BACKWARD",
                    "NOT_TAKEN",
                ):
                    # If we really escape from a block and the current
                    # instruction is not in another block, then there
                    # should be no other nested blocks that we are in.
                    assert len(self.block_stack) == 1
                    self.block_stack.pop()

    else:

        def update_block_stack(self, inst: Instruction) -> None:
            pass

    @property
    def next_instruction(self) -> Instruction:
        assert self.instruction_pointer is not None
        return self.instructions[self.instruction_pointer]

    def step_graph_break(self, continue_inst: Instruction) -> None:
        # generate code from checkpoint
        assert not self.output.output_instructions
        assert self.current_speculation is not None
        # NOTE: adding an assert here since it seems like the only place
        # where we call step_graph_break right now is when the stack is empty,
        # so let's enforce that for now.
        assert not self.stack
        # NOTE: if we support non-empty self.stack in the future, the `stack_pops` argument
        # below should be set to the stack length to ensure that the stack is codegen'd
        # for the rest of the function.
        log.debug("step triggered compile")
        all_stack_locals_metadata = self.output.compile_subgraph(
            self,
            partial_convert=True,
            reason=GraphCompileReason("step_unsupported", [self.frame_summary()]),
        )
        # current frame state
        # cells,
        # [
        #   frame N locals,
        #   frame N-1 stack + locals,
        #   ...,
        #   frame 1 stack + locals,
        # ],
        if self.parent:
            from .eval_frame import skip_code

            # nested graph break
            assert config.nested_graph_breaks
            cg = PyCodegen(self.output.root_tx)

            # codegen cells and frame values only for frame N
            cg.extend_output(
                [
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    create_instruction("BUILD_LIST", arg=1),
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    create_instruction("BUILD_LIST", arg=1),
                ]
            )
            # No need to fix stack, since stack is assumed to be empty here.
            # Do NOT handle_inactive_ctx because we will be skipping this resume code.
            leaf_resume_code, leaf_resume_name = self.create_resume(
                0, continue_inst, all_stack_locals_metadata[0], [], cg, True, False
            )
            skip_code(leaf_resume_code)

            # current frame state
            # cells,
            # [
            #   frame N locals,
            #   frame N-1 stack + locals,
            #   ...,
            #   frame 1 stack + locals,
            # ], [frame N cells], [frame N locals],
            self.codegen_call_resume([leaf_resume_code], [leaf_resume_name], cg)

            # current frame state
            # cells,
            # [
            #   frame N locals,
            #   frame N-1 stack + locals,
            #   ...,
            #   frame 1 stack + locals,
            # ], leaf_resume result

            # pop frame N cells and locals
            cg.extend_output(
                [
                    *create_copy(2),
                    cg.create_load_const(0),
                    create_instruction("DELETE_SUBSCR"),
                    *create_copy(3),
                    cg.create_load_const(0),
                    create_instruction("DELETE_SUBSCR"),
                ]
            )

            # add the leaf_resume result to frame N-1 stack
            num_stack = all_stack_locals_metadata[1].num_stack
            cg.extend_output(
                [
                    create_instruction("BUILD_LIST", arg=1),
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    *create_binary_slice(num_stack, num_stack, True),
                ]
            )
            self.parent.push(UnknownVariable())
            all_stack_locals_metadata[1].num_stack += 1

            # current frame state
            # cells, frame_values
            # extract frame N-1 stack to stack
            cg.extend_output(
                [
                    create_dup_top(),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    *create_binary_slice(0, num_stack + 1),
                ]
            )

            # current frame state
            # cells, frame_values, frame N-1 stack + leaf_resume result
            # remove frame N-1 stack from frame_values
            cg.extend_output(
                # frame_values[0] = frame_values[0][num_stack + 1:]
                [
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    create_dup_top(),
                    *create_binary_slice(num_stack + 1, None),
                    *create_swap(2),
                    cg.create_load_const(0),
                    create_instruction("STORE_SUBSCR"),
                ]
            )

            # current frame state
            # cells, frame_values, frame N-1 stack + leaf_resume result
            # unpack the stack (need to unpack twice since UNPACK_SEQUENCE unpacks in reverse order)
            cg.extend_output(
                [
                    create_instruction("UNPACK_SEQUENCE", arg=num_stack + 1),
                    create_instruction("BUILD_LIST", arg=num_stack + 1),
                    create_instruction("UNPACK_SEQUENCE", arg=num_stack + 1),
                ]
            )

            # call the remaining resume functions
            # current frame state
            # [frame N-1 cells, ..., frame 1 cells],
            # [
            #   frame N-1 locals,
            #   frame N-2 stack + locals,
            #   ...,
            #   frame 1 stack + locals,
            # ], *(frame N-1 stack), leaf_resume result
            self.output.add_output_instructions(
                cg.get_instructions()
                + self.parent.create_call_resume_at(
                    self.parent.next_instruction, all_stack_locals_metadata[1:]
                )
            )
        else:
            # pop cells
            self.output.add_output_instructions(
                [
                    *create_swap(2),
                    create_instruction("POP_TOP"),
                ]
            )
            # load locals from frame values
            cg = PyCodegen(self.output.root_tx)
            self.output.add_output_instructions(
                [
                    cg.create_load_const(-1),
                    cg.create_binary_subscr(),
                ]
            )
            for local, idx in all_stack_locals_metadata[-1].locals_names.items():
                self.output.add_output_instructions(
                    [
                        create_dup_top(),
                        cg.create_load_const(idx),
                        cg.create_binary_subscr(),
                        cg.create_store(local),
                    ]
                )
            self.output.add_output_instructions(
                [
                    create_instruction("POP_TOP"),
                    create_jump_absolute(continue_inst),
                    *self.instructions,
                ]
            )

    def run_ctx_mgr(self) -> Any:
        # NB: Don't push the top level frame summary; set_current_loc will
        # take care of it.  However, DO make sure we attach real_stack to
        # exceptions
        return TracingContext.current_frame(None)

    def run(self) -> None:
        with self.run_ctx_mgr():
            dump_file(self.f_code.co_filename)
            try:
                self.output.push_tx(self)
                self.start_point = self.instruction_pointer
                try:
                    while self.step():
                        pass
                except Exception as e:
                    if self.is_tracing_resume_prologue:
                        raise ResumePrologueTracingError(
                            "Error while tracing through a Dynamo-generated resume function prologue. "
                            "Errors are not allowed when tracing resume function prologues.\n"
                            f"{type(e).__qualname__}: {str(e)}"
                        ).with_traceback(e.__traceback__) from None
                    raise
            except TensorifyScalarRestartAnalysis:
                raise
            except BackendCompilerFailed:
                raise
            except RuntimeError as e:
                # If the root tx fails to handle the graph break, then the caller (convert_frame)
                # will skip the frame and fall back to eager.
                # This code path happens e.g. for bytecodes we don't support
                # or when we are unable to resume from a graph break.
                if (
                    isinstance(e, Unsupported)
                    and isinstance(self, InstructionTranslator)
                    and not self.error_on_graph_break
                    and not self.one_graph
                ):
                    # log graph break if we won't error
                    reason = (
                        "Failed to handle graph break gracefully. "
                        "Skipping the function and falling back to eager. "
                        f"Graph break encountered:\n\n{str(e)}"
                    )
                    self.log_graph_break(
                        self.code_options,
                        reason=reason,
                        exc=e,
                    )

                if hasattr(e, "msg") and "Data-dependent" in e.msg:
                    readable_graph = torch.fx.GraphModule(
                        self.output.nn_modules, self.output.graph
                    ).print_readable(
                        print_output=False, include_stride=True, include_device=True
                    )
                    e.partial_fx_graph = readable_graph  # type: ignore[attr-defined]
                    raise

                raise
            except Exception as e:
                if self.exec_recorder:
                    e.exec_record = self.exec_recorder.get_record()  # type: ignore[attr-defined]

                raise
            finally:
                self.output.pop_tx()
                # Cleanup the outputGraph to delete the held tensors. We perform the
                # cleanup only for InstructionTranslator and not
                # InliningInstructionTranslator. The InliningInstructionTranslator
                # mutates the output object and is restored to original state if
                # there was an exception.
                if isinstance(self, InstructionTranslator):
                    self.output.cleanup()

                    # Note that this call maybe redundant if compile_subgraph is
                    # called. This is ok, because calling exit stack close()
                    # twice is not an issue (second stop is a no op).
                    self.output.mark_bytecode_tracing_stop()

    def push(self, val: Optional[VariableTracker]) -> None:
        assert val is None or isinstance(val, VariableTracker), (
            f"push expects VariableTracker, got {typestr(val)}"
        )
        self.stack.append(val)  # type: ignore[arg-type]

    def push_many(self, vals: list[VariableTracker]) -> None:
        for val in vals:
            self.push(val)

    def pop(self) -> VariableTracker:
        return self.stack.pop()

    def popn(self, n: int) -> list[VariableTracker]:
        return [*reversed([self.pop() for _ in range(n)])]

    def LOAD_FAST(self, inst: Instruction) -> None:
        name = inst.argval
        if self.exec_recorder and name in self.f_locals:
            self.exec_recorder.add_local_var(name, self.f_locals[name])

        try:
            self.push(self.symbolic_locals[name].unwrap())
        except KeyError:
            if name.startswith("."):
                try:
                    # This happens in dict/list comprehensions
                    new_name = name.replace(".", "implicit")
                    self.push(self.symbolic_locals[new_name])
                except KeyError:
                    unimplemented(
                        gb_type="Attempted to read undefined local variable (implicit)",
                        context=f"LOAD_FAST {name}",
                        explanation=f"Could not find an implicit local variable with name `{name}`",
                        hints=[
                            "This happens in dict/list comprehensions",
                            *graph_break_hints.USER_ERROR,
                        ],
                    )
            else:
                unimplemented(
                    gb_type="Attempted to read undefined local variable",
                    context=f"LOAD_FAST {name}",
                    explanation=f"Could not find a local variable with name `{name}`",
                    hints=[*graph_break_hints.USER_ERROR],
                )

        # for continuation functions
        if name.startswith("__stack"):
            self.symbolic_locals.pop(name)

    def LOAD_DEREF(self, inst: Instruction) -> None:
        assert inst.argval in self.cell_and_freevars()
        cell = self.symbolic_locals[inst.argval]
        contents_var = self.output.side_effects.load_cell(cell)
        self.push(contents_var)

        if self.exec_recorder and inst.argval in self.f_locals:
            self.exec_recorder.add_local_var(inst.argval, self.f_locals[inst.argval])

    def STORE_FAST(self, inst: Instruction) -> None:
        name = inst.argval
        loaded_vt = self.pop()
        loaded_vt.set_name_hint(name)
        self.symbolic_locals[name] = loaded_vt
        if name == IS_TRACING_RESUME_PROLOGUE_VARNAME:
            val = loaded_vt.as_python_constant()
            assert type(val) is bool
            self.is_tracing_resume_prologue = val

    def DELETE_FAST(self, inst: Instruction) -> None:
        del self.symbolic_locals[inst.argval]

    def STORE_DEREF(self, inst: Instruction) -> None:  # type: ignore[override]
        assert inst.argval in self.cell_and_freevars()
        cell = self.symbolic_locals[inst.argval]
        val = self.pop()
        self.output.side_effects.store_cell(cell, val)

        assert isinstance(cell, CellVariable)  # tame mypy
        if cell.local_name is not None:
            val.set_name_hint(cell.local_name)  # type: ignore[attr-defined]

    LOAD_CLOSURE = LOAD_FAST

    def _load_const(self, inst: Instruction) -> VariableTracker:
        i = inst.arg
        if i is None:
            return ConstantVariable.create(value=inst.argval)  # type: ignore[return-value]
        val = self._constants_cache[i]
        if not val:
            self._constants_cache[i] = ConstantVariable.create(value=inst.argval)  # type: ignore[call-overload]
            val = self._constants_cache[i]
        assert val is not None
        return val

    def LOAD_CONST(self, inst: Instruction) -> None:
        self.push(self._load_const(inst))

    def _load_global(self, inst: Instruction) -> None:
        name = inst.argval

        if self.exec_recorder:
            if name in self.f_globals:
                self.exec_recorder.add_global_var(name, self.f_globals[name])
            else:
                assert name in self.f_builtins
                self.exec_recorder.builtins[name] = self.f_builtins[name]

        if name not in self.f_globals:
            return self.load_builtin(inst)

        if name in self.symbolic_globals:
            variable = self.output.side_effects[self.symbolic_globals[name]]
            self.push(self.output.side_effects.load_global(variable, name))
            return

        value = self.f_globals[name]
        self.push(VariableTracker.build(self, value, GlobalSource(name)))

    @functools.cached_property
    def nn_modules_globals_vt(self) -> VariableTracker:
        module_name = "torch.nn.modules.module"
        module_source = self.import_source(module_name)
        fglobals_value = _import_module(module_name)
        return VariableTracker.build(self, fglobals_value, module_source)

    def LOAD_GLOBAL(self, inst: Instruction) -> None:
        assert inst.arg is not None
        if sys.version_info >= (3, 11) and sys.version_info < (3, 13) and inst.arg % 2:
            self.PUSH_NULL(inst)
        self._load_global(inst)
        if sys.version_info >= (3, 13) and inst.arg % 2:
            self.PUSH_NULL(inst)

    def STORE_GLOBAL(self, inst: Instruction) -> None:
        value = self.pop()
        name = inst.argval
        source = GlobalSource(name)
        if name not in self.symbolic_globals:
            self.symbolic_globals[name] = object()  # type: ignore[assignment]  # sentinel object
        variable = self.output.side_effects.track_global_existing(
            source, self.symbolic_globals[name]
        )
        if isinstance(value, RemovableHandleVariable):
            unimplemented(
                gb_type="Storing Tensor hook handle in globals",
                context=name,
                explanation="This is not supported.",
                hints=[],
            )
        self.output.side_effects.store_global(variable, name, value)

    # Cache note: This cache only exists for the duration of this
    # InstructionTranslator - so it should be safe to do.
    @cache_method
    def import_source(self, module_name: str) -> GlobalSource:
        """Create an alias to a module for use in guards"""
        if "torch_package" in module_name:
            value = torch.package.package_importer._package_imported_modules[
                module_name
            ]
            alias = (
                module_name.replace(">", "_").replace("<", "_").replace(".", "_dot_")
            )
        else:
            value = _import_module(module_name)
            alias = f"__import_{module_name.replace('.', '_dot_')}"

        if self.package is not None:
            self.package.add_import_source(alias, module_name)
        self.output.import_sources[alias] = module_name
        f_globals = self.output.global_scope
        assert alias not in f_globals or f_globals[alias] is value
        f_globals[alias] = value
        self.output.update_co_names(alias)
        return GlobalSource(alias)

    def resolve_name(self, name: str, package: str, level: int) -> str:
        """
        Copied from the Cpython implementation of __import__
        Resolve a relative module name to an absolute one.
        https://github.com/python/cpython/blob/5a094f0255eea1db58fb2cf14c200971e64ec36e/Lib/importlib/_bootstrap.py#L902
        """
        bits = package.rsplit(".", level - 1)
        if len(bits) < level:
            raise ImportError("attempted relative import beyond top-level package")
        base = bits[0]
        return f"{base}.{name}" if name else base

    def calc_package(self) -> str:
        """
        Copied from the Cpython implementation of __import__
        https://github.com/python/cpython/blob/5a094f0255eea1db58fb2cf14c200971e64ec36e/Lib/importlib/_bootstrap.py#L1090
        """
        package = self.f_globals.get("__package__")
        spec = self.f_globals.get("__spec__")
        if package is not None:
            if spec is not None and package != spec.parent:
                log.warning(
                    "__package__ != __spec__.parent (%r != %r)",
                    package,
                    spec.parent,
                    stacklevel=3,
                )
            return package
        elif spec is not None:
            return spec.parent
        else:
            log.warning(
                "can't resolve package from __spec__ or __package__, "
                "falling back on __name__ and __path__",
                stacklevel=3,
            )
            package = self.f_globals["__name__"]
            if "__path__" not in self.f_globals:
                package = package.rpartition(".")[0]
        return package

    def IMPORT_NAME(self, inst: Instruction) -> None:
        level, fromlist = self.popn(2)
        level = level.as_python_constant()
        fromlist = fromlist.as_python_constant()
        module_name = inst.argval

        # Are we replaying? if so, load recorded module
        recorded_name = (
            f"{ExecutionRecorder.LOCAL_MOD_PREFIX}_{level}_{fromlist}_{module_name}"
        )
        if recorded_name in self.f_globals:
            value = self.f_globals[recorded_name]
            source = GlobalSource(recorded_name)
        else:
            try:
                value = __import__(
                    module_name,
                    fromlist=fromlist,
                    level=level,
                    globals=self.f_globals,
                )
            except ImportError:
                unimplemented(
                    gb_type="Import failure",
                    context=f"module_name: {module_name}, fromlist: {fromlist}, level={level}",
                    explanation="Failure when attempting to import.",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            if level != 0:
                pkg = self.calc_package()
                module_name = self.resolve_name(module_name, pkg, level)

            # For __import__, when the name variable is of the form package.module,
            # normally, the top-level package (the name up till the first dot) is
            # returned, not the module named by module_name. However, when a
            # non-empty fromlist argument is given, the module named by name is
            # returned. Therefore, we set the source correctly here.
            if not fromlist:
                top_level_module_name = module_name.partition(".")[0]
                source = self.import_source(top_level_module_name)
            else:
                source = self.import_source(module_name)

        if self.exec_recorder:
            # pyrefly: ignore [unbound-name]
            self.exec_recorder.add_local_mod(recorded_name, value)

        # pyrefly: ignore [unbound-name]
        if istype(value, (types.ModuleType, DummyModule)):
            # pyrefly: ignore [unbound-name]
            self.push(PythonModuleVariable(value, source=source))
        else:
            unimplemented(
                gb_type="Bad import result",
                # pyrefly: ignore [unbound-name]
                context=typestr(value),
                explanation="Import result is not a Python module.",
                hints=[],
            )

    # fb internal 3.12 opcode
    EAGER_IMPORT_NAME = IMPORT_NAME

    def IMPORT_FROM(self, inst: Instruction) -> None:
        self.DUP_TOP(inst)
        self._load_attr(inst.argval)

    # Cache note: This cache only exists for the duration of this
    # InstructionTranslator - so it should be safe to do.
    @cache_method
    def load_builtin_from_argval(self, argval: Any) -> VariableTracker:
        if argval not in self.f_builtins:
            unimplemented(
                gb_type="failed to find name in frame builtins",
                context="",
                explanation=f"Failed to find name `{argval}` in frame's builtins.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        val = self.f_builtins[argval]

        if callable(val):
            builtins_source = GlobalSource(
                self.output.name_of_builtins_dict_key_in_fglobals
            )
            var_source = DictGetItemSource(builtins_source, argval)
            return VariableTracker.build(self, val, var_source)
        else:
            assert is_builtin_constant(val)
            return ConstantVariable.create(value=val)

    def load_builtin(self, inst: Instruction) -> None:
        self.push(self.load_builtin_from_argval(inst.argval))

    def jump(self, inst: Instruction | BlockStackEntry) -> None:
        assert self.instruction_pointer is not None
        assert self.start_point is not None
        assert inst.target is not None
        get_metrics_context().increment(
            "ir_count", self.instruction_pointer - self.start_point
        )
        self.instruction_pointer = self.indexof[inst.target]
        self.start_point = self.instruction_pointer

    JUMP_FORWARD = jump
    JUMP_ABSOLUTE = jump

    POP_JUMP_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_IF_TRUE = generic_jump(operator.truth, False)
    JUMP_IF_FALSE_OR_POP = generic_jump(operator.not_, True)
    JUMP_IF_TRUE_OR_POP = generic_jump(operator.truth, True)

    def SETUP_LOOP(self, inst: Instruction) -> None:
        # only exists in python<=3.7
        assert inst.target is not None
        self.block_stack.append(BlockStackEntry(inst, inst.target, len(self.stack)))

    def SETUP_EXCEPT(self, inst: Instruction) -> None:
        # only exists in python<=3.7
        assert inst.target is not None
        self.block_stack.append(BlockStackEntry(inst, inst.target, len(self.stack)))

    def POP_BLOCK(self, inst: Instruction) -> None:
        self.block_stack.pop()

    def SETUP_WITH(self, inst: Instruction) -> None:
        self.setup_or_before_with(inst)

    def SETUP_FINALLY(self, inst: Instruction) -> None:
        assert inst.target is not None
        self.block_stack.append(BlockStackEntry(inst, inst.target, len(self.stack)))

    def BEGIN_FINALLY(self, inst: Instruction) -> None:
        self.push(None)

    def WITH_CLEANUP_START(self, inst: Instruction) -> None:
        exit, exc = self.popn(2)
        assert exc is None
        self.push(exc)

        self.push(exit.call_function(self, [ConstantVariable.create(None)] * 3, {}))

    def WITH_CLEANUP_FINISH(self, inst: Instruction) -> None:
        self.popn(2)
        self.push(None)

    def FOR_ITER(self, inst: Instruction) -> None:
        it = self.pop().realize()
        self.push(it)
        try:
            val = it.next_variable(self)
            self.push(val)
        except (StopIteration, exc.ObservedUserStopIteration) as e:
            if isinstance(e, exc.ObservedUserStopIteration):
                exc.handle_observed_exception(self)

            if sys.version_info >= (3, 12):
                # CPython 3.12 actually jumps to the instruction after the END_FOR
                # and performs the action of END_FOR as part of FOR_ITER. We jump
                # to the END_FOR and run it, so we need to make sure 2 values are
                # on the stack for it to pop.
                self.push(ConstantVariable.create(None))
            else:
                # pop the iterator in Python < 3.12
                self.pop()
            self.jump(inst)

    def _create_exception_type(self, val: VariableTracker) -> VariableTracker:
        if isinstance(
            val, (variables.BuiltinVariable, UserDefinedExceptionClassVariable)
        ):
            # Create the instance of the exception type
            # https://github.com/python/cpython/blob/3.11/Python/ceval.c#L6547-L6549
            val = val.call_function(self, [], {})  # type: ignore[arg-type]
        return val

    def _attach_traceback_to_exception(self, exc: ExceptionVals) -> None:
        # based on CPython's PyTraceBack_Here impl
        frame_summary = self.frame_summary()
        tb = exc.var_getattr(
            # pyrefly: ignore [bad-argument-type]
            self,
            "__traceback__",
        )
        assert isinstance(
            tb, (ConstantVariable, TracebackVariable)
        )  # make pyrefly happy
        new_tb = TracebackVariable.from_frame_summary(frame_summary, tb)
        exc.call_method(
            # pyrefly: ignore [bad-argument-type]
            self,
            "__setattr__",
            [ConstantVariable("__traceback__"), new_tb],
            {},
        )

    def _raise_exception_variable(self, val: VariableTracker) -> NoReturn:
        # User can raise exception in 2 ways
        #   1) raise exception type - raise NotImplementedError
        #   2) raise exception instance - raise NotImplementedError("foo")

        # 1) when user raises exception type
        val = self._create_exception_type(val)

        # Handle https://peps.python.org/pep-0479/
        # CPython 3.12+ has a specific bytecode instruction (CALL_INTRINSIC_1 3) for this
        if (
            is_generator(self.f_code)
            and isinstance(val, variables.ExceptionVariable)
            and val.exc_type is StopIteration
        ):
            val = variables.BuiltinVariable(RuntimeError).call_function(self, [], {})  # type: ignore[arg-type]

        # Capture the python_stack when the exception is first raised.
        # This preserves the original exception location even if the exception
        # is later re-raised (e.g., in context manager cleanup).
        # ExceptionVariable and UserDefinedExceptionObjectVariable both have
        # a python_stack attribute.
        if (
            self._isinstance_exception(val)
            and getattr(val, "python_stack", None) is None
        ):
            val.python_stack = torch._guards.TracingContext.extract_stack()  # type: ignore[union-attr]

        # 2) when user raises exception instance
        if self._isinstance_exception(val):
            # Save the exception in a global data structure
            self.exn_vt_stack.set_current_exception(val)  # type: ignore[arg-type]

            observed_exception_type = exc.get_dynamo_observed_exception(val.exc_type)  # type: ignore[attr-defined, union-attr]
            # Pass the stored python_stack to preserve the original exception location
            python_stack = getattr(val, "python_stack", None)
            raise observed_exception_type(
                f"raised exception {val}", real_stack=python_stack
            )

        unimplemented(
            gb_type="Failed to raise exception",
            context=str(exc),
            explanation="Attempted to raise a non-Exception type/value.",
            hints=[*graph_break_hints.USER_ERROR],
        )

    def RAISE_VARARGS(self, inst: Instruction) -> None:
        if inst.arg == 0:
            if not len(self.exn_vt_stack):
                msg = ConstantVariable("No active exception to reraise")
                exc.raise_observed_exception(RuntimeError, self, args=[msg])

            # re-raise the previous exception. Here CPython refers to the exception
            # on top of the exception stack
            assert len(self.exn_vt_stack)
            val = self.exn_vt_stack[-1]
            assert self._isinstance_exception(val), val
            self._raise_exception_variable(val)
        elif inst.arg == 1:
            # raise TOS
            val = self.stack[-1]  # type: ignore[assignment]
            try:
                self._raise_exception_variable(val)
            finally:
                # Update __traceback__ in the raised exception
                curr_exc = self.exn_vt_stack.get_current_exception()
                self._attach_traceback_to_exception(curr_exc)
        else:
            # raise .. from ...
            from_vt = self.pop()
            val = self.pop()  # type: ignore[assignment]
            try:
                self._raise_exception_variable(val)
            finally:
                # Update __cause__/__suppress_context__ in the raised exception
                curr_exc = self.exn_vt_stack.get_current_exception()
                self._attach_traceback_to_exception(curr_exc)
                cause = self._create_exception_type(from_vt)
                curr_exc.call_setattr(self, ConstantVariable("__cause__"), cause)  # type: ignore[arg-type, union-attr, assignment]

    def CLEANUP_THROW(self, inst: Instruction) -> None:
        # https://github.com/python/cpython/pull/96010
        tos = self.stack[-1]
        assert isinstance(tos, ExceptionVariable)
        if tos.exc_type is StopIteration:
            unimplemented(
                gb_type="CLEANUP_THROW with StopIteration",
                context="",
                explanation="Received StopIteration when handling generator.throw/close. This is not supported.",
                hints=[],
            )
        else:
            self.RERAISE(inst)

    def RERAISE(self, inst: Instruction) -> None:
        # https://docs.python.org/3/library/dis.html#opcode-RERAISE
        #   Re-raises the exception currently on top of the stack. If oparg is
        #   non-zero, pops an additional value from the stack which is used to
        #   set f_lasti of the current frame.

        if sys.version_info >= (3, 11):
            # RERAISE is currently supported in a narrow case of `raise ... from None`
            val = self.pop()
            if inst.argval:
                # RERAISE 1
                _ = self.pop()
                self._raise_exception_variable(val)
            else:
                # RERAISE 0
                self.push(val)
                self._raise_exception_variable(val)
        else:
            _exc = self.pop()
            val = self.pop()
            _tb = self.pop()
            self._raise_exception_variable(val)

    def _isinstance_exception(self, val: VariableTracker) -> TypeIs[ExceptionVals]:
        return isinstance(val, ExceptionVals)

    def WITH_EXCEPT_START(self, inst: Instruction) -> None:
        args: list[VariableTracker] = []
        if sys.version_info >= (3, 11):
            fn_loc = 4 if sys.version_info < (3, 14) else 5
            # At the top of the stack are 4 values:
            #    - TOP = exc_info()
            #    - SECOND = previous exception
            #    - THIRD: lasti of exception in exc_info()
            #    - FOURTH: the context.__exit__ bound method
            #    We call FOURTH(type(TOP), TOP, GetTraceback(TOP)).
            #    Then we push the __exit__ return value.
            # In Python 3.14+, there is a NULL placed between the context.__exit__ bound method and the lasti,
            # that is, fn is now the 5th from TOS.
            assert len(self.stack) >= fn_loc
            fn = self.stack[-fn_loc]
            val = self.stack[-1]
            assert self._isinstance_exception(val)
            typ = BuiltinVariable(val.exc_type)  # type: ignore[attr-defined, union-attr]
            tb = val.var_getattr(
                # pyrefly: ignore[bad-argument-type]
                self,
                "__traceback__",
            )
            if sys.version_info >= (3, 14):
                if not isinstance(self.stack[-4], NullVariable):
                    args.append(self.stack[-4])
        else:
            assert len(self.stack) >= 7
            fn = self.stack[-7]
            val = self.stack[-2]
            assert self._isinstance_exception(val)
            typ = BuiltinVariable(val.exc_type)  # type: ignore[attr-defined]

            tb = val.var_getattr(self, "__traceback__")

        args += [typ, val, tb]
        self.call_function(fn, args, {})

    def exception_handler(self, raised_exception: ObservedException) -> None:
        observed_exn_gb_explanation = (
            "Dynamo found no exception handler at the top-level compiled function "
            "when encountering an exception. Exception will propagate outside the compiled region."
        )

        def bubble_exception_to_interpreter() -> None:
            # Bubble the exception to the interpreter
            curr_exc = self.exn_vt_stack.get_current_exception()
            dynamo_exc = exc.get_dynamo_observed_exception(curr_exc.python_type())
            assert isinstance(raised_exception, dynamo_exc)  # sanity check
            unimplemented(
                gb_type="Observed exception",
                context=f"raised exception {curr_exc.python_type_name()}({curr_exc.args})",  # type: ignore[union-attr]
                explanation=observed_exn_gb_explanation,
                hints=[
                    *graph_break_hints.USER_ERROR,
                    *graph_break_hints.SUPPORTABLE,
                ],
                from_exc=raised_exception,
            )

        if sys.version_info >= (3, 11):
            exn_tab_entry = self.current_instruction.exn_tab_entry
            if exn_tab_entry:
                # Implementation is based on https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt

                # 1) pop values from the stack until it matches the stack depth
                # for the handler
                while len(self.stack) > exn_tab_entry.depth:
                    self.pop()

                # 2) if 'lasti' is true, then push the offset that the exception was raised at
                if exn_tab_entry.lasti:
                    self.push(
                        variables.ConstantVariable(self.current_instruction.offset)
                    )

                # 3) push the exception to the stack
                self.push(self.exn_vt_stack.get_current_exception())

                # 4) jump to the handler
                self.jump(exn_tab_entry)  # type: ignore[arg-type]
            else:
                # No handler found. Bubble the exception to the parent
                # instruction translator. We use special exception for this.
                self.stack.clear()

                # attach traceback to the exception and set it as current exception
                curr_exc = self.exn_vt_stack.get_current_exception()
                self._attach_traceback_to_exception(curr_exc)

                if type(self) is InstructionTranslator:
                    bubble_exception_to_interpreter()
                raise raised_exception
        else:
            if len(self.block_stack):
                # base implementation - https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4455

                block_stack_entry = self.block_stack.pop()

                while block_stack_entry.inst.opname == "EXCEPT_HANDLER":
                    # TODO(anijain2305) - This is not tested .. unable to create a testcase
                    # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1456
                    self.popn(3)
                    self.exn_vt_stack.pop()
                    if len(self.block_stack) == 0:
                        # No handler found in this frame. Bubble the exception to the parent
                        # instruction translator.
                        self.stack.clear()
                        if type(self) is InstructionTranslator:
                            unimplemented(
                                gb_type="Observed exception (EXCEPT_HANDLER)",
                                context=str(raised_exception),
                                explanation=observed_exn_gb_explanation
                                + " This graph break is unexpected.",
                                hints=[*graph_break_hints.DYNAMO_BUG],
                                from_exc=raised_exception,
                            )

                        raise raised_exception
                    block_stack_entry = self.block_stack.pop()

                exception_var = self.exn_vt_stack.get_current_exception()
                self.exn_vt_stack.move_current_exception_to_stack()

                # 1) pop values from the stack until it matches the stack depth
                # for the handler
                while len(self.stack) > block_stack_entry.stack_index:
                    self.pop()

                # Push a dummy block stack entry of EXCEPT_HANDLER
                # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1456
                except_handler_inst = Instruction(int(1e6), "EXCEPT_HANDLER", None, 0)
                self.block_stack.append(
                    BlockStackEntry(except_handler_inst, None, len(self.stack))
                )

                # Push old exception
                if len(self.exn_vt_stack) >= 2:
                    old_exception = self.exn_vt_stack[-2]

                    # Push the old exception on to stack - tb, value, type
                    # Traceback is currently mapped to UnknownVariable
                    self.push(variables.UnknownVariable())
                    self.push(old_exception)

                    self.push(variables.BuiltinVariable(old_exception.exc_type))
                else:
                    # Push empty exception tb, value, type
                    self.push(variables.ConstantVariable(None))
                    self.push(variables.ConstantVariable(None))
                    self.push(variables.ConstantVariable(None))

                # Push new exception - tb, val, type
                # Traceback is currently mapped to UnknownVariable
                self.push(variables.UnknownVariable())
                self.push(exception_var)

                self.push(variables.BuiltinVariable(exception_var.exc_type))

                # Jump to target
                self.jump(block_stack_entry)
            else:
                # No handler found. Bubble the exception to the parent
                # instruction translator. We use special exception for this.
                self.stack.clear()
                if type(self) is InstructionTranslator:
                    bubble_exception_to_interpreter()
                raise raised_exception

    def PUSH_EXC_INFO(self, inst: Instruction) -> None:
        # https://docs.python.org/3/library/dis.html#opcode-PUSH_EXC_INFO
        #   Pops a value from the stack. Pushes the current exception to the top
        #   of the stack. Pushes the value originally popped back to the stack.
        #
        # The behavior of this opcode in CPython is a bit different than what it
        # is described. It pops a value from the stack, pushes the top of the
        # exception stack to the interpreter stack and moves the
        # "current exception" to the exception stack.
        #
        # As an example, suppose the stack is in the following state:
        #   + stack = [..., ConstantVariable(1), ConstantVariable(2)]
        #   + current_exception = TypeError
        #   + exception_stack = [ValueError]
        #
        # After PUSH_EXC_INFO is executed
        #   + stack = [..., ConstantVariable(1), ValueError, ConstantVariable(2)]
        #   + current_exception = None
        #   + exception_stack = [ValueError, TypeError]

        val = self.pop()
        if len(self.exn_vt_stack) == 0:
            prev_exc: VariableTracker = ConstantVariable(None)
        else:
            prev_exc = self.exn_vt_stack[-1]
        self.push(prev_exc)
        self.push(val)
        self.exn_vt_stack.move_current_exception_to_stack()

    def POP_EXCEPT(self, inst: Instruction) -> None:
        if sys.version_info >= (3, 11):
            _ = self.pop()
            # This exception is handled and therefore we can clear the error indicator
            assert len(self.exn_vt_stack)
            self.exn_vt_stack.pop()
        else:
            assert len(self.block_stack) > 0
            if self.block_stack[-1].inst.opname != "EXCEPT_HANDLER":
                raise AssertionError(
                    "Bug in Dynamo tracing of exception handling."
                    "Top of the block stack is not EXCEPT_HANDLER."
                )
            self.block_stack.pop()

            self.popn(3)

            # This exception is handled and therefore we can clear the error indicator
            assert len(self.exn_vt_stack)
            self.exn_vt_stack.pop()

    def check_if_exc_matches(self) -> bool:
        assert len(self.stack) >= 2
        expected_exc_types = self.pop()
        if sys.version_info >= (3, 11):
            # CHECK_EXC_MATCH (which is used from 3.11 onwards) does not pop.
            # This is the description from the disassembly doc
            #
            # Performs exception matching for ``except``. Tests whether the ``STACK[-2]``
            # is an exception matching ``STACK[-1]``. Pops ``STACK[-1]`` and pushes the boolean
            # result of the test.
            exc_instance = self.stack[-1]
        else:
            # This is used prior to 3.11 via opcode JUMP_IF_NOT_EXC_MATCH
            # There is no documentation but here is the code pointer that does 2 pops
            # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L3650-L3665
            exc_instance = self.stack.pop()

        # Users can check exception in 3 ways
        # 1) except NotImplementedError --> BuiltinVariable
        # 2) except CustomException --> UserDefinedExceptionClassVariable
        # 3) except (NotImplementedError, AttributeError) -> TupleVariable

        if not isinstance(
            expected_exc_types,
            (
                BuiltinVariable,
                TupleVariable,
                UserDefinedExceptionClassVariable,
                UserDefinedExceptionObjectVariable,
            ),
        ):
            unimplemented(
                gb_type="Exception with bad expected type",
                context=str(expected_exc_types),
                explanation=f"`except ...` has unsupported type {expected_exc_types}.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        if sys.version_info >= (3, 11):
            if not self._isinstance_exception(exc_instance):
                unimplemented(
                    gb_type="Caught non-Exception value",
                    context=str(exc_instance),
                    explanation=f"Except expects to receive an object of Exception type but received {exc_instance}.",
                    hints=[*graph_break_hints.USER_ERROR],
                )

        if isinstance(expected_exc_types, TupleVariable):
            expected_types = expected_exc_types.items
        else:
            expected_types = [
                expected_exc_types,
            ]

        for expected_type in expected_types:
            if not isinstance(
                expected_type,
                (
                    BuiltinVariable,
                    UserDefinedExceptionObjectVariable,
                    UserDefinedExceptionClassVariable,
                ),
            ):
                unimplemented(
                    gb_type="Exception with non-type expectation",
                    context=str(expected_type),
                    explanation=f"`except ...` expects a non-type: {expected_type}.",
                    hints=[*graph_break_hints.USER_ERROR],
                )
            if self._isinstance_exception(exc_instance) and issubclass(
                exc_instance.exc_type,  # type: ignore[union-attr]
                expected_type.fn,  # type: ignore[attr-defined]
            ):
                return True
            elif isinstance(exc_instance, variables.BuiltinVariable) and issubclass(
                exc_instance.fn,
                # pyrefly: ignore [missing-attribute]
                expected_type.fn,
            ):
                return True

        return False

    def CHECK_EXC_MATCH(self, inst: Instruction) -> None:
        self.push(variables.ConstantVariable(self.check_if_exc_matches()))

    def JUMP_IF_NOT_EXC_MATCH(self, inst: Instruction) -> None:
        if not self.check_if_exc_matches():
            self.jump(inst)

    def COMPARE_OP(self, inst: Instruction) -> None:
        if inst.argval == "exception match":
            self.CHECK_EXC_MATCH(inst)
        else:
            self.push(compare_op_handlers[inst.argval](self, self.popn(2), {}))

    def GET_ITER(self, inst: Instruction) -> None:
        self.call_function(BuiltinVariable(iter), [self.pop()], {})

    @break_graph_if_unsupported(
        push=True,
        msg_prefix="Encountered graph break when attempting to trace CALL_FUNCTION: a call to a regular function, e.g. f(x, y)",
    )
    def CALL_FUNCTION(self, inst: Instruction) -> None:
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    @break_graph_if_unsupported(
        push=True,
        msg_prefix="Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args)",
    )
    def CALL_FUNCTION_EX(self, inst: Instruction) -> None:
        kwargsvars: VariableTracker
        if inst.argval == 0:
            kwargsvars = ConstDictVariable({})
            argsvars = self.pop()
        elif inst.argval == 1 or sys.version_info >= (3, 14):
            # Python 3.14+ removed the argval and replaced it with a possibly NULL kwargs
            kwargsvars = self.pop()
            if isinstance(kwargsvars, NullVariable):
                kwargsvars = ConstDictVariable({})
            argsvars = self.pop()
        else:
            unimplemented(
                gb_type="Variadic function call with bad flags",
                context=f"flags: {inst.argval}",
                explanation=f"Attempted to call a variadic function (CALL_FUNCTION_EX) with bad flags {inst.argval}",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )

        if sys.version_info >= (3, 13):
            # 3.13 swapped null and callable
            null = self.pop()
            assert isinstance(null, NullVariable)

        fn = self.pop()

        if sys.version_info >= (3, 11) and sys.version_info < (3, 13):
            null = self.pop()
            assert isinstance(null, NullVariable)

        if not isinstance(
            # pyrefly: ignore [unbound-name]
            argsvars,
            BaseListVariable,
            # pyrefly: ignore [unbound-name]
        ) and argsvars.has_force_unpack_var_sequence(self):
            # pyrefly: ignore [unbound-name]
            argsvars = TupleVariable(argsvars.force_unpack_var_sequence(self))

        # Unpack for cases like fn(**obj) where obj is a map
        # pyrefly: ignore [unbound-name]
        if isinstance(kwargsvars, UserDefinedObjectVariable):
            kwargsvars = BuiltinVariable.call_custom_dict(self, dict, kwargsvars)  # type: ignore[arg-type]

        # pyrefly: ignore [unbound-name]
        if not isinstance(argsvars, BaseListVariable) or not isinstance(
            # pyrefly: ignore [unbound-name]
            kwargsvars,
            ConstDictVariable,
        ):
            unimplemented(
                gb_type="Variadic function call with bad args/kwargs type",
                # pyrefly: ignore [unbound-name]
                context=f"args type: {typestr(argsvars)}, kwargs type: {typestr(kwargsvars)}",
                explanation="Expected args to be a list and kwargs to be a dict",
                hints=[*graph_break_hints.USER_ERROR],
            )

        # Map to a dictionary of str -> VariableTracker
        # pyrefly: ignore [unbound-name, missing-attribute]
        kwargsvars = kwargsvars.keys_as_python_constant()
        # pyrefly: ignore [unbound-name, missing-attribute]
        self.call_function(fn, argsvars.items, kwargsvars)

    @break_graph_if_unsupported(
        push=True,
        msg_prefix="Encountered graph break when attempting to trace CALL_FUNCTION_KW: "
        "a function call with keyword arguments, e.g. f(x=True)",
    )
    def CALL_FUNCTION_KW(self, inst: Instruction) -> None:
        argnames = self.pop()
        args = self.popn(inst.argval)
        fn = self.pop()
        assert isinstance(argnames, TupleVariable) and argnames.is_python_constant()
        argnames = argnames.as_python_constant()
        args, kwargs_list = args[: -len(argnames)], args[-len(argnames) :]
        kwargs = dict(zip(argnames, kwargs_list))
        assert len(kwargs) == len(argnames)
        self.call_function(fn, args, kwargs)

    def LOAD_METHOD_SUPER(self, inst: Instruction) -> None:
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        arg = inst.argval[0]
        argval = self.code_options["co_names"][arg]
        if sys.version_info < (3, 11):
            self._load_attr(argval)
        else:
            self.LOAD_METHOD(dataclasses.replace(inst, argval=argval))

    def LOAD_ATTR_SUPER(self, inst: Instruction) -> None:
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        arg = inst.argval[0]
        argval = self.code_options["co_names"][arg]
        self._load_attr(argval)

    def LOAD_METHOD(self, inst: Instruction) -> None:
        self._load_attr(inst.argval)
        obj = self.pop()
        if sys.version_info >= (3, 13):
            self.push(obj)
            self.PUSH_NULL(inst)
        elif sys.version_info >= (3, 11):
            # always follow the NULL + fn convention, since if obj
            # is actually a method, self is already bound to it, so it
            # doesn't need to be passed in as an arg.
            self.PUSH_NULL(inst)
            self.push(obj)
        else:
            self.push(obj)
            self.push(None)

    def CALL_METHOD(self, inst: Instruction) -> None:
        args = self.popn(inst.argval)
        dummy = self.pop()
        assert dummy is None
        fn = self.pop()
        self.call_function(fn, args, {})

    def _load_attr(self, attr: Any) -> None:
        obj = self.pop()
        result = BuiltinVariable(getattr).call_function(
            self,  # type: ignore[arg-type]
            [obj, ConstantVariable.create(attr)],
            {},
        )
        self.push(result)

    def LOAD_ATTR(self, inst: Instruction) -> None:
        if sys.version_info >= (3, 12):
            # pyrefly: ignore [unsupported-operation]
            if inst.arg % 2:
                self.LOAD_METHOD(inst)
                return
        self._load_attr(inst.argval)

    @break_graph_if_unsupported(
        push=False,
        msg_prefix="Encountered graph break when attempting to trace STORE_ATTR: storing an object's attribute, e.g. x.attr = y",
    )
    def STORE_ATTR(self, inst: Instruction) -> None:
        val, obj = self.popn(2)
        BuiltinVariable(setattr).call_function(
            self,  # type: ignore[arg-type]
            [obj, ConstantVariable.create(inst.argval), val],
            {},
        )

    def DELETE_ATTR(self, inst: Instruction) -> None:
        obj = self.pop()
        BuiltinVariable(delattr).call_function(
            self,  # type: ignore[arg-type]
            [obj, ConstantVariable.create(inst.argval)],
            {},
        )

    @staticmethod
    def codegen_return_with_pops(
        inst: Instruction, num_stack: int
    ) -> list[Instruction]:
        """
        Debug CPython expects the stack to be empty after the return.
        Calling compile_subgraph will push cells and frame values to TOS.
        This function will pop those 2 values from the stack before actually returning.

        Expects the stack to be:
            cells, frame values, current frame stack (0 or 1 values)

        Pops cells and frame values, leaving the current frame stack as TOS.
        A return instruction is included.
        """
        insts = []
        # NOTE: Debug CPython expects the stack to be empty after the return.
        # Expect the current stack to be in the state
        # cells, frame values, current frame stack (0 or 1 values)
        assert num_stack <= 1
        if num_stack == 1:
            insts.extend(create_swap(3))
        return_inst = (
            create_instruction("RETURN_VALUE")
            if inst.opname == "RETURN_VALUE"
            else create_instruction("RETURN_CONST", argval=inst.argval)
        )
        insts.extend(
            [create_instruction("POP_TOP"), create_instruction("POP_TOP"), return_inst]
        )
        return insts

    def create_resume(
        self,
        idx: int,
        resume_inst: Instruction,
        meta: StackLocalsMetadata,
        resume_codes: list[types.CodeType],
        cg: PyCodegen,
        is_leaf: bool,
        handle_inactive_ctx: bool,
    ) -> tuple[types.CodeType, str]:
        """
        Creates the resume function for the frame corresponding to `self`.

        Expects the TOS to be:
            [frame N cells, ..., frame 1 cells],
            [
                frame N stack + locals,
                ...,
                frame 1 stack + locals
            ]

        Some additional codegen may happen to prepare the frame stack + locals values for the generated resume function:
        - inactive context variables in the stack and locals will be replaced by their types
        - if the frame is a leaf frame, prune dead locals

        Regardless of codegen, the stack will be left in the same state as before.

        Args:
            - idx: depth of this frame: 0 corresponds to the leaf frame (frame N), N-1 to the root frame (frame 1).
            - resume_inst: the instruction that this frame should resume at
            - meta: metadata for this frame returned from OutputGraph.compile_subgraph
            - resume_codes: nested resume code objects generated from previous create_resume calls.
            - cg: codegen object to output to
            - is_leaf: True if `self` corresponds to the leaf frame.
            - handle_inactive_ctx: If True, handles inactive context variables as described above. This is necessary
                iff the resume function is traced
        """
        # Handle inactive context variables.
        # The resume function assumes that context variables are the class, NOT the object.
        # e.g. torch.set_grad_enabled(True) will be reconstructed as torch.set_grad_enabled
        # NOTE: if the unsupported instruction modifies the inactive context variable, it may
        # result in silent incorrectness!
        if handle_inactive_ctx:
            for (j, _), j_orig in zip(meta.stack_ctx_args, meta.stack_ctx_idxes_orig):
                # Replace the stack var with the context class
                ctx = cast(ContextWrappingVariable, self.stack[j_orig])
                # frames[idx][j] = reconstructed_ctx
                cg.append_output(create_dup_top())
                ctx.reconstruct_type(cg)
                cg.extend_output(
                    [
                        *create_swap(2),
                        cg.create_load_const(idx),
                        cg.create_binary_subscr(),
                        cg.create_load_const(j),
                        create_instruction("STORE_SUBSCR"),
                    ]
                )

            for name, _ in meta.locals_ctx_args:
                # Replace the local with the context class
                ctx = cast(ContextWrappingVariable, self.symbolic_locals[name])
                # frames[idx][meta.num_stack +meta.locals_names[name]] = reconstructed_ctx
                cg.append_output(create_dup_top())
                ctx.reconstruct_type(cg)
                cg.extend_output(
                    [
                        *create_swap(2),
                        cg.create_load_const(idx),
                        cg.create_binary_subscr(),
                        cg.create_load_const(meta.num_stack + meta.locals_names[name]),
                        create_instruction("STORE_SUBSCR"),
                    ]
                )

        # If the resume instruction is a jump absolute, then resume
        # at the target instead. This handles the case where we
        # graph break again in a nested function before jump-resuming
        # this frame.
        if is_jump_absolute(resume_inst):
            assert resume_inst.target
            resume_inst = resume_inst.target

        resume_name = unique_id(f"__resume_at_{resume_inst.offset}")

        # More locals may have been pruned in the current/leaf frame
        # after the unsupported instruction (e.g. branch).
        # There should not be any pruning in the other frames since
        # the current instruction there should be a CALL.
        if is_leaf:
            reads = livevars_analysis(self.instructions, resume_inst)
            all_argnames = tuple(
                k
                for k in self.symbolic_locals
                if k in reads and k not in self.cell_and_freevars()
            )
            argnames_null_set = set(meta.locals_null_keys)
            argnames = tuple(k for k in all_argnames if k not in argnames_null_set)
            argnames_null = tuple(k for k in all_argnames if k in argnames_null_set)

            # codegen filter for current frame's locals
            # current stack state: frames
            cg.extend_output(
                [
                    create_dup_top(),
                    cg.create_load_const(idx),
                    cg.create_binary_subscr(),
                    create_dup_top(),
                ]
            )
            for arg in argnames:
                # current stack state: frames, frames[i], *(prev locals), frames[i]
                cg.extend_output(
                    [
                        create_dup_top(),
                        cg.create_load_const(meta.num_stack + meta.locals_names[arg]),
                        cg.create_binary_subscr(),
                        *create_swap(2),
                    ],
                )
            # current stack state: frames, frames[i], *(frame i live locals), frames[i]
            cg.extend_output(
                [
                    create_instruction("POP_TOP"),
                    create_instruction("BUILD_LIST", arg=len(argnames)),
                    *create_swap(2),
                    # frames, frames i live locals, frames[i]
                    *create_binary_slice(meta.num_stack, None, True),
                    # frames[i][num_stack:] = frame i live locals
                ]
            )
            # current stack state: frames
        else:
            argnames = tuple(meta.locals_names.keys())
            argnames_null = tuple(meta.locals_null_keys)

        if sys.version_info < (3, 12):
            assert len(argnames_null) == 0, "variables should not be NULL in < 3.12"

        # compile_subgraph did not codegen any NULLs,
        # so we should not count NullVariables
        stack_len = len(self.stack) - len(meta.stack_null_idxes)

        assert self.current_instruction.offset is not None
        new_code: types.CodeType = ContinueExecutionCache.lookup(
            self.f_code,
            self.lineno,
            self.current_instruction.offset,
            resume_inst.offset,
            # pyre: ignore[missing-attribute]
            tuple(b.target.offset for b in self.block_stack),
            stack_len,
            argnames,
            argnames_null,
            tuple(b.resume_fn() for b in self.block_stack),
            handle_inactive_ctx,
            tuple(meta.stack_ctx_args),
            tuple(meta.locals_ctx_args),
            tuple(meta.stack_null_idxes),
            tuple(resume_codes),
            not self.current_instruction_push,
        )

        # Add original GraphModule context to the resume function to handle
        # the case of a graph break while tracing a GraphModule
        orig_graphmodule_maybe = code_context.get_context(self.f_code).get(
            "orig_graphmodule", lambda: None
        )()
        if orig_graphmodule_maybe is not None:
            code_context.get_context(new_code)["orig_graphmodule"] = weakref.ref(
                orig_graphmodule_maybe
            )

        # add resume function to the global scope
        if new_code.co_freevars:
            # expose code object for debugging purposes
            self.output.install_global_unsafe(resume_name, new_code)
            package_name = None
        else:
            # This is safe: we pre-generate a unique name
            self.output.install_global_unsafe(
                resume_name,
                types.FunctionType(new_code, self.f_globals, resume_name),
            )
            package_name = resume_name

        if self.package is not None:
            self.package.add_resume_function(
                new_code, self.f_globals["__name__"], package_name
            )

        counters["resumes"][new_code.co_name] += 1

        return new_code, resume_name

    def create_call_resume_at(
        self,
        inst: Instruction,
        all_stack_locals_metadata: list[StackLocalsMetadata],
    ) -> list[Instruction]:
        """
        Codegen all resume function(s) from the frame stack starting at `self`, call them,
        and return the result.
        Assumes that the unsupported instruction has already been run.

        Expects the TOS to be:
            [
                frame N locals,
                frame N-1 stack + locals,
                ...,
                frame 1 stack + locals
            ], *(frame N stack (post-unsupported instruction))

        Leaves the result of calling the resume functions on the stack and returns it
        (empty stack after return).

        Args:
            - inst: the instruction of the current (deepest) frame to resume at
            - all_stack_locals_metadata: metadata returned from OutputGraph.compile_subgraph - contains
                metadata such as local names, NULL positions, stack length, etc.
        """

        self.instruction_pointer = None

        cg = PyCodegen(self.output.root_tx)

        # NOTE: We do not need to codegen frames whose resume instruction is RETURN_VALUE
        # We could also do something similar for RETURN_CONST, but a lot more code is necessary
        # since we would need to track RETURN_CONST values and inject the constant in the right places.

        # Filter out tx'es that are resuming on RETURN_*.
        txes: list[InstructionTranslatorBase] = []
        idxes: list[int] = []
        resume_insts: list[Instruction] = []
        cur_tx: Optional[InstructionTranslatorBase] = self
        idx = 0
        while cur_tx is not None:
            if cur_tx is self:
                resume_inst = inst
            else:
                resume_inst = cur_tx.next_instruction
            if resume_inst.opname != "RETURN_VALUE":
                txes.append(cur_tx)
                idxes.append(idx)
                resume_insts.append(resume_inst)

            cur_tx = cur_tx.parent
            idx += 1

        current_num_stack = len(self.stack) - len(
            all_stack_locals_metadata[0].stack_null_idxes
        )

        # Every tx is returning - no need to call a resume function.
        if not txes:
            # Pop everything but TOS, then return the TOS.
            # Frame N's stack must have length >= 1 since it's about to RETURN_VALUE.
            # Frame N actually should have stack length == 1, because debug CPython expects
            # empty stacks after return, but there is no guarantee written down anywhere.
            assert current_num_stack >= 1
            cg.extend_output(create_swap(current_num_stack + 2))
            for _ in range(current_num_stack + 1):
                cg.append_output(create_instruction("POP_TOP"))
            cg.append_output(create_instruction("RETURN_VALUE"))

            return cg.get_instructions()

        # Let frame k be the deepest frame where the resume function is not RETURN_VALUE
        # - If k == N, then the frame N stack is prepended to the frame N locals.
        # - If k != N, then frame N's TOS is added to frame k's stack.

        # Rearrange the TOS to be compatible with create_resume and codegen_call_resume:
        #     [
        #         frame N stack + locals,
        #         ...,
        #         frame 1 stack + locals
        #     ]

        # create the stack values that should be moved
        if txes[0] is self:
            # Frame N is non-returning, pack all of frame N's stack to
            # be moved to frame N's frame values
            cg.append_output(create_instruction("BUILD_LIST", arg=current_num_stack))
            # frame N stack is not yet on the frame N's frame values
            stack_insert_idx = 0
            all_stack_locals_metadata[0].num_stack = current_num_stack
        else:
            # Frame N is returning. Let frame k be the deepest non-returning frame.
            # Add frame N's TOS to frame k's stack.
            # pop frame N stack except TOS
            cg.extend_output(create_swap(current_num_stack))
            for _ in range(current_num_stack - 1):
                cg.append_output(create_instruction("POP_TOP"))
            cg.append_output(create_instruction("BUILD_LIST", arg=1))
            # frame k stack is already on frame k's frame values
            stack_insert_idx = all_stack_locals_metadata[idxes[0]].num_stack
            all_stack_locals_metadata[idxes[0]].num_stack += 1
            txes[0].push(UnknownVariable())

        # move the predetermined stack value(s) to the deepest non-returning frame
        cg.extend_output(
            [
                *create_copy(2),
                # frame_values, return_const, frame_values
                cg.create_load_const(idxes[0]),
                cg.create_binary_subscr(),
                *create_binary_slice(stack_insert_idx, stack_insert_idx, True),
                # frame_values[idxes[0]][stack_insert_idx:stack_insert_idx] = frame N stack/[return_const/TOS]
                # frame_values left on top of stack
            ]
        )

        # filter out frame values of skipped tx'es
        filter_insts = []
        for idx in idxes:
            filter_insts.extend(
                [
                    create_dup_top(),
                    cg.create_load_const(idx),
                    cg.create_binary_subscr(),
                    *create_swap(2),
                ]
            )
        # TOS: cells, frame_values[idxes[0]], ..., frame_values[idxes[...]], frame_values
        filter_insts.extend(
            [
                create_instruction("POP_TOP"),
                create_instruction("BUILD_LIST", arg=len(idxes)),
            ]
        )
        # TOS: cells, filtered frame_values

        cg.extend_output(filter_insts)
        # filter out cells of skipped tx'es using the same instructions in filter_insts,
        # but with cells as TOS instead of frame values
        cg.extend_output(
            [
                *create_swap(2),
                *copy.deepcopy(filter_insts),
                *create_swap(2),
            ]
        )
        # TOS: filtered cells, filtered frame_values

        resume_codes: list[types.CodeType] = []
        resume_names = []
        for i, cur_tx in enumerate(txes):
            resume_code, resume_name = cur_tx.create_resume(
                i,
                resume_insts[i],
                all_stack_locals_metadata[idxes[i]],
                resume_codes,
                cg,
                cur_tx is self,
                True,
            )
            resume_codes.append(resume_code)
            resume_names.append(resume_name)

        self.codegen_call_resume(resume_codes, resume_names, cg)
        cg.append_output(create_instruction("RETURN_VALUE"))

        return cg.get_instructions()

    @staticmethod
    def codegen_call_resume(
        resume_codes: list[types.CodeType], resume_names: list[str], cg: PyCodegen
    ) -> None:
        """
        Calls the provided resume functions.

        Expects the TOS to be in the state:
            [frame N cells, ..., frame 1 cells],
            [
                frame N stack + locals,
                frame N-1 stack + locals,
                ...,
                frame 1 stack + locals
            ]

        Pops the cells and frame values, leaving the result of calling the resume functions on TOS.

        Args:
            - resume_codes: list of resume function code objects to call
            - resume_names: list of the corresponding names of the resume functions
            - cg: PyCodegen object to output instructions to
        """
        # NOTE: We will load cells as we load resume functions

        # load resume functions except the root's
        cg.extend_output(create_copy(2))
        for i, (name, code) in enumerate(zip(resume_names, resume_codes)):
            if i == len(resume_names) - 1:
                break
            # stack: cells, frames, *(resume 1, ...), cells
            if code.co_freevars:
                cg.extend_output(
                    [
                        create_dup_top(),
                        cg.create_load_const(i),
                        cg.create_binary_subscr(),
                    ]
                )
                cg.make_function_with_closure(name, code)
            else:
                cg.extend_output(cg.load_function_name(name, False, 0))
            cg.extend_output(create_swap(2))
        cg.extend_output(
            [
                create_instruction("POP_TOP"),
                create_instruction("BUILD_LIST", arg=len(resume_codes) - 1),
            ]
        )

        # stack: cells, frames, [resume 1, ..., resume N - 1]
        # load root resume function
        cg.extend_output(create_swap(3))
        if resume_codes[-1].co_freevars:
            cg.extend_output(
                [
                    cg.create_load_const(-1),
                    cg.create_binary_subscr(),
                ]
            )
            cg.make_function_with_closure(resume_names[-1], resume_codes[-1])
            cg.extend_output(
                [
                    *create_rot_n(3),
                ]
            )
        else:
            cg.extend_output(
                [
                    create_instruction("POP_TOP"),
                    *cg.load_function_name(resume_names[-1], False),
                    *create_rot_n(3),
                ]
            )

        # resume 1, [resume N, ..., resume 2], frames

        # load top level-frame; final stack state should be:
        # first resume function (+ NULL),
        # [
        #     [resume N, ..., resume 2],
        #     [
        #         frame N stack + locals,
        #         ...,
        #         frame 2 stack + locals,
        #     ], *(frame 1 stack + locals)
        # ]
        cg.extend_output(
            [
                create_dup_top(),
                create_dup_top(),
                # frames, frames, frames
                cg.create_load_const(-1),
                cg.create_binary_subscr(),
                # frames, frames, frames[-1]
                *create_swap(2),
                # frames, frames[-1], frames
                cg.create_load_const(-1),
                create_instruction("DELETE_SUBSCR"),
            ]
        )

        # TOS: resume 1, remaining resumes, frames (popped), frame 1 stack + locals
        cg.extend_output(
            [
                *create_rot_n(3),
                create_instruction("BUILD_LIST", arg=2),
                *create_swap(2),
                # [resumes, frames (popped)], frame 1 stack + locals
                create_instruction("LIST_EXTEND", arg=1),
            ]
        )

        # TOS: resume 1, [remaining resumes, frames, *(frame 1 stack + locals)]
        cg.extend_output(create_call_function_ex(False, True))

    def should_compile_partial_graph(self) -> bool:
        if sys.version_info >= (3, 11):
            # Do not compile if current instruction's block is not the top with block
            entry = self.current_instruction.exn_tab_entry
            if entry and (
                not self.block_stack or entry.target is not self.block_stack[-1].target
            ):
                return False
        return (
            all(b.can_restore() for b in self.block_stack)
            and not self.one_graph
            and not self.error_on_graph_break
            and not self.is_tracing_resume_prologue
            and not self.active_generic_context_managers
            # Do not allow nested graph breaks in HOPs
            and self.output.current_tracer.parent is None
        )

    @break_graph_if_unsupported(
        push=False,
        msg_prefix="Encountered graph break when attempting to trace STORE_SUBSCR: trying to store subscript, e.g. x[key] = y",
    )
    def STORE_SUBSCR(self, inst: Instruction) -> None:
        val, obj, key = self.popn(3)
        obj.call_method(self, "__setitem__", [key, val], {})

    def DELETE_SUBSCR(self, inst: Instruction) -> None:
        obj, key = self.popn(2)
        obj.call_method(self, "__delitem__", [key], {})

    def BUILD_TUPLE(self, inst: Instruction) -> None:
        items = self.popn(inst.argval)
        self.push(TupleVariable(items))

    def BUILD_SLICE(self, inst: Instruction) -> None:
        items = self.popn(inst.argval)
        self.push(SliceVariable(items, tx=self))  # type: ignore[arg-type]

    def BUILD_LIST(self, inst: Instruction) -> None:
        items = self.popn(inst.argval)
        self.push(ListVariable(items, mutation_type=ValueMutationNew()))

    def BUILD_SET(self, inst: Instruction) -> None:
        if config.inject_BUILD_SET_unimplemented_TESTING_ONLY:
            unimplemented(
                gb_type="missing BUILD_SET handler",
                context="",
                explanation="Missing BUILD_SET bytecode handler (for testing purposes).",
                hints=[],
            )
        items = self.popn(inst.argval)
        new_set = SetVariable(items, mutation_type=ValueMutationNew())
        self.push(new_set)

    def BUILD_LIST_UNPACK(self, inst: Instruction, cls: type = ListVariable) -> None:
        seqs = self.popn(inst.argval)
        items = []
        for seq in seqs:
            try:
                items.extend(seq.force_unpack_var_sequence(self))
            except NotImplementedError:
                unimplemented(
                    gb_type="Failed to unpack object for BUILD_LIST_UNPACK",
                    context=str(seq),
                    explanation=f"{seq} cannot be unpacked into a list for the BUILD_LIST_UNPACK "
                    "bytecode (`[*x, *y, ...]`).",
                    hints=[*graph_break_hints.USER_ERROR],
                )
        self.push(cls(items, mutation_type=ValueMutationNew()))

    def BUILD_TUPLE_UNPACK(self, inst: Instruction) -> None:
        self.BUILD_LIST_UNPACK(inst, cls=TupleVariable)

    BUILD_TUPLE_UNPACK_WITH_CALL = BUILD_TUPLE_UNPACK

    def BUILD_MAP(self, inst: Instruction) -> None:
        items = self.popn(inst.argval * 2)
        d = dict(zip(items[::2], items[1::2]))
        self.push(ConstDictVariable(d, mutation_type=ValueMutationNew()))

    def BUILD_MAP_UNPACK(self, inst: Instruction) -> None:
        items = self.popn(inst.argval)
        # ensure everything is a dict
        items = [BuiltinVariable(dict).call_function(self, [x], {}) for x in items]  # type: ignore[arg-type]
        result: dict[Any, Any] = {}
        for x in items:
            assert isinstance(x, ConstDictVariable)
            result.update(x.items)
        self.push(
            ConstDictVariable(
                result,
                mutation_type=ValueMutationNew(),
            )
        )

    BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK

    def BUILD_CONST_KEY_MAP(self, inst: Instruction) -> None:
        keys = self.pop()
        values = self.popn(inst.argval)
        assert isinstance(keys, TupleVariable)
        assert keys.is_python_constant()

        keys = keys.force_unpack_var_sequence(self)
        assert len(keys) == len(values)

        self.push(
            ConstDictVariable(
                dict(zip(keys, values)),
                mutation_type=ValueMutationNew(),
            )
        )

    def MAP_ADD(self, inst: Instruction) -> None:
        k, v = self.popn(2)
        assert inst.argval > 0
        assert inst.arg is not None
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ConstDictVariable)
        obj.call_method(self, "__setitem__", (k, v), {})  # type: ignore[arg-type]

    def SET_ADD(self, inst: Instruction) -> None:
        v = self.pop()
        assert inst.argval > 0
        assert inst.arg is not None
        obj = self.stack[-inst.arg]
        assert isinstance(obj, SetVariable)
        assert obj.is_mutable()
        obj.call_method(self, "add", [v], {})  # type: ignore[arg-type]

    def SET_UPDATE(self, inst: Instruction) -> None:
        v = self.pop()
        assert inst.argval > 0
        assert inst.arg is not None
        obj = self.stack[-inst.arg]
        assert isinstance(obj, SetVariable)
        assert obj.is_mutable()
        obj.call_method(self, "update", [v], {})  # type: ignore[arg-type]

    def LIST_APPEND(self, inst: Instruction) -> None:
        v = self.pop()
        assert inst.argval > 0
        assert inst.arg is not None
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ListVariable)
        assert obj.is_mutable()
        self.output.side_effects.mutation(obj)
        obj.items.append(v)

    def MAKE_FUNCTION(self, inst: Instruction) -> None:
        flags = inst.arg
        if sys.version_info < (3, 11):
            fn_name = self.pop()
        code = self.pop()
        if sys.version_info >= (3, 11):
            # MAKE_FUNCTION behavior actually changed in 3.11, see
            # https://github.com/python/cpython/pull/93189/
            assert hasattr(code.value, "co_qualname")  # type: ignore[attr-defined]
            fn_name = ConstantVariable.create(value=code.value.co_qualname)  # type: ignore[attr-defined]
        defaults = None
        closure = None
        annotations = None
        kwdefaults = None

        if sys.version_info < (3, 13):
            # in 3.13, this is handled in SET_FUNCTION_ATTRIBUTE
            if flags is not None:
                if flags & 0x08:
                    closure = self.pop()
                if flags & 0x04:
                    annotations = self.pop()
                if flags & 0x02:
                    kwdefaults = self.pop()
                if flags & 0x01:
                    defaults = self.pop()

        self.push(
            NestedUserFunctionVariable(
                fn_name,
                code,
                self.f_globals,
                defaults,
                kwdefaults,
                annotations,
                closure,
            )
        )

    def UNPACK_SEQUENCE(self, inst: Instruction) -> None:
        seq = self.pop()
        if seq.is_tensor():
            val = seq.unpack_var_sequence(self, idxes=range(inst.argval))  # type: ignore[arg-type]
        elif isinstance(seq, GetAttrVariable) and seq.obj.is_tensor():
            # x, y = a.shape
            proxy = getattr(seq.obj.as_proxy(), seq.name)
            val = [wrap_fx_proxy(self, proxy[i]) for i in range(inst.argval)]
        elif seq.has_force_unpack_var_sequence(self):
            val = seq.force_unpack_var_sequence(self)
        else:
            unimplemented(
                gb_type="Failed to unpack object for UNPACK_SEQUENCE",
                context=str(seq),
                explanation=f"{seq} cannot be unpacked into a list for the UNPACK_SEQUENCE bytecode "
                "(i.e. `a, b, c = d`).",
                hints=[*graph_break_hints.USER_ERROR],
            )
        # pyrefly: ignore [unbound-name]
        if len(val) != inst.argval:
            unimplemented(
                gb_type="Length mismatch when unpacking object for UNPACK_SEQUENCE",
                # pyrefly: ignore [unbound-name]
                context=f"expected length: {inst.argval}, actual: {len(val)}",
                explanation=f"{seq} unpacked to a list for the UNPACK_SEQUENCE bytecode "
                "(i.e. `a, b, c = d`) with unexpected length.",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        # pyrefly: ignore [unbound-name]
        for i in reversed(val):
            self.push(i)

    def UNPACK_EX(self, inst: Instruction) -> None:
        assert 0 <= inst.argval <= 0xFFFF
        prefix = inst.argval & 0xFF  # low byte
        suffix = inst.argval >> 8  # high byte
        seq = self.pop()
        if seq.has_force_unpack_var_sequence(self):
            vals = list(seq.force_unpack_var_sequence(self))
            assert len(vals) >= prefix + suffix
            vals_prefix = vals[:prefix]
            vals_list = vals[prefix : len(vals) - suffix]
            vals_suffix = vals[len(vals) - suffix :]
            for item in reversed(vals_suffix):
                self.push(item)
            self.push(TupleVariable(vals_list))
            for item in reversed(vals_prefix):
                self.push(item)
        else:
            unimplemented(
                gb_type="Failed to unpack object for UNPACK_EX",
                context=str(seq),
                explanation=f"{seq} cannot be unpacked into a list for the UNPACK_EX bytecode.",
                hints=[*graph_break_hints.USER_ERROR],
            )

    @break_graph_if_unsupported(
        push=False, msg_prefix="Encountered intentional debugging graph break"
    )
    def graph_break_on_leaf_function(self, inst: Instruction) -> None:
        if self.is_leaf_tracer:
            unimplemented(
                gb_type="Forced graph break on leaf function",
                context="",
                explanation="Forced graph break for nested graph break testing purposes",
                hints=[
                    "Set torch._dynamo.config.debug_force_graph_break_on_leaf_return = False",
                ],
            )

    def NOP(self, inst: Instruction) -> None:
        # Dynamo-specific testing behavior
        if inst.argval == "GRAPH_BREAK_IF_LEAF":
            self.graph_break_on_leaf_function(inst)

    def POP_TOP(self, inst: Instruction) -> None:
        self.pop()

    def ROT_TWO(self, inst: Instruction) -> None:
        a = self.pop()
        b = self.pop()
        self.push(a)
        self.push(b)

    def ROT_THREE(self, inst: Instruction) -> None:
        a = self.pop()
        b = self.pop()
        c = self.pop()
        self.push(a)
        self.push(c)
        self.push(b)

    def ROT_FOUR(self, inst: Instruction) -> None:
        a = self.pop()
        b = self.pop()
        c = self.pop()
        d = self.pop()
        self.push(a)
        self.push(d)
        self.push(c)
        self.push(b)

    def DUP_TOP(self, inst: Instruction) -> None:
        a = self.pop()
        self.push(a)
        self.push(a)

    def DUP_TOP_TWO(self, inst: Instruction) -> None:
        a = self.pop()
        b = self.pop()
        self.push(b)
        self.push(a)
        self.push(b)
        self.push(a)

    def _convert_value(self, value: VariableTracker, flag: int) -> VariableTracker:
        if flag == 1:
            return BuiltinVariable(str).call_function(self, [value], {})  # type: ignore[arg-type]
        elif flag == 2:
            return BuiltinVariable(repr).call_function(self, [value], {})  # type: ignore[arg-type]
        elif flag == 3:
            return BuiltinVariable(ascii).call_function(self, [value], {})  # type: ignore[arg-type]
        return value

    def _format_value(self, fmt_spec: VariableTracker, flags: int) -> None:
        value = self.pop()
        if isinstance(value, SymNodeVariable):
            from torch._dynamo.variables.lazy import (
                LazySymNodeFormatString,
                LazyVariableTracker,
            )

            value = LazyVariableTracker.create(
                LazySymNodeFormatString(value, fmt_spec), source=value.source
            )
            self.push(value)
            return

        value = self._convert_value(value, flags & 0x03)

        fmt_var = ConstantVariable.create("{:" + fmt_spec.as_python_constant() + "}")

        self.call_function(BuiltinVariable(str.format), [fmt_var, value], {})

    def FORMAT_VALUE(self, inst: Instruction) -> None:
        flags = inst.arg
        assert flags is not None
        if (flags & 0x04) == 0x04:
            fmt_spec = self.pop()
        else:
            fmt_spec = ConstantVariable.create("")

        return self._format_value(fmt_spec, flags)

    def BUILD_STRING(self, inst: Instruction) -> None:
        format_string_parts: list[str] = []
        args: list[VariableTracker] = []
        kwargs: dict[str, VariableTracker] = {}
        assert inst.arg is not None
        for part in self.popn(inst.arg):
            if part.is_python_constant():
                format_string_parts.append("{}")
                args.append(part)
            elif isinstance(part, variables.StringFormatVariable):
                format_string_parts.append(part.format_string)
                args.extend(part.sym_args)
                if set(kwargs.keys()) & set(part.sym_kwargs.keys()):
                    unimplemented(
                        gb_type="BUILD_STRING key conflict",
                        context=f"format_string_parts: {format_string_parts}, kwargs: {kwargs}, part.sym_kwargs: {part.sym_kwargs}",
                        explanation="Failed to build format string due to key conflict",
                        hints=[*graph_break_hints.USER_ERROR],
                    )
                kwargs.update(part.sym_kwargs)
            else:
                unimplemented(
                    gb_type="BUILD_STRING type error",
                    context=str(part),
                    explanation="Format string part type is not correct - expected constant or format string.",
                    hints=[*graph_break_hints.USER_ERROR],
                )
        self.push(
            variables.StringFormatVariable.create(
                "".join(format_string_parts), args, kwargs
            )
        )

    def IS_OP(self, inst: Instruction) -> None:
        assert inst.argval == 0 or inst.argval == 1
        if inst.argval == 0:
            new_argval = "is"
        else:
            new_argval = "is not"
        new_inst = create_instruction("COMPARE_OP", argval=new_argval)
        self.COMPARE_OP(new_inst)

    def CONTAINS_OP(self, inst: Instruction) -> None:
        assert inst.argval == 0 or inst.argval == 1
        left, right = self.popn(2)
        op = inst.argval
        try:
            self.push(right.call_method(self, "__contains__", [left], {}))
        except (
            # right.__contains__ can raise TypeError
            exc.ObservedTypeError,
            # Ideally we should only capture TypeError here but some VTs don't
            # implement hasattr(vt, "__contains__") entirely
            Unsupported,
        ) as excp:  # object doesn't support __contains__
            # Use __iter__ as fallback
            if isinstance(excp, Unsupported):
                if excp.skip_frame:
                    # do not absorb graph break with skip_frame set
                    raise
                excp.remove_from_stats()
            self.push(
                self.inline_user_function_return(
                    VariableTracker.build(self, impl_CONTAINS_OP_fallback),
                    [left, right],
                    {},
                )
            )
        if op == 1:
            self.UNARY_NOT(inst)

    def LIST_EXTEND(self, inst: Instruction) -> None:
        v = self.pop()
        assert inst.argval > 0
        assert inst.arg is not None
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ListVariable)
        assert obj.is_mutable()
        obj.call_method(self, "extend", [v], {})  # type: ignore[arg-type]

    def LIST_TO_TUPLE(self, inst: Instruction) -> None:
        self.push(BuiltinVariable(tuple).call_function(self, [self.pop()], {}))  # type: ignore[arg-type]

    def STOPITERATION_ERROR(self, inst: Instruction) -> None:
        # wrap the generator body in a try: ... except StopIteration: ... which
        # converts the StopIteration into a RuntimeError
        # https://peps.python.org/pep-0479/
        # https://github.com/python/cpython/pull/99006
        # https://github.com/python/cpython/commit/28187141cc34063ef857976ddbca87ba09a882c2
        val = self.stack[-1]
        assert self._isinstance_exception(val)
        if val.exc_type is StopIteration:  # type: ignore[union-attr]
            new_val = variables.BuiltinVariable(RuntimeError).call_function(
                self,  # type: ignore[arg-type]
                [ConstantVariable("generator raised StopIteration")],
                {},
            )
            new_val.call_setattr(self, ConstantVariable("__context__"), val)  # type: ignore[attr-defined]
            new_val.call_setattr(self, ConstantVariable("__cause__"), val)  # type: ignore[attr-defined]
            self.stack[-1] = new_val

    def DICT_MERGE(self, inst: Instruction) -> None:
        v = self.pop()
        assert inst.argval > 0
        assert inst.arg is not None
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ConstDictVariable)
        assert obj.is_mutable()
        obj.call_method(self, "update", [v], {})  # type: ignore[arg-type]

    DICT_UPDATE = DICT_MERGE

    def GEN_START(self, inst: Instruction) -> None:
        self.pop()

    def GET_LEN(self, inst: Instruction) -> None:
        tos = self.stack[-1]
        if tos.is_python_constant():
            self.push(ConstantVariable.create(len(tos.as_python_constant())))
        else:
            self.push(tos.call_method(self, "__len__", [], {}))

    def MATCH_MAPPING(self, inst: Instruction) -> None:
        tos = self.stack[-1]
        assert isinstance(tos, ConstDictVariable)
        if isinstance(tos.items, collections.abc.Mapping):
            self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(False))

    def MATCH_SEQUENCE(self, inst: Instruction) -> None:
        tos = self.stack[-1]
        assert tos.is_python_constant()
        tos_value = tos.as_python_constant()
        if isinstance(tos_value, collections.abc.Sequence) and not isinstance(
            tos_value, (str, bytes, bytearray)
        ):
            self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(False))

    def MATCH_KEYS(self, inst: Instruction) -> None:
        tos = self.stack[-1]
        assert isinstance(tos, TupleVariable)
        keys = tos.unpack_var_sequence(self)  # type: ignore[arg-type]
        tos1 = self.stack[-2]
        assert isinstance(tos1, ConstDictVariable)

        if all(k in tos1 for k in keys):  # type: ignore[attr-defined]
            self.push(TupleVariable([tos1.getitem_const(self, k) for k in keys]))  # type: ignore[attr-defined,arg-type]
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(None))
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(False))

    def LOAD_ASSERTION_ERROR(self, inst: Instruction) -> None:
        self.push(self.load_builtin_from_argval("AssertionError"))

    def LOAD_BUILD_CLASS(self, inst: Instruction) -> None:
        self.push(self.load_builtin_from_argval("__build_class__"))

    UNARY_POSITIVE = stack_op(operator.pos)
    UNARY_NEGATIVE = stack_op(operator.neg)
    UNARY_NOT = stack_op(operator.not_)
    UNARY_INVERT = stack_op(operator.invert)

    BINARY_POWER = stack_op(operator.pow)
    BINARY_MULTIPLY = stack_op(operator.mul)
    BINARY_MATRIX_MULTIPLY = stack_op(operator.matmul)
    BINARY_FLOOR_DIVIDE = stack_op(operator.floordiv)
    BINARY_TRUE_DIVIDE = stack_op(operator.truediv)
    BINARY_MODULO = stack_op(operator.mod)
    BINARY_REMAINDER = stack_op(operator.mod)
    BINARY_ADD = stack_op(operator.add)
    BINARY_SUBTRACT = stack_op(operator.sub)
    BINARY_SUBSCR = break_graph_if_unsupported(
        push=True,
        msg_prefix="Encountered graph break when attempting to trace BINARY_SUBSCR: a binary subscript, e.g. x[attr]",
    )(stack_op(operator.getitem))
    BINARY_LSHIFT = stack_op(operator.lshift)
    BINARY_RSHIFT = stack_op(operator.rshift)
    BINARY_AND = stack_op(operator.and_)
    BINARY_OR = stack_op(operator.or_)
    BINARY_XOR = stack_op(operator.xor)

    INPLACE_POWER = stack_op(operator.ipow)
    INPLACE_MULTIPLY = stack_op(operator.imul)
    INPLACE_MATRIX_MULTIPLY = stack_op(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = stack_op(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = stack_op(operator.itruediv)
    INPLACE_MODULO = stack_op(operator.imod)
    INPLACE_REMAINDER = stack_op(operator.imod)
    INPLACE_ADD = stack_op(operator.iadd)
    INPLACE_SUBTRACT = stack_op(operator.isub)
    INPLACE_LSHIFT = stack_op(operator.ilshift)
    INPLACE_RSHIFT = stack_op(operator.irshift)
    INPLACE_AND = stack_op(operator.iand)
    INPLACE_XOR = stack_op(operator.ixor)
    INPLACE_OR = stack_op(operator.ior)

    # 3.11 opcodes
    def RESUME(self, inst: Instruction) -> None:
        if inst.arg == 0:
            self.append_prefix_inst(inst)
            self.accept_prefix_inst = False
        else:
            assert not self.accept_prefix_inst

    if sys.version_info >= (3, 11):

        def BINARY_OP(self, inst: Instruction) -> None:
            assert inst.arg is not None
            return _binary_op_lookup[inst.arg](self, inst)

    def PRECALL(self, inst: Instruction) -> None:
        pass

    def KW_NAMES(self, inst: Instruction) -> None:
        kw_names = self.code_options["co_consts"][inst.arg]
        assert isinstance(kw_names, tuple)
        for name in kw_names:
            assert isinstance(name, str)
        assert self.kw_names is None
        self.kw_names = ConstantVariable.create(value=kw_names)  # type: ignore[assignment]

    def PUSH_NULL(self, inst: Instruction) -> None:
        self.push(NullVariable())

    def _call(self, inst: Instruction, call_kw: bool = False) -> None:
        # see https://docs.python.org/3.11/library/dis.html#opcode-CALL
        # for convention
        if call_kw:
            # TOS is kw_names for CALL_KW instruction
            assert sys.version_info >= (3, 13)
            kw_names = self.pop()
            assert isinstance(kw_names, TupleVariable) and kw_names.is_python_constant()
            kw_names = kw_names.as_python_constant()
        else:
            kw_names = self.kw_names.value if self.kw_names else ()

        assert inst.arg is not None
        contents = self.popn(inst.arg + 2)
        if sys.version_info >= (3, 13):
            # NULL and callable swapped
            fn = contents[0]
            args = [] if isinstance(contents[1], NullVariable) else [contents[1]]
        else:
            if isinstance(contents[0], NullVariable):
                fn = contents[1]
                args = []
            else:
                fn = contents[0]
                args = [contents[1]]

        if kw_names:
            args = args + contents[2 : -len(kw_names)]

            kwargs_list = contents[-len(kw_names) :]

            kwargs = dict(zip(kw_names, kwargs_list))

            assert len(kwargs) == len(kw_names)
        else:
            args = args + contents[2:]
            kwargs = {}

        try:
            # if call_function fails, need to set kw_names to None, otherwise
            # a subsequent call may have self.kw_names set to an old value
            self.call_function(fn, args, kwargs)
        finally:
            self.kw_names = None

    @break_graph_if_unsupported(
        push=True,
        msg_prefix="Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y)",
    )
    def CALL(self, inst: Instruction) -> None:
        self._call(inst)

    def COPY(self, inst: Instruction) -> None:
        assert inst.arg is not None
        self.push(self.stack[-inst.arg])

    def SWAP(self, inst: Instruction) -> None:
        assert inst.arg is not None
        self.stack[-1], self.stack[-inst.arg] = self.stack[-inst.arg], self.stack[-1]

    JUMP_BACKWARD = jump
    JUMP_BACKWARD_NO_INTERRUPT = jump

    POP_JUMP_FORWARD_IF_TRUE = generic_jump(operator.truth, False)
    POP_JUMP_BACKWARD_IF_TRUE = generic_jump(operator.truth, False)
    POP_JUMP_FORWARD_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_BACKWARD_IF_FALSE = generic_jump(operator.not_, False)

    def CACHE(self, inst: Instruction) -> None:
        pass

    def BEFORE_WITH(self, inst: Instruction) -> None:
        self.setup_or_before_with(inst)

    def enter_ctx(
        self,
        ctx: Union[ContextWrappingVariable, GenericContextWrappingVariable],
        inst: Instruction,
    ) -> VariableTracker:
        if (
            isinstance(ctx, GenericContextWrappingVariable)
            and not ctx.supports_graph_breaks()
        ):
            self.active_generic_context_managers.append(ctx)

        if sys.version_info >= (3, 11):
            # See update_block_stack/create_resume for block stack details.
            # Only push a block if the current instruction's block is a
            # with block that is not nested in a try block - that is, the current
            # instruction's block target is the same as the top block's target.
            if inst.exn_tab_entry and (
                not self.block_stack
                or inst.exn_tab_entry.target is not self.block_stack[-1].target
            ):
                target = None
            else:
                assert self.next_instruction.exn_tab_entry is not None
                target = self.next_instruction.exn_tab_entry.target
        else:
            target = inst.target

        if target:
            if isinstance(self, InstructionTranslator) or config.nested_graph_breaks:
                self.block_stack.append(
                    BlockStackEntry(inst, target, len(self.stack), ctx)
                )
            else:
                self.block_stack.append(BlockStackEntry(inst, target, len(self.stack)))

        return ctx.enter(self)  # type: ignore[arg-type]

    @staticmethod
    def unsupported_ctx_graph_break(ctx: VariableTracker) -> NoReturn:
        unimplemented(
            gb_type="Unsupported context manager",
            context=f"Attempted SETUP_WITH/BEFORE_WITH/LOAD_SPECIAL on {ctx}",
            explanation=f"Dynamo does not know how to enter a `{ctx.python_type_name()}` context manager.",
            hints=[
                "Avoid using the unsupported context manager.",
                "If the context manager seems like it should be supported (e.g. torch.set_grad_enabled), then "
                "it may be the case that it was created outside the compiled region, which Dynamo does not support. "
                "Supported context managers can cross graph break boundaries only if they are local non-closure "
                "variables, or are intermediate values.",
                "File an issue to PyTorch. Simple context managers can potentially be supported, "
                "but note that context managers can't be supported in general",
            ],
        )

    def setup_or_before_with(self, inst: Instruction) -> None:
        ctx = self.pop()
        if not isinstance(
            ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
        ):
            self.unsupported_ctx_graph_break(ctx)

        # Need this redundant check for mypy
        assert isinstance(
            ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
        )

        self.push(WithExitFunctionVariable(ctx, inst.target))
        self.push(self.enter_ctx(ctx, inst))

    def append_prefix_inst(self, inst: Instruction) -> None:
        assert self.accept_prefix_inst
        self.prefix_insts.append(inst)

    def MAKE_CELL(self, inst: Instruction) -> None:
        if sys.version_info >= (3, 12) and not self.accept_prefix_inst:
            # In 3.12+, MAKE_CELL is not longer necessarily a prefix instruction.
            # It can be generated by inlined comprehensions.
            assert isinstance(self.symbolic_locals[inst.argval], NullVariable)
            self.symbolic_locals[inst.argval] = (
                self.output.side_effects.track_cell_new()
            )
        else:
            self.append_prefix_inst(inst)

    def COPY_FREE_VARS(self, inst: Instruction) -> None:
        self.append_prefix_inst(inst)

    def RETURN_GENERATOR(self, inst: Instruction) -> None:
        self.append_prefix_inst(inst)

    # 3.12 opcodes
    # BINARY/STORE_SLICE opcodes are broken down into
    # BUILD_SLICE 2 and BINARY/STORE_SUBSCR

    def END_FOR(self, inst: Instruction) -> None:
        if sys.version_info >= (3, 13):
            self.pop()
        else:
            self.popn(2)

    def LOAD_FAST_CHECK(self, inst: Instruction) -> None:
        if istype(self.symbolic_locals.get(inst.argval, None), NullVariable):
            unimplemented(
                gb_type="LOAD_FAST_CHECK on uninitialized variable",
                context=inst.argval,
                explanation=f"Attempted to load uninitialized local variable {inst.argval}",
                hints=[*graph_break_hints.USER_ERROR],
            )
        self.LOAD_FAST(inst)

    def LOAD_FAST_AND_CLEAR(self, inst: Instruction) -> None:
        if inst.argval not in self.symbolic_locals:
            self.push(NullVariable())
        else:
            self.LOAD_FAST(inst)
        self.symbolic_locals[inst.argval] = NullVariable()

    def LOAD_SUPER_ATTR(self, inst: Instruction) -> None:
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        assert inst.arg is not None
        if inst.arg & 1:
            self.LOAD_METHOD(inst)
        else:
            self._load_attr(inst.argval)

    def CALL_INTRINSIC_1(self, inst: Instruction) -> None:
        if inst.argval == 3:
            # INTRINSIC_STOPITERATION_ERROR
            self.STOPITERATION_ERROR(inst)
        elif inst.argval == 5:
            # INTRINSIC_UNARY_POSITIVE
            self.UNARY_POSITIVE(inst)
        elif inst.argval == 6:
            # INTRINSIC_LIST_TO_TUPLE
            self.push(TupleVariable(self.pop().force_unpack_var_sequence(self)))
        else:
            unimplemented(
                gb_type="Missing CALL_INTRINSIC_1 handler",
                context=f"CALL_INTRINSIC_1 operand: {inst.argval}",
                explanation=f"No handler implemented for CALL_INTRINSIC_1 {inst.argval} instruction.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

    def END_SEND(self, inst: Instruction) -> None:
        tos = self.pop()
        self.pop()
        self.push(tos)

    # 3.13 opcodes
    # fused instructions LOAD_FAST_LOAD_FAST, STORE_FAST_STORE_FAST, STORE_FAST_LOAD_FAST
    # are broken down.
    @break_graph_if_unsupported(
        push=True,
        msg_prefix="Encountered graph break when attempting to trace CALL_KW: "
        "a function call with keyword arguments, e.g. f(x=True)",
    )
    def CALL_KW(self, inst: Instruction) -> None:
        self._call(inst, call_kw=True)

    def TO_BOOL(self, inst: Instruction) -> None:
        # TO_BOOL only precedes a conditional jump or UNARY_NOT (see compile.c in CPython)
        # So we can skip this instruction as long as we remember to codegen a TO_BOOL
        # before conditional jumps/UNARY_NOT.
        assert self.next_instruction.opname in (
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_IF_FALSE",
            "UNARY_NOT",
        )

    def SET_FUNCTION_ATTRIBUTE(self, inst: Instruction) -> None:
        flags = inst.arg
        assert flags is not None
        fn = self.pop()
        assert isinstance(fn, NestedUserFunctionVariable)
        attr = self.pop()

        if flags & 0x10:
            assert sys.version_info >= (3, 14)

            # maybe use Format.VALUE_WITH_FAKE_GLOBALS instead?
            # https://docs.python.org/3/library/annotationlib.html#annotationlib.Format.VALUE_WITH_FAKE_GLOBALS
            attr = attr.call_function(self, [ConstantVariable.create(1)], {})
            fn.annotations = attr
        elif flags & 0x08:
            fn.closure = attr
        elif flags & 0x04:
            fn.annotations = attr
        elif flags & 0x02:
            fn.kwdefaults = attr
        elif flags & 0x01:
            fn.defaults = attr

        self.push(fn)

    def CONVERT_VALUE(self, inst: Instruction) -> None:
        self.push(self._convert_value(self.pop(), inst.argval))

    def FORMAT_SIMPLE(self, inst: Instruction) -> None:
        self._format_value(ConstantVariable.create(""), 0)

    def FORMAT_WITH_SPEC(self, inst: Instruction) -> None:
        self._format_value(self.pop(), 0)

    # 3.14 opcodes
    LOAD_FAST_BORROW = LOAD_FAST
    NOT_TAKEN = NOP
    POP_ITER = POP_TOP

    # See
    # https://github.com/python/cpython/blob/805e3368d6d07e58430654d1365283924fdf4143/Python/ceval.c#L559
    # for the LOAD_SPECIAL table - make sure it matches for Python 3.14+
    _load_special_names = (
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
    )

    def LOAD_SPECIAL(self, inst: Instruction) -> None:
        assert isinstance(inst.arg, int), "expected LOAD_SPECIAL arg to be set to int"
        attr = self._load_special_names[inst.arg]
        if attr in ("__enter__", "__exit__"):
            ctx = self.pop()
            if not isinstance(
                ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
            ):
                self.unsupported_ctx_graph_break(ctx)

            # Need this redundant check for mypy
            assert isinstance(
                ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
            )
            if attr == "__enter__":
                self.push(WithEnterFunctionVariable(ctx))
                self.PUSH_NULL(inst)
            else:
                # WithExitFunctionVariable doesn't really do anything with target for 3.11+
                self.push(WithExitFunctionVariable(ctx, None))
                self.PUSH_NULL(inst)
        else:
            # Implementation is similar to LOAD_METHOD for 3.13+
            self._load_attr(attr)
            obj = self.pop()
            self.push(obj)
            self.PUSH_NULL(inst)

    def LOAD_SMALL_INT(self, inst: Instruction) -> None:
        self.push(ConstantVariable.create(inst.argval))

    # See
    # https://github.com/python/cpython/blob/7519ac294fc5c4fd7fb9cb8dc0edc960688cf887/Python/pylifecycle.c#L814
    # for the common constants - make sure it matches for Python 3.14+.
    # The common constants are all attributes of `builtins`.
    _common_constants = (
        "AssertionError",
        "NotImplementedError",
        "tuple",
        "all",
        "any",
    )

    def LOAD_COMMON_CONSTANT(self, inst: Instruction) -> None:
        assert isinstance(inst.arg, int), (
            "expected LOAD_COMMON_CONSTANT arg to be set to int"
        )
        self.push(self.load_builtin_from_argval(self._common_constants[inst.arg]))

    def is_non_empty_graph(self) -> bool:
        if self.output.count_calls() > 1:
            # perf optimization only
            self.is_non_empty_graph = lambda: True  # type: ignore[method-assign]
            return True
        return False

    def format_frame_summary(
        self, additional_stack_frames: Optional[list[Any]] = None
    ) -> str:
        if additional_stack_frames is None:
            additional_stack_frames = []
        return "".join(
            traceback.format_list(
                [self.frame_summary()] + list(reversed(additional_stack_frames))
            )
        )

    def frame_summary(self) -> traceback.FrameSummary:
        return traceback.FrameSummary(
            getattr(self.f_code, "co_filename", "<unknown>"),
            self.lineno,
            getattr(self.f_code, "co_name", "<unknown>"),
            lookup_line=False,
        )

    def is_co_filename_from_nn_modules(self) -> bool:
        filename = getattr(self.f_code, "co_filename", "<unknown>")
        nn_modules_pattern = re.compile(r".*torch/nn/modules.*")
        return nn_modules_pattern.match(filename) is not None

    def store_global_weakref_by_id(self, prefix: str, value: Any) -> str:
        global_name = self.output.install_global_by_id(prefix, weakref.ref(value))
        install_guard(
            GlobalWeakRefSource(global_name).make_guard(GuardBuilder.WEAKREF_ALIVE)
        )
        return global_name

    @property
    def fake_mode(self) -> Optional[FakeTensorMode]:
        return self.output.tracing_context.fake_mode

    @contextlib.contextmanager
    def strict_translation_mode(
        self, check_fn: Callable[[VariableTracker], bool]
    ) -> Any:
        """
        Strict mode is enabled on a per-VariableTracker level depending on the return value of check_fn(node).
        """
        prior = self.strict_checks_fn
        self.strict_checks_fn = check_fn
        try:
            yield
        finally:
            self.strict_checks_fn = prior

    def speculate(self) -> SpeculationEntry:
        assert self.instruction_pointer is not None
        assert self.instruction_pointer > 0
        return self.speculation_log.next(
            self.f_code.co_filename,
            self.lineno,
            self.instruction_pointer - 1,
            self.instructions[self.instruction_pointer - 1],
        )

    def _make_frame_loc(
        self, filename: str, lineno: Optional[int], fallback_lineno: int
    ) -> tuple[str, int]:
        if lineno is None or lineno < 0:
            return (filename, fallback_lineno)
        return (filename, lineno)

    def _get_frame_loc_chain(
        self, frame_loc: tuple[str, int]
    ) -> tuple[tuple[str, int], ...]:
        frame_loc_chain_list: list[tuple[str, int]] = []

        if config.nested_graph_breaks:
            current_tx: Optional[InstructionTranslatorBase] = self.parent
            while current_tx is not None:
                parent_frame_loc = self._make_frame_loc(
                    current_tx.f_code.co_filename,
                    current_tx.lineno,
                    current_tx.f_code.co_firstlineno,
                )
                frame_loc_chain_list.append(parent_frame_loc)
                current_tx = current_tx.parent

        frame_loc_chain_list.reverse()
        frame_loc_chain_list.append(frame_loc)
        return tuple(frame_loc_chain_list)

    def log_graph_break(
        self,
        code_options: dict[str, Any],
        reason: str,
        exc: Unsupported | StepUnsupported,
    ) -> None:
        if exc.logged:
            return

        user_stack = getattr(exc, "real_stack", None)

        if user_stack is None:
            user_stack = torch._guards.TracingContext.extract_stack()

        try:
            if config.nested_graph_breaks and self.parent is not None:
                frame_loc = self._make_frame_loc(
                    self.f_code.co_filename,
                    self.lineno,
                    self.f_code.co_firstlineno,
                )
            else:
                frame_loc = self._make_frame_loc(
                    user_stack[-1].filename,
                    user_stack[-1].lineno,
                    0,
                )
        except IndexError:
            # first instruction
            frame_loc = (
                code_options["co_filename"],
                code_options["co_firstlineno"],
            )
        frame_loc_chain = self._get_frame_loc_chain(frame_loc)
        stack_above_dynamo_formatted = ""
        if config.verbose:
            stack_above_dynamo = get_stack_above_dynamo()
            stack_above_dynamo_formatted = "".join(
                traceback.format_list(stack_above_dynamo)
            )
        else:
            user_stack = get_stack_above_dynamo() + user_stack  # type: ignore[assignment]
            user_stack = collapse_resume_frames(user_stack)
        user_stack_formatted = "".join(traceback.format_list(user_stack))

        # Add HOP context after the first line of reason if present
        if exc is not None:
            reason = augment_exc_message_with_hop_name(exc, reason)

        user_stack_trace = (
            f"Graph break in user code at {frame_loc[0]}:{frame_loc[1]}\n"
            f"Graph Break Reason: {reason}\n"
            "\nUser code traceback:\n"
        )

        if config.verbose:
            user_stack_trace += (
                f"{stack_above_dynamo_formatted}\n"
                "========== most recent `torch.compile` tracing attempt started here ==========\n\n"
                f"{user_stack_formatted}\n"
                "NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! "
                "This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another "
                "Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python "
                "function, which Dynamo intercepts as a top-level frame.\n"
            )
        else:
            user_stack_trace += str(user_stack_formatted)

        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_graph_break_reason",
                "encoding": "string",
            },
            payload_fn=lambda: f"{user_stack_trace}\n{traceback.format_exc()}",
        )

        # torch._dynamo.explain() formats this a little nicer, and presents a slightly
        # more actionable user code pointer
        gb_type = exc.gb_type if isinstance(exc, Unsupported) else type(exc)
        if (
            graph_break_log.isEnabledFor(logging.DEBUG)
            and not explain
            and graph_break_dup_warning_checker.add((gb_type, frame_loc_chain))  # type: ignore[arg-type]
        ):
            # This log line MUST contain the string "Graph break in user code",
            # This log line is exercised from
            #   python test/dynamo/test_exc.py -k test_graph_break_log
            if config.verbose:
                user_stack_trace += (
                    "\nMost recent bytecode instructions traced (max 20):\n"
                )
                user_stack_trace += "\n".join(self.latest_bytecode_queue) + "\n"

            graph_break_log.debug(
                user_stack_trace,
            )
        else:
            # This log line MUST not contain the string "Graph break in user code",
            # exercised by
            #   python test/dynamo/test_misc.py -k test_duplicate_graph_break_log
            graph_break_log.debug(
                "Graph break (user stack suppressed due to duplicate graph break) in user code at %s:%s\nGraph Break Reason: %s",
                frame_loc[0],
                frame_loc[1],
                reason,
            )

        exc.logged = True

    @staticmethod
    def raise_loop_graph_break(code: types.CodeType, exc: Unsupported) -> NoReturn:
        unimplemented(
            gb_type="graph break in loop",
            context=f"frame skipped: {format_frame_info(code)}",
            explanation="torch.compile detected a graph break in a for/while loop. "
            "Skipping the frame and falling back to eager, as graph breaks in loops are not supported.",
            hints=[*graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK],
            from_exc=exc,
            skip_frame=True,
        )

    def __init__(
        self,
        output: OutputGraph,
        instructions: list[Instruction],
        f_locals: dict[str, Any],
        f_globals: dict[str, Any],
        f_builtins: dict[str, Any],
        code_options: dict[str, Any],
        symbolic_locals: dict[str, VariableTracker],
        symbolic_globals: dict[str, VariableTracker],
        symbolic_torch_function_state: SymbolicTorchFunctionState,
        symbolic_stream_state: SymbolicStreamState,
        f_code: types.CodeType,
        export: bool,
        inline_depth: int,
        speculation_log: SpeculationLog,
        exn_vt_stack: ExceptionStack,
        distributed_state: Optional[DistributedState],
        # This determines whether to use the execution recorder.
        closure: Optional[tuple[types.CellType]] = None,
        package: Optional[CompilePackage] = None,
    ) -> None:
        super().__init__()
        self.speculation_log = speculation_log
        self.distributed_state = distributed_state

        # Mutable state checkpointed by copy_graphstate()
        self.output = output
        self.symbolic_locals = symbolic_locals
        self.symbolic_globals = symbolic_globals
        self.symbolic_torch_function_state = symbolic_torch_function_state
        self.symbolic_stream_state = symbolic_stream_state
        # used to keep cell/freevars alive after pruning symbolic_locals (prune_dead_locals)
        # in order to generate any nested closures
        self.post_prune_cell_and_freevars = None
        self.stack: list[VariableTracker] = []
        self.instruction_pointer = 0
        self.start_point = None
        self.current_instruction = create_instruction("NOP")
        self.current_instruction_push = True
        self.block_stack = []
        # states before SETUP_WITH for checkpointing and fallback
        self.active_generic_context_managers: list[GenericContextWrappingVariable] = []
        self.lineno = -1
        self.kw_names = None
        self.accept_prefix_inst = True
        self.prefix_insts = []
        self.exn_vt_stack = exn_vt_stack
        self.latest_bytecode_queue = deque(maxlen=20)

        # Properties of the input/output code
        self.instructions: list[Instruction] = instructions
        self.indexof: dict[Instruction, int] = get_indexof(self.instructions)
        self.f_locals: dict[str, Any] = (
            f_locals  # needed for recording accessed locals for replay
        )
        self.f_globals: dict[str, Any] = f_globals
        self.f_builtins: dict[str, Any] = f_builtins
        self.code_options: dict[str, Any] = code_options
        self.f_code: types.CodeType = f_code
        self.closure = closure

        # Execution record for replaying errors
        if closure is not None and config.replay_record_enabled:
            self.exec_recorder = ExecutionRecorder(
                code=f_code, closure=closure, code_options=code_options
            )
        else:
            self.exec_recorder = None
        # Stack of module being parsed, current nn.module is at the end of ordered dict.
        # The first field of tuple is the fully qualified name of current module
        # in original hierarchy.  The second field is the type of current nn.module
        self.nn_module_stack: dict[str, tuple[str, type[Any]]] = {}
        self.num_calls: dict[str, int] = {}
        # Flag to indicate whether tracing is used for export.
        self.export = export
        # NOTE: one_graph is used for export/fullgraph=True to always force errors on graph breaks.
        # To toggle erroring/resuming on graph breaks during fullgraph=False compile, self.error_on_graph_break
        # is used instead. Every step(), its value is updated to the global tls.error_on_graph_break.
        # We mirror this value since cleanup may (correctly) inadvertently change tls.error_on_graph_break.
        # This assumes that we cannot both trace a change to tls.error_on_graph_break and graph break on
        # the same instruction.
        self.one_graph = False
        self.error_on_graph_break = False
        # Also do not graph break when tracing resume function prologues
        self.is_tracing_resume_prologue = False

        self.current_speculation = None

        self.strict_checks_fn = None

        self.is_leaf_tracer = True
        self.parent = None
        self.debug_locals = []

        self.package = package

        from .resume_execution import (
            CO_ASYNC_GENERATOR,
            CO_COROUTINE,
            CO_GENERATOR,
            CO_ITERABLE_COROUTINE,
        )

        if f_code.co_flags & (
            CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR
        ):
            self.push(BuiltinVariable(None))

        self.inline_depth = inline_depth
        self.inconsistent_side_effects = False
        self._constants_cache: list[
            Optional[Union[ConstantVariable, SliceVariable]]
        ] = [None] * len(f_code.co_consts)

        self.is_trace_bytecode_log_enabled: Optional[bool] = (
            trace_bytecode_log.isEnabledFor(logging.DEBUG)
        )
        self.is_trace_source_log_enabled: Optional[bool] = (
            trace_source_log.isEnabledFor(logging.DEBUG)
        )
        linecache.lazycache(f_code.co_filename, f_globals)


class InstructionTranslator(InstructionTranslatorBase):
    @staticmethod
    def current_tx() -> InstructionTranslator:
        return tls.current_tx

    @contextlib.contextmanager
    def set_current_tx(self) -> Any:
        prior = getattr(tls, "current_tx", None)
        tls.current_tx = self
        try:
            yield
        finally:
            tls.current_tx = prior

    def __init__(
        self,
        instructions: list[Instruction],
        f_code: types.CodeType,
        f_locals: dict[str, Any],
        f_globals: dict[str, Any],
        f_builtins: dict[str, Any],
        closure: Optional[tuple[Any, ...]],
        torch_function_mode_stack: Any,
        code_options: dict[str, Any],
        compiler_fn: Any,
        one_graph: bool,
        export: bool,
        export_constraints: Any,
        frame_state: Any,
        speculation_log: SpeculationLog,
        exn_vt_stack: ExceptionStack,
        distributed_state: Optional[DistributedState],
        package: Optional[CompilePackage],
    ) -> None:
        _step_logger()(
            logging.INFO,
            f"torchdynamo start tracing {f_code.co_name} {code_options['co_filename']}:{code_options['co_firstlineno']}",
        )
        super().__init__(
            output=OutputGraph(
                code_options,
                compiler_fn,
                self,
                export,
                export_constraints,
                frame_state,
                local_scope=f_locals,
                global_scope=f_globals,
                f_code=f_code,
                torch_function_mode_stack=torch_function_mode_stack,
                one_graph=one_graph,
                package=package,
            ),
            instructions=instructions,
            f_locals=f_locals,
            f_globals=f_globals,
            f_builtins=f_builtins,
            closure=closure,
            code_options=code_options,
            symbolic_locals={},  # set below
            # A global var is inserted only after a STORE_GLOBAL happens to it
            symbolic_globals={},
            symbolic_torch_function_state=None,  # type: ignore[arg-type] # set below
            symbolic_stream_state=None,  # type: ignore[arg-type] # set below
            f_code=f_code,
            export=export,
            inline_depth=0,
            speculation_log=speculation_log,
            exn_vt_stack=exn_vt_stack,
            distributed_state=distributed_state,
            package=package,
        )

        self._throw_if_in_functorch()

        # as soon as we create the tracing context we should keep it active, so any calls
        # into dynamo apis can rely on finding it
        with tracing(self.output.tracing_context), self.set_current_tx():
            self.one_graph: bool = one_graph
            self.export = export
            if self.export:
                assert self.one_graph, (
                    "Export without one graph - something has gone wrong."
                )

            self.symbolic_locals = {}
            # Populate `symbolic_locals` with non-cell variables.
            cell_and_freevars: set[str] = set(self.cell_and_freevars())

            dynamism = code_context.get_context(f_code).get("dynamism", None)
            for name, value in f_locals.items():
                if name not in cell_and_freevars:
                    local_dynamism = None
                    if dynamism:
                        local_dynamism = frozenset(dynamism.get(name, {}).items())
                    var = LazyVariableTracker.create(
                        value,
                        # NOTE: pyrefly currently has issue with init for frozen
                        # dataclass, so ignore these errors
                        LocalSource(
                            name,
                            is_input=True,
                            dynamism=local_dynamism,
                        ),
                    )
                    self.symbolic_locals[name] = var

            # Populate `symbolic_locals` with cells created by this frame,
            # effectively implementing the `MAKE_CELL` instructions.
            side_effects = self.output.side_effects
            for name in self.cellvars():
                if name in f_locals:
                    # This models cells that are also function inputs.
                    value = f_locals[name]
                    # NOTE: root frame inputs that are captured by a nested
                    # function become special cell objects -- they exist in
                    # `f_locals` as contents of the cells, rather than the cells
                    # objects themselves.
                    #
                    # In Dynamo, we choose to represent such input cell objects
                    # as newly created (rather than pre-existing) cell objects,
                    # because
                    #
                    # 1. The reason for representing a pre-existing cell object
                    # is to emit guard or codegen mutations. However, local
                    # cells should never be used for guards. Moreover, at this
                    # point these input cell objects should've never been
                    # accessed by anyone else, since Dynamo intercepts the frame
                    # right after its evaluation starts, i.e., right after these
                    # cell objects are created. So they should have no external
                    # reference, meaning no mutation needs to be propagated.
                    #
                    # 2. This conveniently allows codegen to prune away
                    # mutations to these cells, unless they escape the frame.
                    contents_source = LocalSource(
                        name,
                        is_input=True,
                        is_derefed_cell_contents=True,
                    )
                    contents_var: VariableTracker = LazyVariableTracker.create(
                        value, contents_source
                    )
                    cell_var = side_effects.track_cell_new()
                    side_effects.store_cell(cell_var, contents_var)
                else:
                    cell_var = side_effects.track_cell_new()
                cell_var.local_name = name  # type: ignore[attr-defined]
                self.symbolic_locals[name] = cell_var

            # Populate `symbolic_locals` with cells captured by this frame,
            # effectively implementing the `COPY_FREE_VARS` instruction.
            assert closure is not None
            for name, cell in zip(self.freevars(), closure):
                cell_source = LocalCellSource(name)
                contents_source = LocalSource(name, is_derefed_cell_contents=True)
                try:
                    contents_var = LazyVariableTracker.create(
                        cell.cell_contents, contents_source
                    )
                except ValueError:
                    # Cell has not yet been assigned
                    contents_var = variables.DeletedVariable()
                cell_var = side_effects.track_cell_existing(
                    cell_source, cell, contents_var
                )
                cell_var.local_name = name  # type: ignore[attr-defined]
                self.symbolic_locals[name] = cell_var

            self.symbolic_torch_function_state = SymbolicTorchFunctionState(
                torch_function_mode_stack
            )

            self.symbolic_stream_state = SymbolicStreamState()

            if export:
                # export gets confused if we never realize unused inputs
                # in export mode just eagerly realize everything
                self.symbolic_locals = variables.LazyVariableTracker.realize_all(
                    self.symbolic_locals
                )

    def _throw_if_in_functorch(self) -> None:
        # Fallback to eager in case of a graph break inside vmap
        eager = torch._dynamo.lookup_backend("eager")
        compiler_fn = inspect.getattr_static(
            self.output.compiler_fn, "compiler_fn", self.output.compiler_fn
        )
        ci = torch._C._functorch.peek_interpreter_stack()
        forbidden_keys = (
            torch._C._functorch.TransformType.Vmap,
            torch._C._functorch.TransformType.Grad,
            torch._C._functorch.TransformType.Jvp,
        )

        if ci is not None and ci.key() in forbidden_keys and compiler_fn is not eager:
            name = ci.key().name.lower()
            msg = (
                "If you are reaching here, it means dynamo failed for one of the following reasons:\n"
                # Calling a torch.compiled function
                f"- Calling torch.func.{name}(compiled_fn) function from eager mode is not supported. "
                f"Ensure that torch.func.{name} is also wrapped within a torch.compile function. "
                "For more information, see PyTorch issue #128711.\n"
                # if it reaches here, it means Dynamo failed to inline a functorch function
                f"- torch.func.{name}(fn) requires the function to be inlined by dynamo"
            )
            unimplemented(
                gb_type="Unsupported functorch tracing attempt",
                context="",
                explanation=msg,
                hints=[],
            )

    def get_example_value(self, source: Source) -> Any:
        if isinstance(source, LocalSource):
            return self.f_locals[source.local_name]
        if isinstance(source, GlobalSource):
            return self.f_globals[source.global_name]
        raise KeyError

    def symbolic_locals_contain_module_class(self) -> bool:
        for v in self.symbolic_locals.values():
            if isinstance(v, UserDefinedClassVariable) and issubclass(
                v.as_python_constant(), torch.nn.Module
            ):
                return True
        return False

    def replace_tos_if_return_is_generator(self) -> None:
        if (
            len(self.stack)
            and (tos := self.stack[-1])
            and isinstance(tos, LocalGeneratorObjectVariable)
        ):
            self.stack[-1] = ListIteratorVariable(
                tos.force_unpack_var_sequence(self),
                mutation_type=ValueMutationNew(),
            )

    def _return(self, inst: Instruction) -> None:
        self.replace_tos_if_return_is_generator()
        assert self.instruction_pointer is not None
        assert self.start_point is not None
        get_metrics_context().increment(
            "ir_count", self.instruction_pointer - self.start_point
        )

        if (
            not config.allow_empty_graphs
            and self.output.count_calls() == 0
            and not self.inconsistent_side_effects
            and not self.symbolic_locals_contain_module_class()
            and not self.export
            and not self.one_graph
            and not self.error_on_graph_break
            and not self.is_tracing_resume_prologue
        ):
            # TODO graph break if one_graph is set - this might break things
            raise exc.SkipFrame(
                "No ops traced for the FX graph. `torch.compile` will skip the frame and fall back to eager.\n"
                f"Frame info: {format_frame_info(self.f_code)}"
            )

        self.instruction_pointer = None
        _step_logger()(
            logging.INFO,
            f"torchdynamo done tracing {self.f_code.co_name} ({inst.opname})",
        )
        log.debug("return triggered compile")
        all_stack_locals_metadata = self.output.compile_subgraph(
            self,
            reason=GraphCompileReason(
                "return_value", [self.frame_summary()], graph_break=False
            ),
            # the value to be returned
            stack_pops=1 if inst.opname == "RETURN_VALUE" else 0,
        )
        # check that our stack/locals meta are correct:
        # we should only be tracing 1 frame, and there should not be any NULLs on the stack
        assert len(all_stack_locals_metadata) == 1
        assert not all_stack_locals_metadata[0].stack_null_idxes
        self.output.add_output_instructions(
            self.codegen_return_with_pops(inst, all_stack_locals_metadata[0].num_stack)
        )
        raise ReturnValueOp

    def RETURN_VALUE(self, inst: Instruction) -> None:
        self._return(inst)

    def RETURN_CONST(self, inst: Instruction) -> None:
        self._return(inst)


if sys.version_info >= (3, 11):
    _binary_op_lookup = [
        getattr(
            InstructionTranslator,
            opname[3:] if "INPLACE" in opname else f"BINARY_{opname[3:]}",
        )
        for opname, _ in dis._nb_ops  # type: ignore[attr-defined]
    ]


class InliningInstructionTranslator(InstructionTranslatorBase):
    """Trace and inline a called method"""

    symbolic_result: Optional[VariableTracker]
    # pyrefly: ignore [bad-override]
    parent: InstructionTranslatorBase

    @classmethod
    def inline_call(cls, parent: Any, func: Any, args: Any, kwargs: Any) -> Any:
        tracer = cls.build_inline_tracer(parent, func, args, kwargs)
        return tracer.inline_call_()

    @staticmethod
    def check_inlineable(func: Any) -> trace_rules.SkipResult:
        if func.has_self():
            unimplemented(
                gb_type="Inline attempt with __self__",
                context=str(func),
                explanation="Attempted to inline a function with the `__self__` attribute. "
                "Dynamo is expected to decompose method calls into function calls with a `self` argument.",
                hints=[],
            )

        if isinstance(func, UserFunctionVariable) and inspect.getattr_static(
            func.get_function(), "_torchdynamo_disable", False
        ):
            msg = inspect.getattr_static(
                func.get_function(), "_torchdynamo_disable_msg", None
            )
            unimplemented(
                gb_type="Skip inlining `torch.compiler.disable()`d function",
                context=str(func.get_function()),
                explanation=f"Skip inlining function {func.get_function()} since it was wrapped "
                f"with `torch.compiler.disable` (reason: {msg})",
                hints=[
                    "Remove the `torch.compiler.disable` call",
                ],
            )

        result = trace_rules.check_verbose(func, is_inlined_call=True)
        if result.skipped:
            from torch._dynamo.variables.misc import produce_trampoline_autograd_apply

            # _origin marks this as coming from an internal dynamo known function that is safe to
            # trace through.
            if (
                hasattr(getattr(func, "fn", None), "_origin")
                and func.fn._origin is produce_trampoline_autograd_apply
            ):
                # Known sound
                return trace_rules.SkipResult(
                    False, "allowlist in dynamo known function"
                )
            fn_qualname = func.fn.__qualname__ if hasattr(func, "fn") else ""
            hints = [
                f"Avoid calling the function `{fn_qualname}`.",
            ]
            if "_dynamo" not in func.get_filename():
                hints += [
                    f"Apply `@torch._dynamo.dont_skip_tracing` to the function `{fn_qualname}` "
                    "to force tracing into the function. "
                    "More graph breaks may occur as a result of attempting to trace into the function.",
                    "Please file an issue to PyTorch.",
                ]
            unimplemented(
                gb_type="Attempted to inline function marked as skipped",
                context=f"qualname: {fn_qualname}, name: {func.get_name()}, "
                f"filename: `{func.get_filename()}`, skip reason: {result.reason}",
                explanation=f"Dynamo developers have intentionally marked that the function `{fn_qualname}` "
                "should not be traced.",
                hints=hints,
            )

        return result

    @staticmethod
    def build_inline_tracer(
        parent: Any,
        func: VariableTracker,
        args: list[VariableTracker],
        kwargs: Any,
    ) -> InliningInstructionTranslator:
        assert isinstance(
            func,
            (
                UserFunctionVariable,
                NestedUserFunctionVariable,
                LocalGeneratorFunctionVariable,
                LocalGeneratorObjectVariable,
            ),
        )
        code: types.CodeType = func.get_code()
        result = None
        tracing_ctx = parent.output.tracing_context

        # Check if we have already identified this function to be inline-able.
        # The exception is dont_skip_tracing flag which affects the inline
        # behavior. If the flag is True, don't rely on previous results.
        if not config.dont_skip_tracing and tracing_ctx:
            if previous_result := tracing_ctx.previously_inlined_functions.get(
                code, None
            ):
                result = previous_result

        if result is None:
            if isinstance(func, SkipFunctionVariable):
                unimplemented(
                    gb_type="Attempted to inline function marked as skipped (SkipFunctionVariable)",
                    context=f"Attempted to inline a SkipFunctionVariable {func}",
                    explanation=(
                        "Attempted to inline a function that was previously determined to be marked as intentionally skipped."
                    ),
                    hints=[],
                )
            result = InliningInstructionTranslator.check_inlineable(func)
            assert result.skipped is False

            if not config.dont_skip_tracing and tracing_ctx:
                tracing_ctx.previously_inlined_functions[code] = result

        sub_locals = None
        try:
            sub_locals = func.bind_args(parent, args, kwargs)
        except TypeError as e:
            unimplemented(
                gb_type="failed to bind arguments when attempting to inline",
                context=f"func='{func.get_name()}' {func.get_filename()}:{func.get_code().co_firstlineno}; "
                f"args = {[arg.python_type() for arg in args]}; kwargs = {kwargs}",
                explanation=f"Argument mismatch when attempting to trace function {func.get_name()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
                from_exc=e,
            )

        assert sub_locals is not None

        for v in itertools.chain(sub_locals.values()):
            if not isinstance(v, VariableTracker):
                unimplemented(
                    gb_type="Encountered unconverted argument when attempting to inline",
                    context=f"func: {func}, arg: {v}",
                    explanation="An argument to an inlined function was not successfully converted to a VariableTracker.",
                    hints=[*graph_break_hints.DYNAMO_BUG],
                )

        if code.co_name in ("__setitem__", "__setattr__") and not (
            args and isinstance(args[0], variables.UserDefinedObjectVariable)
        ):
            unimplemented(
                gb_type="Unsupported __setitem__/__setattr__ inline attempt",
                context=f"code name: {code.co_name}, args: {args}",
                explanation=f"Attempted to inline {code.co_name} where first argument (self) is not a user-defined object.",
                hints=[],
            )

        suffix = ""
        # TODO: mlazos, add support for enabling multiple artifact logs
        # with a single alias
        if torch._logging._internal.log_state.is_artifact_enabled("bytecode"):
            suffix = f"\n{dis.Bytecode(code).dis()}"
        if sys.version_info >= (3, 11):
            cur_inst = parent.current_instruction
            parent_code = parent.f_code

            def get_trace_call_log_str() -> str:
                header = parent.get_line_of_code_header(
                    lineno=cur_inst.positions.lineno
                )
                line = get_instruction_source_311(parent_code, cur_inst).rstrip()
                return f"TRACE inlined call {code.co_name} from {header}\n{line}"

            trace_call_log.debug("%s", LazyString(get_trace_call_log_str))
        log.debug("INLINING %s%s, %s", code, suffix, result.reason)

        # Detect inline GraphModule calls in order to propagate node metadata,
        # by checking if the first argument (self) is a variable tracking a GraphModule.
        if args and isinstance(args[0], NNModuleVariable):
            module = parent.output.get_submodule(args[0].module_key)
            if isinstance(module, torch.fx.GraphModule):
                # The inline call might not actually be a call to `forward`,
                # but it is enough to add a context for `forward` in case it is called.
                code_context.get_context(module.forward.__code__)[
                    "orig_graphmodule"
                ] = weakref.ref(module)
        # When we have inline_nn_module turned on, modules resolve to UnspecializedNNModuleVariable
        if args and isinstance(args[0], UnspecializedNNModuleVariable):
            module = args[0].value
            if isinstance(module, torch.fx.GraphModule):
                # The inline call might not actually be a call to `forward`,
                # but it is enough to add a context for `forward` in case it is called.
                code_context.get_context(module.forward.__code__)[
                    "orig_graphmodule"
                ] = weakref.ref(module)

        assert not isinstance(func, SkipFunctionVariable)
        tracer: InliningInstructionTranslator
        if is_generator(code):
            tracer = InliningGeneratorInstructionTranslator(
                parent,
                code,
                sub_locals,
                parent.symbolic_globals,
                parent.symbolic_torch_function_state,
                parent.symbolic_stream_state,
                func,
            )
        else:
            tracer = InliningInstructionTranslator(
                parent,
                code,
                sub_locals,
                parent.symbolic_globals,
                parent.symbolic_torch_function_state,
                parent.symbolic_stream_state,
                func,
            )
        return tracer

    def inline_call_(self) -> VariableTracker:
        parent = self.parent
        code = self.f_code

        strict_ctx: Any = contextlib.nullcontext()
        if parent.strict_checks_fn:
            strict_ctx = self.strict_translation_mode(parent.strict_checks_fn)
        try:
            with strict_ctx:
                self.run()
        except exc.ObservedException as e:
            msg = f"Observed exception DURING INLING {code} : {e}"
            log.debug(msg)
            # bubble up the exception to the parent frame.
            raise
        except Unsupported as e:
            # If this graph break has skip_frame set, unset it
            # since it refers to the current frame and not the parent.
            e.skip_frame = False
            raise
        except Exception:
            log.debug("FAILED INLINING %s", code)
            raise
        finally:
            parent.error_on_graph_break = self.error_on_graph_break

        if self.output.should_exit:
            # graph break
            return ConstantVariable.create(None)  # return dummy variable

        assert self.symbolic_result is not None

        if self.f_globals is parent.f_globals:
            # Merge symbolic_globals back if parent and child are in the same namespace
            parent.symbolic_globals.update(self.symbolic_globals)

        parent.inconsistent_side_effects |= self.inconsistent_side_effects

        log.debug("DONE INLINING %s", code)
        self.output.tracing_context.traced_code.append(code)

        if config.enable_faithful_generator_behavior or (
            isinstance(self, InliningGeneratorInstructionTranslator)
            and self.is_generator_from_ctx_manager
        ):
            if (
                is_generator(code)
                and isinstance(self, InliningGeneratorInstructionTranslator)
                and self.generator_exhausted
            ):
                assert isinstance(self, InliningGeneratorInstructionTranslator)
                # When the generator returns None, we raise StopIteration
                args = []
                if not self.symbolic_result.is_constant_none():
                    args = [self.symbolic_result]
                exc.raise_observed_exception(StopIteration, self, args=args)
            else:
                return self.symbolic_result
        else:
            if is_generator(code):
                assert isinstance(self, InliningGeneratorInstructionTranslator)
                assert self.symbolic_result.is_constant_none()
                return ListIteratorVariable(
                    self.generated_items,
                    mutation_type=ValueMutationNew(),
                )
            else:
                return self.symbolic_result

    def __init__(
        self,
        parent: InstructionTranslatorBase,
        code: types.CodeType,
        symbolic_locals: dict[str, VariableTracker],
        symbolic_globals: dict[str, VariableTracker],
        symbolic_torch_function_state: SymbolicTorchFunctionState,
        symbolic_stream_state: SymbolicStreamState,
        funcvar: BaseUserFunctionVariable | LocalGeneratorObjectVariable,
    ) -> None:
        f_globals = funcvar.get_globals()
        f_builtins = f_globals["__builtins__"]
        if not isinstance(f_builtins, dict):
            f_builtins = f_builtins.__dict__

        # Get the cached instructions. These instructions are safe to cache
        # because we dont mutate them in transform_code_object (those
        # instructions are for the top most Instruction translator).  Also, we
        # have to be careful about not using _cached_cleaned_instructions here
        # because that function is global, while we want the cache to be
        # alive only during a compilation.
        tracing_ctx = parent.output.tracing_context
        instructions = None
        if tracing_ctx:
            if tracing_ctx.previously_cleaned_instructions.get(code):
                instructions = tracing_ctx.previously_cleaned_instructions[code]

        if instructions is None:
            instructions = cleaned_instructions(code)
            propagate_line_nums(instructions)
            if tracing_ctx:
                tracing_ctx.previously_cleaned_instructions[code] = instructions

        super().__init__(
            output=parent.output,
            f_locals={},
            f_globals=f_globals,
            f_builtins=f_builtins,
            symbolic_locals=symbolic_locals,
            symbolic_globals=symbolic_globals,
            symbolic_torch_function_state=symbolic_torch_function_state,
            symbolic_stream_state=symbolic_stream_state,
            instructions=instructions,
            code_options={k: getattr(code, k) for k in get_code_keys()},
            f_code=code,
            export=parent.export,
            inline_depth=parent.inline_depth + 1,
            speculation_log=parent.speculation_log,
            exn_vt_stack=parent.exn_vt_stack,
            distributed_state=parent.distributed_state,
            package=parent.package,
        )
        self.funcvar = funcvar
        self.parent = parent
        self.num_calls = parent.num_calls
        self.symbolic_result = None
        self.nn_module_stack = parent.nn_module_stack.copy()
        self.one_graph = parent.one_graph

    @property
    def fake_mode(self) -> Optional[FakeTensorMode]:
        return self.parent.fake_mode

    def run_ctx_mgr(self) -> Any:
        return TracingContext.current_frame(self.parent.frame_summary())

    def should_compile_partial_graph(self) -> bool:
        if config.nested_graph_breaks:
            if not self.funcvar.should_allow_nested_graph_breaks():
                return False
            if not self.parent.should_compile_partial_graph():
                return False
            return super().should_compile_partial_graph()
        return False  # inlining functions is all-or-nothing

    def create_call_resume_at(
        self,
        inst: Instruction,
        all_stack_locals_metadata: list[StackLocalsMetadata],
    ) -> list[Instruction]:
        if config.nested_graph_breaks:
            return super().create_call_resume_at(inst, all_stack_locals_metadata)
        unimplemented(
            gb_type="Graph break in inlined function",
            context="",
            explanation="Graph breaks in an inlined call are not supported.",
            hints=[],
        )

    def RETURN_VALUE(self, inst: Instruction) -> None:
        self.symbolic_result = self.pop()  # type: ignore[assignment]
        self.instruction_pointer = None
        raise ReturnValueOp

    def RETURN_CONST(self, inst: Instruction) -> None:
        self.symbolic_result = self._load_const(inst)
        self.instruction_pointer = None
        raise ReturnValueOp

    def get_globals_source_and_value(
        self, name: str
    ) -> tuple[Any, VariableTracker, Source]:
        # NamedTuple's `__new__` has a fake global scope that's not an actual
        # module. TODO generalize the check for other non-importable cases.
        # https://github.com/python/cpython/blob/8421b03b16a4852a527256cb7cdce2ab2d318548/Lib/collections/__init__.py#L441-L447
        if "__name__" in self.f_globals and not self.f_globals["__name__"].startswith(
            "namedtuple_"
        ):
            module_name = self.f_globals["__name__"]
            module_source = self.import_source(module_name)
            if "torch_package" in module_name:
                fglobals_value = (
                    torch.package.package_importer._package_imported_modules[
                        module_name
                    ]
                )  # type: ignore[assignment]
            else:
                fglobals_value = _import_module(module_name)
            # Dont use lazy vt because we will do a setattr afterwards
            # TODO: fix InstructionTranslator -> InstructionTranslatorBase
            # pyrefly: ignore[bad-argument-type]
            fglobals_vt = VariableBuilder(self, module_source)(fglobals_value)
            global_source = AttrSource(module_source, name)
        else:
            globals_name = self.output.install_global_by_id(
                "___unnamed_scope", self.f_globals
            )
            globals_source = GlobalSource(globals_name)
            fglobals_value = self.f_globals  # type: ignore[assignment]
            # Dont use lazy vt because we will do a setattr afterwards
            # pyrefly: ignore[bad-argument-type]
            fglobals_vt = VariableBuilder(self, globals_source)(fglobals_value)
            global_source = DictGetItemSource(globals_source, name)  # type: ignore[assignment]

        if is_stdlib(fglobals_value):
            # Users don't inplace mutate a stdlib attribute (like inspect,
            # collections), skip guards that originate from the stdlib modules.
            global_source = SkipGuardSource(global_source)  # type: ignore[assignment]

        return fglobals_value, fglobals_vt, global_source

    def _load_global(self, inst: Instruction) -> None:
        name = inst.argval
        if name not in self.f_globals:
            return self.load_builtin(inst)

        if self.output.global_scope is self.f_globals:
            # If the global scope matches that of the root frame, use handler in
            # root frame instruction translator, to enforce consistency.
            super()._load_global(inst)
        else:
            _, fglobals_vt, global_source = self.get_globals_source_and_value(name)
            if self.output.side_effects.has_pending_mutation_of_attr(fglobals_vt, name):
                self.push(self.output.side_effects.load_attr(fglobals_vt, name))
            else:
                value = self.f_globals[name]
                self.push(VariableTracker.build(self, value, global_source))

    def STORE_GLOBAL(self, inst: Instruction) -> None:
        if self.output.global_scope is self.f_globals:
            # If the global scope matches that of the root frame, use handler in
            # root frame instruction translator, to enforce consistency.
            super().STORE_GLOBAL(inst)
        else:
            value = self.pop()
            if isinstance(value, RemovableHandleVariable):
                unimplemented(
                    gb_type="Storing Tensor hook handle in globals (inline call)",
                    context=inst.argval,
                    explanation="This is not supported.",
                    hints=[],
                )
            name = inst.argval
            _fglobals_value, fglobals_vt, _ = self.get_globals_source_and_value(name)
            self.output.side_effects.store_attr(fglobals_vt, name, value)


class InliningGeneratorInstructionTranslator(InliningInstructionTranslator):
    generated_items: list[VariableTracker]
    # Flag whether or not the InlineGenerator should consume the entire iterator

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.generated_items = []
        self.generator_exhausted = False
        self.is_generator_from_ctx_manager = False

    def should_compile_partial_graph(self) -> bool:
        # resuming on graph break on inlined generator not supported
        return False

    def YIELD_VALUE(self, inst: Instruction) -> None:
        top = self.pop()
        self.generated_items.append(top)
        if len(self.generated_items) > MAX_ITERATOR_LIMIT:
            raise exc.InfiniteGeneratorError
        self.push(ConstantVariable.create(None))
        if (
            config.enable_faithful_generator_behavior
            or self.is_generator_from_ctx_manager
        ):
            self.symbolic_result = top
            # Stop tracing
            raise YieldValueOp

    def GET_YIELD_FROM_ITER(self, inst: Instruction) -> None:
        tos = self.stack[-1]
        if not isinstance(tos, ListIteratorVariable):
            self.pop()
            res = BuiltinVariable(iter).call_function(self, [tos], {})  # type: ignore[arg-type]
            self.push(res)

    def RETURN_VALUE(self, inst: Instruction) -> None:
        self.generator_exhausted = True
        return super().RETURN_VALUE(inst)

    def RETURN_CONST(self, inst: Instruction) -> None:
        self.generator_exhausted = True
        return super().RETURN_CONST(inst)

    def YIELD_FROM(self, inst: Instruction) -> None:
        assert len(self.stack) >= 2
        val = self.pop()
        tos = self.stack[-1]
        if not val.is_constant_none():
            # invoke send
            # Unreachable code - if you hit this, you are implementing generator support and have
            # lifted the `unimplemented("generator")` in frame conversion. This codepath handles
            # subgenerator and lines up with this line in Python 3.10
            # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2599
            unimplemented(
                gb_type="Unreachable sub-generator code",
                context="",
                explanation="Should only be encountered while implementing generator support.",
                hints=[],
            )

        try:
            val = tos.next_variable(self)
        except (StopIteration, exc.ObservedUserStopIteration) as ex:
            if isinstance(ex, exc.ObservedUserStopIteration):
                exc.handle_observed_exception(self)

            # The iterator is exhausted. Stop the loop and return.
            self.pop()
            self.push(ConstantVariable.create(ex.value))
        else:
            # Repeat the YIELD_FROM instruction in the next eval loop
            assert (
                isinstance(self.instruction_pointer, int)
                and self.instruction_pointer > 0
            )
            self.instruction_pointer -= 1

            self.push(val)
            # Add the value to yield into generated_items and replace the top of the stack with None
            self.YIELD_VALUE(inst)

    def SEND(self, inst: Instruction) -> None:
        assert len(self.stack) >= 2
        val = self.pop()
        tos = self.stack[-1]
        if isinstance(tos, (IteratorVariable, LocalGeneratorObjectVariable)) or (
            isinstance(tos, UserDefinedObjectVariable)
            and isinstance(tos.value, collections.abc.Iterator)
        ):
            if val.is_constant_none():
                try:
                    val = tos.next_variable(self)  # type: ignore[arg-type]
                except (StopIteration, exc.ObservedUserStopIteration) as ex:
                    # To implement SEND, we have to look at the implementation
                    # when the iterator returns StopIteration. This translates to this code
                    # 3.11: https://github.com/python/cpython/blob/3.11/Python/ceval.c#L2613-L2619
                    # 3.12: https://github.com/python/cpython/blob/3.12/Python/bytecodes.c#L863-L866
                    # The implementation is different in 3.11 and 3.12. In 3.12, we rely
                    # on END_SEND to clean up. In 3.11, SEND does the cleanup as well.
                    if sys.version_info < (3, 12):
                        self.pop()  # Python 3.12 uses new opcode END_SEND
                    self.push(ConstantVariable.create(ex.value))
                    self.jump(inst)
                else:
                    self.push(val)
            else:
                # invoke send
                # Unreachable code - if you hit this, you are implementing generator support and have
                # lifted the `unimplemented("generator")` in frame conversion. This codepath handles
                # subgenerator and lines up with this line in Python 3.11
                # https://github.com/python/cpython/blob/3.11/Python/ceval.c#L2597
                unimplemented(
                    gb_type="Unreachable sub-generator code",
                    context="",
                    explanation="Should only be encountered while implementing generator support.",
                    hints=[],
                )
        else:
            unimplemented(
                gb_type="SEND with bad type",
                context=f"TOS type: {typestr(tos)}",
                explanation=f"Attempted to SEND with unsupported type {typestr(tos)}.",
                hints=[],
            )
