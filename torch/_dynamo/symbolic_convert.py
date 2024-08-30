# mypy: allow-untyped-defs
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
import typing
import weakref
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)
from unittest.mock import patch

import torch
import torch._logging
from torch._guards import tracing, TracingContext

from . import config, exc, logging as torchdynamo_logging, trace_rules, variables
from .bytecode_analysis import (
    get_indexof,
    JUMP_OPNAMES,
    livevars_analysis,
    propagate_line_nums,
)
from .bytecode_transformation import (
    cleaned_instructions,
    create_call_function,
    create_instruction,
    create_jump_absolute,
    create_swap,
    get_code_keys,
    Instruction,
    is_generator,
    unique_id,
)
from .code_context import code_context
from .codegen import PyCodegen
from .exc import ArgsMismatchError, BackendCompilerFailed, unimplemented, Unsupported
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import ContinueExecutionCache, ReenterWith
from .source import (
    AttrSource,
    GetItemSource,
    GlobalSource,
    GlobalWeakRefSource,
    LocalSource,
    Source,
    TorchFunctionModeStackSource,
)
from .trace_rules import is_builtin_constant, is_forbidden
from .utils import (
    counters,
    get_fake_value,
    get_instruction_source_311,
    get_torch_function_mode_stack,
    graph_break_dup_warning_checker,
    istype,
    LazyString,
    proxy_args_kwargs,
)
from .variables.base import is_side_effect_safe, MutableLocal, typestr, VariableTracker
from .variables.builder import VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable
from .variables.ctx_manager import (
    ContextWrappingVariable,
    GenericContextWrappingVariable,
    WithExitFunctionVariable,
)
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import (
    BaseUserFunctionVariable,
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .variables.iter import MAX_ITERATOR_LIMIT
from .variables.lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    SliceVariable,
    TupleVariable,
)
from .variables.misc import (
    ClosureVariable,
    GetAttrVariable,
    InlinedClosureVariable,
    NullVariable,
    PythonModuleVariable,
    UnknownVariable,
)
from .variables.nn_module import NNModuleVariable, UnspecializedNNModuleVariable
from .variables.tensor import supported_comparison_ops, SymNodeVariable, TensorVariable


if TYPE_CHECKING:
    from .variables.torch_function import TorchFunctionModeVariable

from .variables.user_defined import (
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedObjectVariable,
)


log = logging.getLogger(__name__)
graph_break_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")
trace_call_log = torch._logging.getArtifactLogger(__name__, "trace_call")
trace_source_log = torch._logging.getArtifactLogger(__name__, "trace_source")
trace_bytecode_log = torch._logging.getArtifactLogger(__name__, "trace_bytecode")
tls = threading.local()
compare_op_handlers: Dict[str, Any] = {
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


PT2_ISSUE_TRACKER_URL = "https://github.com/pytorch/pytorch/issues/new?&labels=oncall%3A+pt2&projects=&template=pt2-bug-report.yml"


@dataclasses.dataclass
class SpeculationEntry:
    filename: str
    lineno: int
    instruction_pointer: int
    inst: Instruction  # for debugging only
    failed: bool = False
    reason: Optional[GraphCompileReason] = None

    def fail_and_restart_analysis(self):
        """
        Start tracing of the current frame over again, and don't take this branch.
        """
        self.failed = True
        if self.reason is not None:
            restart_reason = self.reason.reason
        else:
            restart_reason = "Unknown fail_and_restart_analysis"
        raise exc.SpeculationRestartAnalysis(restart_reason=restart_reason)


@dataclasses.dataclass
class SpeculationLog:
    """
    SpeculationLog replaces the prior copy_graphstate/restore_graphstate
    checkpointing.  Rather than saving/restoring state, we restart the
    dynamo conversion process over from the beginning -- but when we
    hit the start of the speculation that failed, we instead generate
    a graph break.
    """

    entries: List[SpeculationEntry] = dataclasses.field(default_factory=list)
    index: int = 0

    def restart(self):
        self.index = 0

    def clear(self):
        self.entries.clear()
        self.index = 0

    def next(
        self, filename: str, lineno: int, instruction_pointer, inst
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
        assert (
            entry.instruction_pointer == instruction_pointer
            and entry.filename == filename
            and entry.lineno == lineno
        ), f"""
SpeculationLog diverged at index {self.index} (log had {len(self.entries)} entries):
- Expected: {entry.filename}:{entry.lineno} ({entry.inst.opname} at ip={entry.instruction_pointer})
- Actual: {filename}:{lineno} ({inst.opname} at ip={instruction_pointer})
{prev_entry_msg}
There are two usual reasons why this may have occured:
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
        self.index += 1
        return entry


@dataclasses.dataclass
class LocalState:
    input_sizes: Dict[str, List[int]] = dataclasses.field(default_factory=dict)
    input_strides: Dict[str, List[int]] = dataclasses.field(default_factory=dict)


# Mutable box that is shared across restarts
@dataclasses.dataclass
class DistributedState:
    compile_pg: Any
    local_state: LocalState
    all_states: Optional[List[LocalState]] = None


@functools.lru_cache(None)
def _step_logger():
    return torchdynamo_logging.get_step_logger(log)


@dataclasses.dataclass
class BlockStackEntry:
    # Current instruction that pushes something to block_stack
    inst: Instruction
    target: Instruction
    stack_index: Optional[int] = None
    with_context: Optional[
        Union[ContextWrappingVariable, GenericContextWrappingVariable]
    ] = None

    def can_restore(self):
        return self.with_context is not None

    def resume_fn(self):
        assert self.stack_index is not None
        if (
            self.with_context
            and hasattr(self.with_context, "target_values")
            and self.with_context.target_values
        ):
            return ReenterWith(self.stack_index, tuple(self.with_context.target_values))
        else:
            return ReenterWith(self.stack_index)

    def exit(self, tx):
        assert self.with_context is not None
        return self.with_context.exit(tx)


class ReturnValueOp(Exception):
    pass


def stack_op(fn: typing.Callable[..., object]):
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self: "InstructionTranslator", inst: Instruction):
        self.push(fn_var.call_function(self, self.popn(nargs), {}))

    return impl


def _detect_and_normalize_assert_statement(
    self: "InstructionTranslatorBase",
    truth_fn: typing.Callable[[object], bool],
    push: bool,
):
    # Detect if this jump instruction is assert and normalize the assert
    # by pushing dummy error message when nothing is given.
    #
    # Python 3.9 assertion is in following format:
    # 18 POP_JUMP_IF_TRUE       28
    # 20 LOAD_ASSERTION_ERROR
    # 22 LOAD_CONST               3 ('Assert message') -> optional instruction
    # 24 CALL_FUNCTION            1                    -> optional instruction
    # 26 RAISE_VARARGS
    #
    # Python 3.8 assertion is in following format:
    # 18 POP_JUMP_IF_TRUE       28
    # 20 LOAD_GLOBAL              0 (Assertion type)
    # 22 LOAD_CONST               3 ('Assert message') -> optional instruction
    # 24 CALL_FUNCTION            1                    -> optional instruction
    # 26 RAISE_VARARGS            1

    if (truth_fn is not operator.truth) or push:
        return False

    assert isinstance(self.instruction_pointer, int)
    current_instruction_pointer = self.instruction_pointer
    inst = self.instructions[current_instruction_pointer]
    # Detect LOAD_ASSERTION_ERROR or LOAD_GLOBAL 0
    if sys.version_info < (3, 9):
        if inst.opname != "LOAD_GLOBAL" or inst.argval != "AssertionError":
            return False
    else:
        if inst.opname != "LOAD_ASSERTION_ERROR":
            return False

    current_instruction_pointer += 1

    # Use dummy error message if its hard to extract
    error_msg = "assertion error"

    inst = self.instructions[current_instruction_pointer]
    # DETECT RAISE_VARARGS or LOAD CONST
    if inst.opname == "LOAD_CONST":
        if not isinstance(inst.argval, str):
            return False
        error_msg = inst.argval

        # if it is LOAD_CONSTANT, it must be followed by CALL_FUNCTION
        # (PRECALL for Python 3.11, CALL for Python 3.12+)
        current_instruction_pointer += 1
        inst = self.instructions[current_instruction_pointer]
        if inst.opname not in ("CALL_FUNCTION", "PRECALL", "CALL"):
            return False

        # for Python 3.11, PRECALL should be followed by CALL, then RAISE_VARARGS
        # for Python != 3.11, CALL_FUNCTION/CALL should be followed by RAISE_VARARGS
        current_instruction_pointer += 1
        if inst.opname == "PRECALL":
            current_instruction_pointer += 1
        inst = self.instructions[current_instruction_pointer]

    if inst.opname != "RAISE_VARARGS":
        return False

    self.push(ConstantVariable.create(error_msg))

    return True


def generic_jump(truth_fn: typing.Callable[[object], bool], push: bool):
    def jump_graph_break(self, inst, value, extra_msg=""):
        if not self.should_compile_partial_graph():
            unimplemented("should_compile_partial_graph=False")
        # compile a partial subgraph prefix then jump into user code
        if self.maybe_has_backedge():
            msg = (
                "Skipping frame because there is a graph break in a for/while loop\n"
                f"{self.frame_summary()}"
            )
            log.info(msg)
            raise exc.SkipFrame(msg)

        self.push(value)
        log.debug("generic_jump triggered compile")
        self.output.compile_subgraph(
            self,
            reason=GraphCompileReason(
                f"generic_jump {typestr(value)}{extra_msg}", [self.frame_summary()]
            ),
        )
        self.pop()

        if_next = self.create_call_resume_at(self.next_instruction)
        if push:
            self.push(value)
        if_jump = self.create_call_resume_at(inst.target)

        if sys.version_info >= (3, 13):
            # 3.13 requires stack[-1] to be bool type
            self.output.add_output_instructions([create_instruction("TO_BOOL")])

        self.output.add_output_instructions(
            [create_instruction(inst.opname, target=if_jump[0])] + if_next + if_jump
        )

    def inner(self: "InstructionTranslatorBase", inst: Instruction):
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
                else:
                    jump_graph_break(self, inst, value)

            # TODO maybe should respect DtoH sync intention of users later??
            # Manually insert torch._assert_async instead of python assert and jump over
            # assert related instructions as we don't need them anymore.

            # if we see Tensor as assert statement, no need to call scalar_tensor
            if isinstance(value, TensorVariable):
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
                        "Assertion failed on symbolic shapes. Did you make sure eager mode succeeds?"
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
            if truth_fn(value.as_python_constant()):
                if push:
                    self.push(value)
                self.jump(inst)
        elif (
            isinstance(value, (TensorVariable)) and self.should_compile_partial_graph()
        ):
            jump_graph_break(self, inst, value)
        elif isinstance(value, NNModuleVariable):
            # Equivalent of "self.nn_module is not None"
            mod = self.output.get_submodule(value.module_key)
            if truth_fn(mod):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, UnspecializedNNModuleVariable):
            mod = value.value
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
                result = x.call_function(self, [], {})  # type: ignore[arg-type]
                if isinstance(result, ConstantVariable) and isinstance(
                    result.value, (bool, int)
                ):
                    if truth_fn(result.value):
                        if push:
                            self.push(value)
                        self.jump(inst)
                else:
                    unimplemented(
                        "generic_jump on UserDefined with __bool__ returning non-constant"
                    )
            # __bool__ or __len__ is non-function or not existed in the user defined object
            else:
                if truth_fn(True):
                    if push:
                        self.push(value)
                    self.jump(inst)
        elif not isinstance(value, TensorVariable) and value.has_unpack_var_sequence(
            self
        ):
            if truth_fn(len(value.unpack_var_sequence(self))):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, SymNodeVariable):
            try:
                eval_result = value.evaluate_expr(self.output)
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
                # TODO link the torch.cond doc later
                raise exc.UserError(
                    exc.UserErrorType.DYNAMIC_CONTROL_FLOW,
                    "Dynamic control flow is not supported at the moment. Please use "
                    "functorch.experimental.control_flow.cond to explicitly capture the control flow.",
                    case_name="cond_operands",
                )

    return inner


explain = False


def break_graph_if_unsupported(*, push):
    def decorator(inner_fn):
        @functools.wraps(inner_fn)
        def wrapper(self: "InstructionTranslatorBase", inst: Instruction):
            speculation = self.speculate()
            if speculation.failed:
                assert speculation.reason is not None
                return handle_graph_break(self, inst, speculation.reason)
            try:
                return inner_fn(self, inst)
            except Unsupported as excp:
                if self.generic_context_manager_depth > 0:
                    # We don't support graph break under GenericContextWrappingVariable,
                    # If there is, we roll back to the checkpoint and fall back.
                    excp.remove_from_stats()
                    unimplemented("Graph break under GenericContextWrappingVariable")

                if isinstance(excp, exc.UncapturedHigherOrderOpError):
                    raise

                if not self.should_compile_partial_graph():
                    raise

                user_stack = excp.real_stack
                # TODO: Also report the traceback from the parent frame
                try:
                    frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                except IndexError:
                    # first instruction
                    code_options = self.code_options
                    frame_loc = (
                        code_options["co_filename"],
                        code_options["co_firstlineno"],
                    )
                # torch._dynamo.explain() formats this a little nicer, and presents a slightly
                # more actionable user code pointer
                if (
                    graph_break_log.isEnabledFor(logging.DEBUG)
                    and not explain
                    and graph_break_dup_warning_checker.add(frame_loc)
                ):
                    user_stack_formatted = "".join(traceback.format_list(user_stack))
                    # This log line is exercised from
                    #   python test/dynamo/test_exc.py -k test_graph_break_log
                    graph_break_log.debug(
                        "Graph break: from user code at:\n%s",
                        user_stack_formatted,
                        exc_info=True,
                    )
                else:
                    # This log line MUST NOT contain the string "Graph break",
                    # exercised by
                    #   python test/dynamo/test_misc.py -k test_duplicate_graph_break_log
                    log.debug(
                        "Unsupported break in user code at %s:%s (details suppressed)",
                        *frame_loc,
                    )

                if self.maybe_has_backedge():
                    msg = (
                        "Skipping frame because there is a graph break in a for/while loop\n"
                        f"{self.frame_summary()}"
                    )
                    log.info(msg)
                    raise exc.SkipFrame(msg) from excp

                excp.remove_from_stats()
                excp.add_to_stats("graph_break")
                speculation.reason = GraphCompileReason(excp.msg, user_stack)
            speculation.fail_and_restart_analysis()

        def handle_graph_break(
            self: "InstructionTranslatorBase",
            inst: Instruction,
            reason: GraphCompileReason,
        ):
            self.output.compile_subgraph(self, reason=reason)
            cg = PyCodegen(self)
            cleanup: List[Instruction] = []
            # Reconstruct the context variable CLASS in the block stack
            for b in self.block_stack:
                assert b.with_context is not None
                assert isinstance(b.with_context, ContextWrappingVariable)
                b.with_context.reconstruct_type(cg)
                cg.extend_output(b.resume_fn().try_except(cg.code_options, cleanup))
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
                self.output.add_output_instructions(
                    create_call_function(inst.arg, False)
                )
            else:
                # copy instruction, but without exception table data
                assert inst.target is None
                inst_copy = copy.copy(inst)
                inst_copy.exn_tab_entry = None
                self.output.add_output_instructions([inst_copy])

            self.output.add_output_instructions(cleanup)

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
            self.popn(push - stack_effect)

            for _ in range(push):
                self.push(UnknownVariable())
            self.output.add_output_instructions(
                self.create_call_resume_at(self.next_instruction)
            )

        return wrapper

    return decorator


class BytecodeDistpatchTableMeta(type):
    """Installs a `cls.dispatch_table` on every subclass to speed up calls to self.OPCODE()"""

    def __init__(cls, name, bases, dct) -> None:
        super().__init__(name, bases, dct)

        def _missing(opname, *args):
            unimplemented(f"missing: {opname}")

        dispatch_table = {
            op: getattr(cls, opname, functools.partial(_missing, opname))
            for opname, op in dis.opmap.items()
        }
        cls.dispatch_table = [dispatch_table.get(i) for i in range(2**8)]


class InstructionTranslatorBase(
    metaclass=BytecodeDistpatchTableMeta,
):
    output: OutputGraph
    symbolic_locals: Dict[str, VariableTracker]
    symbolic_globals: Dict[str, VariableTracker]
    symbolic_torch_function_mode_stack: Deque["TorchFunctionModeVariable"]
    stack: List[VariableTracker]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    block_stack: List[BlockStackEntry]
    lineno: int
    kw_names: Optional[ConstantVariable]
    accept_prefix_inst: bool
    prefix_insts: List[Instruction]
    inline_depth: int
    inconsistent_side_effects: bool
    current_speculation: Optional[SpeculationEntry]
    dispatch_table: List[Any]
    exn_vt_stack: List[VariableTracker]
    exec_recorder: Optional[ExecutionRecorder]
    strict_checks_fn: Optional[Callable[[VariableTracker], bool]]

    def mark_inconsistent_side_effects(self):
        """
        InstructionTranslator has encountered instructions which may cause
        dynamo to see a different version of history from eager
        See: https://github.com/pytorch/pytorch/issues/110765
        """
        self.inconsistent_side_effects = True

    def maybe_has_backedge(self):
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
        # perform a SkipFrame instead of potentially applying optimizations. For
        # false negatives (where an edge that should be marked as a backedge
        # isn't), multiple graphs may be generated if there's a break in the
        # graph during a for loop. In general, its better to have fewer false
        # negatives so that Dynamo does not skip the whole frame.

        cur_offset = self.current_instruction.offset
        assert self.instruction_pointer is not None
        for inst in self.instructions[self.instruction_pointer :]:
            if inst.opname in ("RETURN_VALUE", "RETURN_CONST"):
                return False
            if inst.opname in JUMP_OPNAMES:
                jump_offset = inst.argval
                if jump_offset < cur_offset:
                    return True
        return False

    def cell_and_freevars(self):
        if not hasattr(self, "_cell_and_freevars"):
            self._cell_and_freevars = tuple(
                self.code_options["co_cellvars"] or []
            ) + tuple(self.code_options["co_freevars"] or [])

            # An inlined function might depend on the freevar of the parent
            # function. So, recursively obtain parent cell and freevars.
            if isinstance(self, InliningInstructionTranslator):
                self._cell_and_freevars += self.parent.cell_and_freevars()
        return self._cell_and_freevars

    def prune_dead_locals(self):
        reads = livevars_analysis(self.instructions, self.current_instruction)
        # implicit use by super()
        # reads = reads | {"__class__"}
        # output variables?
        reads = reads | set(self.cell_and_freevars())
        self.symbolic_locals = {
            k: v for k, v in self.symbolic_locals.items() if k in reads
        }
        self.output.side_effects.prune_dead_object_new(self)

    def call_function(
        self,
        fn: VariableTracker,
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ):
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

    def inline_user_function_return(self, fn, args, kwargs):
        """
        A call to some user defined function by inlining it.
        """
        return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

    def get_line_of_code_header(self, lineno=None):
        if lineno is None:
            lineno = self.lineno
        inline_depth_str = (
            f" (inline depth: {self.inline_depth})" if self.inline_depth > 0 else ""
        )
        funcname = get_funcname(self.f_code.co_filename, lineno)
        funcname_str = "" if funcname is None else f" ({funcname})"
        return f"{self.f_code.co_filename}:{lineno} in {self.f_code.co_name}{funcname_str}{inline_depth_str}"

    def get_log_starts_line_log_str(self):
        log_str = f"TRACE starts_line {self.get_line_of_code_header()}\n"
        line = linecache.getline(self.f_code.co_filename, self.lineno).rstrip()
        log_str += f"    {line}"
        return log_str

    def starts_line(self, lineno):
        if self.lineno == lineno:
            return
        self.lineno = lineno
        TracingContext.set_current_loc(
            self.f_code.co_filename, lineno, self.f_code.co_name
        )
        from torch._logging.structured import dump_file

        dump_file(self.f_code.co_filename)
        if trace_source_log.isEnabledFor(logging.DEBUG):
            trace_source_log.debug("%s", LazyString(self.get_log_starts_line_log_str))

    def step(self):
        """Process exactly one instruction, return False we should exit"""
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
            if self.current_speculation.failed:
                return self.step_graph_break(inst)

        if trace_bytecode_log.isEnabledFor(logging.DEBUG):
            trace_bytecode_log.debug(
                "TRACE %s %s %s", inst.opname, inst.argval, self.stack
            )

        self.update_block_stack(inst)

        try:
            self.dispatch_table[inst.opcode](self, inst)
            return not self.output.should_exit
        except exc.ObservedException as e:
            self.exception_handler(e)
            return True
        except ReturnValueOp:
            return False
        except Unsupported:
            if self.current_speculation is None:
                log.debug("empty checkpoint")
                raise
            log.debug("step triggered compile", exc_info=True)

        self.current_speculation.fail_and_restart_analysis()

    if sys.version_info >= (3, 11):

        def update_block_stack(self, inst):
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
                if self.block_stack and inst.opname not in ("NOP", "JUMP_BACKWARD"):
                    # If we really escape from a block and the current
                    # instruction is not in another block, then there
                    # should be no other nested blocks that we are in.
                    assert len(self.block_stack) == 1
                    self.block_stack.pop()

    else:

        def update_block_stack(self, inst):
            pass

    @property
    def next_instruction(self):
        return self.instructions[self.instruction_pointer]  # type: ignore[index]

    def step_graph_break(self, continue_inst):
        # generate code from checkpoint
        assert not self.output.output_instructions
        assert self.current_speculation is not None
        self.output.compile_subgraph(
            self,
            partial_convert=True,
            reason=GraphCompileReason("step_unsupported", [self.frame_summary()]),
        )
        self.output.add_output_instructions(
            [create_jump_absolute(continue_inst)] + self.instructions
        )

    def run_ctx_mgr(self):
        # NB: Don't push the top level frame summary; set_current_loc will
        # take care of it.  However, DO make sure we attach real_stack to
        # exceptions
        return TracingContext.current_frame(None)

    def run(self):
        with self.run_ctx_mgr():
            try:
                self.output.push_tx(self)
                while self.step():
                    pass
            except BackendCompilerFailed:
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

    def push(self, val: Optional[VariableTracker], name: Any = None):
        assert val is None or isinstance(
            val, VariableTracker
        ), f"push expects VariableTracker, got {typestr(val)}"
        self.stack.append(val)  # type: ignore[arg-type]
        if sys.version_info >= (3, 13):
            self.name_stack.append(name)
            assert len(self.stack) == len(self.name_stack)

    def push_many(self, vals: List[VariableTracker]):
        for val in vals:
            self.push(val)

    def pop(self) -> VariableTracker:
        if sys.version_info >= (3, 13):
            assert len(self.stack) == len(self.name_stack)
            self.name_stack.pop()
        return self.stack.pop()

    def popn(self, n: int) -> List[VariableTracker]:
        return [*reversed([self.pop() for _ in range(n)])]

    def _load_closure(self, name):
        return ClosureVariable(name=name)

    def _load_fast(self, name):
        if self.exec_recorder and name in self.f_locals:
            self.exec_recorder.add_local_var(name, self.f_locals[name])

        try:
            self.push(self.symbolic_locals[name].unwrap(), name=name)
        except KeyError:
            if sys.version_info >= (3, 13) and name in self.cell_and_freevars():
                # 3.13 merged LOAD_CLOSURE into LOAD_FAST
                # If we fail to LOAD_FAST, then we probably should have done LOAD_CLOSURE.
                # Closure variable creation is actually done in SET_FUNCTION_ATTRIBUTE,
                # but we'll do it again here so that we don't need to push a dummy variable.
                # We shouldn't actually be doing anything with this variable anyway.
                self.push(self._load_closure(name), name=name)
            elif name.startswith("."):
                try:
                    # This happens in dict/list comprehensions
                    new_name = name.replace(".", "implicit")
                    self.push(self.symbolic_locals[new_name], name=new_name)
                except KeyError:
                    unimplemented("undefined LOAD_FAST (implicit)")
            else:
                unimplemented("undefined LOAD_FAST")

        # for continuation functions
        if name.startswith("___stack"):
            self.symbolic_locals.pop(name)

    def LOAD_FAST(self, inst):
        self._load_fast(inst.argval)

    def LOAD_DEREF(self, inst):
        assert inst.argval in self.cell_and_freevars()

        if self.exec_recorder and inst.argval in self.f_locals:
            self.exec_recorder.add_local_var(inst.argval, self.f_locals[inst.argval])

        if inst.argval not in self.symbolic_locals:
            unimplemented(f"undefined LOAD_DEREF {inst.argval}")
        self.push(self.symbolic_locals[inst.argval])

    def _store_fast(self, name):
        loaded_vt = self.pop()
        loaded_vt.set_name_hint(name)
        self.symbolic_locals[name] = loaded_vt

    def STORE_FAST(self, inst):
        self._store_fast(inst.argval)

    def DELETE_FAST(self, inst):
        del self.symbolic_locals[inst.argval]

    STORE_DEREF = STORE_FAST

    def LOAD_CLOSURE(self, inst):
        self.push(self._load_closure(inst.argval))

    def _load_const(self, inst):
        i = inst.arg
        if i is None:
            return ConstantVariable.create(value=inst.argval)
        val = self._constants_cache[i]
        if not val:
            self._constants_cache[i] = val = ConstantVariable.create(value=inst.argval)
        return val

    def LOAD_CONST(self, inst):
        self.push(self._load_const(inst))

    def _load_global(self, inst):
        name = inst.argval

        if self.exec_recorder:
            if name in self.f_globals:
                self.exec_recorder.add_global_var(name, self.f_globals[name])
            else:
                assert name in self.f_builtins
                self.exec_recorder.builtins[name] = self.f_builtins[name]

        if name in self.symbolic_globals:
            variable = self.output.side_effects[self.symbolic_globals[name]]
            self.push(self.output.side_effects.load_global(variable, name))
            return

        try:
            value = self.f_globals[name]
        except KeyError:
            return self.load_builtin(inst)

        source = GlobalSource(name)
        self.push(VariableBuilder(self, source)(value))

    @functools.cached_property
    def nn_modules_globals_vt(self):
        module_name = "torch.nn.modules.module"
        module_source = self.import_source(module_name)
        fglobals_value = importlib.import_module(module_name)  # type: ignore[assignment]
        return VariableBuilder(self, module_source)(fglobals_value)

    def LOAD_GLOBAL(self, inst):
        if sys.version_info >= (3, 11) and sys.version_info < (3, 13) and inst.arg % 2:
            self.PUSH_NULL(inst)
        self._load_global(inst)
        if sys.version_info >= (3, 13) and inst.arg % 2:
            self.PUSH_NULL(inst)

    def STORE_GLOBAL(self, inst):
        value = self.pop()
        name = inst.argval
        source = GlobalSource(name)
        if name not in self.symbolic_globals:
            self.symbolic_globals[name] = object()  # type: ignore[assignment]  # sentinel object
        variable = self.output.side_effects.track_global_existing(
            source, self.symbolic_globals[name]
        )
        if isinstance(value, RemovableHandleVariable):
            unimplemented("Storing handles in globals - NYI")
        self.output.side_effects.store_global(variable, name, value)

    def import_source(self, module_name):
        """Create an alias to a module for use in guards"""
        if "torch_package" in module_name:
            value = torch.package.package_importer._package_imported_modules[
                module_name
            ]
            alias = (
                module_name.replace(">", "_").replace("<", "_").replace(".", "_dot_")
            )
        else:
            value = importlib.import_module(module_name)
            alias = f"__import_{module_name.replace('.', '_dot_')}"
        f_globals = self.output.global_scope
        assert alias not in f_globals or f_globals[alias] is value
        f_globals[alias] = value
        self.output.update_co_names(alias)
        return GlobalSource(alias)

    def resolve_name(self, name, package, level):
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

    def calc_package(self):
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

    def IMPORT_NAME(self, inst):
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
                unimplemented("import a module that does not exist")

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
            self.exec_recorder.add_local_mod(recorded_name, value)

        if istype(value, (types.ModuleType, DummyModule)):
            self.push(PythonModuleVariable(value, source=source))
        else:
            unimplemented(f"IMPORT_NAME {typestr(value)}")

    def IMPORT_FROM(self, inst):
        self.DUP_TOP(inst)
        self._load_attr(inst)

    def load_builtin_from_argval(self, argval):
        if argval not in self.f_builtins:
            raise NameError(f"name '{argval}' is not defined")
        val = self.f_builtins[argval]

        if callable(val):
            builtins_source = GlobalSource(
                self.output.name_of_builtins_dict_key_in_fglobals
            )
            var_source = GetItemSource(builtins_source, argval)
            self.push(VariableBuilder(self, var_source)(val))
        else:
            assert is_builtin_constant(val)
            self.push(ConstantVariable.create(value=val))

    def load_builtin(self, inst):
        self.load_builtin_from_argval(inst.argval)

    def jump(self, inst):
        self.instruction_pointer = self.indexof[inst.target]

    JUMP_FORWARD = jump
    JUMP_ABSOLUTE = jump

    POP_JUMP_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_IF_TRUE = generic_jump(operator.truth, False)
    JUMP_IF_FALSE_OR_POP = generic_jump(operator.not_, True)
    JUMP_IF_TRUE_OR_POP = generic_jump(operator.truth, True)

    def SETUP_LOOP(self, inst):
        # only exists in python<=3.7
        self.block_stack.append(BlockStackEntry(inst, inst.target))

    def SETUP_EXCEPT(self, inst):
        # only exists in python<=3.7
        self.block_stack.append(BlockStackEntry(inst, inst.target))

    def POP_BLOCK(self, inst):
        self.block_stack.pop()

    def SETUP_WITH(self, inst):
        self.setup_or_before_with(inst)

    def SETUP_FINALLY(self, inst):
        self.block_stack.append(BlockStackEntry(inst, inst.target))

    def BEGIN_FINALLY(self, inst):
        self.push(None)

    def WITH_CLEANUP_START(self, inst):
        exit, exc = self.popn(2)
        assert exc is None
        self.push(exc)
        self.push(exit.call_function(self, [ConstantVariable.create(None)] * 3, {}))

    def WITH_CLEANUP_FINISH(self, inst):
        self.popn(2)
        self.push(None)

    def CALL_FINALLY(self, inst):
        """
        pushes the address of the next instruction onto the stack and increments
        bytecode counter by delta
        """
        # Python 3.8 only
        addr = self.indexof[self.next_instruction]
        self.push(ConstantVariable.create(addr))
        self.jump(inst)

    def END_FINALLY(self, inst):
        # Python 3.8 only
        # https://docs.python.org/3.8/library/dis.html#opcode-END_FINALLY
        tos = self.pop()
        if isinstance(tos, ConstantVariable):
            self.instruction_pointer = tos.as_python_constant()
        else:
            pass

    def POP_FINALLY(self, inst):
        # Python 3.8 only
        preserve_tos = inst.argval
        if preserve_tos:
            tos = self.pop()
        _ = self.pop()
        if preserve_tos:
            self.push(tos)  # type: ignore[possibly-undefined]

    def FOR_ITER(self, inst):
        it = self.pop().realize()
        try:
            val = it.next_variable(self)
            self.push(it)
            self.push(val)
        except (StopIteration, exc.ObservedUserStopIteration) as e:
            if isinstance(e, exc.ObservedUserStopIteration):
                exc.handle_observed_exception(self)

            # leave iterator upon exhaustion in 3.12
            if sys.version_info >= (3, 12):
                # CPython 3.12 actually jumps to the instruction after the END_FOR
                # and performs the action of END_FOR as part of FOR_ITER. We jump
                # to the END_FOR and run it, so we need to make sure 2 values are
                # on the stack for it to pop.
                self.push(it)
                self.push(ConstantVariable.create(None))
            self.jump(inst)

    def _raise_exception_variable(self, inst):
        val = self.pop()
        # User can raise exception in 2 ways
        #   1) raise exception type - raise NotImplementedError
        #   2) raise execption instance - raise NotImplemetedError("foo")

        # 1) when user raises exception type
        if isinstance(val, variables.BuiltinVariable):
            # Create the instance of the exception type
            # https://github.com/python/cpython/blob/3.11/Python/ceval.c#L6547-L6549
            val = val.call_function(self, [], {})  # type: ignore[arg-type]

        # Save the exception in a global data structure
        self.exn_vt_stack.append(val)

        # 2) when user raises exception instance
        if isinstance(val, variables.ExceptionVariable):
            if observed_exception_type := exc.observed_exception_map.get(val.exc_type):
                raise observed_exception_type(f"raised exception {val}")
            raise exc.ObservedException(f"raised exception {val}")
        unimplemented(f"raise {exc}")

    def RAISE_VARARGS(self, inst):
        if inst.arg == 0:
            unimplemented("re-raise")
        elif inst.arg == 1:
            self._raise_exception_variable(inst)
        else:
            # Support raise .. from None ... Dynamo does not track __cause__ and other attributes of exception. So we
            # ignore `from None` part.
            from_vt = self.pop()
            if isinstance(from_vt, ConstantVariable) and from_vt.value is None:
                self._raise_exception_variable(inst)
            unimplemented("raise ... from ...")

    def RERAISE(self, inst):
        if sys.version_info >= (3, 11):
            # RERAISE is currently supported in a narrow case of `raise ... from None`
            self._raise_exception_variable(inst)
        unimplemented("RERAISE")

    def exception_handler(self, raised_exception):
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
                assert len(self.exn_vt_stack)
                self.push(self.exn_vt_stack[-1])

                # 4) jump to the handler
                self.jump(exn_tab_entry)
            else:
                # No handler found. Bubble the exception to the parent
                # instruction translater. We use special exception for this.
                self.stack.clear()
                if type(self) is InstructionTranslator:
                    raise Unsupported("Observed exception")
                raise raised_exception
        else:
            if len(self.block_stack):
                # base implementation - https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4455

                assert len(self.exn_vt_stack)
                exception_var = self.exn_vt_stack[-1]

                block_stack_entry = self.block_stack.pop()

                while block_stack_entry.inst.opname == "EXCEPT_HANDLER":
                    # TODO(anijain2305) - This is not tested .. unable to create a testcase
                    # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1456
                    self.popn(3)
                    if len(self.block_stack) == 0:
                        # No handler found in this frame. Bubble the exception to the parent
                        # instruction translater.
                        self.stack.clear()
                        if type(self) is InstructionTranslator:
                            raise Unsupported("Observed exception")
                        raise raised_exception
                    block_stack_entry = self.block_stack.pop()

                if block_stack_entry.inst.opname != "SETUP_FINALLY":
                    unimplemented(
                        "exception is raised when top of the block stack "
                        "is not exception handler (e.g. try .. with .. except). "
                        f"Current TOS is {block_stack_entry.inst}"
                    )

                # Push a dummy block stack entry of EXCEPT_HANDLER
                # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1456
                except_handler_inst = Instruction(1e6, "EXCEPT_HANDLER", None, 0)
                self.block_stack.append(BlockStackEntry(except_handler_inst, None))

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
                # instruction translater. We use special exception for this.
                self.stack.clear()
                if type(self) is InstructionTranslator:
                    raise Unsupported("Observed exception")
                raise raised_exception

    def PUSH_EXC_INFO(self, inst):
        val = self.pop()
        assert len(self.exn_vt_stack)
        self.push(self.exn_vt_stack[-1])
        self.push(val)

    def POP_EXCEPT(self, inst):
        if sys.version_info >= (3, 11):
            val = self.pop()
            assert isinstance(val, variables.ExceptionVariable)

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

    def check_if_exc_matches(self):
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

        # Users can check exception in 2 ways
        # 1) except NotImplementedError --> BuilinVariable
        # 2) except (NotImplemetedError, AttributeError) -> TupleVariable

        if not isinstance(expected_exc_types, (BuiltinVariable, TupleVariable)):
            unimplemented(
                f"except has an unsupported types of objects {expected_exc_types}"
            )

        if sys.version_info >= (3, 11):
            if not isinstance(exc_instance, variables.ExceptionVariable):
                unimplemented(
                    f"except expects to recieve an object of exception type but received {exc_instance}"
                )

        if isinstance(expected_exc_types, TupleVariable):
            expected_types = expected_exc_types.items
        else:
            expected_types = [
                expected_exc_types,
            ]

        for expected_type in expected_types:
            if not isinstance(expected_type, BuiltinVariable):
                unimplemented(
                    f"except has an unsupported types of object {expected_type}"
                )
            if isinstance(exc_instance, variables.ExceptionVariable) and issubclass(
                exc_instance.exc_type, expected_type.fn
            ):
                return True
            elif isinstance(exc_instance, variables.BuiltinVariable) and issubclass(
                exc_instance.fn, expected_type.fn
            ):
                return True

        return False

    def CHECK_EXC_MATCH(self, inst):
        self.push(variables.ConstantVariable(self.check_if_exc_matches()))

    def JUMP_IF_NOT_EXC_MATCH(self, inst):
        if not self.check_if_exc_matches():
            self.jump(inst)

    def COMPARE_OP(self, inst):
        if inst.argval == "exception match":
            self.CHECK_EXC_MATCH(inst)
        else:
            self.push(compare_op_handlers[inst.argval](self, self.popn(2), {}))

    def GET_ITER(self, inst):
        self.call_function(BuiltinVariable(iter), [self.pop()], {})

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION(self, inst):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION_EX(self, inst):
        kwargsvars: VariableTracker
        if inst.argval == 0:
            kwargsvars = ConstDictVariable({})
            argsvars = self.pop()
        elif inst.argval == 1:
            kwargsvars = self.pop()
            argsvars = self.pop()
        else:
            unimplemented("CALL_FUNCTION_EX")

        if sys.version_info >= (3, 13):
            # 3.13 swapped null and callable
            null = self.pop()
            assert isinstance(null, NullVariable)

        fn = self.pop()

        if sys.version_info >= (3, 11) and sys.version_info < (3, 13):
            null = self.pop()
            assert isinstance(null, NullVariable)

        if isinstance(fn, GetAttrVariable) and isinstance(fn.obj, TensorVariable):
            # realize is requires for Python 3.8
            kwargsvars = kwargsvars.realize()
            if fn.name == "view" and isinstance(
                argsvars, (ConstantVariable, TensorVariable)
            ):
                # Hack to handle special case in some bert models.  Converts
                # x.view(*shape) into x.view(shape), which is correct for view()
                # but not generally.  See test_transpose_for_scores().
                argsvars = TupleVariable([argsvars])
            elif (
                fn.name == "random_"
                and isinstance(argsvars, TupleVariable)
                and len(argsvars.items) == 0
                and isinstance(kwargsvars, ConstDictVariable)
                and ConstantVariable.create("from") in kwargsvars
            ):
                # `from`` is python keyword. Adding random_ with `from` in the
                # Fx graph causes syntax error. Even if we convert the kwargs to
                # args, aot_autograd/inductor while lowering generates
                # aten.random.from, again causing syntax errors. Since this
                # usecase is uncommon, graph break.
                unimplemented("random_ op is called with from keyword")
            elif (
                fn.name == "uniform_"
                and isinstance(argsvars, TupleVariable)
                and len(argsvars.items) == 0
                and isinstance(kwargsvars, ConstDictVariable)
                and ConstantVariable.create("from") in kwargsvars
            ):
                # `from`` is python keyword. Adding uniform_ with `from` in the
                # Fx graph causes syntax error. Even if we convert the kwargs to
                # args, aot_autograd/inductor while lowering generates
                # aten.uniform.from, again causing syntax errors. Since this
                # usecase is uncommon, graph break.
                unimplemented("uniform_ op is called with from keyword")

        if not isinstance(
            argsvars, BaseListVariable
        ) and argsvars.has_unpack_var_sequence(self):
            argsvars = TupleVariable(argsvars.unpack_var_sequence(self))

        # Unpack for cases like fn(**obj) where obj is a map
        if isinstance(kwargsvars, UserDefinedObjectVariable):
            kwargsvars = BuiltinVariable.call_custom_dict(self, dict, kwargsvars)  # type: ignore[arg-type]

        if not isinstance(argsvars, BaseListVariable) or not isinstance(
            kwargsvars, ConstDictVariable
        ):
            unimplemented(f"non-static call {typestr(argsvars)} {typestr(kwargsvars)}")

        # Map to a dictionary of str -> VariableTracker
        kwargsvars = kwargsvars.keys_as_python_constant()
        self.call_function(fn, argsvars.items, kwargsvars)

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION_KW(self, inst):
        argnames = self.pop()
        args = self.popn(inst.argval)
        fn = self.pop()
        assert isinstance(argnames, TupleVariable) and argnames.is_python_constant()
        argnames = argnames.as_python_constant()
        args, kwargs_list = args[: -len(argnames)], args[-len(argnames) :]
        kwargs = dict(zip(argnames, kwargs_list))
        assert len(kwargs) == len(argnames)
        self.call_function(fn, args, kwargs)

    def LOAD_METHOD_SUPER(self, inst):
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        arg = inst.argval[0]
        argval = self.code_options["co_names"][arg]
        if sys.version_info < (3, 11):
            self._load_attr(dataclasses.replace(inst, argval=argval))
        else:
            self.LOAD_METHOD(dataclasses.replace(inst, argval=argval))

    def LOAD_ATTR_SUPER(self, inst):
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        arg = inst.argval[0]
        argval = self.code_options["co_names"][arg]
        self._load_attr(dataclasses.replace(inst, argval=argval))

    def LOAD_METHOD(self, inst):
        self._load_attr(inst)
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

    def CALL_METHOD(self, inst):
        args = self.popn(inst.argval)
        dummy = self.pop()
        assert dummy is None
        fn = self.pop()
        self.call_function(fn, args, {})

    def _load_attr(self, inst):
        obj = self.pop()
        result = BuiltinVariable(getattr).call_function(
            self, [obj, ConstantVariable.create(inst.argval)], {}  # type: ignore[arg-type]
        )
        self.push(result)

    def LOAD_ATTR(self, inst):
        if sys.version_info >= (3, 12):
            if inst.arg % 2:
                self.LOAD_METHOD(inst)
                return
        self._load_attr(inst)

    def STORE_ATTR(self, inst):
        speculation = self.speculate()
        if speculation.failed:
            return self.store_attr_graph_break(inst)
        val, obj = self.popn(2)

        if isinstance(obj, NNModuleVariable) and not isinstance(val, ConstantVariable):
            # We don't allow side effects during export on non-constant values
            # https://github.com/pytorch/torchdynamo/issues/1475
            assert (
                not self.export
            ), f"Mutating module attribute {inst.argval} during export."

        try:
            BuiltinVariable(setattr).call_function(
                self, [obj, ConstantVariable.create(inst.argval), val], {}  # type: ignore[arg-type]
            )
            return
        except Unsupported as e:
            if not self.should_compile_partial_graph():
                raise
            log.debug("STORE_ATTR triggered compile", exc_info=True)
            e.remove_from_stats()
            e.add_to_stats("graph_break")
        speculation.fail_and_restart_analysis()

    def store_attr_graph_break(self, inst):
        if not self.should_compile_partial_graph():
            unimplemented("should_compile_partial_graph=False")
        self.output.compile_subgraph(
            self, reason=GraphCompileReason("store_attr", [self.frame_summary()])
        )
        self.output.add_output_instructions([copy.copy(inst)])
        self.popn(2)
        self.output.add_output_instructions(
            self.create_call_resume_at(self.next_instruction)
        )

    def DELETE_ATTR(self, inst):
        obj = self.pop()
        BuiltinVariable(delattr).call_function(
            self, [obj, ConstantVariable.create(inst.argval)], {}  # type: ignore[arg-type]
        )

    def create_call_resume_at(self, offset):
        raise AssertionError(
            f"create_call_resume_at not overridden by subclass {type(self)}"
        )

    def should_compile_partial_graph(self) -> bool:
        raise AssertionError(
            f"should_compile_partial_graph not overridden by subclass {type(self)}"
        )

    @break_graph_if_unsupported(push=0)
    def STORE_SUBSCR(self, inst):
        val, obj, key = self.popn(3)
        obj.call_method(self, "__setitem__", [key, val], {})

    def DELETE_SUBSCR(self, inst):
        obj, key = self.popn(2)
        obj.call_method(self, "__delitem__", [key], {})

    def BUILD_TUPLE(self, inst):
        name_tuple = None
        if sys.version_info >= (3, 13):
            name_tuple = tuple(self.name_stack[-inst.argval :])
        items = self.popn(inst.argval)
        self.push(TupleVariable(items), name=name_tuple)

    def BUILD_SLICE(self, inst):
        items = self.popn(inst.argval)
        self.push(SliceVariable(items))

    def BUILD_LIST(self, inst):
        items = self.popn(inst.argval)
        self.push(ListVariable(items, mutable_local=MutableLocal()))

    def BUILD_SET(self, inst):
        if config.inject_BUILD_SET_unimplemented_TESTING_ONLY:
            unimplemented("missing: BUILD_SET")
        items = self.popn(inst.argval)
        new_set = SetVariable(items, mutable_local=MutableLocal())
        self.push(new_set)

    def BUILD_LIST_UNPACK(self, inst, cls=ListVariable):
        seqs = self.popn(inst.argval)
        items = []
        for seq in seqs:
            try:
                items.extend(seq.unpack_var_sequence(self))
            except NotImplementedError:
                unimplemented(f"BUILD_LIST_UNPACK {seq}")
        self.push(cls(items, mutable_local=MutableLocal()))

    def BUILD_TUPLE_UNPACK(self, inst):
        self.BUILD_LIST_UNPACK(inst, cls=TupleVariable)

    BUILD_TUPLE_UNPACK_WITH_CALL = BUILD_TUPLE_UNPACK

    def BUILD_MAP(self, inst):
        items = self.popn(inst.argval * 2)
        d = dict(zip(items[::2], items[1::2]))
        self.push(ConstDictVariable(d, mutable_local=MutableLocal()))

    def BUILD_MAP_UNPACK(self, inst):
        items = self.popn(inst.argval)
        # ensure everything is a dict
        items = [BuiltinVariable(dict).call_function(self, [x], {}) for x in items]  # type: ignore[arg-type]
        result = {}
        for x in items:
            assert isinstance(x, ConstDictVariable)
            result.update(x.items)
        self.push(
            ConstDictVariable(
                result,
                mutable_local=MutableLocal(),
            )
        )

    BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK

    def BUILD_CONST_KEY_MAP(self, inst):
        keys = self.pop()
        values = self.popn(inst.argval)
        assert isinstance(keys, TupleVariable)
        assert keys.is_python_constant()

        keys = keys.unpack_var_sequence(self)
        assert len(keys) == len(values)

        self.push(
            ConstDictVariable(
                dict(zip(keys, values)),
                mutable_local=MutableLocal(),
            )
        )

    def MAP_ADD(self, inst):
        k, v = self.popn(2)
        assert inst.argval > 0
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ConstDictVariable)
        obj.call_method(self, "__setitem__", (k, v), {})  # type: ignore[arg-type]

    def SET_ADD(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, SetVariable)
        assert obj.mutable_local
        return obj.call_method(self, "add", [v], {})

    def SET_UPDATE(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, SetVariable)
        assert obj.mutable_local
        obj.call_method(self, "update", [v], {})

    def LIST_APPEND(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ListVariable)
        assert obj.mutable_local
        self.output.side_effects.mutation(obj)
        obj.items.append(v)

    def MAKE_FUNCTION(self, inst):
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
                closure_scope=self,
            )
        )

    def UNPACK_SEQUENCE(self, inst):
        seq = self.pop()
        if isinstance(seq, TensorVariable):
            val = seq.unpack_var_sequence(self, idxes=range(inst.argval))  # type: ignore[arg-type]
        elif isinstance(seq, GetAttrVariable) and isinstance(seq.obj, TensorVariable):
            # x, y = a.shape
            proxy = getattr(seq.obj.as_proxy(), seq.name)
            val = [wrap_fx_proxy(self, proxy[i]) for i in range(inst.argval)]
        elif seq.has_unpack_var_sequence(self):
            val = seq.unpack_var_sequence(self)
        else:
            unimplemented(f"UNPACK_SEQUENCE {seq}")
        if len(val) != inst.argval:
            unimplemented("UNPACK_SEQUENCE length mismatch")
        for i in reversed(val):
            self.push(i)

    def UNPACK_EX(self, inst):
        assert 0 <= inst.argval <= 0xFFFF
        prefix = inst.argval & 0xFF  # low byte
        suffix = inst.argval >> 8  # high byte
        seq = self.pop()
        if seq.has_unpack_var_sequence(self):
            vals = list(seq.unpack_var_sequence(self))
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
            unimplemented(f"UNPACK_EX {seq}")

    def NOP(self, inst):
        pass

    def POP_TOP(self, inst):
        self.pop()

    def ROT_TWO(self, inst):
        a = self.pop()
        b = self.pop()
        self.push(a)
        self.push(b)

    def ROT_THREE(self, inst):
        a = self.pop()
        b = self.pop()
        c = self.pop()
        self.push(a)
        self.push(c)
        self.push(b)

    def ROT_FOUR(self, inst):
        a = self.pop()
        b = self.pop()
        c = self.pop()
        d = self.pop()
        self.push(a)
        self.push(d)
        self.push(c)
        self.push(b)

    def DUP_TOP(self, inst):
        a = self.pop()
        self.push(a)
        self.push(a)

    def DUP_TOP_TWO(self, inst):
        a = self.pop()
        b = self.pop()
        self.push(b)
        self.push(a)
        self.push(b)
        self.push(a)

    def FORMAT_VALUE(self, inst):
        flags = inst.arg
        if (flags & 0x04) == 0x04:
            fmt_spec = self.pop()
        else:
            fmt_spec = ConstantVariable.create("")

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
        if (flags & 0x03) == 0x01:
            value = BuiltinVariable(str).call_function(self, [value], {})  # type: ignore[arg-type]
        elif (flags & 0x03) == 0x02:
            value = BuiltinVariable(repr).call_function(self, [value], {})  # type: ignore[arg-type]
        elif (flags & 0x03) == 0x03:
            value = BuiltinVariable(ascii).call_function(self, [value], {})  # type: ignore[arg-type]

        fmt_var = ConstantVariable.create("{:" + fmt_spec.as_python_constant() + "}")

        self.call_function(BuiltinVariable(str.format), [fmt_var, value], {})

    def BUILD_STRING(self, inst):
        format_string_parts: List[str] = []
        args: List[VariableTracker] = []
        kwargs: Dict[str, VariableTracker] = {}
        for part in self.popn(inst.arg):
            if isinstance(part, ConstantVariable):
                format_string_parts.append("{}")
                args.append(part)
            elif isinstance(part, variables.StringFormatVariable):
                format_string_parts.append(part.format_string)
                args.extend(part.sym_args)
                if set(kwargs.keys()) & set(part.sym_kwargs.keys()):
                    unimplemented(
                        f"BUILD_STRING key conflict {kwargs} & {part.sym_kwargs}"
                    )
                kwargs.update(part.sym_kwargs)
            else:
                unimplemented(f"BUILD_STRING {part}")
        self.push(
            variables.StringFormatVariable.create(
                "".join(format_string_parts), args, kwargs
            )
        )

    def IS_OP(self, inst):
        assert inst.argval == 0 or inst.argval == 1
        if inst.argval == 0:
            new_argval = "is"
        else:
            new_argval = "is not"
        new_inst = create_instruction("COMPARE_OP", argval=new_argval)
        self.COMPARE_OP(new_inst)

    def CONTAINS_OP(self, inst):
        assert inst.argval == 0 or inst.argval == 1
        left, right = self.popn(2)
        op = inst.argval
        self.push(right.call_method(self, "__contains__", [left], {}))
        if op == 1:
            self.UNARY_NOT(inst)

    def LIST_EXTEND(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ListVariable)
        assert obj.mutable_local
        obj.call_method(self, "extend", [v], {})

    def LIST_TO_TUPLE(self, inst):
        self.push(BuiltinVariable(tuple).call_function(self, [self.pop()], {}))  # type: ignore[arg-type]

    def DICT_MERGE(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ConstDictVariable)
        assert obj.mutable_local
        obj.call_method(self, "update", [v], {})

    DICT_UPDATE = DICT_MERGE

    def GEN_START(self, inst):
        self.pop()

    def GET_LEN(self, inst):
        tos = self.stack[-1]
        if tos.is_python_constant():
            self.push(ConstantVariable.create(len(tos.as_python_constant())))
        else:
            self.push(tos.call_method(self, "__len__", [], {}))

    def MATCH_MAPPING(self, inst):
        tos = self.stack[-1]
        assert isinstance(tos, ConstDictVariable)
        if isinstance(tos.items, collections.abc.Mapping):
            self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(False))

    def MATCH_SEQUENCE(self, inst):
        tos = self.stack[-1]
        assert tos.is_python_constant()
        tos_value = tos.as_python_constant()
        if isinstance(tos_value, collections.abc.Sequence) and not isinstance(
            tos_value, (str, bytes, bytearray)
        ):
            self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(False))

    def MATCH_KEYS(self, inst):
        tos = self.stack[-1]
        tos1 = self.stack[-2]
        assert isinstance(tos1, ConstDictVariable)

        if all(k in tos1 for k in tos):  # type: ignore[attr-defined]
            self.push(TupleVariable([tos1.getitem_const(self, k) for k in tos]))  # type: ignore[attr-defined,arg-type]
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(None))
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(False))

    def LOAD_ASSERTION_ERROR(self, inst):
        self.load_builtin_from_argval("AssertionError")

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
    BINARY_SUBSCR = break_graph_if_unsupported(push=1)(stack_op(operator.getitem))
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
    def RESUME(self, inst):
        if inst.arg == 0:
            self.append_prefix_inst(inst)
            self.accept_prefix_inst = False
        else:
            assert not self.accept_prefix_inst

    if sys.version_info >= (3, 11):

        def BINARY_OP(self, inst):
            return _binary_op_lookup[inst.arg](self, inst)

    def PRECALL(self, inst):
        pass

    def KW_NAMES(self, inst):
        kw_names = self.code_options["co_consts"][inst.arg]
        assert isinstance(kw_names, tuple)
        for name in kw_names:
            assert isinstance(name, str)
        assert self.kw_names is None
        self.kw_names = ConstantVariable.create(value=kw_names)  # type: ignore[assignment]

    def PUSH_NULL(self, inst):
        self.push(NullVariable())

    def _call(self, inst, call_kw=False):
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

    @break_graph_if_unsupported(push=1)
    def CALL(self, inst):
        self._call(inst)

    def COPY(self, inst):
        self.push(self.stack[-inst.arg])

    def SWAP(self, inst):
        self.stack[-1], self.stack[-inst.arg] = self.stack[-inst.arg], self.stack[-1]

    JUMP_BACKWARD = jump
    JUMP_BACKWARD_NO_INTERRUPT = jump

    POP_JUMP_FORWARD_IF_TRUE = generic_jump(operator.truth, False)
    POP_JUMP_BACKWARD_IF_TRUE = generic_jump(operator.truth, False)
    POP_JUMP_FORWARD_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_BACKWARD_IF_FALSE = generic_jump(operator.not_, False)

    def CACHE(self, inst):
        pass

    def BEFORE_WITH(self, inst):
        self.setup_or_before_with(inst)

    def setup_or_before_with(self, inst):
        ctx = self.pop()
        if not isinstance(
            ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
        ):
            unimplemented(f"{inst.opname} {ctx}")

        if isinstance(ctx, GenericContextWrappingVariable):
            self.generic_context_manager_depth += 1

        # Need this redundant check for mypy
        assert isinstance(
            ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
        )

        exit = WithExitFunctionVariable(
            ctx,
            inst.target,
        )

        if sys.version_info >= (3, 11):
            # See create_call_resume_at for block stack details.
            # Only push a block if the current instruction's block is a
            # with block that is not nested in a try block - that is, the current
            # instruction's block target is the same as the top block's target.
            if inst.exn_tab_entry and (
                not self.block_stack
                or inst.exn_tab_entry.target is not self.block_stack[-1].target
            ):
                target = None
            else:
                target = self.next_instruction.exn_tab_entry.target
        else:
            target = inst.target

        if target:
            if isinstance(self, InstructionTranslator):
                self.block_stack.append(
                    BlockStackEntry(inst, target, len(self.stack), ctx)
                )
            else:
                self.block_stack.append(BlockStackEntry(inst, target))

        self.push(exit)
        self.push(ctx.enter(self))

    def append_prefix_inst(self, inst):
        assert self.accept_prefix_inst
        self.prefix_insts.append(inst)

    def MAKE_CELL(self, inst):
        if sys.version_info >= (3, 12) and not self.accept_prefix_inst:
            # In 3.12+, MAKE_CELL is not longer necessarily a prefix instruction.
            # It can be generated by inlined comprehensions.
            assert isinstance(self.symbolic_locals[inst.argval], NullVariable)
            self.symbolic_locals[
                inst.argval
            ] = self.output.side_effects.track_cell_new()
        else:
            self.append_prefix_inst(inst)

    def COPY_FREE_VARS(self, inst):
        self.append_prefix_inst(inst)

    def RETURN_GENERATOR(self, inst):
        self.append_prefix_inst(inst)

    # 3.12 opcodes
    # BINARY/STORE_SLICE opcodes are broken down into
    # BUILD_SLICE 2 and BINARY/STORE_SUBSCR

    def END_FOR(self, inst):
        if sys.version_info >= (3, 13):
            self.pop()
        else:
            self.popn(2)

    def LOAD_FAST_CHECK(self, inst):
        if isinstance(self.symbolic_locals[inst.argval], NullVariable):
            unimplemented("LOAD_FAST_CHECK on uninitialized variable")
        self.LOAD_FAST(inst)

    def LOAD_FAST_AND_CLEAR(self, inst):
        if inst.argval not in self.symbolic_locals:
            self.push(NullVariable())
        else:
            self.LOAD_FAST(inst)
        self.symbolic_locals[inst.argval] = NullVariable()

    def LOAD_SUPER_ATTR(self, inst):
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        if inst.arg & 1:
            self.LOAD_METHOD(inst)
        else:
            self._load_attr(inst)

    def CALL_INTRINSIC_1(self, inst):
        if inst.argval == 5:
            # INTRINSIC_UNARY_POSITIVE
            self.UNARY_POSITIVE(inst)
        elif inst.argval == 6:
            # INTRINSIC_LIST_TO_TUPLE
            self.push(TupleVariable(self.pop().unpack_var_sequence(self)))
        else:
            unimplemented(f"missing CALL_INTRINSIC_1 operand {inst.argval}")

    def END_SEND(self, inst):
        tos = self.pop()
        self.pop()
        self.push(tos)

    # 3.13 opcodes
    # fused instructions LOAD_FAST_LOAD_FAST, STORE_FAST_STORE_FAST, STORE_FAST_LOAD_FAST
    # are broken down.
    @break_graph_if_unsupported(push=1)
    def CALL_KW(self, inst):
        self._call(inst, call_kw=True)

    def TO_BOOL(self, inst):
        # TO_BOOL only precedes a conditional jump or UNARY_NOT (see compile.c in CPython)
        # So we can skip this instruction as long as we remember to codegen a TO_BOOL
        # before conditional jumps/UNARY_NOT.
        assert self.next_instruction.opname in (
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_IF_FALSE",
            "UNARY_NOT",
        )

    def SET_FUNCTION_ATTRIBUTE(self, inst):
        flags = inst.arg
        fn = self.pop()
        assert isinstance(fn, NestedUserFunctionVariable)
        attr_names = self.name_stack[-1]
        attr = self.pop()

        if flags & 0x08:
            # 3.13 merged LOAD_CLOSURE into LOAD_FAST, so we won't know if a given LOAD_FAST
            # is meant to load a closure variable or not. Our workaround is to maintain a stack
            # of LOAD_FAST variable names and tuples (self.name_stack). So if we are indeed
            # constructing a closure tuple, we can use self.name_stack to construct the closure
            # variables here.
            assert isinstance(attr_names, tuple) and all(
                isinstance(name, str) for name in attr_names
            )
            fn.closure = TupleVariable(
                [self._load_closure(name) for name in attr_names]
            )
            fn.closure_scope = self
        elif flags & 0x04:
            fn.annotations = attr
        elif flags & 0x02:
            fn.kwdefaults = attr
        elif flags & 0x01:
            fn.defaults = attr

        self.push(fn)

    def _format_value_313(self, fmt_spec):
        value = self.pop()
        if isinstance(value, SymNodeVariable):
            value = ConstantVariable.create(str(value.sym_num))

        fmt_var = ConstantVariable.create("{:" + fmt_spec.as_python_constant() + "}")

        self.call_function(BuiltinVariable(str.format), [fmt_var, value], {})

    def FORMAT_SIMPLE(self, inst):
        self._format_value_313(ConstantVariable.create(""))

    def FORMAT_WITH_SPEC(self, inst):
        self._format_value_313(self.pop())

    def is_non_empty_graph(self):
        if self.output.count_calls() > 1:
            # perf optimization only
            self.is_non_empty_graph = lambda: True  # type: ignore[method-assign]
            return True
        return False

    def format_frame_summary(self, additional_stack_frames=None):
        if additional_stack_frames is None:
            additional_stack_frames = []
        return "".join(
            traceback.format_list(
                [self.frame_summary()] + list(reversed(additional_stack_frames))
            )
        )

    def frame_summary(self):
        return traceback.FrameSummary(
            getattr(self.f_code, "co_filename", "<unknown>"),
            self.lineno,
            getattr(self.f_code, "co_name", "<unknown>"),
            lookup_line=False,
        )

    def is_co_filename_from_nn_modules(self):
        filename = getattr(self.f_code, "co_filename", "<unknown>")
        nn_modules_pattern = re.compile(r".*torch/nn/modules.*")
        return nn_modules_pattern.match(filename) is not None

    def store_global_weakref_by_id(self, prefix, value):
        global_name = self.output.install_global_by_id(prefix, weakref.ref(value))
        install_guard(
            GlobalWeakRefSource(global_name).make_guard(GuardBuilder.WEAKREF_ALIVE)
        )
        return global_name

    @property
    def fake_mode(self):
        return self.output.tracing_context.fake_mode

    def find_symbolic_locals_name(self, tensor_variable):
        for key, value in self.symbolic_locals.items():
            if value is tensor_variable:
                return key
        return None

    @contextlib.contextmanager
    def strict_translation_mode(self, check_fn: Callable[[VariableTracker], bool]):
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

    def __init__(
        self,
        output: OutputGraph,
        instructions: List[Instruction],
        f_locals: Dict[str, Any],
        f_globals: Dict[str, Any],
        f_builtins: Dict[str, Any],
        code_options: Dict[str, Any],
        symbolic_locals: Dict[str, VariableTracker],
        symbolic_globals: Dict[str, VariableTracker],
        symbolic_torch_function_mode_stack: Deque["TorchFunctionModeVariable"],
        f_code: types.CodeType,
        export: bool,
        inline_depth: int,
        speculation_log: SpeculationLog,
        distributed_state: Optional[DistributedState],
    ) -> None:
        super().__init__()
        self.speculation_log = speculation_log
        self.distributed_state = distributed_state

        # Mutable state checkpointed by copy_graphstate()
        self.output = output
        self.symbolic_locals = symbolic_locals
        self.symbolic_globals = symbolic_globals
        self.symbolic_torch_function_mode_stack = symbolic_torch_function_mode_stack
        self.stack = []
        # stack of variable names for tracking 3.13 closures
        self.name_stack: list[Any] = []
        self.instruction_pointer = 0
        self.current_instruction = create_instruction("NOP")
        self.block_stack = []
        # states before SETUP_WITH for checkpointing and fallback
        self.generic_context_manager_depth = 0
        self.lineno = -1
        self.kw_names = None
        self.accept_prefix_inst = True
        self.prefix_insts = []
        self.exn_vt_stack = []

        # Properties of the input/output code
        self.instructions: List[Instruction] = instructions
        self.indexof: Dict[Instruction, int] = get_indexof(self.instructions)
        self.f_locals: Dict[
            str, Any
        ] = f_locals  # needed for recording accessed locals for replay
        self.f_globals: Dict[str, Any] = f_globals
        self.f_builtins: Dict[str, Any] = f_builtins
        self.code_options: Dict[str, Any] = code_options
        self.f_code: types.CodeType = f_code

        # Execution record for replaying errors
        if config.replay_record_enabled:
            self.exec_recorder = ExecutionRecorder(
                code=f_code, code_options=code_options
            )
        else:
            self.exec_recorder = None
        # Stack of module being parsed, current nn.module is at the end of ordered dict.
        # The first field of tuple is the fully qualified name of current module
        # in original hierarchy.  The second field is the type of current nn.module
        self.nn_module_stack: Dict[str, Tuple[str, Type[Any]]] = {}
        # Flag to indicate whether tracing is used for export.
        self.export = export
        self.one_graph = False

        self.current_speculation = None

        self.strict_checks_fn = None

        if sys.version_info >= (3, 10):
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
        self._constants_cache: List[Optional[VariableTracker]] = [None] * len(
            f_code.co_consts
        )
        linecache.lazycache(f_code.co_filename, f_globals)


class InstructionTranslator(InstructionTranslatorBase):
    mutated_closure_cell_contents: Set[str]

    @staticmethod
    def current_tx() -> "InstructionTranslator":
        return tls.current_tx

    @contextlib.contextmanager
    def set_current_tx(self):
        prior = getattr(tls, "current_tx", None)
        tls.current_tx = self
        try:
            yield
        finally:
            tls.current_tx = prior

    def __init__(
        self,
        instructions: List[Instruction],
        f_code,
        f_locals,
        f_globals,
        f_builtins,
        code_options,
        compiler_fn,
        one_graph,
        export,
        export_constraints,
        mutated_closure_cell_contents: Set[str],
        frame_state,
        speculation_log: SpeculationLog,
        distributed_state: Optional[DistributedState],
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
            ),
            instructions=instructions,
            f_locals=f_locals,
            f_globals=f_globals,
            f_builtins=f_builtins,
            code_options=code_options,
            symbolic_locals={},  # set below
            # A global var is inserted only after a STORE_GLOBAL happens to it
            symbolic_globals={},
            symbolic_torch_function_mode_stack=collections.deque(),
            f_code=f_code,
            export=export,
            inline_depth=0,
            speculation_log=speculation_log,
            distributed_state=distributed_state,
        )

        self._throw_if_in_functorch()

        # as soon as we create the tracing context we should keep it active, so any calls
        # into dynamo apis can rely on finding it
        with tracing(self.output.tracing_context), self.set_current_tx():
            self.one_graph: bool = one_graph
            self.export = export
            self.mutated_closure_cell_contents = mutated_closure_cell_contents
            if self.export:
                assert (
                    self.one_graph
                ), "Export without one graph - something has gone wrong."

            vars = list(code_options["co_varnames"])
            cells_and_freevars = [x for x in self.cell_and_freevars() if x not in vars]
            vars.extend(cells_and_freevars)
            cells_and_freevars_set = set(cells_and_freevars)

            self.symbolic_locals = {
                k: variables.LazyVariableTracker.create(
                    f_locals[k],
                    source=LocalSource(k, cell_or_freevar=k in cells_and_freevars_set),
                )
                for k in vars
                if k in f_locals
            }

            self._init_torch_function_mode_stack()

            self.debug_locals: List[Tuple[VariableTracker, List[VariableTracker]]] = []
            if export:
                # export gets confused if we never realize unused inputs
                # in export mode just eagerly realize everything
                self.symbolic_locals = variables.LazyVariableTracker.realize_all(
                    self.symbolic_locals
                )

            self._freevars_ids = {}
            for name in self.code_options["co_freevars"]:
                if name in f_locals:
                    self._freevars_ids[name] = id(f_locals[name])

    def _throw_if_in_functorch(self):
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
            unimplemented(msg)

    def _init_torch_function_mode_stack(self):
        from .variables.torch_function import TorchFunctionModeStackVariable

        TorchFunctionModeStackVariable.reset()

        self.symbolic_torch_function_mode_stack: Deque[
            TorchFunctionModeVariable
        ] = collections.deque()
        # We want to retrieve all modes to properly reconstruct the stack if needed
        py_stack = get_torch_function_mode_stack(filter_ignored=False)

        if py_stack:
            isinstance(py_stack[0], torch.utils._device.DeviceContext)

        for i, val in enumerate(py_stack):
            self.symbolic_torch_function_mode_stack.append(
                variables.LazyVariableTracker.create(
                    val, source=TorchFunctionModeStackSource(i)
                )
            )

    def get_example_value(self, source: Source):
        if isinstance(source, LocalSource):
            return self.f_locals[source.local_name]
        if isinstance(source, GlobalSource):
            return self.f_globals[source.global_name]
        raise KeyError

    def run(self):
        super().run()

    def match_nested_cell(self, name, cell):
        """Match a cell in this method to one in a function we are inlining"""
        try:
            value = cell.cell_contents
        except ValueError:
            return None
        # TODO(jansel): check the id of the cell rather than the contents
        if id(value) != self._freevars_ids.get(name):
            return None
        return self.symbolic_locals[name]

    def should_compile_partial_graph(self):
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
            and self.generic_context_manager_depth == 0
        )

    def create_call_resume_at(self, inst):
        self.instruction_pointer = None

        if inst.opname == "RETURN_VALUE":
            return [create_instruction("RETURN_VALUE")]
        elif inst.opname == "RETURN_CONST":
            return [create_instruction("RETURN_CONST", argval=inst.argval)]

        reads = livevars_analysis(self.instructions, inst)
        all_argnames = tuple(
            k
            for k in self.symbolic_locals.keys()
            if k in reads and k not in self.cell_and_freevars()
        )
        # NOTE: do not use isinstance, since it realizes lazy VT's
        argnames = tuple(
            k
            for k in all_argnames
            if not type.__instancecheck__(NullVariable, self.symbolic_locals[k])
        )
        argnames_null = tuple(
            k
            for k in all_argnames
            if type.__instancecheck__(NullVariable, self.symbolic_locals[k])
        )
        if sys.version_info < (3, 12):
            assert len(argnames_null) == 0, "variables should not be NULL in < 3.12"

        cg = PyCodegen(self)

        # Handle inactive context variables.
        # The resume function assumes that context variables are the class, NOT the object.
        # e.g. torch.set_grad_enabled(True) will be reconstructed as torch.set_grad_enabled
        stack_ctx_vars = []
        for i, var in enumerate(self.stack):
            if type.__instancecheck__(ContextWrappingVariable, var):
                ctx = cast(ContextWrappingVariable, var)
                target_values = (
                    () if ctx.target_values is None else tuple(ctx.target_values)
                )
                stack_ctx_vars.append((i, target_values))
                # Replace the current stack var with the context class
                ctx.reconstruct_type(cg)
                cg.extend_output(create_swap(len(self.stack) - i + 1))
                cg.append_output(create_instruction("POP_TOP"))

        argnames_ctx_vars = []
        for name in argnames:
            if type.__instancecheck__(
                ContextWrappingVariable, var := self.symbolic_locals[name]
            ):
                ctx = cast(ContextWrappingVariable, var)
                target_values = (
                    () if ctx.target_values is None else tuple(ctx.target_values)
                )
                argnames_ctx_vars.append((name, target_values))
                # Replace the local with the context class
                ctx.reconstruct_type(cg)
                cg.append_output(create_instruction("STORE_FAST", argval=name))

        # Python does not allow null to be an arg to a function, so
        # we remove nulls from the stack and restore them in the
        # prologue of the resume function

        # sorted list of indices of nulls on the stack
        null_idxes: List[int] = []
        if sys.version_info >= (3, 11):
            # find indices of NullVariables
            for i, var in enumerate(self.stack):
                if type.__instancecheck__(NullVariable, var):
                    null_idxes.append(i)
            # generate bytecode to pop the nulls
            null_cnt = 0
            for i, var in enumerate(reversed(self.stack)):
                if type.__instancecheck__(NullVariable, var):
                    for j in range(2, i + 2 - null_cnt):
                        cg.append_output(create_instruction("SWAP", arg=j))
                    cg.extend_output(cg.pop_null())
                    null_cnt += 1

        # we popped all nulls from the stack at runtime,
        # so we should not count NullVariables
        stack_len = len(self.stack) - len(null_idxes)
        nargs = stack_len + len(argnames)

        name = unique_id(f"__resume_at_{inst.offset}")

        new_code: types.CodeType = ContinueExecutionCache.lookup(
            self.f_code,
            self.lineno,
            inst.offset,
            tuple(b.target.offset for b in self.block_stack),
            stack_len,
            argnames,
            argnames_null,
            tuple(b.resume_fn() for b in self.block_stack),
            tuple(stack_ctx_vars),
            tuple(argnames_ctx_vars),
            tuple(null_idxes),
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

        if new_code.co_freevars:
            # expose code object for debugging purposes
            self.output.install_global_unsafe(name, new_code)
            cg.make_function_with_closure(name, new_code, True, stack_len)
        else:
            # This is safe: we pre-generate a unique name
            self.output.install_global_unsafe(
                name, types.FunctionType(new_code, self.f_globals, name)
            )
            cg.extend_output(cg.load_function_name(name, True, stack_len))

        cg.extend_output([cg.create_load(k) for k in argnames])
        cg.extend_output(create_call_function(nargs, False))
        cg.append_output(create_instruction("RETURN_VALUE"))
        return cg.get_instructions()

    def symbolic_locals_contain_module_class(self):
        for v in self.symbolic_locals.values():
            if isinstance(v, UserDefinedClassVariable) and issubclass(
                v.as_python_constant(), torch.nn.Module
            ):
                return True
        return False

    def _return(self, inst):
        if (
            self.output.count_calls() == 0
            and not self.inconsistent_side_effects
            and not self.symbolic_locals_contain_module_class()
            and not self.export
        ):
            raise exc.SkipFrame("because no content in function call")
        self.instruction_pointer = None
        _step_logger()(
            logging.INFO,
            f"torchdynamo done tracing {self.f_code.co_name} ({inst.opname})",
        )
        log.debug("%s triggered compile", inst.opname)
        self.output.compile_subgraph(
            self,
            reason=GraphCompileReason(
                "return_value", [self.frame_summary()], graph_break=False
            ),
        )
        return_inst = (
            create_instruction("RETURN_VALUE")
            if inst.opname == "RETURN_VALUE"
            else create_instruction("RETURN_CONST", argval=inst.argval)
        )
        self.output.add_output_instructions([return_inst])
        raise ReturnValueOp

    def RETURN_VALUE(self, inst):
        self._return(inst)

    def RETURN_CONST(self, inst):
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

    symbolic_result: Optional[TensorVariable]

    @classmethod
    def inline_call(cls, parent, func, args, kwargs):
        with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
            return cls.inline_call_(parent, func, args, kwargs)

    @staticmethod
    def check_inlineable(func):
        if func.has_self():
            unimplemented("inline with __self__")

        result = trace_rules.check_verbose(func, is_inlined_call=True)
        if result.skipped:
            from torch._dynamo.variables.misc import produce_trampoline_autograd_apply

            # _origin marks this as coming from an internal dynamo known function that is safe to
            # trace through.
            if hasattr(getattr(func, "fn", None), "_origin") and func.fn._origin in [
                produce_trampoline_autograd_apply,
            ]:
                # Known sound
                return trace_rules.SkipResult(
                    False, "allowlist in dynamo known function"
                )
            fn_qualname = func.fn.__qualname__ if hasattr(func, "fn") else ""
            unimplemented(
                f"'inline in skipfiles: {fn_qualname} | {func.get_name()} {func.get_filename()}, {result.reason}'"
            )

        if isinstance(func, UserFunctionVariable) and inspect.getattr_static(
            func.get_function(), "_torchdynamo_disable", False
        ):
            unimplemented(
                f"call torch._dynamo.disable() wrapped function {func.get_function()}"
            )
        else:
            return result

    @staticmethod
    def inline_call_(
        parent, func: VariableTracker, args: List[VariableTracker], kwargs
    ):
        if isinstance(func, SkipFunctionVariable):
            unimplemented("inline with functions in skip files")
        assert isinstance(
            func,
            (UserFunctionVariable, NestedUserFunctionVariable),
        )
        result = InliningInstructionTranslator.check_inlineable(func)
        assert result.skipped is False
        try:
            sub_locals, closure_cells = func.bind_args(parent, args, kwargs)
        except TypeError as e:
            # Wrap the general TypeError during bind_args() to the internal ArgsMismatchError with detailed info
            raise ArgsMismatchError(  # noqa: B904
                "{reason}.\n  func = {func}, args = {args}, kwargs = {kwargs}".format(
                    reason=str(e),
                    func=f"'{func.get_name()}' {func.get_filename()}:{func.get_code().co_firstlineno}",
                    args=[arg.python_type() for arg in args],
                    kwargs=kwargs,
                ),
            )

        for v in itertools.chain(sub_locals.values(), closure_cells.values()):
            if not isinstance(v, VariableTracker):
                unimplemented(f"unconverted arg {v}")

        code: types.CodeType = func.get_code()
        if code.co_name in ("__setitem__", "__setattr__") and not (
            args
            and isinstance(
                args[0],
                (variables.CustomizedDictVariable, variables.UserDefinedObjectVariable),
            )
        ):
            unimplemented(f"inline {code.co_name}")

        suffix = ""
        # TODO: mlazos, add support for enabling multiple artifact logs
        # with a single alias
        if torch._logging._internal.log_state.is_artifact_enabled("bytecode"):
            suffix = f"\n{dis.Bytecode(code).dis()}"
        if sys.version_info >= (3, 11):
            cur_inst = parent.current_instruction
            parent_code = parent.f_code
            header = parent.get_line_of_code_header(lineno=cur_inst.positions.lineno)

            def get_trace_call_log_str():
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

        tracer: InliningInstructionTranslator
        if is_generator(code):
            tracer = InliningGeneratorInstructionTranslator(
                parent,
                code,
                sub_locals,
                parent.symbolic_globals,
                parent.symbolic_torch_function_mode_stack,
                closure_cells,
                func,
            )
        else:
            tracer = InliningInstructionTranslator(
                parent,
                code,
                sub_locals,
                parent.symbolic_globals,
                parent.symbolic_torch_function_mode_stack,
                closure_cells,
                func,
            )

        strict_ctx: Any = contextlib.nullcontext()
        if parent.strict_checks_fn:
            strict_ctx = tracer.strict_translation_mode(parent.strict_checks_fn)
        try:
            with strict_ctx:
                tracer.run()
        except exc.ObservedException as e:
            msg = f"Observed exception DURING INLING {code} : {e}"
            # TODO(anijain2305) - This works but we should probably have a
            # global/central data structure for the exception stack.
            parent.exn_vt_stack.extend(tracer.exn_vt_stack)
            log.debug(msg)
            # bubble up the exception to the parent frame.
            raise
        except exc.SkipFrame as e:
            msg = f"SKIPPED INLINING {code}: {e}"
            log.debug(msg)
            raise Unsupported(msg) from e
        except Exception as e:
            log.debug("FAILED INLINING %s", code)
            raise
        assert tracer.symbolic_result is not None
        func.export_freevars(parent, tracer)

        if tracer.f_globals is parent.f_globals:
            # Merge symbolic_globals back if parent and child are in the same namespace
            parent.symbolic_globals.update(tracer.symbolic_globals)

        parent.inconsistent_side_effects |= tracer.inconsistent_side_effects

        log.debug("DONE INLINING %s", code)

        if is_generator(code):
            assert isinstance(tracer, InliningGeneratorInstructionTranslator)
            assert tracer.symbolic_result.as_python_constant() is None
            return ListIteratorVariable(
                tracer.generated_items,
                mutable_local=MutableLocal(),
            )
        else:
            return tracer.symbolic_result

    def __init__(
        self,
        parent: InstructionTranslatorBase,
        code: types.CodeType,
        symbolic_locals: Dict[str, VariableTracker],
        symbolic_globals: Dict[str, VariableTracker],
        symbolic_torch_function_mode_stack: Deque["TorchFunctionModeVariable"],
        closure_cells: Dict[str, VariableTracker],
        funcvar: BaseUserFunctionVariable,
    ) -> None:
        f_globals = funcvar.get_globals()  # type: ignore[attr-defined]
        f_builtins = f_globals["__builtins__"]
        if not isinstance(f_builtins, dict):
            f_builtins = f_builtins.__dict__
        instructions = cleaned_instructions(code)
        propagate_line_nums(instructions)
        super().__init__(
            output=parent.output,
            f_locals={},
            f_globals=f_globals,
            f_builtins=f_builtins,
            symbolic_locals=symbolic_locals,
            symbolic_globals=symbolic_globals,
            symbolic_torch_function_mode_stack=symbolic_torch_function_mode_stack,
            instructions=instructions,
            code_options={k: getattr(code, k) for k in get_code_keys()},
            f_code=code,
            export=parent.export,
            inline_depth=parent.inline_depth + 1,
            speculation_log=parent.speculation_log,
            distributed_state=parent.distributed_state,
        )
        self.parent = parent
        self.symbolic_result = None
        self.closure_cells = closure_cells
        self.nn_module_stack = parent.nn_module_stack.copy()
        self.one_graph = parent.one_graph

    @property
    def fake_mode(self):
        return self.parent.fake_mode

    def run_ctx_mgr(self):
        return TracingContext.current_frame(self.parent.frame_summary())

    def STORE_DEREF(self, inst):
        if inst.argval in self.closure_cells:
            cell = self.closure_cells[inst.argval]
            val = self.pop()
            if isinstance(cell, ClosureVariable):
                if not self.output.is_root_tracer():
                    unimplemented(
                        "HigherOrderOperator: Mutating a variable not in the current scope (ClosureVariable)"
                    )
                self.output.root_tx.symbolic_locals[cell.name] = val
            else:
                self.output.side_effects.store_cell(cell, val)
        else:
            maybe_cell = self.symbolic_locals.get(inst.argval)
            if isinstance(
                maybe_cell,
                variables.NewCellVariable,
            ):
                self.output.side_effects.store_cell(
                    self.symbolic_locals[inst.argval], self.pop()
                )
            else:
                if (
                    maybe_cell is not None
                    and maybe_cell.source.name()
                    not in self.output.root_tx.mutated_closure_cell_contents
                ):
                    # Why is the source name here unique?
                    # mutated_closure_cell_contents is a per-frame
                    # concept, and sources identify, e.g., particular
                    # locals from the frame.  If you had two locals,
                    # they'll get different source names, and therefore
                    # differ here.
                    self.output.root_tx.mutated_closure_cell_contents.add(
                        maybe_cell.source.name()
                    )
                    raise exc.UnspecializeRestartAnalysis
                unimplemented("write to __closure__ while inlining")

    def LOAD_DEREF(self, inst):
        if inst.argval in self.closure_cells:
            cell = self.closure_cells[inst.argval]
            if isinstance(cell, ClosureVariable):
                self.push(self.output.root_tx.symbolic_locals[cell.name])
            else:
                self.push(self.output.side_effects.load_cell(cell))
        else:
            maybe_sym_local = self.symbolic_locals.get(inst.argval, None)
            if isinstance(maybe_sym_local, variables.NewCellVariable):
                self.push(self.output.side_effects.load_cell(maybe_sym_local))
            else:
                super().LOAD_DEREF(inst)

    def _load_closure(self, name):
        assert name in self.cell_and_freevars()
        if name in self.closure_cells:
            return self.closure_cells[name]
        else:
            return InlinedClosureVariable(name=name)

    def check_replace_is_safe(self, oldvar):
        if not is_side_effect_safe(oldvar.mutable_local):
            unimplemented(
                "HigherOrderOperator: Mutating a variable not in the current scope (replace_all)"
            )

    def should_compile_partial_graph(self):
        return False  # inlining functions is all-or-nothing

    def create_call_resume_at(self, offset):
        unimplemented("cant resume while inlining")

    def RETURN_VALUE(self, inst):
        self.symbolic_result = self.pop()  # type: ignore[assignment]
        self.instruction_pointer = None
        raise ReturnValueOp

    def RETURN_CONST(self, inst):
        self.symbolic_result = self._load_const(inst)
        self.instruction_pointer = None
        raise ReturnValueOp

    def get_globals_source_and_value(self, name):
        if "__name__" in self.f_globals:
            module_name = self.f_globals["__name__"]
            module_source = self.import_source(module_name)
            if "torch_package" in module_name:
                fglobals_value = torch.package.package_importer._package_imported_modules[module_name]  # type: ignore[assignment]
            else:
                fglobals_value = importlib.import_module(module_name)  # type: ignore[assignment]
            fglobals_vt = VariableBuilder(self, module_source)(fglobals_value)
            global_source = AttrSource(module_source, name)
        else:
            globals_name = self.output.install_global_by_id(
                "___unnamed_scope", self.f_globals
            )
            globals_source = GlobalSource(globals_name)
            fglobals_value = self.f_globals  # type: ignore[assignment]
            fglobals_vt = VariableBuilder(self, globals_source)(fglobals_value)
            global_source = GetItemSource(globals_source, name)  # type: ignore[assignment]
        return fglobals_value, fglobals_vt, global_source

    def _load_global(self, inst):
        if self.output.global_scope is self.f_globals:
            super()._load_global(inst)
        else:
            name = inst.argval

            _, fglobals_vt, global_source = self.get_globals_source_and_value(name)
            if self.output.side_effects.has_pending_mutation_of_attr(fglobals_vt, name):
                self.push(self.output.side_effects.load_attr(fglobals_vt, name))
            else:
                try:
                    value = self.f_globals[name]
                except KeyError:
                    return self.load_builtin(inst)

                self.push(VariableBuilder(self, global_source)(value))

    def STORE_GLOBAL(self, inst):
        if self.f_globals is self.parent.f_globals:
            super().STORE_GLOBAL(inst)
        else:
            value = self.pop()
            if isinstance(value, RemovableHandleVariable):
                unimplemented("Storing handles in globals - NYI")
            name = inst.argval
            _, fglobals_vt, _ = self.get_globals_source_and_value(name)
            self.output.side_effects.store_attr(fglobals_vt, name, value)


class InliningGeneratorInstructionTranslator(InliningInstructionTranslator):
    generated_items: List[VariableTracker]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generated_items = []

    def YIELD_VALUE(self, inst: Instruction):
        self.generated_items.append(self.pop())
        if len(self.generated_items) > MAX_ITERATOR_LIMIT:
            unimplemented(
                "Too many yield values in generator. Maybe you are inlining an infinite generator. "
                f"If not, please report a bug at {PT2_ISSUE_TRACKER_URL}",
            )
        self.push(ConstantVariable.create(None))

    def GET_YIELD_FROM_ITER(self, inst):
        tos = self.stack[-1]
        if not isinstance(tos, ListIteratorVariable):
            self.pop()
            res = BuiltinVariable(iter).call_function(self, [tos], {})  # type: ignore[arg-type]
            self.push(res)

    def YIELD_FROM(self, inst):
        assert len(self.stack) >= 2
        val = self.pop()
        tos = self.stack[-1]
        if not (isinstance(val, ConstantVariable) and val.value is None):
            # invoke send
            # Unreachable code - if you hit this, you are implementing generator support and have
            # lifted the `unimplemented("generator")` in frame conversion. This codepath handles
            # subgenerator and lines up with this line in Python 3.10
            # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2599
            unimplemented("Unreachable sub-generator code")

        try:
            val = tos.next_variable(self)
        except (StopIteration, exc.ObservedUserStopIteration) as ex:
            if isinstance(ex, exc.ObservedUserStopIteration):
                exc.handle_observed_exception(self)

            # The iterator is exhausted. Stop the loop and return.
            self.pop()
            self.push(ConstantVariable.create(ex.value))
        else:
            self.push(val)
            # Add the value to yield into generated_items and replace the top of the stack with None
            self.YIELD_VALUE(inst)

            # Repeat the YIELD_FROM instruction in the next eval loop
            assert (
                isinstance(self.instruction_pointer, int)
                and self.instruction_pointer > 0
            )
            self.instruction_pointer -= 1

    def SEND(self, inst):
        assert len(self.stack) >= 2
        val = self.pop()
        tos = self.stack[-1]
        if isinstance(tos, ListIteratorVariable) or (
            isinstance(tos, UserDefinedObjectVariable)
            and isinstance(tos.value, collections.abc.Iterator)
        ):
            if isinstance(val, ConstantVariable) and val.value is None:
                try:
                    val = tos.next_variable(self)
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
                unimplemented("Unreachable sub-generator code")
        else:
            unimplemented(f"SEND {typestr(tos)}")
