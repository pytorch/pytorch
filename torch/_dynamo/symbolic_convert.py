import collections
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
import sys
import traceback
import types
import typing
import weakref
from collections.abc import Sized
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type
from unittest.mock import patch

import torch
import torch._logging
from torch._guards import Checkpointable, tracing, TracingContext

from . import (
    allowed_functions,
    config,
    exc,
    logging as torchdynamo_logging,
    side_effects,
    skipfiles,
    variables,
)
from .allowed_functions import is_allowed, is_builtin_constant
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
    Instruction,
    is_generator,
    unique_id,
)
from .code_context import code_context
from .codegen import PyCodegen
from .exc import ArgsMismatchError, BackendCompilerFailed, unimplemented, Unsupported
from .funcname_cache import get_funcname
from .guards import GuardBuilder
from .output_graph import GraphCompileReason, OutputGraph, OutputGraphState
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import ContinueExecutionCache, ReenterWith
from .source import (
    AttrSource,
    GetItemSource,
    GlobalSource,
    GlobalWeakRefSource,
    LocalSource,
)
from .utils import (
    counters,
    get_fake_value,
    get_instruction_source_311,
    graph_break_dup_warning_checker,
    istype,
    LazyString,
    proxy_args_kwargs,
)
from .variables.base import is_side_effect_safe, MutableLocal, typestr, VariableTracker
from .variables.builder import VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable, EnumVariable
from .variables.ctx_manager import (
    ContextWrappingVariable,
    GenericContextWrappingVariable,
    WithExitFunctionVariable,
)
from .variables.dicts import ConstDictVariable
from .variables.functions import (
    BaseUserFunctionVariable,
    NestedUserFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .variables.lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    SetVariable,
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
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
    supported_const_comparison_ops,
    supported_tensor_comparison_ops,
    SymNodeVariable,
    TensorVariable,
)
from .variables.torch import TorchVariable
from .variables.user_defined import (
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedObjectVariable,
    UserDefinedVariable,
)

log = logging.getLogger(__name__)
graph_break_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")
trace_call_log = torch._logging.getArtifactLogger(__name__, "trace_call")
trace_source_log = torch._logging.getArtifactLogger(__name__, "trace_source")


@functools.lru_cache(None)
def _step_logger():
    return torchdynamo_logging.get_step_logger(log)


@dataclasses.dataclass
class BlockStackEntry:
    target: Instruction
    stack_index: Optional[int] = None
    with_context: ContextWrappingVariable = None

    def can_restore(self):
        return self.with_context is not None

    def resume_fn(self):
        assert self.stack_index is not None
        if self.with_context and self.with_context.target_values:
            return ReenterWith(self.stack_index, tuple(self.with_context.target_values))
        else:
            return ReenterWith(self.stack_index)

    def exit(self, tx):
        return self.with_context.exit(tx)


class InstructionTranslatorGraphState(NamedTuple):
    output: OutputGraphState
    symbolic_locals: Dict[str, VariableTracker]
    stack: List[VariableTracker]
    block_stack: List[BlockStackEntry]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    next_instruction: Optional[Instruction]
    lineno: int

    def diff(self, other: "InstructionTranslatorGraphState") -> Optional[str]:
        for k in self._fields:
            if k == "output":
                return self.output.diff(other.output, prefix=f"{k}.")
            sv = getattr(self, k)
            ov = getattr(other, k)
            if sv != ov:
                return f"{k} mismatch: {sv} != {ov}"
        return None


def stack_op(fn: typing.Callable[..., object]):
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self: "InstructionTranslatorBase", inst: Instruction):
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
        # (PRECALL for Python 3.11+)
        current_instruction_pointer += 1
        inst = self.instructions[current_instruction_pointer]
        if inst.opname not in ("CALL_FUNCTION", "PRECALL"):
            return False

        # for Python 3.11+, PRECALL should be followed by CALL, then RAISE_VARARGS
        # for Python < 3.11, CALL_FUNCTION should be followed by RAISE_VARARGS
        current_instruction_pointer += 1
        if inst.opname == "PRECALL":
            current_instruction_pointer += 1
        inst = self.instructions[current_instruction_pointer]

    if inst.opname != "RAISE_VARARGS":
        return False

    self.push(ConstantVariable.create(error_msg))

    return True


def generic_jump(truth_fn: typing.Callable[[object], bool], push: bool):
    def inner(self: "InstructionTranslatorBase", inst: Instruction):
        value: VariableTracker = self.pop()
        self.output.guards.update(value.guards)
        if (
            config.rewrite_assert_with_torch_assert
            and _detect_and_normalize_assert_statement(self, truth_fn, push)
        ):
            error_msg: VariableTracker = self.pop()
            self.output.guards.update(error_msg.guards)
            # Skip over things like `assert True`
            if value.is_python_constant() and bool(value.as_python_constant()):
                self.jump(inst)
                return

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

            scalar_to_tensor_proxy = self.output.create_proxy(
                "call_function", torch.scalar_tensor, *proxy_args_kwargs((value,), {})
            )

            scalar_to_tensor = wrap_fx_proxy(
                self,
                scalar_to_tensor_proxy,
                example_value=get_fake_value(scalar_to_tensor_proxy.node, self),
                **VariableTracker.propagate([value]),
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
                push and self.push(value)
                self.jump(inst)
        elif (
            isinstance(value, (TensorVariable)) and self.should_compile_partial_graph()
        ):
            # compile a partial subgraph prefix then jump into user code
            if self.has_backedge():
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
                    f"generic_jump {typestr(value)}", [self.frame_summary()]
                ),
            )
            self.pop()

            if_next = self.create_call_resume_at(self.next_instruction)
            push and self.push(value)
            if_jump = self.create_call_resume_at(inst.target)

            self.output.add_output_instructions(
                [create_instruction(inst.opname, target=if_jump[0])] + if_next + if_jump
            )
        elif isinstance(value, NNModuleVariable):
            # Equivalent of "self.nn_module is not None"
            mod = self.output.get_submodule(value.module_key)
            if truth_fn(mod):
                push and self.push(value)
                self.jump(inst)
        elif isinstance(value, UserDefinedObjectVariable):
            x = value.var_getattr(self, "__bool__")
            # if __bool__ is missing, trying __len__ to infer a truth value.
            if x.is_python_constant() and x.as_python_constant() is None:
                x = value.var_getattr(self, "__len__")
            # __bool__ or __len__ is function
            if isinstance(x, UserMethodVariable):
                state = self.copy_graphstate()
                result = x.call_function(self, [], {})
                if isinstance(result, ConstantVariable) and isinstance(
                    result.value, (bool, int)
                ):
                    self.output.guards.update(result.guards)
                    if truth_fn(result.value):
                        push and self.push(value)
                        self.jump(inst)
                else:
                    # rollback to the state before the __bool__ or __len__ inline
                    self.restore_graphstate(state)
                    unimplemented(
                        "generic_jump on UserDefined with __bool__ returning non-constant"
                    )
            # __bool__ or __len__ is non-function or not existed in the user defined object
            else:
                if truth_fn(True):
                    push and self.push(value)
                    self.jump(inst)
        elif not isinstance(value, TensorVariable) and value.has_unpack_var_sequence(
            self
        ):
            if truth_fn(len(value.unpack_var_sequence(self))):
                push and self.push(value)
                self.jump(inst)
        elif isinstance(value, SymNodeVariable):
            eval_result = value.evaluate_expr(self.output)
            if truth_fn(eval_result):
                push and self.push(value)
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
            state = self.copy_graphstate()
            reason = None
            try:
                TracingContext.set_current_loc(
                    self.f_code.co_filename, self.lineno, self.f_code.co_name
                )
                return inner_fn(self, inst)
            except Unsupported as excp:
                if self.has_backedge() and self.should_compile_partial_graph():
                    msg = (
                        "Skipping frame because there is a graph break in a for/while loop\n"
                        f"{self.frame_summary()}"
                    )
                    log.info(msg)
                    raise exc.SkipFrame(msg) from excp

                if len(self.states_before_block) > 0:
                    # We don't support graph break under GenericContextWrappingVariable,
                    # If there is, we roll back to the checkpoint and fall back.
                    excp.remove_from_stats()
                    state = self.states_before_block.pop()
                    self.restore_graphstate(state)
                    ctx = state.stack[-1]
                    assert isinstance(ctx, GenericContextWrappingVariable)
                    unimplemented(f"Graph break under {ctx}")

                if isinstance(excp, exc.UncapturedHigherOrderOpError):
                    raise

                if not self.should_compile_partial_graph():
                    raise

                log.debug("break_graph_if_unsupported triggered compile", exc_info=True)

                user_stack = excp.real_stack
                # TODO: Also report the traceback from the parent frame
                user_stack_formatted = "".join(traceback.format_list(user_stack))
                frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                # torch._dynamo.explain() formats this a little nicer, and presents a slightly
                # more actionable user code pointer
                if (
                    graph_break_log.isEnabledFor(logging.DEBUG)
                    and not explain
                    and graph_break_dup_warning_checker.add(frame_loc)
                ):
                    graph_break_log.debug(
                        "Graph break: %s from user code at:\n%s",
                        excp,
                        user_stack_formatted,
                    )

                excp.remove_from_stats()
                excp.add_to_stats("graph_break")
                reason = GraphCompileReason(excp.msg, user_stack)
            self.restore_graphstate(state)

            self.output.compile_subgraph(self, reason=reason)
            cg = PyCodegen(self)
            cleanup: List[Instruction] = []
            # Reconstruct the context variables in the block stack
            for b in self.block_stack:
                self.output.add_output_instructions(
                    [
                        *b.with_context.reconstruct(cg),
                        *b.resume_fn().try_except(cg.code_options, cleanup),
                    ]
                )

            if sys.version_info >= (3, 11) and inst.opname == "CALL":
                kw_names = (
                    self.kw_names.as_python_constant()
                    if self.kw_names is not None
                    else ()
                )
                if len(kw_names) > 0:
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

            if sys.version_info >= (3, 11) and inst.opname == "CALL":
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


class InstructionTranslatorBase(Checkpointable[InstructionTranslatorGraphState]):
    output: OutputGraph
    symbolic_locals: Dict[str, VariableTracker]
    symbolic_globals: Dict[str, VariableTracker]
    stack: List[VariableTracker]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    next_instruction: Optional[Instruction]
    block_stack: List[BlockStackEntry]
    lineno: int
    kw_names: Optional[ConstantVariable]
    accept_prefix_inst: bool
    prefix_insts: List[Instruction]
    inline_depth: int

    checkpoint: Optional[Tuple[Instruction, InstructionTranslatorGraphState]]
    random_calls: List[
        Tuple[Callable[..., object], Tuple[object, ...], Dict[str, object]]
    ]

    def has_backedge(self):
        cur_offset = self.current_instruction.offset
        assert self.instruction_pointer is not None
        for inst in self.instructions[self.instruction_pointer :]:
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
        return self._cell_and_freevars

    def prune_dead_locals(self):
        reads = livevars_analysis(self.instructions, self.current_instruction)
        # implicit use by super()
        # reads = reads | {"__class__"}
        # output variables?
        reads = reads | set(self.cell_and_freevars())
        self.symbolic_locals = collections.OrderedDict(
            [(k, v) for k, v in self.symbolic_locals.items() if k in reads]
        )
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
        if (
            inner_fn
            and callable(inner_fn)
            and hasattr(inner_fn, "_dynamo_forbidden")
            and inner_fn._dynamo_forbidden
        ):
            raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
        self.push(fn.call_function(self, args, kwargs))

    def update_locals_and_stack(self, oldvar: VariableTracker, newvar: VariableTracker):
        def repl(v: VariableTracker):
            if v.mutable_local is oldvar.mutable_local:
                return newvar
            return v

        def skip(v: VariableTracker):
            return oldvar.mutable_local not in v.recursively_contains

        cache: Dict[int, Tuple[object, object]] = dict()
        self.output.side_effects.apply(repl, cache, skip_fn=skip)
        self.stack = [
            VariableTracker.apply(repl, x, cache, skip_fn=skip) for x in self.stack
        ]
        for k, x in self.symbolic_locals.items():
            self.symbolic_locals[k] = VariableTracker.apply(
                repl, x, cache, skip_fn=skip
            )

    def replace_all(self, oldvar: VariableTracker, newvar: VariableTracker):
        if isinstance(oldvar.mutable_local, side_effects.MutableSideEffects):
            newvar = self.output.side_effects.mutation(oldvar, newvar)
        else:
            assert isinstance(oldvar.mutable_local, variables.base.MutableLocal)
            newvar = newvar.clone(mutable_local=variables.base.MutableLocal())
        self.update_locals_and_stack(oldvar, newvar)
        return newvar

    def inline_user_function_return(self, fn, args, kwargs):
        """
        A call to some user defined function by inlining it.
        """
        state = self.copy_graphstate()
        try:
            result = InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
            self.output.guards.update(fn.guards)
            return result
        except Exception:
            self.restore_graphstate(state)
            raise

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

    def log_starts_line(self):
        trace_source_log.debug("%s", LazyString(self.get_log_starts_line_log_str))

    def step(self):
        """Process exactly one instruction, return False we should exit"""
        assert isinstance(self.instruction_pointer, int)
        inst = self.instructions[self.instruction_pointer]
        self.current_instruction = inst
        self.instruction_pointer += 1
        if self.instruction_pointer < len(self.instructions):
            self.next_instruction = self.instructions[self.instruction_pointer]
        else:
            self.instruction_pointer = None
            self.next_instruction = None
        if inst.starts_line and self.lineno != inst.starts_line:
            self.lineno = inst.starts_line
            self.log_starts_line()

        if len(self.stack) == 0 and self.should_compile_partial_graph():
            self.checkpoint = inst, self.copy_graphstate()

        log.debug("TRACE %s %s %s", inst.opname, inst.argval, self.stack)

        # 3.11 no longer uses a block stack, but we still keep track of one
        # so that we know which contexts are currently active.
        # For our purposes, all exception table entries with the same target
        # are considered to be part of the same "block".
        if sys.version_info >= (3, 11):
            entry = inst.exn_tab_entry
            if not (
                # still in the same block
                self.block_stack
                and entry
                and self.block_stack[-1].target is entry.target
            ):
                if not entry:
                    # no longer in any block
                    # It is possible for NOPs to be between two instructions
                    # in the same block, but the NOPs are not covered by an
                    # exception table entry. In this case, assume that we
                    # are still in the same block.
                    if self.block_stack and inst.opname != "NOP":
                        # If we really escape from a block and the current
                        # instruction is not in another block, then there
                        # should be no other nested blocks that we are in.
                        assert len(self.block_stack) == 1
                        self.block_stack.pop()
                elif (
                    # current instruction is in the previous block
                    len(self.block_stack) > 1
                    and self.block_stack[-2].target is entry.target
                ):
                    # exit the current block
                    self.block_stack.pop()
                else:
                    # current instruction is in a new block
                    # push block to stack - note, BEFORE_WITH blocks won't
                    # be pushed here since BEFORE_WITH pushes the block, and
                    # the current instruction would be counted as being in that block.
                    self.block_stack.append(
                        BlockStackEntry(entry.target, len(self.stack))
                    )

        try:
            if not hasattr(self, inst.opname):
                unimplemented(f"missing: {inst.opname}")
            TracingContext.set_current_loc(
                self.f_code.co_filename, self.lineno, self.f_code.co_name
            )
            getattr(self, inst.opname)(inst)

            return inst.opname != "RETURN_VALUE"
        except Unsupported:
            if self.empty_checkpoint():
                log.debug("empty checkpoint")
                raise

            log.debug("step triggered compile", exc_info=True)

        # generate code from checkpoint
        assert not self.output.output_instructions
        assert self.checkpoint is not None
        continue_inst, state = self.checkpoint
        self.restore_graphstate(state)
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
                while (
                    self.instruction_pointer is not None
                    and not self.output.should_exit
                    and self.step()
                ):
                    pass
            except BackendCompilerFailed:
                raise
            except Exception as e:
                if config.replay_record_enabled:
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

    def push(self, val: Optional[VariableTracker]):
        assert val is None or isinstance(
            val, VariableTracker
        ), f"push expects VariableTracker, got {typestr(val)}"
        self.stack.append(val)

    def push_many(self, vals: List[VariableTracker]):
        for val in vals:
            self.push(val)

    def pop(self) -> VariableTracker:
        return self.stack.pop()

    def popn(self, n: int) -> List[VariableTracker]:
        assert n >= 0
        return list(reversed([self.pop() for _ in range(n)]))

    def LOAD_FAST(self, inst):
        name = inst.argval

        if name in self.f_locals and config.replay_record_enabled:
            self.exec_recorder.add_local_var(name, self.f_locals[name])

        if name.startswith(".") and name not in self.symbolic_locals:
            # This happens in dict/list comprehensions
            name = name.replace(".", "implicit")
        assert name not in self.cell_and_freevars()
        if name not in self.symbolic_locals:
            unimplemented("undefined LOAD_FAST")
        self.push(self.symbolic_locals[name])
        if name.startswith("___stack"):
            self.symbolic_locals.pop(name)

    def LOAD_DEREF(self, inst):
        assert inst.argval in self.cell_and_freevars()

        if inst.argval in self.f_locals and config.replay_record_enabled:
            self.exec_recorder.add_local_var(inst.argval, self.f_locals[inst.argval])

        if inst.argval not in self.symbolic_locals:
            unimplemented(f"undefined LOAD_DEREF {inst.argval}")
        self.push(self.symbolic_locals[inst.argval])

    def STORE_FAST(self, inst):
        loaded_vt = self.pop()
        name = inst.argval
        loaded_vt = loaded_vt.rename(self, name)
        self.symbolic_locals[name] = loaded_vt

    def DELETE_FAST(self, inst):
        del self.symbolic_locals[inst.argval]

    STORE_DEREF = STORE_FAST

    def LOAD_CLOSURE(self, inst):
        self.push(ClosureVariable(name=inst.argval))

    def LOAD_CONST(self, inst):
        # For empty tuples, create empty TupleVariable
        if isinstance(inst.argval, tuple) and not inst.argval:
            self.push(TupleVariable([]))
        else:
            self.push(ConstantVariable.create(value=inst.argval))

    def get_global_source(self, name):
        if self.output.global_scope is self.f_globals:
            source = GlobalSource(name)
        else:
            if "__name__" in self.f_globals:
                source = AttrSource(
                    self.import_source(self.f_globals["__name__"]), name
                )
            else:
                mangled_name = f"___unnamed_scope_{id(self.f_globals)}"
                if mangled_name not in self.output.global_scope:
                    self.output.install_global(mangled_name, self.f_globals)
                source = GetItemSource(GlobalSource(mangled_name), name)
        return source

    def LOAD_GLOBAL(self, inst):
        if sys.version_info >= (3, 11):
            if inst.arg % 2:
                self.PUSH_NULL(inst)

        name = inst.argval

        if config.replay_record_enabled:
            if name in self.f_globals:
                self.exec_recorder.add_global_var(name, self.f_globals[name])
            else:
                assert name in self.f_builtins
                self.exec_recorder.builtins[name] = self.f_builtins[name]

        if inst.argval == "AssertionError":
            unimplemented("assert with non-string message")

        if name in self.symbolic_globals:
            variable = self.output.side_effects[self.symbolic_globals[name]]
            self.push(self.output.side_effects.load_global(variable, name))
            return

        try:
            value = self.f_globals[name]
        except KeyError:
            return self.load_builtin(inst)

        source = self.get_global_source(name)
        self.push(VariableBuilder(self, source)(value))

    def STORE_GLOBAL(self, inst):
        value = self.pop()
        name = inst.argval
        source = self.get_global_source(name)
        if name not in self.symbolic_globals:
            self.symbolic_globals[name] = object()  # sentinel object
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
            value = __import__(
                module_name,
                fromlist=fromlist,
                level=level,
                globals=self.f_globals,
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

        if config.replay_record_enabled:
            self.exec_recorder.add_local_mod(recorded_name, value)

        if is_allowed(value):
            self.push(TorchVariable(value, source=source))
        elif istype(value, (types.ModuleType, DummyModule)):
            self.push(PythonModuleVariable(value, source=source))
        else:
            unimplemented(f"IMPORT_NAME {typestr(value)}")

    def IMPORT_FROM(self, inst):
        self.DUP_TOP(inst)
        self.LOAD_ATTR(inst)

    def load_builtin(self, inst):
        if inst.argval not in self.f_builtins:
            raise NameError(f"name '{inst.argval}' is not defined")
        val = self.f_builtins[inst.argval]

        if callable(val):
            self.push(VariableBuilder(self, GlobalSource(inst.argval))(val))
        else:
            assert is_builtin_constant(val)
            self.push(ConstantVariable.create(value=val))

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
        self.block_stack.append(BlockStackEntry(inst.target))

    def SETUP_EXCEPT(self, inst):
        # only exists in python<=3.7
        self.block_stack.append(BlockStackEntry(inst.target))

    def POP_BLOCK(self, inst):
        self.block_stack.pop()

    def SETUP_WITH(self, inst):
        self.setup_or_before_with(inst)

    def SETUP_FINALLY(self, inst):
        self.block_stack.append(BlockStackEntry(inst.target))

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

    def END_FINALLY(self, inst):
        tos = self.pop()
        assert tos is None

    def POP_FINALLY(self, inst):
        preserve_tos = inst.argval
        if preserve_tos:
            tos = self.pop()
        assert self.pop() is None
        if preserve_tos:
            self.push(tos)

    def FOR_ITER(self, inst):
        it = self.pop()
        if isinstance(it, ListIteratorVariable):
            self.output.guards.update(it.guards)
            try:
                val, next_iter = it.next_variables()
                self.replace_all(it, next_iter)
                self.push(next_iter)
                self.push(val)
            except StopIteration:
                self.jump(inst)
        else:
            unimplemented(f"FOR_ITER {typestr(it)}")

    def COMPARE_OP(self, inst):
        left, right = self.popn(2)
        left = left.as_specialized(self)
        right = right.as_specialized(self)
        options = VariableTracker.propagate([left, right])
        op = inst.argval
        supported_any = dict(
            itertools.chain(
                supported_tensor_comparison_ops.items(),
                supported_const_comparison_ops.items(),
            )
        )
        if (
            isinstance(
                left,
                (
                    TensorVariable,
                    SymNodeVariable,
                    NNModuleVariable,
                    BaseListVariable,
                    UserDefinedVariable,
                    BaseUserFunctionVariable,
                    ConstDictVariable,
                ),
            )
            and isinstance(right, ConstantVariable)
            and right.value is None
            and op in supported_const_comparison_ops
        ):
            # <non-None> is None
            self.push(
                ConstantVariable.create(
                    supported_const_comparison_ops[op](object(), right.value), **options
                )
            )

        elif (
            left.is_python_constant()
            and right.is_python_constant()
            and op in supported_any
        ):
            # constant fold
            self.push(
                ConstantVariable.create(
                    supported_any[op](
                        left.as_python_constant(), right.as_python_constant()
                    ),
                    **options,
                )
            )
        elif op in ("in", "not in"):
            self.push(right.call_method(self, "__contains__", [left], {}))
            if op == "not in":
                self.UNARY_NOT(inst)
        else:
            self.push(
                BuiltinVariable(supported_any[op], **options).call_function(
                    self, [left, right], {}
                )
            )

    def GET_ITER(self, inst):
        self.call_function(BuiltinVariable(iter), [self.pop()], {})

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION(self, inst):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION_EX(self, inst):
        if inst.argval == 0:
            kwargsvars = ConstDictVariable({}, dict)
            argsvars = self.pop()
        elif inst.argval == 1:
            kwargsvars = self.pop()
            argsvars = self.pop()
        else:
            unimplemented("CALL_FUNCTION_EX")
        fn = self.pop()
        if sys.version_info >= (3, 11):
            null = self.pop()
            assert isinstance(null, NullVariable)
        self.output.guards.update(argsvars.guards)
        self.output.guards.update(kwargsvars.guards)

        if (
            isinstance(fn, GetAttrVariable)
            and isinstance(fn.obj, TensorVariable)
            and fn.name == "view"
            and isinstance(argsvars, (ConstantVariable, TensorVariable))
        ):
            # Hack to handle special case in some bert models.  Converts
            # x.view(*shape) into x.view(shape), which is correct for view()
            # but not generally.  See test_transpose_for_scores().
            argsvars = TupleVariable([argsvars])

        if not isinstance(
            argsvars, BaseListVariable
        ) and argsvars.has_unpack_var_sequence(self):
            argsvars = TupleVariable(argsvars.unpack_var_sequence(self))

        if not isinstance(argsvars, BaseListVariable) or not isinstance(
            kwargsvars, ConstDictVariable
        ):
            unimplemented(f"non-static call {typestr(argsvars)} {typestr(kwargsvars)}")

        self.call_function(fn, argsvars.items, kwargsvars.items)

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

    def LOAD_METHOD(self, inst):
        self.LOAD_ATTR(inst)
        obj = self.pop()
        if sys.version_info >= (3, 11):
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

    def LOAD_ATTR(self, inst):
        obj = self.pop()
        result = BuiltinVariable(getattr).call_function(
            self, [obj, ConstantVariable.create(inst.argval)], {}
        )
        self.push(result)

    def STORE_ATTR(self, inst):
        prior = self.copy_graphstate()
        val, obj = self.popn(2)

        if isinstance(obj, NNModuleVariable):
            # We don't allow side effects during export
            # https://github.com/pytorch/torchdynamo/issues/1475
            assert (
                not self.export
            ), f"Mutating module attribute {inst.argval} during export."

        try:
            self.output.guards.update(
                BuiltinVariable(setattr)
                .call_function(
                    self, [obj, ConstantVariable.create(inst.argval), val], {}
                )
                .guards
            )
            return
        except Unsupported as e:
            if not self.should_compile_partial_graph():
                raise
            log.debug("STORE_ATTR triggered compile", exc_info=True)
            e.remove_from_stats()
            e.add_to_stats("graph_break")
            self.restore_graphstate(prior)

        # break the graph
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
        self.output.guards.update(
            BuiltinVariable(delattr)
            .call_function(self, [obj, ConstantVariable.create(inst.argval)], {})
            .guards
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
        result = obj.call_method(self, "__setitem__", [key, val], {})
        # no result is pushed, so need to lift the guards to global
        self.output.guards.update(result.guards)

    def BUILD_TUPLE(self, inst):
        items = self.popn(inst.argval)
        options = VariableTracker.propagate(items)
        self.push(TupleVariable(items, **options))

    def BUILD_SLICE(self, inst):
        items = self.popn(inst.argval)
        options = VariableTracker.propagate(items)
        self.push(
            SliceVariable(
                [x.as_specialized(self) for x in items],
                **options,
            )
        )

    def BUILD_LIST(self, inst):
        items = self.popn(inst.argval)
        options = VariableTracker.propagate(items)
        self.push(ListVariable(items, mutable_local=MutableLocal(), **options))

    def BUILD_SET(self, inst):
        if config.inject_BUILD_SET_unimplemented_TESTING_ONLY:
            unimplemented("missing: BUILD_SET")
        items = self.popn(inst.argval)
        options = VariableTracker.propagate(items)
        new_set = SetVariable(items, mutable_local=MutableLocal(), **options)
        self.push(new_set)

    def BUILD_LIST_UNPACK(self, inst, cls=ListVariable):
        seqs = self.popn(inst.argval)
        options = VariableTracker.propagate(seqs)
        items = list()
        for seq in seqs:
            try:
                items.extend(seq.unpack_var_sequence(self))
            except NotImplementedError:
                unimplemented(f"BUILD_LIST_UNPACK {seq}")
        self.push(cls(items, mutable_local=MutableLocal(), **options))

    def BUILD_TUPLE_UNPACK(self, inst):
        self.BUILD_LIST_UNPACK(inst, cls=TupleVariable)

    BUILD_TUPLE_UNPACK_WITH_CALL = BUILD_TUPLE_UNPACK

    def BUILD_MAP(self, inst):
        items = self.popn(inst.argval * 2)
        options = VariableTracker.propagate(items)
        result = dict()
        for k, v in zip(items[::2], items[1::2]):
            assert (
                isinstance(k, (ConstantVariable, EnumVariable, BuiltinVariable))
                or (isinstance(k, TensorVariable) and k.specialized_value is not None)
                or k.is_python_constant()
            )

            result[ConstDictVariable.get_key(k)] = v
        assert len(result) == len(items) / 2
        self.push(
            ConstDictVariable(result, dict, mutable_local=MutableLocal(), **options)
        )

    def BUILD_MAP_UNPACK(self, inst):
        items = self.popn(inst.argval)
        # ensure everything is a dict
        items = [BuiltinVariable(dict).call_function(self, [x], {}) for x in items]
        result = dict()
        for x in items:
            assert isinstance(x, ConstDictVariable)
            result.update(x.items)
        self.push(
            ConstDictVariable(
                result,
                dict,
                mutable_local=MutableLocal(),
                **VariableTracker.propagate(items),
            )
        )

    BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK

    def BUILD_CONST_KEY_MAP(self, inst):
        keys = self.pop()
        values = self.popn(inst.argval)
        options = VariableTracker.propagate([keys] + values)
        assert isinstance(keys, TupleVariable)
        assert keys.is_python_constant()
        keys = keys.as_python_constant()
        assert istype(keys, tuple)
        assert len(keys) == len(values)
        self.push(
            ConstDictVariable(
                dict(zip(keys, values)),
                dict,
                mutable_local=MutableLocal(),
                **options,
            )
        )

    def MAP_ADD(self, inst):
        k, v = self.popn(2)
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ConstDictVariable)
        assert obj.mutable_local
        items = dict(obj.items)
        items[k.as_python_constant()] = v
        self.replace_all(
            obj,
            ConstDictVariable(
                items,
                obj.user_cls,
                **VariableTracker.propagate([obj, k, v]),
            ),
        )

    def SET_ADD(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, SetVariable)
        assert obj.mutable_local
        return obj.call_method(self, "add", [v], {})

    def LIST_APPEND(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ListVariable)
        assert obj.mutable_local
        # only copy if the new obj contains other mutables
        new_rec_contains = obj.recursively_contains
        if v.recursively_contains or v.mutable_local:
            new_rec_contains = obj.recursively_contains.union(v.recursively_contains)

            if v.mutable_local:
                new_rec_contains.add(v.mutable_local)

        self.replace_all(
            obj,
            ListVariable(
                obj.items + [v],
                recursively_contains=new_rec_contains,
                regen_guards=False,
                **VariableTracker.propagate([obj, v]),
            ),
        )

    def MAKE_FUNCTION(self, inst):
        flags = inst.arg
        old_stack = list(self.stack)
        if sys.version_info < (3, 11):
            fn_name = self.pop()
        code = self.pop()
        if sys.version_info >= (3, 11):
            # MAKE_FUNCTION behavior actually changed in 3.11, see
            # https://github.com/python/cpython/pull/93189/
            assert hasattr(code.value, "co_qualname")
            fn_name = ConstantVariable.create(value=code.value.co_qualname)
        defaults = None
        closure = None
        annotations = None
        kwdefaults = None

        if flags & 0x08:
            closure = self.pop()
        if flags & 0x04:
            annotations = self.pop()
        if flags & 0x02:
            kwdefaults = self.pop()
        if flags & 0x01:
            defaults = self.pop()

        options = VariableTracker.propagate(old_stack[len(self.stack) :])
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
                **options,
            )
        )

    def UNPACK_SEQUENCE(self, inst):
        seq = self.pop()
        if isinstance(seq, (BaseListVariable, SetVariable)):
            self.output.guards.update(seq.guards)
            val = seq.unpack_var_sequence(self)
        elif seq.is_python_constant() and isinstance(seq, ConstantVariable):
            val = seq.unpack_var_sequence(self)
        elif isinstance(seq, TensorVariable):
            val = seq.unpack_var_sequence(self, idxes=range(inst.argval))
        elif isinstance(seq, GetAttrVariable) and isinstance(seq.obj, TensorVariable):
            # x, y = a.shape
            proxy = getattr(seq.obj.as_proxy(), seq.name)
            options = VariableTracker.propagate(self)
            val = [wrap_fx_proxy(self, proxy[i], **options) for i in range(inst.argval)]
        else:
            unimplemented(f"UNPACK_SEQUENCE {seq}")
        assert len(val) == inst.argval
        for i in reversed(val):
            self.push(i)

    def UNPACK_EX(self, inst):
        assert 0 <= inst.argval <= 0xFFFF
        prefix = inst.argval & 0xFF  # low byte
        suffix = inst.argval >> 8  # high byte
        seq = self.pop()
        options = VariableTracker.propagate(seq)
        if seq.has_unpack_var_sequence(self):
            vals = list(seq.unpack_var_sequence(self))
            assert len(vals) >= prefix + suffix
            vals_prefix = vals[:prefix]
            vals_list = vals[prefix : len(vals) - suffix]
            vals_suffix = vals[len(vals) - suffix :]
            for item in reversed(vals_suffix):
                self.push(item.add_options(options))
            self.push(TupleVariable(vals_list, **options))
            for item in reversed(vals_prefix):
                self.push(item.add_options(options))
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
            value = ConstantVariable.create(str(value.sym_num))
        if (flags & 0x03) == 0x01:
            value = BuiltinVariable(str).call_function(self, [value], {})
        elif (flags & 0x03) == 0x02:
            value = BuiltinVariable(repr).call_function(self, [value], {})
        elif (flags & 0x03) == 0x03:
            value = BuiltinVariable(ascii).call_function(self, [value], {})

        fmt_var = ConstantVariable.create(
            "{:" + fmt_spec.as_python_constant() + "}"
        ).add_options(fmt_spec)

        self.call_function(BuiltinVariable(str.format), [fmt_var, value], {})

    def BUILD_STRING(self, inst):
        result = ""
        for _ in range(inst.arg):
            str_var = self.pop()
            assert isinstance(str_var, ConstantVariable)
            result = str_var.value + result
        self.push(ConstantVariable.create(value=result))

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
        self.push(BuiltinVariable(tuple).call_function(self, [self.pop()], {}))

    def DICT_MERGE(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
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
        assert tos.is_python_constant()
        keys = tos.as_python_constant()
        tos1 = self.stack[-2]
        assert isinstance(tos1, ConstDictVariable)
        match_obj = tos1.items
        if all(key in match_obj for key in keys):
            self.push(TupleVariable([match_obj[key] for key in keys]))
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(None))
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(False))

    def LOAD_ASSERTION_ERROR(self, inst):
        unimplemented("assert with non-string message")

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

    def BINARY_OP(self, inst):
        if sys.version_info >= (3, 11):
            opname = dis._nb_ops[inst.arg][0][3:]
            if opname.startswith("INPLACE"):
                return getattr(self, "INPLACE_" + opname[8:])(inst)
            return getattr(self, "BINARY_" + opname)(inst)
        else:
            unimplemented("BINARY_OP requires Python 3.11+")

    def PRECALL(self, inst):
        pass

    def KW_NAMES(self, inst):
        kw_names = self.code_options["co_consts"][inst.arg]
        assert isinstance(kw_names, tuple)
        for name in kw_names:
            assert isinstance(name, str)
        assert self.kw_names is None
        self.kw_names = ConstantVariable.create(value=kw_names)

    def PUSH_NULL(self, inst):
        self.push(NullVariable())

    @break_graph_if_unsupported(push=1)
    def CALL(self, inst):
        # see https://docs.python.org/3.11/library/dis.html#opcode-CALL
        # for convention
        contents = self.popn(inst.arg + 2)
        if isinstance(contents[0], NullVariable):
            fn = contents[1]
            args = []
        else:
            fn = contents[0]
            args = [contents[1]]
        kw_names = self.kw_names.value if self.kw_names else ()
        if kw_names:
            args = args + contents[2 : -len(kw_names)]
            kwargs_list = contents[-len(kw_names) :]
            kwargs = dict(zip(kw_names, kwargs_list))
            assert len(kwargs) == len(kw_names)
        else:
            args = args + contents[2:]
            kwargs = {}
        self.call_function(fn, args, kwargs)
        self.kw_names = None

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
        state = self.copy_graphstate()
        ctx = self.pop()
        if not isinstance(ctx, ContextWrappingVariable):
            unimplemented(f"{inst.opname} {ctx}")

        if isinstance(ctx, GenericContextWrappingVariable):
            # Save the checkpoint to restore if there is
            # graph break under the GenericContextWrappingVariable.
            self.states_before_block.append(state)

        self.output.guards.update(ctx.guards)

        exit = WithExitFunctionVariable(
            ctx,
            inst.target,
            **VariableTracker.propagate(ctx),
        )
        if sys.version_info >= (3, 11):
            # see create_call_resume_at for block stack details
            assert self.next_instruction
            assert self.next_instruction.exn_tab_entry
            target = self.next_instruction.exn_tab_entry.target
        else:
            target = inst.target
        if isinstance(self, InstructionTranslator):
            self.block_stack.append(BlockStackEntry(target, len(self.stack), ctx))
        else:
            self.block_stack.append(BlockStackEntry(target))

        self.push(exit)
        self.push(ctx.enter(self))

    def append_prefix_inst(self, inst):
        assert self.accept_prefix_inst
        self.prefix_insts.append(inst)

    def MAKE_CELL(self, inst):
        self.append_prefix_inst(inst)

    def COPY_FREE_VARS(self, inst):
        self.append_prefix_inst(inst)

    def RETURN_GENERATOR(self, inst):
        self.append_prefix_inst(inst)

    def copy_graphstate(self) -> InstructionTranslatorGraphState:
        """Create a checkpoint of the current state by copying everything"""
        return InstructionTranslatorGraphState(
            self.output.copy_graphstate(),
            collections.OrderedDict(self.symbolic_locals),
            list(self.stack),
            list(self.block_stack),
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
            self.lineno,
        )

    def restore_graphstate(self, state: InstructionTranslatorGraphState):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            output_state,
            self.symbolic_locals,
            self.stack,
            self.block_stack,
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
            self.lineno,
        ) = state
        self.output.restore_graphstate(output_state)

    def empty_checkpoint(self):
        if self.checkpoint is None:
            return True
        output_graphstate = self.checkpoint[1][0]
        graphstate = self.checkpoint[1][1:]
        state = (*output_graphstate, *graphstate)
        for obj in state:
            if isinstance(obj, Sized):
                if len(obj) != 0:
                    return False
        return True

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

    def store_global_weakref(self, name, value):
        self.output.guards.add(
            GlobalWeakRefSource(name).make_guard(GuardBuilder.WEAKREF_ALIVE)
        )
        if name not in self.output.global_scope:
            self.output.install_global(name, weakref.ref(value))

    @property
    def fake_mode(self):
        return self._fake_mode

    def find_symbolic_locals_name(self, tensor_variable):
        for key, value in self.symbolic_locals.items():
            if value is tensor_variable:
                return key
        return None

    @contextlib.contextmanager
    def strict_translation_mode(self):
        self.strict_checks_enabled = True
        try:
            yield
        finally:
            self.strict_checks_enabled = False

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
        f_code: types.CodeType,
        export: bool,
        inline_depth: int,
    ):
        super().__init__()

        # Mutable state checkpointed by copy_graphstate()
        self.output = output
        self.symbolic_locals = symbolic_locals
        self.symbolic_globals = symbolic_globals
        self.stack = []
        self.instruction_pointer = 0
        self.current_instruction = create_instruction("NOP")
        self.next_instruction = None
        self.block_stack = []
        # states before SETUP_WITH for checkpointing and fallback
        self.states_before_block: List[InstructionTranslatorGraphState] = []
        self.lineno = code_options["co_firstlineno"]
        self.kw_names = None
        self.accept_prefix_inst = True
        self.prefix_insts = []

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
        self.exec_recorder = ExecutionRecorder(code=f_code, code_options=code_options)
        # Stack of module being parsed, current nn.module is at the end of ordered dict.
        # The first field of tuple is the fully qualified name of current module
        # in original hierarchy.  The second field is the type of current nn.module
        self.nn_module_stack: Dict[str, Tuple[str, Type[Any]]] = {}
        # Flag to indicate whether tracing is used for export.
        self.export = export

        self._fake_mode = output.tracing_context.fake_mode

        self.checkpoint = None
        self.random_calls = []

        self.strict_checks_enabled = False

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
        linecache.lazycache(f_code.co_filename, f_globals)
        self.log_starts_line()


class InstructionTranslator(InstructionTranslatorBase):
    mutated_closure_cell_contents: Set[str]

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
    ):
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
            symbolic_locals=collections.OrderedDict(),  # set below
            # A global var is inserted only after a STORE_GLOBAL happens to it
            symbolic_globals=collections.OrderedDict(),
            f_code=f_code,
            export=export,
            inline_depth=0,
        )

        # as soon as we create the tracing context we should keep it active, so any calls
        # into dynamo apis can rely on finding it
        with tracing(self.output.tracing_context):
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

            self.symbolic_locals = collections.OrderedDict(
                (
                    k,
                    VariableBuilder(
                        self,
                        LocalSource(k, cell_or_freevar=k in cells_and_freevars_set),
                    )(f_locals[k]),
                )
                for k in vars
                if k in f_locals
            )

            self.init_local_index_guards_hack()

            self._freevars_ids = dict()
            for name in self.code_options["co_freevars"]:
                if name in f_locals:
                    self._freevars_ids[name] = id(f_locals[name])

    def init_local_index_guards_hack(self):
        # symbolic_locals contains the mapping from original f_locals to the
        # Variable objects. During the Variable building phase, each object also
        # has its associated guards. At the end, we will accumulate these
        # guards.
        #
        # One way of handling these guards is to just accumulate all of them
        # right now. However, many f_locals might not be used in the frame and
        # thus can unnecessarily increase guard execution overhead.  Therefore,
        # we selectively update output.guards as we run the Python Bytecode
        # instruction by instruction.
        #
        # An exception here is list/dict variables. Guards related to these
        # variables have indexed access, like Tensor_match on args[0], and if
        # args is not used in this frame, we will miss a LIST_LENGTH check like
        # len(args) == 2. Missing the LIST_LENGTH check causes problem for the
        # next invocation when args is not a list, and args[0] is a runtime
        # error. Therefore, we recursively add guards for list/dict variable here.
        for val in self.symbolic_locals.values():
            if isinstance(
                val, (ListIteratorVariable, BaseListVariable, ConstDictVariable)
            ):
                local_guards = VariableTracker.propagate(val)["guards"]
                index_guards = [
                    guard
                    for guard in local_guards
                    if guard.create_fn
                    in (
                        GuardBuilder.LIST_LENGTH,
                        GuardBuilder.DICT_KEYS,
                        GuardBuilder.ODICT_KEYS,
                        GuardBuilder.TUPLE_ITERATOR_LEN,
                    )
                ]
                self.output.guards.update(index_guards)

    def run(self):
        super().run()

    def match_nested_cell(self, name, cell):
        """Match a cell in this method to one in a function we are inlining"""
        value = cell.cell_contents
        # TODO(jansel): check the id of the cell rather than the contents
        if id(value) != self._freevars_ids.get(name):
            return None
        return self.symbolic_locals[name]

    def should_compile_partial_graph(self):
        return all(b.can_restore() for b in self.block_stack) and not self.one_graph

    def create_call_resume_at(self, inst):
        self.instruction_pointer = None

        if inst.opname == "RETURN_VALUE":
            return [create_instruction("RETURN_VALUE")]

        reads = livevars_analysis(self.instructions, inst)
        argnames = tuple(
            k
            for k in self.symbolic_locals.keys()
            if k in reads and k not in self.cell_and_freevars()
        )

        cg = PyCodegen(self)

        # Python does not allow null to be an arg to a function, so
        # we remove nulls from the stack and restore them in the
        # prologue of the resume function

        # sorted list of indices of nulls on the stack
        null_idxes: List[int] = []
        if sys.version_info >= (3, 11):
            # find indices of NullVariables
            for i, var in enumerate(self.stack):
                if isinstance(var, NullVariable):
                    null_idxes.append(i)
            # generate bytecode to pop the nulls
            null_cnt = 0
            for i, var in enumerate(reversed(self.stack)):
                if isinstance(var, NullVariable):
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
            tuple(b.resume_fn() for b in self.block_stack),
            tuple(null_idxes),
        )

        # Add original GraphModule context to the resume function to handle
        # the case of a graph break while tracing a GraphModule
        orig_graphmodule_maybe = code_context.get_context(self.f_code).get(
            "orig_graphmodule", None
        )
        if orig_graphmodule_maybe is not None:
            code_context.get_context(new_code)[
                "orig_graphmodule"
            ] = orig_graphmodule_maybe

        if new_code.co_freevars:
            cg.make_function_with_closure(name, new_code, True, stack_len)
        else:
            self.output.install_global(
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

    def RETURN_VALUE(self, inst):
        if (
            self.output.count_calls() == 0
            and not self.symbolic_locals_contain_module_class()
            and not self.export
        ):
            raise exc.SkipFrame("because no content in function call")
        self.instruction_pointer = None
        _step_logger()(
            logging.INFO,
            f"torchdynamo done tracing {self.f_code.co_name} (RETURN_VALUE)",
        )
        log.debug("RETURN_VALUE triggered compile")
        self.output.compile_subgraph(
            self,
            reason=GraphCompileReason(
                "return_value", [self.frame_summary()], graph_break=False
            ),
        )
        self.output.add_output_instructions([create_instruction("RETURN_VALUE")])


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

        if func.get_name() == "patched_init":
            unimplemented("Patched init cannot be inlined.")

        try:
            if id(func.get_function()) in allowed_functions._disallowed_function_ids:
                unimplemented(f"inlining disallowed: {func.get_function()}")
        except NotImplementedError:
            pass  # closures

        result = skipfiles.check_verbose(func.get_filename(), extra_check=True)
        if result.skipped:
            from torch._dynamo.variables.misc import (
                produce_trampoline_autograd_apply,
                produce_trampoline_autograd_bwd,
                produce_trampoline_autograd_fwd,
            )

            # _origin marks this as coming from an internal dynamo known function that is safe to
            # trace through.
            if hasattr(func.fn, "_origin") and func.fn._origin in [
                produce_trampoline_autograd_fwd,
                produce_trampoline_autograd_apply,
                produce_trampoline_autograd_bwd,
            ]:
                # Known sound
                return skipfiles.SkipResult(False, "allowlist in dynamo known function")
            unimplemented(
                f"'inline in skipfiles: {func.fn.__qualname__} | {func.get_name()} {func.get_filename()}, {result.reason}'"
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
            raise ArgsMismatchError(
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
            args is not None
            and len(args) > 0
            and isinstance(args[0], variables.CustomizedDictVariable)
        ):
            unimplemented(f"inline {code.co_name}")

        suffix = ""
        # TODO: mlazos, add support for enabling multiple artifact logs
        # with a single alias
        if torch._logging._internal.log_state.is_artifact_enabled("output_code"):
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
                ] = module

        tracer: InliningInstructionTranslator
        if is_generator(code):
            tracer = InliningGeneratorInstructionTranslator(
                parent, code, sub_locals, parent.symbolic_globals, closure_cells, func
            )
        else:
            tracer = InliningInstructionTranslator(
                parent, code, sub_locals, parent.symbolic_globals, closure_cells, func
            )

        strict_ctx: Any = contextlib.nullcontext()
        if parent.strict_checks_enabled:
            strict_ctx = tracer.strict_translation_mode()
        try:
            with strict_ctx:
                tracer.run()
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

        log.debug("DONE INLINING %s", code)

        if is_generator(code):
            assert isinstance(tracer, InliningGeneratorInstructionTranslator)
            assert tracer.symbolic_result.as_python_constant() is None
            return ListIteratorVariable(
                tracer.generated_items,
                mutable_local=MutableLocal(),
                **VariableTracker.propagate(tracer.symbolic_result),
            )
        else:
            return tracer.symbolic_result

    def __init__(
        self,
        parent: InstructionTranslatorBase,
        code: types.CodeType,
        symbolic_locals: Dict[str, VariableTracker],
        symbolic_globals: Dict[str, VariableTracker],
        closure_cells: Dict[str, VariableTracker],
        funcvar: BaseUserFunctionVariable,
    ):
        f_globals = funcvar.get_globals()
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
            instructions=instructions,
            code_options={k: getattr(code, k) for k in dir(code)},
            f_code=code,
            export=parent.export,
            inline_depth=parent.inline_depth + 1,
        )
        self.parent = parent
        self.symbolic_result = None
        self.closure_cells = closure_cells
        self.nn_module_stack = parent.nn_module_stack.copy()

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
                    raise exc.RestartAnalysis()
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

    def LOAD_CLOSURE(self, inst):
        assert inst.argval in self.cell_and_freevars()
        if inst.argval in self.closure_cells:
            self.push(self.closure_cells[inst.argval])
        else:
            self.push(InlinedClosureVariable(name=inst.argval))

    def check_replace_is_safe(self, oldvar):
        if not is_side_effect_safe(oldvar.mutable_local):
            unimplemented(
                "HigherOrderOperator: Mutating a variable not in the current scope (replace_all)"
            )

    def replace_all(self, oldvar: VariableTracker, newvar: VariableTracker):
        self.check_replace_is_safe(oldvar)
        newvar = super().replace_all(oldvar, newvar)
        # recursively check and update parent's locals and stack in case oldvar is from parent
        translator: InstructionTranslatorBase = self
        while hasattr(translator, "parent"):
            translator = translator.parent  # type: ignore[attr-defined]
            translator.update_locals_and_stack(oldvar, newvar)
        return newvar

    def should_compile_partial_graph(self):
        return False  # inlining functions is all-or-nothing

    def create_call_resume_at(self, offset):
        unimplemented("cant resume while inlining")

    def RETURN_VALUE(self, inst):
        self.symbolic_result = self.pop()
        self.instruction_pointer = None


class InliningGeneratorInstructionTranslator(InliningInstructionTranslator):
    generated_items: List[VariableTracker]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_items = []

    def YIELD_VALUE(self, inst: Instruction):
        self.generated_items.append(self.pop())
        # TODO(jansel): figure out why this is needed, it isn't in the docs for YIELD_VALUE
        self.push(ConstantVariable.create(None))

    def GET_YIELD_FROM_ITER(self, inst):
        tos = self.stack[-1]
        if not isinstance(tos, ListIteratorVariable):
            self.pop()
            res = BuiltinVariable(iter).call_function(self, [tos], {})
            self.push(res)
        return self.YIELD_FROM(inst)

    def YIELD_FROM(self, inst):
        while True:
            tos = self.stack[-1]
            if isinstance(tos, ConstantVariable) and tos.value is None:
                self.pop()
                return
            if isinstance(tos, ListIteratorVariable):
                self.output.guards.update(tos.guards)
                try:
                    val, next_iter = tos.next_variables()
                    self.replace_all(tos, next_iter)
                    self.push(val)
                    # TODO(voz): Unclear if we need the push None in YIELD_VALUE?
                    self.YIELD_VALUE(inst)
                    self.pop()
                    self.push(next_iter)
                except StopIteration:
                    return
            else:
                unimplemented(f"YIELD_FROM {typestr(tos)}")

    def SEND(self, inst):
        assert len(self.stack) >= 2
        val = self.pop()
        tos = self.stack[-1]
        if isinstance(tos, ListIteratorVariable):
            if isinstance(val, ConstantVariable) and val.value is None:
                self.push(val)
                self.instruction_pointer = self.indexof[inst.target]
            else:
                # invoke send
                # Unreachable code - if you hit this, you are implementing generator support and have
                # lifted the `unimplemented("generator")` in frame conversion. This codepath handles
                # subgenerator and lines up with this line in Python 3.11
                # https://github.com/python/cpython/blob/3.11/Python/ceval.c#L2597
                unimplemented("Unreachable sub-generator code")
        else:
            unimplemented(f"SEND {typestr(tos)}")
