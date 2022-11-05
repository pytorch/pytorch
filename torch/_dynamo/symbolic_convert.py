import collections
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import logging
import operator
import sys
import traceback
import types
import typing
import weakref
from typing import Any, Dict, Iterable, List
from unittest.mock import patch

import torch

from . import (
    allowed_functions,
    config,
    exc,
    logging as torchdynamo_logging,
    side_effects,
    skipfiles,
    variables,
)
from .allowed_functions import is_allowed, is_builtin_callable, is_builtin_constant
from .bytecode_analysis import livevars_analysis
from .bytecode_transformation import (
    cleaned_instructions,
    create_instruction,
    Instruction,
    is_generator,
    unique_id,
)
from .codegen import PyCodegen
from .exc import BackendCompilerFailed, unimplemented, Unsupported
from .guards import GuardBuilder
from .output_graph import GraphCompileReason, OutputGraph
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
    fake_tensors_available,
    graph_break_dup_warning_checker,
    istype,
)
from .variables.base import MutableLocal, typestr, VariableTracker
from .variables.builder import VariableBuilder
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable
from .variables.dicts import ConstDictVariable
from .variables.functions import (
    BaseUserFunctionVariable,
    NestedUserFunctionVariable,
    UserFunctionVariable,
)
from .variables.lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    SliceVariable,
    TupleVariable,
)
from .variables.misc import (
    ClosureVariable,
    ContextWrappingVariable,
    GetAttrVariable,
    GradModeVariable,
    PythonModuleVariable,
    UnknownVariable,
    WithExitFunctionVariable,
)
from .variables.nn_module import NNModuleVariable
from .variables.tensor import DynamicShapeVariable, TensorVariable
from .variables.torch import TorchVariable
from .variables.user_defined import UserDefinedVariable

log = logging.getLogger(__name__)


@functools.lru_cache(None)
def _step_logger():
    return torchdynamo_logging.get_step_logger(log)


@dataclasses.dataclass
class BlockStackEntry:
    target: Instruction
    stack_index: int = None
    with_context: ContextWrappingVariable = None

    def can_restore(self):
        return self.with_context is not None

    def resume_fn(self):
        assert self.stack_index is not None
        return ReenterWith(self.stack_index)

    def exit(self, tx):
        return self.with_context.exit(tx)


def stack_op(fn: typing.Callable):
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self: "InstructionTranslatorBase", inst: Instruction):
        self.push(fn_var.call_function(self, self.popn(nargs), {}))

    return impl


def generic_jump(truth_fn: typing.Callable, push: bool):
    def inner(self: "InstructionTranslatorBase", inst: Instruction):
        value: VariableTracker = self.pop()
        self.output.guards.update(value.guards)
        if value.is_python_constant():
            if truth_fn(value.as_python_constant()):
                push and self.push(value)
                self.jump(inst)
        elif (
            isinstance(value, (TensorVariable)) and self.should_compile_partial_graph()
        ):
            # compile a partial subgraph prefix then jump into user code
            self.push(value)
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
                [(create_instruction(inst.opname, target=if_jump[0]))]
                + if_next
                + if_jump
            )
        elif not isinstance(value, TensorVariable) and value.has_unpack_var_sequence(
            self
        ):
            if truth_fn(len(value.unpack_var_sequence(self))):
                push and self.push(value)
                self.jump(inst)
        elif isinstance(value, DynamicShapeVariable):
            eval_result = value.evaluate_expr(self.output)
            if truth_fn(eval_result):
                push and self.push(value)
                self.jump(inst)
        else:
            unimplemented(f"generic_jump {typestr(value)}")

    return inner


explain = False


def break_graph_if_unsupported(*, push):
    def decorator(inner_fn):
        @functools.wraps(inner_fn)
        def wrapper(self: "InstructionTranslatorBase", inst: Instruction):
            state = self.copy_graphstate()
            reason = None
            try:
                return inner_fn(self, inst)
            except Unsupported as exc:
                if not self.should_compile_partial_graph():
                    raise
                user_stack = [self.frame_summary()] + list(reversed(exc.real_stack))
                user_stack_formatted = "".join(traceback.format_list(user_stack))
                frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                # torch._dynamo.explain() formats this a little nicer, and presents a slightly
                # more actionable user code pointer
                if (
                    config.print_graph_breaks
                    and not explain
                    and graph_break_dup_warning_checker.add(frame_loc)
                ):
                    log.warning(
                        f"Graph break: {exc} from user code at {user_stack_formatted}"
                    )

                exc.remove_from_stats()
                exc.add_to_stats("graph_break")
                reason = GraphCompileReason(exc.msg, user_stack)
            self.restore_graphstate(state)
            self.output.compile_subgraph(self, reason=reason)
            self.popn(push - dis.stack_effect(inst.opcode, inst.arg))

            for _ in range(push):
                self.push(UnknownVariable())

            resume_call_insts = self.create_call_resume_at(self.next_instruction)
            # Check if there is a block stack entry with GradModeVariable. And
            # wrap the instruction causing the graph break inside a try..finally
            # block. See more details at
            # https://github.com/pytorch/torchdynamo/issues/207
            cleanup = []
            if len(self.block_stack) == 1 and isinstance(
                self.block_stack[0].with_context, GradModeVariable
            ):
                ctx_variable = self.block_stack[0].with_context

                cg = PyCodegen(self)
                setup_finally, cleanup = ctx_variable.reconstruct(
                    cg, resume_call_insts[0]
                )
                self.output.add_output_instructions(setup_finally)

            self.output.add_output_instructions([inst])

            # Add the cleanup instructions from try..finally block
            self.output.add_output_instructions(cleanup)
            self.output.add_output_instructions(
                resume_call_insts,
            )

        return wrapper

    return decorator


class InstructionTranslatorBase(object):
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
        self.push(fn.call_function(self, args, kwargs))

    def update_locals_and_stack(self, oldvar: VariableTracker, newvar: VariableTracker):
        def repl(v: VariableTracker):
            if v.mutable_local is oldvar.mutable_local:
                return newvar
            return v

        cache = dict()
        self.output.side_effects.apply(repl, cache)
        self.stack = [VariableTracker.apply(repl, x, cache) for x in self.stack]
        for k, x in self.symbolic_locals.items():
            self.symbolic_locals[k] = VariableTracker.apply(repl, x, cache)

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

    def step(self):
        """Process exactly one instruction, return False we should exit"""
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
            log.debug(f"TRACE starts_line {self.f_code.co_filename}:{self.lineno}")

        if len(self.stack) == 0 and self.should_compile_partial_graph():
            self.checkpoint = inst, self.copy_graphstate()

        log.debug(f"TRACE {inst.opname} {inst.argval} {self.stack}")

        try:
            if not hasattr(self, inst.opname):
                unimplemented(f"missing: {inst.opname}")
            getattr(self, inst.opname)(inst)

            return inst.opname != "RETURN_VALUE"
        except BackendCompilerFailed:
            raise
        except Unsupported as exc:
            exc.real_stack.append(self.frame_summary())
            if self.empty_checkpoint():
                raise
        except Exception as exc:
            real_stack = getattr(exc, "real_stack", [])
            real_stack.append(self.frame_summary())
            exc.real_stack = real_stack
            raise

        # generate code from checkpoint
        assert not self.output.output_instructions
        continue_inst, state = self.checkpoint
        self.restore_graphstate(state)
        self.output.compile_subgraph(self, partial_convert=True)
        self.output.add_output_instructions(
            [create_instruction("JUMP_ABSOLUTE", target=continue_inst)]
            + self.instructions
        )

    def run(self):
        try:
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
                e.exec_record = self.exec_recorder.get_record()
            raise
        finally:
            # Cleanup the outputGraph to delete the held tensors. We perform the
            # cleanup only for InstructionTranslator and not
            # InliningInstructionTranslator. The InliningInstructionTranslator
            # mutates the output object and is restored to original state if
            # there was an exception.
            if isinstance(self, InstructionTranslator):
                self.output.cleanup()

    def push(self, val):
        assert val is None or isinstance(
            val, VariableTracker
        ), f"push expects VariableTracker, got {typestr(val)}"
        self.stack.append(val)

    def push_many(self, vals: List[TensorVariable]):
        for val in vals:
            self.push(val)

    def pop(self) -> TensorVariable:
        return self.stack.pop()

    def popn(self, n: int) -> List[TensorVariable]:
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
        self.symbolic_locals[inst.argval] = self.pop()

    def DELETE_FAST(self, inst):
        del self.symbolic_locals[inst.argval]

    STORE_DEREF = STORE_FAST

    def LOAD_CLOSURE(self, inst):
        self.push(ClosureVariable(name=inst.argval))

    def LOAD_CONST(self, inst):
        self.push(ConstantVariable(value=inst.argval))

    def get_global_source(self, name):
        if self.output.root_globals is self.f_globals:
            source = GlobalSource(name)
        else:
            if "__name__" in self.f_globals:
                source = AttrSource(
                    self.import_source(self.f_globals["__name__"]), name
                )
            else:
                mangled_name = f"___unnamed_scope_{id(self.f_globals)}"
                if mangled_name not in self.output.root_globals:
                    self.output.install_global(mangled_name, self.f_globals)
                source = GetItemSource(GlobalSource(mangled_name), name)
        return source

    def LOAD_GLOBAL(self, inst):
        name = inst.argval

        if config.replay_record_enabled:
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
        self.output.side_effects.store_global(variable, name, value)

    def import_source(self, module_name):
        """Create an alias to a module for use in guards"""
        value = importlib.import_module(module_name)
        alias = f"__import_{module_name.replace('.', '_dot_')}"
        f_globals = self.output.root_globals
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
        return "{}.{}".format(base, name) if name else base

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
                    "__package__ != __spec__.parent "
                    f"({package!r} != {spec.parent!r})",
                    ImportWarning,
                    stacklevel=3,
                )
            return package
        elif spec is not None:
            return spec.parent
        else:
            log.warning(
                "can't resolve package from __spec__ or __package__, "
                "falling back on __name__ and __path__",
                ImportWarning,
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
        assert inst.argval in self.f_builtins
        val = self.f_builtins[inst.argval]

        if callable(val):
            assert is_builtin_callable(val)
            self.push(VariableBuilder(self, GlobalSource(inst.argval))(val))
        else:
            assert is_builtin_constant(val)
            self.push(ConstantVariable(value=val))

    def jump(self, inst):
        self.instruction_pointer = self.indexof[id(inst.target)]

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
        ctx = self.pop()
        if not isinstance(ctx, ContextWrappingVariable):
            unimplemented(f"SETUP_WITH {ctx}")
        self.output.guards.update(ctx.guards)

        if isinstance(self, InstructionTranslator):
            self.block_stack.append(BlockStackEntry(inst.target, len(self.stack), ctx))
        else:
            # can't restore this while inlining
            self.block_stack.append(BlockStackEntry(inst.target))
        self.push(
            WithExitFunctionVariable(
                ctx,
                inst.target,
                **VariableTracker.propagate(ctx),
            )
        )
        self.push(ctx.enter(self))

    def SETUP_FINALLY(self, inst):
        self.block_stack.append(BlockStackEntry(inst.target))

    def BEGIN_FINALLY(self, inst):
        self.push(None)

    def WITH_CLEANUP_START(self, inst):
        exit, exc = self.popn(2)
        if sys.version_info < (3, 8):
            assert exc.is_python_constant()
            assert exc.as_python_constant() is None
        else:
            assert exc is None
        self.push(exc)
        self.push(exit.call_function(self, [ConstantVariable(None)] * 3, {}))

    def WITH_CLEANUP_FINISH(self, inst):
        self.popn(2)
        self.push(None)

    def END_FINALLY(self, inst):
        tos = self.pop()
        if sys.version_info < (3, 8):
            # python3.7 and 3.8 can have END_FINALLY without BEGIN_FINALLY
            assert tos is None or (
                tos.is_python_constant() and tos.as_python_constant() is None
            )
        else:
            assert tos is None

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
        supported_is_const = {
            "is": operator.is_,
            "is not": operator.is_not,
            "==": operator.eq,
            "!=": operator.ne,
        }
        supported_tensors = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }
        supported_any = dict(
            itertools.chain(supported_tensors.items(), supported_is_const.items())
        )
        if (
            isinstance(
                left,
                (
                    TensorVariable,
                    DynamicShapeVariable,
                    NNModuleVariable,
                    BaseListVariable,
                    UserDefinedVariable,
                    BaseUserFunctionVariable,
                    ConstDictVariable,
                ),
            )
            and isinstance(right, ConstantVariable)
            and right.value is None
            and op in supported_is_const
        ):
            # <non-None> is None
            self.push(
                ConstantVariable(
                    supported_is_const[op](object(), right.value), **options
                )
            )
        elif (
            left.is_python_constant()
            and right.is_python_constant()
            and op in supported_any
        ):
            # constant fold
            self.push(
                ConstantVariable(
                    supported_any[op](
                        left.as_python_constant(), right.as_python_constant()
                    ),
                    **options,
                )
            )
        elif (
            isinstance(left, TensorVariable) or isinstance(right, TensorVariable)
        ) and op in supported_tensors:
            self.push(
                TensorVariable.create(
                    self,
                    supported_tensors[op](left.as_proxy(), right.as_proxy()),
                    **options,
                )
            )
        elif (
            isinstance(left, DynamicShapeVariable)
            or isinstance(right, DynamicShapeVariable)
        ) and op in supported_tensors:
            self.push(
                DynamicShapeVariable.create(
                    self,
                    supported_tensors[op](left.as_proxy(), right.as_proxy()),
                    dyn_shape=None,
                    **options,
                )
            )
        elif op in ("in", "not in"):
            self.push(right.call_method(self, "__contains__", [left], {}))
            if op == "not in":
                self.UNARY_NOT(inst)
        elif (
            isinstance(left, UserFunctionVariable)
            and isinstance(right, UserFunctionVariable)
            and op in supported_is_const
        ):
            self.push(
                ConstantVariable(supported_is_const[op](left.fn, right.fn), **options)
            )
        else:
            unimplemented(f"COMPARE_OP {typestr(left)} {op} {typestr(right)}")

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
        assert isinstance(argnames, ConstantVariable)
        argnames = argnames.value
        args, kwargs = args[: -len(argnames)], args[-len(argnames) :]
        kwargs = dict(zip(argnames, kwargs))
        assert len(kwargs) == len(argnames)
        self.call_function(fn, args, kwargs)

    def LOAD_METHOD(self, inst):
        self.LOAD_ATTR(inst)
        self.push(self.pop())
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
            self, [obj, ConstantVariable(inst.argval)], {}
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
                .call_function(self, [obj, ConstantVariable(inst.argval), val], {})
                .guards
            )
            return
        except Unsupported as e:
            if not self.should_compile_partial_graph():
                raise
            e.remove_from_stats()
            e.add_to_stats("graph_break")
            self.restore_graphstate(prior)

        # break the graph
        self.output.compile_subgraph(
            self, reason=GraphCompileReason("store_attr", [self.frame_summary()])
        )
        self.output.add_output_instructions([inst])
        self.popn(2)
        self.output.add_output_instructions(
            self.create_call_resume_at(self.next_instruction)
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
            assert isinstance(k, ConstantVariable) or (
                isinstance(k, TensorVariable) and k.specialized_value is not None
            )

            result[ConstDictVariable.get_key(k)] = v
        assert len(result) == len(items) / 2
        self.push(
            ConstDictVariable(result, dict, mutable_local=MutableLocal(), **options)
        )

    def BUILD_CONST_KEY_MAP(self, inst):
        keys = self.pop()
        values = self.popn(inst.argval)
        options = VariableTracker.propagate([keys] + values)
        assert isinstance(keys, ConstantVariable)
        keys = keys.value
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
        if sys.version_info < (3, 8):
            v, k = self.popn(2)
        else:
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

    def LIST_APPEND(self, inst):
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ListVariable)
        assert obj.mutable_local
        self.replace_all(
            obj,
            ListVariable(
                obj.items + [v],
                **VariableTracker.propagate([obj, v]),
            ),
        )

    def MAKE_FUNCTION(self, inst):
        flags = inst.arg
        old_stack = list(self.stack)
        fn_name = self.pop()
        code = self.pop()
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
        # TODO(jansel): rewrite this using unpack_var_sequence
        seq = self.pop()
        options = VariableTracker.propagate([seq])
        if isinstance(seq, BaseListVariable):
            if len(seq.items) != inst.argval:
                unimplemented(f"UNPACK_SEQUENCE {seq}")
            self.output.guards.update(seq.guards)
            for i in reversed(seq.items):
                self.push(i)
        elif seq.is_python_constant() and isinstance(seq, ConstantVariable):
            val = seq.as_python_constant()
            assert len(val) == inst.argval
            for i in reversed(val):
                self.push(ConstantVariable(i, **options))
        elif isinstance(seq, TensorVariable):
            proxy = seq.as_proxy()
            for i in reversed(range(inst.argval)):
                self.push(TensorVariable.create(self, proxy[i], **options))
        elif isinstance(seq, GetAttrVariable) and isinstance(seq.obj, TensorVariable):
            # x, y = a.shape
            proxy = getattr(seq.obj.as_proxy(), seq.name)
            for i in reversed(range(inst.argval)):
                self.push(TensorVariable.create(self, proxy[i], **options))
        else:
            unimplemented(f"UNPACK_SEQUENCE {seq}")

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
            fmt_spec = ConstantVariable("")

        value = self.pop()
        if isinstance(value, DynamicShapeVariable):
            value = ConstantVariable(str(value.dyn_shape))
        if (flags & 0x03) == 0x01:
            value = BuiltinVariable(str).call_function(self, [value], {})
        elif (flags & 0x03) == 0x02:
            value = BuiltinVariable(repr).call_function(self, [value], {})
        elif (flags & 0x03) == 0x03:
            value = BuiltinVariable(ascii).call_function(self, [value], {})

        fmt_var = ConstantVariable(
            "{:" + fmt_spec.as_python_constant() + "}"
        ).add_options(fmt_spec)

        self.call_function(BuiltinVariable(str.format), [fmt_var, value], {})

    def BUILD_STRING(self, inst):
        result = ""
        for _ in range(inst.arg):
            str_var = self.pop()
            assert isinstance(str_var, ConstantVariable)
            result = str_var.value + result
        self.push(ConstantVariable(value=result))

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

    def GEN_START(self, inst):
        self.pop()

    def GET_LEN(self, inst):
        tos = self.stack[-1]
        if tos.is_python_constant():
            self.push(ConstantVariable(len(tos.as_python_constant())))
        else:
            self.push(tos.call_method(self, "__len__", [], {}))

    def MATCH_MAPPING(self, inst):
        tos = self.stack[-1]
        assert isinstance(tos, ConstDictVariable)
        if isinstance(tos.items, collections.abc.Mapping):
            self.push(ConstantVariable(True))
        else:
            self.push(ConstantVariable(False))

    def MATCH_SEQUENCE(self, inst):
        tos = self.stack[-1]
        assert tos.is_python_constant()
        tos_value = tos.as_python_constant()
        if isinstance(tos_value, collections.abc.Sequence) and not isinstance(
            tos_value, (str, bytes, bytearray)
        ):
            self.push(ConstantVariable(True))
        else:
            self.push(ConstantVariable(False))

    def MATCH_KEYS(self, inst):
        tos = self.stack[-1]
        assert tos.is_python_constant()
        keys = tos.as_python_constant()
        tos1 = self.stack[-2]
        assert isinstance(tos1, ConstDictVariable)
        match_obj = tos1.items
        if all(key in match_obj for key in keys):
            self.push(TupleVariable(list(match_obj[key] for key in keys)))
            self.push(ConstantVariable(True))
        else:
            self.push(ConstantVariable(None))
            self.push(ConstantVariable(False))

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
    INPLACE_ADD = stack_op(operator.iadd)
    INPLACE_SUBTRACT = stack_op(operator.isub)
    INPLACE_LSHIFT = stack_op(operator.ilshift)
    INPLACE_RSHIFT = stack_op(operator.irshift)
    INPLACE_AND = stack_op(operator.iand)
    INPLACE_XOR = stack_op(operator.ixor)
    INPLACE_OR = stack_op(operator.ior)

    def copy_graphstate(self):
        """Create a checkpoint of the current state by copying everything"""
        return (
            self.output.copy_graphstate(),
            collections.OrderedDict(self.symbolic_locals),
            list(self.stack),
            list(self.block_stack),
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
            self.lineno,
        )

    def restore_graphstate(self, state):
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
            if isinstance(obj, Iterable):
                if len(obj) != 0:
                    return False
        return True

    def format_frame_summary(self, additional_stack_frames=None):
        if additional_stack_frames is None:
            additional_stack_frames = []
        return "".join(
            traceback.format_list(
                ([self.frame_summary()] + list(reversed(additional_stack_frames)))
            )
        )

    def frame_summary(self):
        return traceback.FrameSummary(
            getattr(self.f_code, "co_filename", "<unknown>"),
            self.lineno,
            getattr(self.f_code, "co_name", "<unknown>"),
            lookup_line=False,
        )

    def store_dict_key(self, name, value):
        self.output.guards.add(
            GlobalWeakRefSource(name).make_guard(GuardBuilder.WEAKREF_ALIVE)
        )
        if name not in self.output.root_globals:
            self.output.install_global(name, weakref.ref(value))

    @property
    def fake_mode(self):
        return self._fake_mode

    def find_symbolic_locals_name(self, tensor_variable):
        for key, value in self.symbolic_locals.items():
            if value is tensor_variable:
                return key
        return None

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
    ):
        super(InstructionTranslatorBase, self).__init__()

        # Mutable state checkpointed by copy_graphstate()
        self.output: OutputGraph = output
        self.symbolic_locals: Dict[str, VariableTracker] = symbolic_locals
        self.symbolic_globals: Dict[str, VariableTracker] = symbolic_globals
        self.stack: List[VariableTracker] = []
        self.instruction_pointer: int = 0
        self.current_instruction: Instruction = create_instruction("NOP")
        self.next_instruction: typing.Optional[Instruction] = None
        self.block_stack: List[BlockStackEntry] = []
        self.lineno: int = code_options.get("co_firstlineno")

        # Properties of the input/output code
        self.instructions: List[Instruction] = instructions
        self.indexof: Dict[int, int] = {id(i): n for n, i in enumerate(instructions)}
        self.f_locals: Dict[
            str, Any
        ] = f_locals  # needed for recording accessed locals for replay
        self.f_globals: Dict[str, Any] = f_globals
        self.f_builtins: Dict[str, Any] = f_builtins
        self.code_options: Dict[str, Any] = code_options
        self.f_code: types.CodeType = f_code

        # Execution record for replaying errors
        self.exec_recorder = ExecutionRecorder(code=f_code, code_options=code_options)
        # Stack of module being parsed, current nn.module is at the end of ordered dict
        self.nn_module_stack: Dict[str, str] = {}
        # Flag to indicate whether tracing is used for export.
        self.export = export

        if fake_tensors_available:
            with torch._subclasses.FakeTensorMode(
                throw_on_data_dependent_ops=True,
                shape_env=output.shape_env,
            ) as fake_mode:
                pass
            self._fake_mode = fake_mode

        self.checkpoint = None
        self.random_calls: List[tuple] = []

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


class InstructionTranslator(InstructionTranslatorBase):
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
    ):
        super(InstructionTranslator, self).__init__(
            output=OutputGraph(f_globals, code_options, compiler_fn, self),
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
        )
        self.one_graph: bool = one_graph
        self.export = export
        if self.export:
            assert (
                self.one_graph
            ), "Export without one graph - something has gone wrong."

        vars = list(code_options["co_varnames"])
        vars.extend(x for x in self.cell_and_freevars() if x not in vars)
        self.symbolic_locals = collections.OrderedDict(
            (k, VariableBuilder(self, LocalSource(k))(f_locals[k]))
            for k in vars
            if k in f_locals
        )

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

        self._freevars_ids = dict()
        for name in self.code_options["co_freevars"]:
            if name in f_locals:
                self._freevars_ids[name] = id(f_locals[name])

    def run(self):
        _step_logger()(logging.INFO, f"torchdynamo start tracing {self.f_code.co_name}")
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
        nargs = len(self.stack) + len(argnames)

        name = unique_id(f"__resume_at_{inst.offset}")

        new_code: types.CodeType = ContinueExecutionCache.lookup(
            self.f_code,
            self.lineno,
            inst.offset,
            len(self.stack),
            argnames,
            tuple(b.resume_fn() for b in self.block_stack),
        )

        cg = PyCodegen(self)

        if new_code.co_freevars:
            cg.make_function_with_closure(name, new_code, len(self.stack))
        else:
            self.output.install_global(
                name, types.FunctionType(new_code, self.f_globals, name)
            )
            cg.extend_output(cg.load_function_name(name, len(self.stack)))

        cg.extend_output([cg.create_load(k) for k in argnames])
        cg.extend_output(
            [
                create_instruction("CALL_FUNCTION", nargs),
                create_instruction("RETURN_VALUE"),
            ]
        )
        return cg.get_instructions()

    def RETURN_VALUE(self, inst):
        if self.output.count_calls() == 0 and not self.export:
            raise exc.SkipFrame()
        self.instruction_pointer = None
        _step_logger()(logging.INFO, f"torchdynamo done tracing {self.f_code.co_name}")
        self.output.compile_subgraph(self)
        self.output.add_output_instructions([create_instruction("RETURN_VALUE")])


class InliningInstructionTranslator(InstructionTranslatorBase):
    """Trace and inline a called method"""

    @classmethod
    def inline_call(cls, parent, func, args, kwargs):
        with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
            return cls.inline_call_(parent, func, args, kwargs)

    @staticmethod
    def inline_call_(parent, func, args, kwargs):
        assert isinstance(
            func,
            (UserFunctionVariable, NestedUserFunctionVariable),
        )
        if func.has_self():
            unimplemented("inline with __self__")

        if func.get_name() == "patched_init":
            unimplemented("Patched init cannot be inlined.")

        try:
            if id(func.get_function()) in allowed_functions._disallowed_function_ids:
                unimplemented(f"inlining disallowed: {func.get_function()}")
        except NotImplementedError:
            pass  # closures

        if skipfiles.check(
            func.get_filename()
        ) and not skipfiles.is_torch_inline_allowed(func.get_filename()):
            unimplemented(
                f"inline in skipfiles: {func.get_name()} {func.get_filename()}"
            )

        try:
            sub_locals, closure_cells = func.bind_args(parent, args, kwargs)
        except TypeError as exc:
            log.warning(
                f"{func.get_filename()} {func.get_function()} {args} {kwargs} {exc}"
            )
            unimplemented("arg mismatch inlining")

        for v in itertools.chain(sub_locals.values(), closure_cells.values()):
            if not isinstance(v, VariableTracker):
                unimplemented(f"unconverted arg {v}")

        code: types.CodeType = func.get_code()
        if code.co_name in ("__setitem__", "__setattr__"):
            unimplemented(f"inline {code.co_name}")

        log.debug(f"INLINING {code} \n {dis.Bytecode(code).dis()} \n")

        if is_generator(code):
            tracer = InliningGeneratorInstructionTranslator(
                parent, code, sub_locals, parent.symbolic_globals, closure_cells, func
            )
        else:
            tracer = InliningInstructionTranslator(
                parent, code, sub_locals, parent.symbolic_globals, closure_cells, func
            )

        tracer.run()
        assert tracer.symbolic_result is not None
        func.export_freevars(parent, tracer)

        if tracer.f_globals is parent.f_globals:
            # Merge symbolic_globals back if parent and child are in the same namespace
            parent.symbolic_globals.update(tracer.symbolic_globals)

        log.debug(f"DONE INLINING {code}")

        if is_generator(code):
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
        super(InliningInstructionTranslator, self).__init__(
            output=parent.output,
            f_locals={},
            f_globals=f_globals,
            f_builtins=f_builtins,
            symbolic_locals=symbolic_locals,
            symbolic_globals=symbolic_globals,
            instructions=cleaned_instructions(code),
            code_options={k: getattr(code, k) for k in dir(code)},
            f_code=code,
            export=parent.export,
        )
        self.parent = parent
        self.symbolic_result = None
        self.closure_cells = closure_cells
        self.nn_module_stack = parent.nn_module_stack.copy()

    @property
    def fake_mode(self):
        return self.parent.fake_mode

    def STORE_DEREF(self, inst):
        if inst.argval in self.closure_cells:
            cell = self.closure_cells[inst.argval]
            val = self.pop()
            if isinstance(cell, ClosureVariable):
                self.output.root_tx.symbolic_locals[cell.name] = val
            else:
                self.output.side_effects.store_cell(cell, val)
        else:
            if isinstance(
                self.symbolic_locals.get(inst.argval),
                variables.NewCellVariable,
            ):
                self.output.side_effects.store_cell(
                    self.symbolic_locals[inst.argval], self.pop()
                )
            else:
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
        self.push(self.closure_cells[inst.argval])

    def replace_all(self, oldvar: VariableTracker, newvar: VariableTracker):
        newvar = super().replace_all(oldvar, newvar)
        # recursively check and update parent's locals and stack in case oldvar is from parent
        translator = self
        while hasattr(translator, "parent"):
            translator = translator.parent
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
    def __init__(self, *args, **kwargs):
        super(InliningGeneratorInstructionTranslator, self).__init__(*args, **kwargs)
        self.generated_items = []

    def YIELD_VALUE(self, inst: Instruction):
        self.generated_items.append(self.pop())
        # TODO(jansel): figure out why this is needed, it isn't in the docs for YIELD_VALUE
        self.push(ConstantVariable(None))
