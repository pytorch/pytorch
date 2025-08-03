# mypy: ignore-errors

"""
Function-related variable tracking classes for Dynamo's symbolic execution.

This module contains classes that track different types of functions during graph
compilation, including:
- User-defined functions and methods
- Built-in functions and methods
- Wrapped functions (e.g. from decorators)
- Special function types (e.g. functools.partial)
- Triton kernels and related function types

These classes are responsible for:
- Tracking function calls and their arguments
- Managing function closures and cell variables
- Handling function attributes and special methods
- Maintaining guards for function identity and closure contents
- Supporting function inlining and specialization
- Enabling proper symbolic execution of different function types

The variable trackers here work together with the rest of Dynamo to enable
accurate graph capture while handling Python's various function-related behaviors.
"""

import builtins
import functools
import inspect
import itertools
import logging
import sys
import traceback
import types
from collections.abc import Sequence
from types import FunctionType
from typing import Any, Callable, Optional, TYPE_CHECKING, TypeVar
from typing_extensions import Never
from unittest.mock import patch
from weakref import WeakKeyDictionary

import torch
from torch._dynamo.exc import get_stack_above_dynamo

from .. import config, graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function, create_rot_n, is_generator
from ..exc import (
    get_dynamo_observed_exception,
    handle_observed_exception,
    InfiniteGeneratorError,
    ObservedException,
    ObservedGeneratorExit,
    ObservedUserStopIteration,
    raise_observed_exception,
    SkipFrame,
    unimplemented_v2,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import (
    check_constant_args,
    check_unspec_or_constant_args,
    cmp_name_to_op_mapping,
    counters,
    identity,
    is_function,
    is_wrapper_or_member_descriptor,
    istype,
    make_cell,
)
from .base import (
    AsPythonConstantNotImplementedError,
    AttributeMutationNew,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable


try:
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._higher_order_ops.triton_kernel_wrap import (
        TritonGridType,
        TritonKernelType,
    )


_F = TypeVar("_F", bound=Callable)
CO_VARARGS = 0x04
CO_VARKEYWORDS = 0x08


# Module‐level cache keyed by the function object
_spec_cache = WeakKeyDictionary()


class FunctionSpec:
    def __init__(self, func: FunctionType):
        code = func.__code__
        vn = code.co_varnames

        self.posonly_count = code.co_posonlyargcount
        self.arg_count = code.co_argcount
        self.kwonly_count = code.co_kwonlyargcount

        self.posonly_names = vn[: self.posonly_count]
        self.pos_or_kw_names = vn[self.posonly_count : self.arg_count]
        self.all_pos_names = self.posonly_names + self.pos_or_kw_names
        self.kwonly_names = vn[self.arg_count : self.arg_count + self.kwonly_count]

        off = self.arg_count + self.kwonly_count
        self.varargs_name = vn[off] if code.co_flags & CO_VARARGS else None
        off += 1 if self.varargs_name else 0
        self.varkw_name = vn[off] if code.co_flags & CO_VARKEYWORDS else None

    def update_defaults(self, func: FunctionType):
        # Defaults can change from function call to function call. So re-update
        # them on every call.
        self.defaults = func.__defaults__ or ()
        self.kwdefaults = func.__kwdefaults__ or {}

        # Map positional‐default names → their index in self.defaults
        self.pos_default_map = dict(
            zip(self.all_pos_names[-len(self.defaults) :], range(len(self.defaults)))
        )


def _get_spec(func: FunctionType) -> FunctionSpec:
    spec = _spec_cache.get(func)
    if spec is None:
        spec = FunctionSpec(func)
        _spec_cache[func] = spec
    return spec


def bind_args_cached(func, tx, fn_source, args, kwargs):
    spec = _get_spec(func)
    spec.update_defaults(func)
    ba = {}
    rem_kw = dict(kwargs)

    # 1) Bind all positional (pos-only + pos-or-kw)
    for i, name in enumerate(spec.all_pos_names):
        if i < len(args):
            ba[name] = wrap_bound_arg(tx, args[i])
        elif name in rem_kw:
            if name in spec.posonly_names:
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[ConstantVariable.create(f"{name} is positional-only")],
                )
            ba[name] = wrap_bound_arg(tx, rem_kw.pop(name))
        elif name in spec.pos_default_map:
            idx = spec.pos_default_map[name]
            default_source = None
            if fn_source and not (
                ConstantVariable.is_literal(spec.defaults[idx])
                and config.skip_guards_on_constant_func_defaults
            ):
                default_source = DefaultsSource(fn_source, idx)
            ba[name] = wrap_bound_arg(tx, spec.defaults[idx], default_source)
        else:
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    ConstantVariable.create(
                        f"Missing required positional argument: {name}"
                    )
                ],
            )

    # 2) *args
    extra = args[len(spec.all_pos_names) :]
    if spec.varargs_name:
        ba[spec.varargs_name] = wrap_bound_arg(tx, tuple(extra))
    elif extra:
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                ConstantVariable.create(
                    f"Too many positional arguments: got {len(args)}, expected {len(spec.all_pos_names)}"
                )
            ],
        )

    # 3) Keyword-only
    for name in spec.kwonly_names:
        if name in rem_kw:
            ba[name] = wrap_bound_arg(tx, rem_kw.pop(name))
        elif name in spec.kwdefaults:
            kwdefault_source = None
            if fn_source:
                kwdefault_source = DefaultsSource(fn_source, name, is_kw=True)
            ba[name] = wrap_bound_arg(tx, spec.kwdefaults[name], kwdefault_source)
        else:
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    ConstantVariable.create(
                        f"Missing required keyword-only argument: {name}"
                    )
                ],
            )

    # 4) **kwargs
    if spec.varkw_name:
        ba[spec.varkw_name] = wrap_bound_arg(tx, rem_kw)
    elif rem_kw:
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                ConstantVariable.create(f"Unexpected keyword arguments: {list(rem_kw)}")
            ],
        )

    return ba


def wrap_bound_arg(tx: "InstructionTranslator", val, source=None):
    # Source propagation is best effort since not every object we encounter has a source to begin with.
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        return VariableTracker.build(tx, val)
    else:
        # Create a lazy variable to avoid guarding on __defaults__ unless really
        # needed.
        return variables.LazyVariableTracker.create(val, source)


def wrap_args_kwargs(tx: "InstructionTranslator", result):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v)


def init_cellvars(parent, result: dict[str, VariableTracker], code):
    """
    Update `result` to add mapping from local name to new cells created
    directly by `code`, or update SideEffects in `parent` if the a local cell is
    already in `result` (cell argument).
    """
    side_effects = parent.output.side_effects

    for name in code.co_cellvars:
        new_cell = side_effects.track_cell_new()
        if name in result:
            # This handles when a function argument is a cell (e.g., captured by
            # a nested func). See `MAKE_CELL` bytecode for more info.
            side_effects.store_cell(new_cell, result.pop(name))
        result[name] = new_cell


def _create_nested_fn(
    code, f_globals, name, defaults, closure, kwdefaults, annotations
):
    from types import FunctionType

    func = FunctionType(code, f_globals, name, defaults, closure)
    func.__kwdefaults__ = kwdefaults

    if isinstance(annotations, tuple):
        from itertools import pairwise

        annotations = dict(pairwise(annotations))

    # TypeError: __annotations__ must be set to a dict object
    assert annotations is None or isinstance(annotations, dict)
    func.__annotations__ = annotations

    return func


fn_known_dunder_attrs = {
    "__annotations__",
    "__defaults__",
    "__kwdefaults__",
    "__code__",
    "__globals__",
    "__closure__",
    "__doc__",
}


def fn_var_getattr(tx, fn, source, name):
    source = source and AttrSource(source, name)
    try:
        subobj = inspect.getattr_static(fn, name)
    except AttributeError:
        # function does not have a __getattr__ or __getattribute__ method,
        # so we can safely assume that this attribute is absent
        raise_observed_exception(AttributeError, tx)

    # Special handling for known dunder attributes
    if name in fn_known_dunder_attrs:
        subobj = getattr(fn, name)
    if source:
        return variables.LazyVariableTracker.create(subobj, source)
    return VariableTracker.build(tx, subobj)


class BaseUserFunctionVariable(VariableTracker):
    def get_filename(self):
        return self.get_code().co_filename

    def get_name(self):
        return self.get_code().co_name

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        result = False

        try:
            result = hasattr(self.get_function(), name)
        except NotImplementedError:
            if name == "__name__" and isinstance(self, NestedUserFunctionVariable):
                result = True
        return variables.ConstantVariable.create(result)

    def inspect_parameter_names(self):
        return list(inspect.signature(self.get_function()).parameters)

    def closure_vars(self, tx):
        return {}


class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    _nonvar_fields = {
        "fn",
        "is_constant",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        return cls(value, source=source)

    def __init__(self, fn, is_constant=False, **kwargs) -> None:
        super().__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            self.is_constant = False

        # TODO putting this here to avoid duplication, because we could hit this
        # from several paths (e.g., SuperVariable or `var_getattr`s).
        if not isinstance(fn, (types.FunctionType, torch.jit.ScriptFunction)):
            unimplemented_v2(
                gb_type="can't handle functions not implemented in python ",
                context=f"{fn}",
                explanation="Dynamo can only handle functions defined in python",
                hints=[
                    "Move usage of this function out of `torch.compile` region",
                    *graph_break_hints.INFERENCE_MODE,
                ],
            )
        # TODO(anijain2305) - Replace directly calling UserFunctionVariable with
        # VariableBuilder, which handles the wrapping of _torchdynamo_inline.
        # unpack @torch._dynamo.optimize()(fn) wrapped function
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        self.fn: types.FunctionType = fn

    def as_python_constant(self):
        if istype(self, UserFunctionVariable):
            return self.fn
        # subclasses (such as methods) usually aren't a constant
        return super().as_python_constant()

    def self_args(self):
        return []

    def get_function(self):
        return self.fn

    def get_code(self):
        return self.fn.__code__

    def python_type(self):
        return types.FunctionType

    def has_self(self):
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self):
        return self.fn.__globals__

    def bind_args(self, parent, args, kwargs) -> dict[str, VariableTracker]:
        """
        Assume `args` and `kwargs` are VariableTracker arguments for a call to
        this function, create new bindings for initial locals.
        """
        assert not self.is_constant

        fn: types.FunctionType = self.fn

        if not isinstance(fn, FunctionType):
            raise TypeError("Only supports regular Python functions.")
        root_tx = parent.output.root_tx
        result = bind_args_cached(fn, root_tx, self.source, args, kwargs)

        init_cellvars(parent, result, fn.__code__)
        closure = self.fn.__closure__ or ()
        assert len(closure) == len(self.fn.__code__.co_freevars)
        for idx, name, cell in zip(
            itertools.count(), self.fn.__code__.co_freevars, closure
        ):
            # TODO refactor these 3 branches.
            side_effects = parent.output.side_effects
            if cell in side_effects:
                cell_var = side_effects[cell]

            elif self.source:
                closure_cell = GetItemSource(
                    AttrSource(self.source, "__closure__"), idx
                )
                closure_cell_contents = AttrSource(closure_cell, "cell_contents")
                try:
                    contents_var = VariableTracker.build(
                        parent, cell.cell_contents, closure_cell_contents
                    )
                except ValueError:
                    # Cell has not yet been assigned
                    contents_var = variables.DeletedVariable()
                cell_var = side_effects.track_cell_existing(
                    closure_cell, cell, contents_var
                )

            else:
                # TODO figure out why source isn't available here, and whether
                # we can fix that and remove this branch.
                try:
                    contents_var = VariableTracker.build(parent, cell.cell_contents)
                except ValueError:
                    # Cell has not yet been assigned
                    contents_var = variables.DeletedVariable()
                cell_var = side_effects.track_cell_existing(None, cell, contents_var)

            result[name] = cell_var

        return result

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)
        return fn_var_getattr(tx, self.fn, self.source, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        result = hasattr(self.fn, name)
        return variables.ConstantVariable.create(result)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # Handle patch_dynamo_config call
        if self.fn is torch._dynamo.patch_dynamo_config:
            try:
                args_const = [arg.as_python_constant() for arg in args]
                kwargs_const = {
                    key: val.as_python_constant() for key, val in kwargs.items()
                }
                changes = torch._dynamo.patch_dynamo_config(
                    *args_const, **kwargs_const
                ).changes
                return variables.DynamoConfigPatchVariable(changes)
            except AsPythonConstantNotImplementedError as e:
                raise RuntimeError(
                    "Cannot convert patch_dynamo_config args/kwargs to constants. "
                    "Please fix your call to patch_dynamo_config by using simpler inputs. "
                    f"args: {args}, kwargs: {kwargs}"
                ) from e
        elif self.fn is torch._dynamo.set_fullgraph:
            try:
                bound = inspect.signature(self.fn).bind(*args, **kwargs)
                fullgraph = bound.arguments["fullgraph"].as_python_constant()
                assert isinstance(fullgraph, bool)
                return variables.SetFullgraphVariable(fullgraph)
            except Exception as e:
                raise RuntimeError(
                    "Improper set_fullgraph() call. Please fix your call to set_fullgraph(). "
                    f"args: {args}, kwargs: {kwargs}"
                ) from e
        # Handle a `nonstrict_trace(fn)` call
        elif self.fn is torch._dynamo.nonstrict_trace:
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            fn_var = bound.args[0]
            if not isinstance(fn_var, BaseUserFunctionVariable):
                typ = fn_var.python_type()
                msg = f"`nonstrict_trace` expects a callable, but got value of type <{typ.__name__}>"
                unimplemented_v2(
                    gb_type="TypeError from user code",
                    context=f"call_function({self.value}, {args}, {kwargs})",
                    explanation=msg,
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

            if not isinstance(fn_var, UserFunctionVariable):
                fn_name = fn_var.get_name()
                msg = f"Applying `nonstrict_trace` to function <{fn_name}>; however, `nonstrict_trace` currently requires the function to be defined outside `torch.compile` region."  # noqa: B950
                unimplemented_v2(
                    gb_type="Limitation of `nonstrict_trace",
                    context=f"{self}",
                    explanation=msg,
                    hints=[
                        f"make sure definition of {fn_name} is outside ",
                        "`torch.compile` region",
                    ],
                )

            fn = fn_var.fn
            return variables.TorchInGraphFunctionVariable(fn, nonstrict_traceable=True)

        if self.is_constant:
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), args, kwargs
            )

        if (
            not tx.output.current_tracer.unsafe_allow_externally_visible_side_effects
            and self.fn
            is torch._dynamo.utils._disable_side_effect_safety_checks_for_current_subtracer
        ):
            with torch._dynamo.side_effects.allow_externally_visible_side_effects_in_subtracer(
                tx
            ):
                return super().call_function(tx, args, kwargs)

        if (
            tx.output.current_tracer.under_activation_checkpoint
            and not tx.output.current_tracer.allow_side_effects_under_checkpoint
        ):
            try:
                from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
            except Exception:
                FSDPState = None
            if FSDPState is not None and self.fn in [
                FSDPState._pre_forward,
                FSDPState._post_forward,
            ]:
                with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
                    return super().call_function(tx, args, kwargs)
        return super().call_function(tx, args, kwargs)


class BuiltinMethodVariable(BaseUserFunctionVariable):
    def __init__(self, fn, is_constant=False, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(fn, types.BuiltinMethodType)
        self.fn = fn

    @staticmethod
    def is_supported_builtin_method(obj):
        method_self = obj.__self__
        method_name = obj.__name__

        # TODO(anijain2305) - Add support for more builtin methods
        # Supports tuple.__new__ and frozenset({....}).__contains__
        return (method_self is tuple and method_name == "__new__") or (
            type(method_self) is frozenset and method_name == "__contains__"
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        method_self = self.fn.__self__
        name = self.fn.__name__
        obj_source = self.source and AttrSource(self.source, "__self__")
        obj_vt = VariableTracker.build(tx, method_self, obj_source)
        return obj_vt.call_method(tx, name, args, kwargs)


class LocalGeneratorObjectVariable(VariableTracker):
    def __init__(
        self,
        code: types.CodeType,
        f_globals,
        inline_tracer: Optional["InstructionTranslator"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.code = code
        self.f_globals = f_globals
        self.inline_tracer = inline_tracer

    def get_code(self):
        return self.code

    def get_filename(self):
        return self.get_code().co_filename

    def get_name(self):
        return self.get_code().co_name

    def get_function(self):
        raise NotImplementedError

    def has_self(self):
        return False

    def __name__(self):
        return self.get_name()

    def __str__(self):
        return f"{self.__class__.__name__}({self.get_name()})"

    __repr__ = __str__

    def reconstruct(self, codegen: "PyCodegen"):
        from torch._dynamo.side_effects import disallow_side_effects_in_generator
        from torch._dynamo.symbolic_convert import (
            InstructionTranslator,
            save_and_restart_speculation_log,
            temporarely_allow_writes_to_output_graph,
        )

        tx = InstructionTranslator.current_tx()
        save = save_and_restart_speculation_log(tx)
        disallow = disallow_side_effects_in_generator(tx)
        temp = temporarely_allow_writes_to_output_graph(tx)

        with save, disallow, temp:
            tracer = self._get_inline_tracer(tx)
            if not tracer.generator_exhausted:
                self.remaining_items = self.force_unpack_var_sequence(tx)
            variables.ListIteratorVariable(self.remaining_items).reconstruct(codegen)

    def bind_args(self, tx, args, kwargs):
        return self.fn.bind_args(tx, args, kwargs)

    def get_globals(self):
        return self.f_globals

    def python_type(self):
        return types.GeneratorType

    def _get_inline_tracer(self, tx):
        from torch._dynamo.symbolic_convert import InliningInstructionTranslator

        if self.inline_tracer is None:
            self.inline_tracer = InliningInstructionTranslator.build_inline_tracer(
                tx, self, [], {}
            )
        return self.inline_tracer

    def next_variable(self, tx):
        tracer = self._get_inline_tracer(tx)

        if self._is_generator_exhausted():
            raise_observed_exception(StopIteration, tx)

        try:
            # Hierarchically, tx can be seen as the parent of the inline tracer
            # created on call_function. Any exception needs to be propagated to tx
            # for Dynamo to behave correctly
            with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
                return tracer.inline_call_()
        except ObservedException as e:
            tracer.generator_exhausted = True
            raise e
        except InfiniteGeneratorError:
            # test/dynamo/test_misc.py::test_iterator_limit
            raise
        except Unsupported as e:
            torch._dynamo.eval_frame.skip_code(self.get_code())
            raise SkipFrame from e
        finally:
            counters["unimplemented"] |= counters["inline_call"]

    def call_obj_hasattr(self, tx, name):
        if name in self.python_type().__dict__:
            return ConstantVariable.create(True)
        return ConstantVariable.create(False)

    def has_unpack_var_sequence(self, tx):
        return False

    def has_force_unpack_var_sequence(self, tx) -> builtins.bool:
        return True

    def force_unpack_var_sequence(self, tx) -> list[VariableTracker]:
        result = []
        self.force_apply_to_var_sequence(tx, result.append)
        return result

    def force_apply_to_var_sequence(self, tx, fn) -> None:
        while True:
            try:
                fn(self.next_variable(tx))
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                break

    def _setup_exception(self, tx, exc):
        tracer = self._get_inline_tracer(tx)
        try:
            tracer._raise_exception_variable(exc)
        except ObservedException as e:
            # if no handler is available (i.e. user code doesn't catch it), the
            # exception is raised again.
            tracer.exception_handler(e)

    def _is_generator_just_started(self):
        return self.inline_tracer is None or self.inline_tracer.instruction_pointer == 0

    def _is_generator_exhausted(self):
        return getattr(self.inline_tracer, "generator_exhausted", False)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__next__":
            return self.next_variable(tx)
        elif name == "__iter__":
            # iter(gen) returns itself
            return self
        elif name == "send":
            # Sends a value into the generator function. Returns the next value
            # yielded by the generator, or raises StopIteration if the generator
            # exits without yielding another value
            if self._is_generator_just_started() and len(args):
                # can't send non-None value to a just-started generator
                # Test: GeneratorCPythonTests.test_send_non_none_to_new_gen
                if not all(
                    isinstance(arg, ConstantVariable) and arg.value is None
                    for arg in args
                ):
                    raise_observed_exception(TypeError, tx)
            tracer = self._get_inline_tracer(tx)
            tracer.push_many(args)
            return self.next_variable(tx)
        elif name == "close":
            # * Raises a GeneratorExit at the point where the generator function was paused.
            # * If the generator function catches the exception and returns a
            # value, this value is returned from close() - Python 3.13+
            # * If the generator function is already closed, or raises GeneratorExit
            # (by not catching the exception), close() returns None.
            # * If the generator yields a value, a RuntimeError is raised.
            # * If the generator raises any other exception, it is propagated to the caller.
            # * If the generator has already exited due to an exception or normal
            # exit, close() returns None and has no other effect.

            # Return None if close is called on a just-started generator
            # See test GeneratorCloseCpythonTests::test_close_not_started

            tracer = self._get_inline_tracer(tx)
            if self._is_generator_just_started() or self._is_generator_exhausted():
                tracer.generator_exhausted = True
                return variables.ConstantVariable(None)

            # Raise GeneratorExit to see if user code catches it. Any other exception
            # is propagated to the parent frame.
            try:
                self._setup_exception(
                    tx, variables.ExceptionVariable(GeneratorExit, ())
                )
                # There's an extra block on Python 3.12+ to handle StopIteration
                # see: https://github.com/python/cpython/blob/8f93dd8a8f237b277abad20d566df90c5cbd7f1e/Objects/genobject.c#L394-L397
                #
                #   1           0 RETURN_GENERATOR
                #               2 POP_TOP
                #               4 RESUME                   0

                #   2           6 LOAD_CONST               1 (1)
                #               8 YIELD_VALUE              1
                #              10 RESUME                   1
                #              12 POP_TOP
                #              14 RETURN_CONST             0 (None)
                #         >>   16 CALL_INTRINSIC_1         3 (INTRINSIC_STOPITERATION_ERROR)
                #              18 RERAISE                  1
                # ExceptionTable:
                #   4 to 14 -> 16 [0] lasti
                if (
                    sys.version_info >= (3, 12)
                    and tracer.next_instruction.opname == "CALL_INTRINSIC_1"
                ):
                    tracer.generator_exhausted = True
                    return variables.ConstantVariable(None)
            except ObservedGeneratorExit:
                # If it doesn't catch, we just return None, as per the text above
                tracer.generator_exhausted = True
                return variables.ConstantVariable(None)

            try:
                # Raise RuntimeError if the generator yields any other value
                if self.next_variable(tx):
                    raise_observed_exception(RuntimeError, tx)
            except ObservedGeneratorExit:
                tracer.generator_exhausted = True
                return variables.ConstantVariable(None)
            except ObservedUserStopIteration:
                # In Python 3.13+, one can capture GeneratorExit and return a value
                # See test_generator.py::test_close_capture_GeneratorExit_return
                # https://discuss.python.org/t/let-generator-close-return-stopiteration-value/24786/26
                # https://github.com/python/cpython/pull/104771
                assert tracer.symbolic_result is not None
                return tracer.symbolic_result
        elif name == "throw":
            # * Raises an exception at the point where the generator was paused, and
            # returns the next value yielded by the generator.
            # * If the generator exits without yielding, raise StopIteration
            # * If the generator function does not catch the passed-in exception,
            # or raises a different exception, then that exception propagates to the caller.

            # Setup the exception table and jump target in case of try...finally
            tracer = self._get_inline_tracer(tx)
            try:
                # In Python 3.9, the exception is represented as a triple (typ, val, tb)
                # In such cases, we re-raise the exception object given to avoid
                # creating a new object, so that IS_OP works.
                # See: https://github.com/pytorch/pytorch/pull/146496
                self._setup_exception(tx, args[1] if len(args) == 3 else args[0])
            except ObservedException:  # noqa: TRY203
                # propagate the exception back to the parent caller
                raise

            retval = self.next_variable(tx)

            # The exception raised before is still active. We need to check the exception
            # table one more time to find the next target. But why? Let’s walk
            # through an example and its generated bytecode: https://godbolt.org/z/ebdTbMv8M
            #
            #     z = 0
            #     def whoo():
            #         global z
            #         z = 0
            #         try:
            #             yield 1
            #         except ValueError:
            #             yield 2
            #         finally:
            #             z += 1
            #         z += 10
            #
            #     gen = whoo()
            #     next(gen)
            #     gen.throw(ValueError)
            #     print('z', z)  -> z = 1
            #
            #              ...
            #         >>   58 PUSH_EXC_INFO
            #
            #   8          60 LOAD_GLOBAL              2 (ValueError)
            #              70 CHECK_EXC_MATCH
            #              72 POP_JUMP_IF_FALSE        7 (to 88)
            #              74 POP_TOP
            #
            #   9          76 LOAD_CONST               3 (2)
            #              78 YIELD_VALUE              3      <------ ValueError is still active here
            #              80 RESUME                   1
            #              82 POP_TOP
            #              84 POP_EXCEPT
            #              86 jump_backward           34 (to 20)
            #              ...
            #
            #     ExceptionTable:
            #     4 to 8 -> 124 [0] lasti
            #     12 to 18 -> 58 [0]
            #     20 to 56 -> 124 [0] lasti
            #     58 to 82 -> 90 [1] lasti     <------ move to 90
            #     84 to 86 -> 96 [0]
            #     88 to 88 -> 90 [1] lasti
            #     90 to 94 -> 96 [0]
            #     96 to 116 -> 118 [1] lasti
            #     118 to 122 -> 124 [0] lasti
            #
            # In this scenario, a generator can yield after `throw()` is called. Even
            # after the exception is raised a few lines above, it remains active
            # within the `78 YIELD_VALUE` instruction. When the generator resumes
            # after the second yield on instruction `80 RESUME`, we cannot simply
            # return the control flow to the next instruction. Instead, one must
            # check the exception table (or equivalent) to find the next target
            # In this case, it says the instruction pointer must be moved to 90.
            #
            # Without this step, if we let the trace proceed to the next
            # instruction, it would follow the control flow where the exception
            # raised by `throw()` was handled and swallowed, potentially leading
            # to incorrect behavior.
            exc_type = type("__InternalThrowException", (Exception,), {})

            try:
                self._setup_exception(tx, variables.ExceptionVariable(exc_type, ()))
                self.next_variable(tx)
            except get_dynamo_observed_exception(exc_type):
                # We should get back the exception raised before.
                pass
            else:
                raise_observed_exception(RuntimeError, tracer)
            return retval

        super().call_method(tx, name, args, kwargs)


class ContextlibContextManagerLocalGeneratorObjectVariable(
    LocalGeneratorObjectVariable
):
    """
    .. note::

        This is only used when the function is annotated with @contextlib.contextmanager

        It is a special case of a generator function as we do not allow return a context manager
        from a torch.compile function.
    """


class LocalGeneratorFunctionVariable(BaseUserFunctionVariable):
    """functions that behaves like iterators

    .. note::

        This is a wrapper around (Nested)UserFunctionVariable
    """

    def __init__(
        self,
        vt: VariableTracker,
        *,
        generator_cls=LocalGeneratorObjectVariable,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vt = vt
        self.generator_cls = generator_cls

    def __getattr__(self, name):
        if name in self.__class__.__dict__.keys():
            return getattr(self, name)
        return getattr(self.vt, name)

    def _build_inline_tracer(self, tx, args, kwargs):
        from torch._dynamo.symbolic_convert import InliningInstructionTranslator

        return InliningInstructionTranslator.build_inline_tracer(
            tx,
            self,
            args,
            kwargs,
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if not is_generator(self.vt.get_code()):
            unimplemented_v2(
                gb_type="non-generator contextlib.contextmanager",
                context=str(self.vt.get_code()),
                explanation="Cannot compile function decorated with `@contextlib.contextmanager` that is not a generator"
                ", i.e. does not use `yield`",
                hints=[
                    "Use `yield` in the function body instead of `return`.",
                    "Remove the `@contextlib.contextmanager` decorator.",
                ],
            )

        inline_tracer = self._build_inline_tracer(tx, args, kwargs)
        code = self.vt.get_code()
        f_globals = self.vt.get_globals()

        # calling a generator returns a generator object
        return self.generator_cls(
            code,
            f_globals,
            inline_tracer,
            source=self.source,
        )


class FunctionDecoratedByContextlibContextManagerVariable(
    LocalGeneratorFunctionVariable
):
    """
    .. note::

        This is only used when the function is annotated with @contextlib.contextmanager
    """

    def __init__(self, vt, **kwargs):
        super().__init__(
            vt,
            generator_cls=ContextlibContextManagerLocalGeneratorObjectVariable,
            **kwargs,
        )

    def _build_inline_tracer(self, tx, args, kwargs):
        # NOTE: This only exists to not break support for context manager when
        # config.enable_faithful_generator_behavior = False and
        # config.enable_trace_contextlib = True. In case the former is false,
        # Dynamo should still be able to trace through @contextmanager functions
        tracer = super()._build_inline_tracer(tx, args, kwargs)
        assert isinstance(
            tracer,
            torch._dynamo.symbolic_convert.InliningGeneratorInstructionTranslator,
        )
        tracer.is_generator_from_ctx_manager = True
        return tracer


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs) -> None:
        super().__init__(fn=fn, **kwargs)
        self.obj = obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def self_args(self):
        return [self.obj]

    def python_type(self):
        return types.MethodType

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # NOTE this is to handle methods annotated by `nonstrict_trace`. Usually
        # a `nonstrict_trace`-ed function will be wrapped by
        # `VariableTracker.build` and route to `TorchInGraphFunctionVariable`,
        # but in the case of method, we manually wrap it with `UserMethodVariable`
        # inside `UserDefinedObjectVariable.var_getattr`.
        #
        # We might be able to simplify this away by canonicalizing the
        # function/method wrapping code paths.
        from ..trace_rules import is_nonstrict_trace_callable

        if is_nonstrict_trace_callable(self.fn):
            call_args = [*self.self_args(), *args]
            var = variables.TorchInGraphFunctionVariable(
                self.fn, nonstrict_traceable=True
            )
            return var.call_function(tx, call_args, kwargs)

        # For nn.Module methods, redirecting to NNModuleVariable.call_method for optimized solution
        # rather than simple inlining. E.g, putting `call_method` op in FX graph for `forward` method
        # since we ensure `forward` of allowed modules can be traced by AOT safely.
        # Note this is not only for allowed modules, as user customized modules can extend from
        # allowed modules but using parent's `forward` method, which is also covered by this branch.

        # If we are tracing the higher order op, we want Dynamo to step inside
        # the module call so that Dynamo can see the underlying parameters and
        # buffers and raise them as inputs to the graph. The is_root_tracer
        # check bypasses the if condition for non-root tracers and directly
        # calls the super().call_function at the end, which is basically
        # equivalent of inlining the method.
        if tx.output.is_root_tracer() and isinstance(
            self.obj, variables.NNModuleVariable
        ):
            module_attr = getattr(self.fn, "__module__", "")
            # inline torch.nn.utils.parametrize
            if (
                module_attr is not None
                and module_attr.startswith("torch.nn.")
                and module_attr != "torch.nn.utils.parametrize"
                or self.is_constant
            ):
                return self.obj.call_method(
                    tx, self.fn.__name__, args, kwargs, constant=self.is_constant
                )
        elif (
            _fsdp_param_group is not None
            and self.fn is _fsdp_param_group.FSDPParamGroup.use_training_state
        ):
            return variables.TorchCtxManagerClassVariable(self.fn).call_function(
                tx, (self.obj, *args), kwargs
            )
        if self.is_constant:
            fn = getattr(self.obj.value, self.fn.__name__)
            return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
        return super().call_function(tx, args, kwargs)

    def inspect_parameter_names(self):
        return super().inspect_parameter_names()[1:]

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        source = self.source and AttrSource(self.source, name)
        if name == "__self__":
            return self.obj
        if name == "__func__":
            return VariableTracker.build(tx, self.fn, source)
        return super().var_getattr(tx, name)


class WrappedUserMethodVariable(UserMethodVariable):
    def __init__(self, wrapped, context, **kwargs) -> None:
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, wrapped.obj, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen(self.context))
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


class WrappedUserFunctionVariable(UserFunctionVariable):
    def __init__(self, wrapped, context, **kwargs) -> None:
        kwargs.pop("fn", None)
        super().__init__(wrapped.fn, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen(self.context))
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


def invoke_and_store_as_constant(tx: "InstructionTranslator", fn, name, args, kwargs):
    def convert(x):
        if isinstance(x, variables.TensorVariable):
            return x.get_real_value()
        return x.as_python_constant()

    args = [convert(x) for x in args]
    kwargs = {k: convert(v) for k, v in kwargs.items()}
    res = fn(*args, **kwargs)
    return tx.output.register_attr_or_module(
        res,
        name,
        source=ConstantSource(name),
    )


class NestedUserFunctionVariable(BaseUserFunctionVariable):
    _nonvar_fields = {
        "f_globals",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    def __init__(
        self,
        fn_name,
        code,
        f_globals,
        defaults,
        kwdefaults,
        annotations,
        closure,
        # This is present when this function is created by
        # `functools.wrap(wrapped_fn)(this_fn)`.
        wrapped_fn=None,
        **kwargs,
    ) -> None:
        if kwargs.get("mutation_type") is None:
            kwargs.update(mutation_type=AttributeMutationNew())
        super().__init__(**kwargs)
        assert isinstance(fn_name.as_python_constant(), str)
        assert isinstance(code.as_python_constant(), types.CodeType)
        assert isinstance(f_globals, dict)
        self.fn_name = fn_name
        self.code = code
        self.f_globals = f_globals
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.annotations = annotations
        self.closure = closure
        self.wrapped_fn: Optional[VariableTracker] = wrapped_fn

    def self_args(self):
        return []

    def get_code(self):
        return self.code.as_python_constant()

    def python_type(self):
        return types.FunctionType

    def get_function(self):
        if self.closure:
            raise NotImplementedError
        func = types.FunctionType(
            self.code.as_python_constant(),
            self.f_globals,
            self.fn_name.as_python_constant(),
        )
        if self.defaults:
            func.__defaults__ = self.defaults.as_python_constant()
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.as_python_constant()
        if self.annotations:
            annotations = self.annotations.as_python_constant()
            if isinstance(annotations, tuple):
                from itertools import pairwise

                annotations = dict(pairwise(annotations))

            # TypeError: __annotations__ must be set to a dict object
            assert isinstance(annotations, dict)
            func.__annotations__ = annotations
        return func

    def call_setattr(
        self,
        tx: "InstructionTranslator",
        name_var: VariableTracker,
        val: VariableTracker,
    ):
        tx.output.side_effects.store_attr(self, name_var.value, val)
        return ConstantVariable(None)

    def call_method(self, tx, name, args, kwargs):
        if name == "__setattr__":
            return self.call_setattr(tx, *args)
        return super().call_method(tx, name, args, kwargs)

    def has_closure(self):
        return self.closure is not None

    def const_getattr(self, tx, name):
        if name == "__name__":
            return self.fn_name.as_python_constant()
        return super().const_getattr(tx, name)

    def has_self(self):
        return False

    def get_globals(self):
        return self.f_globals

    def bind_args(self, parent, args, kwargs):
        code = self.get_code()
        func = types.FunctionType(
            code,
            self.f_globals,
            self.fn_name.as_python_constant(),
            tuple(self.defaults.items) if self.defaults else None,
            tuple(make_cell(None) for _ in range(len(self.get_code().co_freevars))),
        )
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.keys_as_python_constant()
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())
        wrap_args_kwargs(parent.output.root_tx, result)
        init_cellvars(parent, result, code)

        for idx, name in enumerate(code.co_freevars):
            assert name not in result
            cell = self.closure.items[idx]
            result[name] = cell

        return result

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.load_import_from(__name__, "_create_nested_fn")
        )
        codegen(self.code)
        codegen.extend_output([codegen.create_load_const_unchecked(self.f_globals)])
        codegen(ConstantVariable.create(self.code.value.co_name))

        if self.defaults:
            codegen(self.defaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.closure:
            codegen(self.closure)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.kwdefaults:
            codegen(self.kwdefaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.annotations:
            try:
                annotations = self.annotations.as_python_constant()
                codegen.extend_output(
                    [codegen.create_load_const_unchecked(annotations)]
                )
            except NotImplementedError:
                codegen(self.annotations)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        codegen.extend_output(create_call_function(7, False))

        if self.wrapped_fn:
            codegen.add_push_null(
                lambda: codegen.load_import_from("functools", "wraps")
            )
            codegen(self.wrapped_fn)
            codegen.extend_output(create_call_function(1, False))
            codegen.extend_output(create_rot_n(2))
            codegen.extend_output(create_call_function(1, True))

        # codegen attributes
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
        if tx.output.side_effects.has_pending_mutation(self):
            for name, value in tx.output.side_effects.store_attr_mutations[
                self
            ].items():
                codegen.dup_top()
                codegen(value)
                codegen.extend_output(create_rot_n(2))
                codegen.store_attr(name)


class WrappedNestedUserFunctionVariable(NestedUserFunctionVariable):
    def __init__(self, wrapped, context, **kwargs) -> None:
        kwargs.pop("fn_name", None)
        kwargs.pop("code", None)
        kwargs.pop("f_globals", None)
        kwargs.pop("defaults", None)
        kwargs.pop("kwdefaults", None)
        kwargs.pop("annotations", None)
        kwargs.pop("closure", None)
        kwargs.pop("wrapped_fn", None)
        super().__init__(
            wrapped.fn_name,
            wrapped.code,
            wrapped.f_globals,
            wrapped.defaults,
            wrapped.kwdefaults,
            wrapped.annotations,
            wrapped.closure,
            wrapped.wrapped_fn,
        )
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen(self.context))
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


class SkipFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "value",
        "reason",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, value, reason=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.reason = reason

    def as_python_constant(self):
        return self.value

    @classmethod
    def create_with_source(cls, value, source):
        if not is_wrapper_or_member_descriptor(value):
            # These descriptors are not guaranteed to return the same object on
            # attribute lookup. They are unlikely to be changed, so we can skip
            # guarding them.
            install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if inspect.getattr_static(self.value, "_torchdynamo_disable", False):
            msg = inspect.getattr_static(self.value, "_torchdynamo_disable_msg", None)
            unimplemented_v2(
                gb_type="Skip calling `torch.compiler.disable()`d function",
                context=str(self.value),
                explanation=f"Skip calling function `{self.value}` since it was wrapped "
                f"with `torch.compiler.disable` (reason: {msg})",
                hints=[
                    "Remove the `torch.compiler.disable` call",
                ],
            )
        elif self.value is torch._dynamo.graph_break:
            graph_break_msg = kwargs.get("msg", None)
            if graph_break_msg:
                graph_break_msg = graph_break_msg.as_python_constant()
            unimplemented_v2(
                gb_type="Call to `torch._dynamo.graph_break()`",
                context=f"Called `torch._dynamo.graph_break()` with args `{args}`, kwargs `{kwargs}`",
                explanation=f"User-inserted graph break. Message: {graph_break_msg}",
                hints=[
                    "Remove the `torch._dynamo.graph_break()` call.",
                ],
            )
        elif self.value is torch._dynamo.skip_frame:
            skip_frame_msg = kwargs.get("msg", None)
            if skip_frame_msg:
                skip_frame_msg = skip_frame_msg.as_python_constant()
            raise SkipFrame(
                f"Skip frame due to `torch._dynamo.skip_frame()`. Message: {skip_frame_msg}"
            )
        else:
            if config.dont_skip_tracing:
                from .builder import SourcelessBuilder

                # re-build the function, attempting to not skip
                rebuilt_fn = SourcelessBuilder.create(tx, self.value)
                # if we still get SkipFunctionVariable, then we *really* should skip this function
                if not isinstance(rebuilt_fn, SkipFunctionVariable):
                    return rebuilt_fn.call_function(tx, args, kwargs)
            qualname = getattr(self.value, "__qualname__", "<unknown qualname>")
            module_or = getattr(self.value, "__module__", None)
            module_name = "<unknown module>" if module_or is None else str(module_or)
            try:
                path = inspect.getfile(self.value)
                explanation = (
                    f"Dynamo developers have intentionally marked that the function `{qualname}` "
                    f"in file `{path}` should not be traced."
                )
                hints = [
                    f"Avoid calling the function `{qualname}`.",
                ]
                # TODO improve trace_rules reasoning to provide better hints.
                # How do we tell that a function/file should NOT be removed from skip files?
                # Do a very basic check for now.
                if "_dynamo" not in path:
                    hints += [
                        f"Apply `@torch._dynamo.dont_skip_tracing` to the function `{qualname}` "
                        "to force tracing into the function. "
                        "More graph breaks may occur as a result of attempting to trace into the function.",
                        "Please file an issue to PyTorch.",
                    ]
            except TypeError:
                known_python_builtin_modules = {"_abc", "_warnings"}
                if module_or in known_python_builtin_modules:
                    explanation = (
                        f"Dynamo does not know how to trace the Python builtin "
                        f"`{module_name}.{qualname}`."
                    )
                    hints = [
                        "If you are attempting to call a logging function (e.g. `_warnings.warn`), "
                        "you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.",
                        "Please file an issue on GitHub "
                        "so the PyTorch team can add support for it. ",
                    ]
                elif module_or is not None and module_or.startswith("optree"):
                    explanation = f"Dynamo cannot trace optree C/C++ function {module_name}.{qualname}."
                    hints = [
                        " Consider using torch.utils._pytree - "
                        "https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py"
                    ]
                    # also warn on it because most users won't see the graph break message
                    torch._dynamo.utils.warn_once(explanation + "\n" + "\n".join(hints))
                else:
                    explanation = (
                        f"Dynamo does not know how to trace the builtin `{module_name}.{qualname}.` "
                        f"This function is either a Python builtin (e.g. _warnings.warn) "
                        f"or a third-party C/C++ Python extension (perhaps created with pybind)."
                    )
                    hints = [
                        "If it is a Python builtin, please file an issue on GitHub "
                        "so the PyTorch team can add support for it and see the next case for a workaround.",
                        "If it is a third-party C/C++ Python extension, please "
                        "either wrap it into a PyTorch-understood custom operator "
                        "(see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html "
                        "for more details) or, if it is traceable, use "
                        "`torch.compiler.allow_in_graph`.",
                    ]
                    # also warn on it because most users won't see the graph break message
                    torch._dynamo.utils.warn_once(explanation + "\n" + "\n".join(hints))
            if qualname == "allow_in_graph":
                explanation = (
                    "Found an allow_in_graph decorator to a function which "
                    "is created inside the parent function that is getting "
                    "compiled. This is not supported for now."
                )
                hints = []
            reason = self.reason if self.reason else "<missing reason>"
            unimplemented_v2(
                gb_type="Attempted to call function marked as skipped",
                context=f"module: {module_name}, qualname: {qualname}, skip reason: {reason}",
                explanation=explanation,
                hints=hints,
            )

    def call_obj_hasattr(self, tx: "InstructionTranslator", name):
        return variables.ConstantVariable.create(hasattr(self.value, name))

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)

        return fn_var_getattr(tx, self.value, self.source, name)


class WrappedSkipFunctionVariable(SkipFunctionVariable):
    def __init__(self, wrapped, context, **kwargs) -> None:
        kwargs.pop("value", None)
        kwargs.pop("reason", None)
        super().__init__(wrapped.value, reason=wrapped.reason, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen(self.context))
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


class WrapperUserFunctionVariable(VariableTracker):
    """
    Used to represent a wrapper object that contains the actual callable as an
    attribute. For example, torch.jit.script/trace have the original function at
    their _torchdynamo_inline attribute. Similarly, functions with
    __script_if_tracing_wrapper have the original attr at "__original_fn".
    """

    def __init__(self, wrapper_obj, attr_to_trace, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wrapper_obj = wrapper_obj
        self.attr_to_trace = attr_to_trace

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name == self.attr_to_trace:
            val = getattr(self.wrapper_obj, self.attr_to_trace)
            source = self.source and AttrSource(self.source, name)
            return VariableTracker.build(tx, val, source)

        return super().var_getattr(tx, name)

    def self_args(self):
        return []

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if hasattr(self.wrapper_obj, "cache_info"):
            target_fn = getattr(self.wrapper_obj, self.attr_to_trace, None)
            module_name = getattr(target_fn, "__module__", "") or ""

            if module_name.split(".", maxsplit=1)[0] != "torch":
                msg = (
                    "Dynamo detected a call to a `functools.lru_cache`-wrapped "
                    "function. Dynamo ignores the cache wrapper and directly "
                    "traces the wrapped function. Silent incorrectness is only "
                    "a *potential* risk, not something we have observed. "
                    'Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.'
                )

                torch._dynamo.utils.warn_once(msg)

                dynamo_logger = torch._dynamo.utils.logging.getLogger("torch._dynamo")
                if dynamo_logger.isEnabledFor(logging.DEBUG):
                    user_stack = torch._guards.TracingContext.extract_stack()
                    user_stack = get_stack_above_dynamo() + user_stack
                    frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                    user_stack_formatted = "".join(traceback.format_list(user_stack))
                    user_stack_trace = f"call to a lru_cache wrapped function at: {frame_loc[0]}:{frame_loc[1]}\n"
                    user_stack_trace += str(user_stack_formatted)
                    dynamo_logger.debug(user_stack_trace)

        all_args = self.self_args() + args
        return variables.UserFunctionVariable(
            polyfills.getattr_and_trace
        ).call_function(
            tx,
            [self, variables.ConstantVariable(self.attr_to_trace), *all_args],
            kwargs,
        )


class WrapperUserMethodVariable(WrapperUserFunctionVariable):
    """
    Similar to WrapperUserFunctionVariable, but for methods. The only delta is
    saving the vt for `self` object of the method which is then used by
    WrapperUserFunctionVariable in `call_function` method.
    """

    def __init__(self, wrapper_obj, attr_to_trace, self_obj, **kwargs) -> None:
        super().__init__(wrapper_obj, attr_to_trace, **kwargs)
        self.obj = self_obj

    def self_args(self):
        return [self.obj]


def _traceable_collective_remaps():
    # We can't rely on importing from distributed, since it's not always built
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import (
            traceable_collective_remaps,
        )

        return traceable_collective_remaps
    return {}


def _traceable_collectives_source(tx: "InstructionTranslator", fn):
    assert torch.distributed.is_available(), "Illegal invocation."
    assert fn in _traceable_collective_remaps().values()

    inner_name = fn.__name__
    path_source = tx.import_source("torch.distributed._functional_collectives")
    return AttrSource(path_source, inner_name)


class CollectiveFunctionRewriteVariable(UserFunctionVariable):
    """
    Some of the torch.distributed.* collective APIs are possible to rewrite to 'traceable' collectives.

    This class provides both a way to check if a function is remappable, and perform the remapping.

    In the case that a function is 'remappable' but only for some combinations of call-time arguments,
    we check the args at `call_function` time and fall back to graph-breaking if needed.  This is no worse
    than status-quo as we currently graph-break on all distributed.* collectives.
    """

    def __init__(self, fn, *, replacement_var, **kwargs) -> None:
        super().__init__(fn, **kwargs)
        assert isinstance(replacement_var, UserFunctionVariable)
        self.replacement_var = replacement_var

    @staticmethod
    def create(tx: "InstructionTranslator", old_fn, source, **options):
        new_fn, new_source = CollectiveFunctionRewriteVariable.rewrite(tx, old_fn)
        return CollectiveFunctionRewriteVariable(
            old_fn,
            replacement_var=UserFunctionVariable(new_fn, source=new_source, **options),
            source=source,
            **options,
        )

    @staticmethod
    def can_rewrite(variable):
        return (
            inspect.isfunction(variable) and variable in _traceable_collective_remaps()
        )

    @staticmethod
    def rewrite(tx: "InstructionTranslator", fn):
        new_fn = _traceable_collective_remaps()[fn]
        return new_fn, _traceable_collectives_source(tx, new_fn)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # call_function must check any unsupported arguments and graph-break.
        # It's safe to assume args/kwargs from orig_fn map 1:1 to args/kwargs of remapped_fn,
        # since that's the contract for putting a mapping in `traceable_collective_remaps`
        import torch.distributed as dist
        from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

        # Merge args into kwargs so positional and keyword args
        # can be processed the same way.
        signature = inspect.signature(self.fn)
        kwargs = dict(signature.bind(*args, **kwargs).arguments)
        args = ()

        if "async_op" in kwargs and kwargs["async_op"].as_python_constant():
            unimplemented_v2(
                gb_type="async_op=True for distributed collectives",
                context=f"{self.fn}, {args=}, {kwargs=}",
                explanation=f"`torch.compile` doesn't support `async_op=True for {self.fn}",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        if self.fn in (
            dist.all_reduce,
            dist.reduce_scatter_tensor,
            dist._reduce_scatter_base,
        ):
            reduce_op_var = kwargs.get("op")
            reduce_op = (
                reduce_op_var.value
                if reduce_op_var is not None
                else signature.parameters["op"].default
            )
            if reduce_op not in REDUCE_OP_TO_STR:
                raise ValueError(f"Unsupported all_reduce op: {reduce_op}")
            kwargs["op"] = variables.ConstantVariable.create(
                REDUCE_OP_TO_STR[reduce_op]
            )
        return self.replacement_var.call_function(tx, args, kwargs)


class FunctoolsWrapsVariable(UserFunctionVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if not kwargs and len(args) == 1:

            def wraps(fn):
                if isinstance(fn, variables.NestedUserFunctionVariable):
                    return fn.clone(wrapped_fn=args[0])
                unimplemented_v2(
                    gb_type="functools.wraps",
                    context=f"{fn}",
                    explanation="`torch.compile` can't trace `functools.wraps` on functions defined outside the compile region",
                    hints=[
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            return variables.LambdaVariable(wraps)

        return super().call_function(tx, args, kwargs)


class CollectionsNamedTupleFunction(UserFunctionVariable):
    def as_python_constant(self):
        return self.fn

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        constant_args = check_constant_args(args, kwargs)
        if constant_args:
            value = self.fn(
                *[x.as_python_constant() for x in args],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            return variables.UserDefinedClassVariable(
                value, mutation_type=ValueMutationNew()
            )
        unimplemented_v2(
            gb_type="namedtuple construction",
            context=f"{args=}, {kwargs=}",
            explanation="`torch.compile` only support certain input types for namedtuple",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class FunctoolsPartialVariable(VariableTracker):
    def __init__(self, func: VariableTracker, args, keywords, **kwargs) -> None:
        super().__init__(**kwargs)
        self.func = func
        assert isinstance(args, list)
        self.args = args
        assert isinstance(keywords, dict)
        self.keywords = keywords
        # fake_value is used for id calculation. Creating this value and id'ng
        # on it is sufficient for the tracing purposes.
        self.fake_value = functools.partial(identity)

    def python_type(self):
        return functools.partial

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(lambda: codegen.load_import_from("functools", "partial"))
        codegen(self.func)
        if self.args:
            codegen.foreach(self.args)
        if not self.keywords:
            codegen.extend_output(create_call_function(len(self.args) + 1, False))
            return

        codegen.foreach(self.keywords.values())
        keys = tuple(self.keywords.keys())
        codegen.extend_output(
            codegen.create_call_function_kw(len(keys) + len(self.args) + 1, keys, False)
        )

    def get_function(self):
        return self.as_python_constant()

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        merged_args = self.args + args
        merged_kwargs = {**self.keywords, **kwargs}
        return self.func.call_function(tx, merged_args, merged_kwargs)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        # functools.partial uses slots, so attributes are constant
        return variables.ConstantVariable.create(
            hasattr(functools.partial(identity), name)
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        source = self.source and AttrSource(self.source, name)
        # Handle __slots__
        if name == "func":
            return self.func
        if name == "args":
            return variables.ListVariable(self.args, source=source)
        if name == "keywords":
            items = {ConstantVariable.create(k): v for k, v in self.keywords.items()}
            return variables.ConstDictVariable(items, source=source)
        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)
        raise_observed_exception(AttributeError, tx)

    def as_python_constant(self):
        return functools.partial(
            self.func.as_python_constant(),
            *[arg.as_python_constant() for arg in self.args],
            **{k: v.as_python_constant() for k, v in self.keywords.items()},
        )

    def guard_as_python_constant(self):
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
        return functools.partial(
            self.func.guard_as_python_constant(),
            *[v.guard_as_python_constant() for v in self.args],
            **{k: v.guard_as_python_constant() for k, v in self.keywords.items()},
        )


class PolyfilledFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "fn",
        "wrapped_fn",
        "traceable_fn",
        *VariableTracker._nonvar_fields,
    }

    @classmethod
    @functools.cache
    def _get_polyfill_handlers(cls) -> dict[Callable[..., Any], types.FunctionType]:
        return {}

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))

        return cls(value, source=source)

    def __init__(self, fn: _F, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn: _F = fn

        handler = self._get_polyfill_handlers().get(fn, fn)
        assert callable(handler), f"Polyfill handler {handler} is not callable for {fn}"
        for candidate_attr in (
            "__torch_dynamo_polyfill__",  # registered polyfill
            "__python_implementation__",  # self handler from third-party libraries
        ):
            candidate = getattr(handler, candidate_attr, None)
            if candidate:
                assert callable(candidate)
                traceable_fn = candidate
                break
        else:
            raise RuntimeError(
                f"Polyfill handler {handler} does not have a traceable function"
            )

        self.wrapped_fn: _F = handler
        self.traceable_fn: _F = traceable_fn

    @property
    def polyfill_fn(self) -> _F:
        return self.traceable_fn

    def can_constant_fold_through(self):
        return getattr(
            self.wrapped_fn, "__torch_dynamo_can_constant_fold_through__", False
        )

    def get_function(self):
        return self.as_python_constant()

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        ):
            result = (
                self.fn(  # use the original function which is faster than the polyfill
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                )
            )
            return VariableTracker.build(tx, result)

        # Special case for sum on tuple/list of ints
        if (
            self.fn is builtins.sum
            and len(args) == 1
            and not kwargs
            and isinstance(args[0], (variables.ListVariable, variables.TupleVariable))
            and all(
                (isinstance(x, variables.ConstantVariable) and isinstance(x.value, int))
                or (isinstance(x, variables.SymNodeVariable) and x.python_type() is int)
                for x in args[0].items
            )
        ):
            return variables.SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function",
                    torch.sym_sum,
                    (tuple(a.as_proxy() for a in args[0].items),),
                    {},
                ),
                sym_num=torch.sym_sum(
                    [
                        (
                            x.value
                            if isinstance(x, variables.ConstantVariable)
                            else x.sym_num
                        )
                        for x in args[0].items
                    ]
                ),
            )

        traceable_function_variable = VariableTracker.build(tx, self.traceable_fn)
        return traceable_function_variable.call_function(tx, args, kwargs)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__call__":
            return self.call_function(tx, args, kwargs)

        method = getattr(self.fn, name, None)
        assert method is not None, f"Member {name} not found in {self.fn}"
        assert is_function(method), f"Member {name} is not callable in {self.fn}"
        options = {}
        if self.source:
            options["source"] = AttrSource(self.source, name)
        polyfilled_method_variable = PolyfilledFunctionVariable(method, **options)
        return polyfilled_method_variable.call_function(tx, args, kwargs)

    def as_python_constant(self):
        return self.fn


class TracebackVariable(VariableTracker):
    # We don't track traceback. A call to any function in this module is a no-op
    def call_function(self, tx, args, kwargs): ...


class SysFunctionVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def exc_info(self, tx):
        if len(tx.exn_vt_stack):
            exn = tx.exn_vt_stack[-1]
            typ = exn.exc_type
            tb = None
            items = [
                VariableTracker.build(tx, typ),
                exn,
                VariableTracker.build(tx, tb),
            ]
        else:
            items = [
                variables.ConstantVariable(None),
                variables.ConstantVariable(None),
                variables.ConstantVariable(None),
            ]
        return variables.TupleVariable(items)

    def exception(self, tx):
        return self.exc_info(tx).items[1]

    def call_function(self, tx, args, kwargs):
        if self.value is sys.exc_info:
            return self.exc_info(tx)
        assert self.value is sys.exception
        return self.exception(tx)


from torch._higher_order_ops.triton_kernel_wrap import (
    create_tma_experimental_metadata,
    create_tma_stable_metadata,
    TMADescriptorMetadata,
    TritonHOPifier,
)


class DynamoTritonHOPifier(TritonHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        raise Unsupported(msg)

    def is_callable(self, maybe_callable: Any) -> bool:
        return isinstance(
            maybe_callable, (NestedUserFunctionVariable, UserFunctionVariable)
        )

    def get_value(self, val: Any) -> Any:
        return val.value

    def check_grid(self, grid) -> tuple[torch.fx.proxy.Proxy, ...]:
        from .lists import BaseListVariable

        if isinstance(grid, BaseListVariable):
            return grid.as_proxy()
        else:
            unimplemented_v2(
                gb_type="unsupported grid type for triton hop check_grid",
                context=f"grid type = {type(grid)}",
                explanation="`torch.compile` only supports list-like grid for check_grid",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

    def call_grid(self, grid, meta, tx):
        meta = {variables.ConstantVariable.create(k): v for k, v in meta.items()}
        grid = grid.call_function(tx, [meta], {})
        return grid

    # We use this function to wrap call_prune_configs
    def call_user_defined_fn(self, user_fn, args, kwargs, tx, variable):
        from .builder import SourcelessBuilder

        wrapped_user_function = SourcelessBuilder.create(tx, user_fn)
        result = wrapped_user_function.call_function(tx, args, kwargs)
        return result

    def wrap_user_defined_obj(self, user_obj, tx, variable, name):
        from .builder import VariableBuilder

        wrapped_user_obj = VariableBuilder(
            tx, AttrSource(variable.kernel_source, f"{name}")
        )._wrap(user_obj)
        return wrapped_user_obj

    def maybe_unpack_configs(self, configs, tx):
        # unpack the list of configs
        configs = configs.unpack_var_sequence(tx)

        # guard_as_python_constant inserts guards for Dynamo to check if the configs object changed.
        configs = [config.guard_as_python_constant() for config in configs]

        return configs

    def maybe_unpack_heuristic_result(self, result: Any) -> Any:
        if not result.is_python_constant():
            self.raise_unsupported(
                "@triton.heuristics must return constant values because configs can only contain constant values."
            )

        return result.guard_as_python_constant()

    # We need to override call_getitem here so that we can add the source in the case
    # where we call the triton kernel with a grid
    def call_getitem(
        self,
        variable: "TritonKernelVariable",
        args: Sequence[Any],
    ) -> "TritonKernelVariable":
        # __getitem__ should only be called if we don't already have a grid
        # Only grid needs to be passed
        if variable.grid is not None or len(args) != 1:
            self.raise_unsupported(
                "Triton kernels should be called with only a single grid"
            )
        return type(variable)(
            kernel=variable.kernel,
            kernel_idx=variable.kernel_idx,
            grid=args[0],
            kernel_source=variable.source,
        )

    def call_HOP(self, variable, grids, combined_args_raw, tx) -> ConstantVariable:
        from .constant import ConstantVariable
        from .dicts import ConstDictVariable

        # as we can only pass tensors as non-const args in fx graph,
        # here we replace TMA descriptors
        # (TMADescriptorExperimentalVariable and TMADescriptorStableVariable
        # instances) with the underlying tensors, while moving the
        # TMA descriptor-related metadata to a separate argument,
        # so that we can reconstruct the TMA descriptors downstream
        tma_descriptor_metadata: TMADescriptorMetadata = {}
        for k in list(combined_args_raw.keys()):
            v = combined_args_raw[k]
            if isinstance(
                v, (TMADescriptorExperimentalVariable, TMADescriptorStableVariable)
            ):
                tma_descriptor_metadata[k] = v.to_metadata()
                combined_args_raw[k] = v.get_tensor()

        combined_args = {
            variables.ConstantVariable.create(k): v
            for k, v in combined_args_raw.items()
        }

        from torch._higher_order_ops.triton_kernel_wrap import (
            kernel_side_table,
            triton_kernel_wrapper_mutation,
        )

        # Combine args and kwargs and pass as a dict so that if user defined triton
        # kernel uses variables as 'grid' or 'kernel', it does not conflict with
        # parameters of the wrapper function
        constant_args = {
            k: v.as_python_constant()
            for k, v in combined_args_raw.items()
            if isinstance(v, ConstantVariable)
        }
        non_constant_args = {
            k: v
            for k, v in combined_args.items()
            if not isinstance(v, ConstantVariable)
        }

        for v in non_constant_args.values():
            v = v.realize()
            if not isinstance(v, (variables.TensorVariable, variables.SymNodeVariable)):
                self.raise_unsupported(
                    f"Unexpected argument type for a Triton kernel: {repr(v)}."
                )

        constant_args_idx = kernel_side_table.add_constant_args(constant_args)
        meta = ConstDictVariable(non_constant_args, dict)
        tx.output.create_proxy(
            "call_function",
            triton_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": variable.kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": grids,
                "tma_descriptor_metadata": tma_descriptor_metadata,
                "kwargs": meta.as_proxy(),
            },
        )

        return variables.ConstantVariable(
            None,
        )


dynamo_triton_hopifier_singleton = DynamoTritonHOPifier()


class TritonKernelVariable(VariableTracker):
    grid: "TritonGridType"
    kernel: "TritonKernelType"
    kernel_idx: Optional[int]
    kernel_source: "AttrSource"

    def __init__(self, kernel, kernel_idx, grid, **kwargs) -> None:
        self.kernel_source = kwargs.pop("kernel_source", None)
        super().__init__(**kwargs)
        dynamo_triton_hopifier_singleton.init_variable(self, kernel, kernel_idx, grid)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return dynamo_triton_hopifier_singleton.call_triton_kernel(
            self, args, kwargs, tx
        )

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__":
            return dynamo_triton_hopifier_singleton.call_getitem(self, args)
        elif name == "run":
            return dynamo_triton_hopifier_singleton.call_run(self, args, kwargs, tx)

        # Bail out to parent's implementation
        return super().call_method(tx, name, args, kwargs)

    def specialize_symbolic(self, arg: Any) -> Any:
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, SymNodeVariable):
            return ConstantVariable.create(arg.evaluate_expr())
        return arg


class TMADescriptorExperimentalVariable(VariableTracker):
    def __init__(
        self,
        data_ptr: "variables.DataPtrVariable",
        dims: "list[ConstantVariable]",
        block_dims: "list[ConstantVariable]",
        element_size: "ConstantVariable",
        **kwargs,
    ):
        assert isinstance(data_ptr, variables.DataPtrVariable)
        super().__init__(**kwargs)
        self.data_ptr = data_ptr
        self.dims = dims
        self.block_dims = block_dims
        self.element_size = element_size

    def to_metadata(self):
        return create_tma_experimental_metadata(
            [dim.as_proxy() for dim in self.dims],
            [dim.as_proxy() for dim in self.block_dims],
            self.element_size.as_proxy(),
        )

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                "triton.tools.experimental_descriptor",
                f"create_{len(self.dims)}d_tma_descriptor",
            )
        )
        self.data_ptr.reconstruct(codegen)
        args = [*self.dims, *self.block_dims, self.element_size]
        codegen.foreach(args)
        codegen.call_function(len(args) + 1, False)

    def get_tensor(self):
        return self.data_ptr.from_tensor


class TMADescriptorStableVariable(VariableTracker):
    def __init__(
        self,
        tensor: "variables.TensorVariable",
        block_shape: "variables.ListVariable",
        **kwargs,
    ):
        assert isinstance(tensor, variables.TensorVariable)
        super().__init__(**kwargs)
        self.tensor = tensor
        self.block_shape = block_shape

    def to_metadata(self):
        return create_tma_stable_metadata(
            self.block_shape.as_proxy(),
        )

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                "triton.tools.tensor_descriptor",
                "TensorDescriptor",
            )
        )
        codegen.load_method("from_tensor")
        self.tensor.reconstruct(codegen)
        codegen(self.block_shape)
        codegen.call_method(2)

    def get_tensor(self) -> "variables.TensorVariable":
        return self.tensor


class CreateTMADescriptorExperimentalVariable(VariableTracker):
    def __init__(
        self,
        rank: int,
        **kwargs,
    ) -> None:
        assert rank in (1, 2)
        super().__init__(**kwargs)
        self.rank = rank

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        ptr = kwargs["ptr"] if "ptr" in kwargs else args[0]

        if not isinstance(ptr, variables.DataPtrVariable):
            raise Unsupported(
                "Please ensure there were no graph breaks between "
                f"create_{self.rank}d_tma_descriptor and the upstream "
                ".data_ptr() call."
            )

        if self.rank == 1:
            assert len(args) + len(kwargs) == 4
            dims = [
                kwargs["dim"] if "dim" in kwargs else args[1],
            ]
            block_dims = [
                kwargs["block_dim"] if "block_dim" in kwargs else args[2],
            ]
        else:
            assert len(args) + len(kwargs) == 6
            dims = [
                kwargs["dim1"] if "dim1" in kwargs else args[1],
                kwargs["dim0"] if "dim0" in kwargs else args[2],
            ]
            block_dims = [
                kwargs["block_dim1"] if "block_dim1" in kwargs else args[3],
                kwargs["block_dim0"] if "block_dim0" in kwargs else args[4],
            ]
        element_size = kwargs["element_size"] if "element_size" in kwargs else args[-1]

        return TMADescriptorExperimentalVariable(
            data_ptr=ptr,
            dims=dims,
            block_dims=block_dims,
            element_size=element_size,
        )


class CreateTMADescriptorStableVariable(VariableTracker):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        tensor = kwargs["tensor"] if "tensor" in kwargs else args[0]
        block_shape = kwargs["block_shape"] if "block_shape" in kwargs else args[1]

        return TMADescriptorStableVariable(
            tensor=tensor,
            block_shape=block_shape,
        )
