"""
This module provides functionality for resuming Python execution at specific points in code,
primarily used by PyTorch Dynamo for control flow handling and optimization. It implements
bytecode transformation and execution state management to enable:

- Resuming execution at arbitrary points in Python bytecode
- Managing context managers and their state across execution boundaries
- Transforming and generating new code objects with preserved execution state
- Supporting Python 3.11+ exception handling and block management
- Restoring torch function mode stacks and other execution context

The module is critical for PyTorch Dynamo's ability to optimize code while preserving
Python semantics and execution state.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import sys
import types
import weakref
from typing import Any, cast, NoReturn, TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.utils.weak import _InternalTensorWeakRef, WeakIdKeyDictionary

from .bytecode_transformation import (
    add_push_null,
    bytecode_from_template,
    create_binary_subscr,
    create_call_function,
    create_instruction,
    create_jump_absolute,
    create_load_const,
    Instruction,
    overwrite_instruction,
    transform_code_object,
    unique_id,
)
from .utils import ExactWeakKeyDictionary


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from contextlib import AbstractContextManager

    from .output_graph import CodeOptions


# taken from code.h in cpython
CO_OPTIMIZED = 0x0001
CO_NEWLOCALS = 0x0002
CO_VARARGS = 0x0004
CO_VARKEYWORDS = 0x0008
CO_NESTED = 0x0010
CO_GENERATOR = 0x0020
CO_NOFREE = 0x0040
CO_COROUTINE = 0x0080
CO_ITERABLE_COROUTINE = 0x0100
CO_ASYNC_GENERATOR = 0x0200

# trace_rules.py import this constant for consistency
TORCH_DYNAMO_RESUME_IN_PREFIX = "torch_dynamo_resume_in"
IS_TRACING_RESUME_PROLOGUE_VARNAME = "__is_tracing_resume_prologue"
RESUME_ARGS_VARNAME = "__torch_dynamo_resume_args"


def _boxed_resume_arg_name(code: types.CodeType) -> str | None:
    metadata = ContinueExecutionCache.generated_code_metadata.get(code)
    if metadata is None:
        return None
    return metadata.boxed_resume_arg_name


def _boxed_resume_arg_names(code: types.CodeType) -> tuple[str, ...]:
    metadata = ContinueExecutionCache.generated_code_metadata.get(code)
    if metadata is None:
        return ()
    return metadata.boxed_resume_arg_names


def _boxed_resume_arg_sources(code: types.CodeType) -> dict[str, dict[int, str]]:
    metadata = ContinueExecutionCache.generated_code_metadata.get(code)
    if metadata is None:
        return {}
    return metadata.boxed_resume_arg_sources


def _boxed_resume_arg_external_lifetime_roots(
    code: types.CodeType,
) -> set[tuple[str, int]]:
    metadata = ContinueExecutionCache.generated_code_metadata.get(code)
    if metadata is None:
        return set()
    return {
        (owner_name, index)
        for owner_name, indexes in metadata.boxed_resume_arg_external_lifetime_roots.items()
        for index in indexes
    }


def _is_boxed_resume_code(code: types.CodeType) -> bool:
    return _boxed_resume_arg_name(code) is not None


def _boxed_resume_local_source_info(
    code: types.CodeType,
) -> dict[int, tuple[str, bool, bool, bool]]:
    metadata = ContinueExecutionCache.generated_code_metadata.get(code)
    if metadata is None:
        return {}
    original_code = metadata.code
    input_end = original_code.co_argcount + original_code.co_kwonlyargcount
    input_names = set(original_code.co_varnames[:input_end])
    varargs_name = None
    varkw_name = None
    if original_code.co_flags & CO_VARARGS:
        varargs_name = original_code.co_varnames[input_end]
        input_names.add(varargs_name)
        input_end += 1
    if original_code.co_flags & CO_VARKEYWORDS:
        varkw_name = original_code.co_varnames[input_end]
        input_names.add(varkw_name)
    return {
        idx: (
            name,
            name in input_names,
            name == varargs_name,
            name == varkw_name,
        )
        for idx, name in metadata.boxed_resume_local_argname_indexes.items()
    }


def _boxed_resume_arg_indexes_to_clear(code: types.CodeType) -> tuple[int, ...]:
    metadata = ContinueExecutionCache.generated_code_metadata.get(code)
    if metadata is None:
        return ()
    return metadata.boxed_resume_arg_indexes_to_clear


def _maybe_clear_tensor_resume_arg(resume_args: list[Any], idx: int) -> None:
    if (
        idx < len(resume_args)
        and isinstance(resume_args[idx], Tensor)
        and not torch._C._autograd._top_saved_tensors_default_hooks(True)
        and not _resume_arg_has_observable_destruction(resume_args[idx])
    ):
        resume_args[idx] = None


class _ResumeArgSnapshot:
    """Read-only runtime owner for a value needed during event replay."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        object.__setattr__(self, "_value", value)

    @property
    def value(self) -> Any:
        return object.__getattribute__(self, "_value")

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("resume snapshots are read-only")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("resume snapshots are read-only")


def _make_resume_arg_snapshot(value: Any) -> _ResumeArgSnapshot:
    return _ResumeArgSnapshot(value)


def _resume_arg_owner_value(owner: Any, index: int) -> tuple[bool, Any]:
    if isinstance(owner, _ResumeArgSnapshot):
        return (True, owner.value) if index == 0 else (False, None)
    if isinstance(owner, list) and 0 <= index < len(owner):
        return True, owner[index]
    return False, None


def _clear_resume_arg_owner_value(owner: Any, index: int) -> None:
    if isinstance(owner, _ResumeArgSnapshot):
        if index == 0:
            object.__setattr__(owner, "_value", None)
    elif isinstance(owner, list) and 0 <= index < len(owner):
        owner[index] = None


def _clear_resume_arg_path(resume_args: Any, path: tuple[Any, ...]) -> None:
    """Drop a fully materialized resume-carrier slot."""
    if len(path) == 1 and isinstance(path[0], int):
        _clear_resume_arg_owner_value(resume_args, path[0])


def _refresh_frame_locals_if_resume_carrier_mutated(
    resume_args: list[Any], expected_size: int
) -> None:
    """Discard stale generated-frame owners after reentrant carrier mutation.

    On Python 3.12 and earlier, reading ``frame.f_locals`` caches strong
    references that later STORE_FAST/DELETE_FAST instructions do not update.
    A finalizer can materialize that cache while clearing a Dynamo-only resume
    carrier, an observation that has no eager source-frame counterpart. Refresh
    the generated frame after replay so its cache reflects the completed
    source events rather than extending their values' lifetimes.
    """
    if len(resume_args) != expected_size:
        sys._getframe(1).f_locals


def _snapshot_resume_arg_identities(resume_args: Any) -> tuple[int, ...]:
    """Capture carrier-slot identity without adding another strong owner."""
    if isinstance(resume_args, _ResumeArgSnapshot):
        return (id(resume_args.value),)
    return tuple(map(id, resume_args))


def _clear_resume_arg_path_if_unchanged(
    resume_args: Any,
    path: tuple[Any, ...],
    expected_ids: tuple[int, ...],
    preserve_observable_lifetime: bool,
) -> None:
    """Clear only the carrier value observed before reentrant event replay."""
    if not (len(path) == 1 and isinstance(path[0], int)):
        return
    index = path[0]
    exists, value = _resume_arg_owner_value(resume_args, index)
    if not exists or index >= len(expected_ids):
        return
    if id(value) != expected_ids[index]:
        # User finalization reentered this generated frame and replaced the
        # compiler-only carrier slot. Retain the injected value to frame exit;
        # releasing it at the source value's boundary would create an eager-
        # impossible callback before the next user-code observation.
        return
    if preserve_observable_lifetime:
        _maybe_clear_resume_arg_path(resume_args, path)
    else:
        _clear_resume_arg_path(resume_args, path)


_WEAKREF_CALLBACK_DESCRIPTOR = vars(weakref.ReferenceType)["__callback__"]


def _is_internal_lifetime_weakref(ref: weakref.ReferenceType[Any]) -> bool:
    if type(ref) is _InternalTensorWeakRef:
        return True
    if type(ref) in weakref.ProxyTypes:
        return False
    callback = _WEAKREF_CALLBACK_DESCRIPTOR.__get__(ref, type(ref))
    if not (
        type(callback) is types.FunctionType
        and callback.__module__ == "torch.utils.weak"
        and callback.__qualname__ == "WeakIdKeyDictionary.__init__.<locals>.remove"
    ):
        return False
    defaults = callback.__defaults__
    if defaults is None or len(defaults) != 1:
        return False
    owner_ref = defaults[0]
    if type(owner_ref) is not weakref.ReferenceType:
        return False
    owner = owner_ref()
    return (
        owner is not None
        and type(owner) is WeakIdKeyDictionary
        and owner._is_internal_lifetime_observer
    )


def _resume_arg_has_observable_destruction(
    value: Any,
) -> bool:
    """Whether ``value`` currently has an observable destruction hook.

    External strong-owner provenance is tracked while tracing and propagated
    through resume metadata.  Runtime cleanup only needs to recheck mutable
    object properties (weakrefs, ``__del__``, and nested contents); walking all
    GC referrers here would put a heap-wide scan in every compiled invocation.
    """
    safe_leaf_types = (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        bytearray,
        range,
        object,
    )
    worklist = [value]
    seen: set[int] = set()
    while worklist:
        current = worklist.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        if type(current) in (list, tuple):
            worklist.extend(current)
        elif type(current) is dict:
            worklist.extend(current.keys())
            worklist.extend(current.values())
        elif type(current) in (set, frozenset):
            worklist.extend(current)
        elif type(current) in (Tensor, Parameter):
            if inspect.getattr_static(type(current), "__del__", None) is not None:
                return True
            user_refs = [
                ref
                for ref in weakref.getweakrefs(current)
                if not _is_internal_lifetime_weakref(ref)
            ]
            if user_refs:
                return True
            worklist.append(object.__getattribute__(current, "__dict__"))
        elif type(current) in safe_leaf_types:
            continue
        else:
            return True
    return False


def _maybe_clear_resume_arg_path(resume_args: Any, path: tuple[Any, ...]) -> None:
    """Clear a carrier slot unless frame-lifetime ownership is observable."""
    if not (len(path) == 1 and isinstance(path[0], int)):
        return
    index = path[0]
    exists, value = _resume_arg_owner_value(resume_args, index)
    if not exists:
        return
    # Non-inline saved-tensor hooks intentionally do not guard Dynamo's cache:
    # a graph first compiled without hooks can run later with hooks installed.
    # A pack hook can add a weakref to a graph input after this pre-graph
    # cleanup point, so retain the eager frame owner for the duration of the
    # resume whenever hooks are currently active.
    if torch._C._autograd._top_saved_tensors_default_hooks(True):
        return
    if not _resume_arg_has_observable_destruction(value):
        _clear_resume_arg_owner_value(resume_args, index)


def create_clear_resume_arg(resume_args_varname: str, index: int) -> list[Instruction]:
    return [
        create_load_const(None),
        create_instruction("LOAD_FAST", argval=resume_args_varname),
        create_load_const(index),
        create_instruction("STORE_SUBSCR"),
    ]


# If is_resume - this codegen is for a resume function
def _initial_push_null(insts: list[Instruction]) -> None:
    if sys.version_info >= (3, 11):
        insts.append(create_instruction("PUSH_NULL"))
        if sys.version_info < (3, 13):
            insts.append(create_instruction("SWAP", arg=2))


# Generates bytecode from template and splits the code where LOAD_FAST dummy is present.
def _bytecode_from_template_with_split(
    template: Callable[..., Any],
    stack_index: int,
    varname_map: dict[str, Any] | None = None,
) -> tuple[list[Instruction], list[Instruction]]:
    template_code = bytecode_from_template(template, varname_map=varname_map)
    template_code.append(create_instruction("POP_TOP"))

    # adjust exception table entry depth
    for inst in template_code:
        if inst.exn_tab_entry:
            inst.exn_tab_entry.depth += stack_index

    # search for LOAD_FAST dummy and replace it with 2 NOPs (we can break up the bytecode between them)
    dummy_idx, dummy_inst = next(
        (
            (i, inst)
            for i, inst in enumerate(template_code)
            if inst.opname in ("LOAD_FAST", "LOAD_FAST_BORROW")
            and inst.argval == "dummy"
        ),
        (None, None),
    )
    if dummy_idx is None:
        raise AssertionError("LOAD_FAST dummy instruction not found in template code")
    if dummy_inst is None:
        raise AssertionError("LOAD_FAST dummy instruction instance is None")

    # replace LOAD_FAST dummy with first NOP marking exception area
    overwrite_instruction(dummy_inst, [create_instruction("NOP")])

    # POP_TOP follows LOAD_FAST dummy - replace with NOP marking end of exception area
    if template_code[dummy_idx + 1].opname != "POP_TOP":
        raise AssertionError(
            f"Expected POP_TOP after LOAD_FAST dummy, got {template_code[dummy_idx + 1].opname}"
        )
    overwrite_instruction(template_code[dummy_idx + 1], [create_instruction("NOP")])

    return template_code[: dummy_idx + 1], template_code[dummy_idx + 1 :]


def _try_except_tf_mode_template(dummy: Any) -> None:
    # NOTE: Make sure this name matches what is generated by symbolic_convert:import_source
    # on torch._dynamo.utils.
    # pyrefly: ignore [unknown-name]
    global __import_torch_dot__dynamo_dot_utils
    try:
        dummy
    except:
        __import_torch_dot__dynamo_dot_utils._pop_torch_function_stack()  # type: ignore[name-defined]
        raise


@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int
    target_values: tuple[Any, ...] | None = None

    def try_except_torch_function_mode(
        self, code_options: CodeOptions, cleanup: list[Instruction]
    ) -> list[Instruction]:
        """
        Codegen based off of:
        try:
            (rest)
        except:
            (pop the torch function mode that was pushed for this context)
            raise
        """
        setup_try_except, epilogue = _bytecode_from_template_with_split(
            _try_except_tf_mode_template,
            self.stack_index,
        )
        cleanup[:] = epilogue + cleanup

        return setup_try_except

    # If we do not want to destroy the stack, we can do the same thing as a
    # `SETUP_WITH` block, only that we store the context manager in a local_symbol
    def try_finally(
        self, code_options: CodeOptions, cleanup: list[Instruction]
    ) -> list[Instruction]:
        """
        Codegen based off of:
        load args
        enter context
        try:
            (rest)
        finally:
            exit context
        """
        # NOTE: we assume that TOS is a context manager CLASS!
        # pyrefly: ignore [implicit-any]
        load_args = []
        if self.target_values:
            load_args = [create_load_const(val) for val in self.target_values]
        ctx_name = unique_id(f"___context_manager_{self.stack_index}")
        if ctx_name not in code_options["co_varnames"]:
            code_options["co_varnames"] += (ctx_name,)
        for name in ["__enter__", "__exit__"]:
            if name not in code_options["co_names"]:
                code_options["co_names"] += (name,)

        create_ctx: list[Instruction] = []
        _initial_push_null(create_ctx)
        create_ctx.extend(
            [
                *load_args,
                *create_call_function(len(load_args), False),
                create_instruction("STORE_FAST", argval=ctx_name),
            ]
        )

        def _template(ctx: AbstractContextManager[Any], dummy: Any) -> None:
            ctx.__enter__()
            try:
                dummy
            finally:
                ctx.__exit__(None, None, None)

        setup_try_finally, epilogue = _bytecode_from_template_with_split(
            _template, self.stack_index, varname_map={"ctx": ctx_name}
        )
        cleanup[:] = epilogue + cleanup
        return create_ctx + setup_try_finally

    def __call__(
        self, code_options: dict[str, Any], cleanup: list[Instruction]
    ) -> tuple[list[Instruction], Instruction | None]:
        """
        Codegen based off of:
        with ctx(args):
            (rest)
        """
        # NOTE: we assume that TOS is a context manager CLASS!
        # pyrefly: ignore [implicit-any]
        load_args = []
        if self.target_values:
            load_args = [create_load_const(val) for val in self.target_values]

        create_ctx: list[Instruction] = []
        # Do not push NULL in Python 3.14+ since the NULL should be on the symbolic stack.
        if sys.version_info < (3, 14):
            _initial_push_null(create_ctx)
        create_ctx.extend(
            [
                *load_args,
                *create_call_function(len(load_args), False),
            ]
        )

        def _template(ctx: AbstractContextManager[Any], dummy: Any) -> None:
            with ctx:
                dummy

        setup_with, epilogue = _bytecode_from_template_with_split(
            _template, self.stack_index
        )
        cleanup[:] = epilogue + cleanup

        load_fast_ctx_inst = next(
            (
                inst
                for inst in setup_with
                if inst.opname in ("LOAD_FAST", "LOAD_FAST_BORROW")
                and inst.argval == "ctx"
            ),
            None,
        )
        if load_fast_ctx_inst is None:
            raise AssertionError("LOAD_FAST ctx instruction not found in setup_with")
        # ctx already loaded on stack before the template - no need to LOAD_FAST
        overwrite_instruction(load_fast_ctx_inst, [create_instruction("NOP")])

        # 3.11+ only
        push_exc_info_gen = (
            inst for inst in epilogue if inst.opname == "PUSH_EXC_INFO"
        )
        push_exc_info_inst = next(push_exc_info_gen, None)
        # expect only 1 PUSH_EXC_INFO in epilogue
        if next(push_exc_info_gen, None) is not None:
            raise AssertionError("Expected only 1 PUSH_EXC_INFO in epilogue")

        return create_ctx + setup_with, push_exc_info_inst


@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: list[Instruction] = dataclasses.field(default_factory=list)
    boxed_resume_arg_name: str | None = None
    # Includes the active boxed argument and older boxed carriers forwarded as
    # locals through stacked continuation functions.
    boxed_resume_arg_names: tuple[str, ...] = ()
    boxed_resume_arg_sources: dict[str, dict[int, str]] = dataclasses.field(
        default_factory=dict
    )
    # Carrier roots that may be observed outside the generated frame. Stack
    # values cross an opaque graph-break boundary, while source inputs remain
    # user-owned; both need CPython frame-lifetime semantics in later resumes.
    boxed_resume_arg_external_lifetime_roots: dict[str, frozenset[int]] = (
        dataclasses.field(default_factory=dict)
    )
    boxed_resume_local_argname_indexes: dict[int, str] = dataclasses.field(
        default_factory=dict
    )
    boxed_resume_arg_indexes_to_clear: tuple[int, ...] = ()
    # Python 3.11+ fields
    # NOTE: Python 3.11 removed blocks, but for our purposes, a "block" consists
    # of instructions of all exception table entries that have the same target.

    # map from PUSH_EXC_INFO's in the prefix to original block target offset
    prefix_block_target_offset_remap: list[int] = dataclasses.field(
        default_factory=list
    )
    # per-offset map from new block target offsets to original block target offsets
    block_target_offset_remap: dict[tuple[int, int], dict[int, int]] = (
        dataclasses.field(default_factory=dict)
    )


def _filter_iter(
    l1: Iterable[Any],
    l2: Iterable[Any],
    cond: Callable[[Any, Any], bool],
) -> list[Any]:
    """
    Two-pointer conditional filter.
    e.g. _filter_iter(insts, sorted_offsets, lambda i, o: i.offset == o)
    returns the instructions with offsets in sorted_offsets
    """
    it = iter(l2)
    res: list[Instruction] = []
    try:
        cur = next(it)
        for val in l1:
            if cond(val, cur):
                res.append(val)
                cur = next(it)
    except StopIteration:
        pass
    return res


def _load_tuple_and_call(tup: tuple[Any, ...]) -> list[Instruction]:
    insts: list[Instruction] = []
    _initial_push_null(insts)
    insts.extend(create_load_const(val) for val in tup)
    insts.extend(create_call_function(len(tup), False))
    return insts


class ContinueExecutionCache:
    cache = ExactWeakKeyDictionary()
    generated_code_metadata = ExactWeakKeyDictionary()

    @classmethod
    def lookup(
        cls, code: types.CodeType, lineno: int, init_offset: int, *key: Any
    ) -> types.CodeType:
        if code not in cls.cache:
            cls.cache[code] = {}
        key = tuple(key)
        if key not in cls.cache[code]:
            cls.cache[code][key] = cls.generate(code, lineno, init_offset, *key)
        return cls.cache[code][key]

    @classmethod
    def generate(
        cls,
        code: types.CodeType,
        lineno: int,
        init_offset: int,
        resume_offset: int,
        setup_fn_target_offsets: tuple[int, ...],  # only used in Python 3.11+
        nstack: int,
        argnames: tuple[str, ...],
        argnames_null: tuple[str, ...],
        setup_fns: tuple[ReenterWith, ...],
        handle_inactive_ctx: bool,
        stack_ctx_vars: tuple[tuple[int, tuple[Any, ...]], ...],
        argnames_ctx_vars: tuple[tuple[str, tuple[Any, ...]], ...],
        null_idxes: tuple[int, ...],
        # mainly used to ensure distinct code objects per stack trace,
        # which prevents excessive recompilation of inner frames
        nested_code_objs: tuple[types.CodeType],
        inherited_boxed_resume_arg_sources: tuple[
            tuple[str, tuple[tuple[int, str], ...]], ...
        ],
        inherited_boxed_resume_arg_external_lifetime_roots: tuple[
            tuple[str, tuple[int, ...]], ...
        ],
        external_lifetime_argnames: tuple[str, ...],
        tensor_resume_arg_indexes: tuple[int, ...],
        entry_clear_resume_arg_indexes: tuple[int, ...],
        force_boxed_resume: bool,
        # Are we currently graph breaking on an instruction that doesn't push
        # its result to the stack? If so, and we are not the leaf resume, then we need to pop
        # the result of calling the next resume function.
        pop_nested_resume_result: bool,
    ) -> types.CodeType:
        if resume_offset is None:
            raise AssertionError("resume_offset must not be None")
        if code.co_flags & (
            CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR
        ):
            raise AssertionError(
                "Cannot generate resume function for generator, coroutine, or async generator"
            )
        if not (code.co_flags & CO_OPTIMIZED):
            raise AssertionError("Code object must have CO_OPTIMIZED flag set")
        if code in ContinueExecutionCache.generated_code_metadata:
            return cls.generate_based_on_original_code_object(
                code,
                lineno,
                init_offset,
                resume_offset,
                setup_fn_target_offsets,
                nstack,
                argnames,
                argnames_null,
                setup_fns,
                handle_inactive_ctx,
                stack_ctx_vars,
                argnames_ctx_vars,
                null_idxes,
                nested_code_objs,
                inherited_boxed_resume_arg_sources,
                inherited_boxed_resume_arg_external_lifetime_roots,
                external_lifetime_argnames,
                tensor_resume_arg_indexes,
                entry_clear_resume_arg_indexes,
                force_boxed_resume,
                pop_nested_resume_result,
            )

        is_py311_plus = sys.version_info >= (3, 11)
        meta = ResumeFunctionMetadata(code)

        def update(
            instructions: list[Instruction], code_options: dict[str, Any]
        ) -> None:
            meta.instructions = copy.deepcopy(instructions)

            resume_arg_names = ["__nested_resume_fns", "__nested_frame_values"]
            resume_arg_names += [f"___stack{i}" for i in range(nstack)]
            resume_arg_names.extend(v for v in argnames if v not in resume_arg_names)
            frame_value_names = set(argnames)
            frame_value_names.update(f"___stack{i}" for i in range(nstack))
            future_deleted_fast_names = {
                inst.argval
                for inst in instructions
                if inst.opname == "DELETE_FAST"
                and inst.offset is not None
                and inst.offset >= resume_offset
            }
            future_replaced_fast_names = {
                inst.argval
                for inst in instructions
                if inst.opname in ("DELETE_FAST", "STORE_FAST")
                and inst.offset is not None
                and inst.offset >= resume_offset
            }
            replaced_frame_value_names = future_replaced_fast_names & frame_value_names
            # Leaf resumes need a boxed carrier when they restore operand-stack
            # values. Lifetime-boundary continuations also need one so the old
            # continuation can transfer ownership before the next one runs.
            # Keep other straight-line, stackless resumes positional unless a
            # source DELETE_FAST requires early release; boxing those frames
            # adds provenance and guards on incidental carrier aliases.
            deleted_frame_value_names = future_deleted_fast_names & frame_value_names
            boxed_resume = (
                force_boxed_resume
                or bool(entry_clear_resume_arg_indexes)
                or bool(deleted_frame_value_names)
                or (not nested_code_objs and bool(nstack))
            )
            resume_args_varname = RESUME_ARGS_VARNAME
            if boxed_resume:
                unavailable_names = (
                    set(resume_arg_names)
                    | set(argnames_null)
                    | set(code_options["co_varnames"])
                )
                while resume_args_varname in unavailable_names:
                    resume_args_varname = unique_id(RESUME_ARGS_VARNAME)
            meta.boxed_resume_arg_name = resume_args_varname if boxed_resume else None
            meta.boxed_resume_local_argname_indexes = (
                {
                    idx: name
                    for idx, name in enumerate(resume_arg_names)
                    if name in argnames
                }
                if boxed_resume
                else {}
            )
            meta.boxed_resume_arg_names = tuple(
                dict.fromkeys(
                    (
                        *(name for name, _ in inherited_boxed_resume_arg_sources),
                        *([resume_args_varname] if boxed_resume else []),
                    )
                )
            )
            meta.boxed_resume_arg_sources = {
                name: dict(sources)
                for name, sources in inherited_boxed_resume_arg_sources
            }
            meta.boxed_resume_arg_external_lifetime_roots = {
                name: frozenset(indexes)
                for name, indexes in inherited_boxed_resume_arg_external_lifetime_roots
            }
            if boxed_resume:
                meta.boxed_resume_arg_sources[resume_args_varname] = (
                    meta.boxed_resume_local_argname_indexes
                )
                external_lifetime_indexes = set(range(2, 2 + nstack))
                external_lifetime_indexes.update(
                    index
                    for index, name in meta.boxed_resume_local_argname_indexes.items()
                    if name in external_lifetime_argnames
                )
                if external_lifetime_indexes:
                    meta.boxed_resume_arg_external_lifetime_roots[
                        resume_args_varname
                    ] = frozenset(external_lifetime_indexes)
            args = [resume_args_varname] if boxed_resume else resume_arg_names
            nested_resume_args_varname = None
            if nested_code_objs and not _is_boxed_resume_code(nested_code_objs[-1]):
                unavailable_names = (
                    set(args)
                    | set(resume_arg_names)
                    | set(argnames_null)
                    | set(code_options["co_varnames"])
                )
                nested_resume_args_varname = "__nested_resume_args"
                while nested_resume_args_varname in unavailable_names:
                    nested_resume_args_varname = unique_id("__nested_resume_args")
            freevars = tuple(code_options["co_cellvars"] or []) + tuple(
                code_options["co_freevars"] or []
            )
            freevars = tuple(sorted(freevars))
            code_options["co_name"] = (
                f"{TORCH_DYNAMO_RESUME_IN_PREFIX}_{code_options['co_name']}_at_{lineno}"
            )
            if is_py311_plus:
                qualified_path = code_options["co_qualname"].rsplit(".", maxsplit=1)
                if len(qualified_path) == 1:
                    code_options["co_qualname"] = code_options["co_name"]
                else:
                    if len(qualified_path) != 2:
                        raise AssertionError(
                            f"Expected qualified path to have 2 parts, got {len(qualified_path)}"
                        )
                    module_name, co_name = qualified_path
                    code_options["co_qualname"] = (
                        f"{module_name}.{TORCH_DYNAMO_RESUME_IN_PREFIX}_{co_name}_at_{lineno}"
                    )
            code_options["co_firstlineno"] = lineno
            code_options["co_cellvars"] = ()
            code_options["co_freevars"] = freevars
            code_options["co_argcount"] = len(args)
            code_options["co_posonlyargcount"] = 0
            code_options["co_kwonlyargcount"] = 0
            code_options["co_varnames"] = tuple(
                args
                + (
                    [v for v in resume_arg_names if v not in args]
                    if boxed_resume
                    else []
                )
                + [v for v in argnames_null if v not in args]
                + (
                    [nested_resume_args_varname]
                    if nested_resume_args_varname is not None
                    else []
                )
                + [
                    v
                    for v in code_options["co_varnames"]
                    if v not in args and v not in resume_arg_names
                ]
                + [IS_TRACING_RESUME_PROLOGUE_VARNAME]
            )
            code_options["co_flags"] = code_options["co_flags"] & ~(
                CO_VARARGS | CO_VARKEYWORDS
            )
            target = next(i for i in instructions if i.offset == resume_offset)

            resume_arg_indexes_to_clear = set(tensor_resume_arg_indexes)
            resume_arg_indexes_to_clear.update(entry_clear_resume_arg_indexes)
            # The generated prologue has already copied frame values into real
            # fast locals.  A boxed slot for a subsequently replaced local is
            # therefore redundant ownership; retaining it would delay the
            # old value's finalizer if this resume later falls back to eager.
            resume_arg_indexes_to_clear.update(
                idx
                for idx, name in enumerate(resume_arg_names)
                if name in replaced_frame_value_names
            )
            if (
                target.opname == "STORE_FAST"
                and target.argval in future_deleted_fast_names
                and nstack
            ):
                resume_arg_indexes_to_clear.add(1 + nstack)
            meta.boxed_resume_arg_indexes_to_clear = (
                tuple(sorted(resume_arg_indexes_to_clear)) if boxed_resume else ()
            )

            prefix = []
            if is_py311_plus:
                if freevars:
                    prefix.append(
                        create_instruction("COPY_FREE_VARS", arg=len(freevars))
                    )
                prefix.append(create_instruction("RESUME", arg=0))

            # Set is_tracing_resume_prologue to prevent graph breaks.
            # This doesn't really do anything at runtime, but dynamo will trace this
            # and will know that we're in a resume function prologue.
            prefix.extend(
                [
                    create_instruction("LOAD_CONST", argval=True),
                    create_instruction(
                        "STORE_FAST", argval=IS_TRACING_RESUME_PROLOGUE_VARNAME
                    ),
                ]
            )
            if boxed_resume:
                for idx, name in enumerate(resume_arg_names):
                    prefix.extend(
                        [
                            create_instruction("LOAD_FAST", argval=resume_args_varname),
                            create_instruction("LOAD_CONST", argval=idx),
                            create_binary_subscr(),
                            create_instruction("STORE_FAST", argval=name),
                        ]
                    )
                for idx in sorted(resume_arg_indexes_to_clear):
                    if idx >= 2 + nstack:
                        prefix.extend(create_clear_resume_arg(resume_args_varname, idx))

            cleanup: list[Instruction] = []
            hooks = {fn.stack_index: fn for fn in setup_fns}
            hook_target_offsets = {
                fn.stack_index: setup_fn_target_offsets[i]
                for i, fn in enumerate(setup_fns)
            }
            offset_to_inst = {inst.offset: inst for inst in instructions}
            # map old hook targets to new targets generated by the hook
            # pyrefly: ignore [implicit-any]
            old_hook_target_remap = {}
            stack_i = 0
            null_i = 0
            stack_ctx_vars_d = dict(stack_ctx_vars)  # type: ignore[var-annotated,arg-type]
            for i in range(nstack + len(null_idxes)):
                if null_i < len(null_idxes) and null_idxes[null_i] == i:
                    prefix.append(create_instruction("PUSH_NULL"))
                    null_i += 1
                else:
                    prefix.append(
                        create_instruction("LOAD_FAST", argval=f"___stack{stack_i}")
                    )
                    if handle_inactive_ctx and stack_i in stack_ctx_vars_d:
                        # NOTE: we assume that current stack var is a context manager CLASS!
                        # Load args for context variable and construct it
                        prefix.extend(_load_tuple_and_call(stack_ctx_vars_d[stack_i]))
                    if boxed_resume:
                        prefix.extend(
                            create_clear_resume_arg(resume_args_varname, 2 + stack_i)
                        )
                    prefix.append(
                        create_instruction("DELETE_FAST", argval=f"___stack{stack_i}")
                    )
                    stack_i += 1

                if i in hooks:
                    hook = hooks.pop(i)
                    hook_insts, exn_target = hook(code_options, cleanup)
                    prefix.extend(hook_insts)
                    if is_py311_plus:
                        hook_target_offset = hook_target_offsets.pop(i)
                        old_hook_target = offset_to_inst[hook_target_offset]
                        meta.prefix_block_target_offset_remap.append(hook_target_offset)
                        old_hook_target_remap[old_hook_target] = exn_target

            if is_py311_plus:
                # reverse the mapping since targets of later/nested contexts are inserted
                # into the mapping later, but show up earlier in the prefix.
                meta.prefix_block_target_offset_remap = list(
                    reversed(meta.prefix_block_target_offset_remap)
                )

            if hooks:
                raise AssertionError(f"Unprocessed hooks remaining: {hooks}")

            # NOTE: we assume that local var is a context manager CLASS!
            # initialize inactive context vars in argnames
            if handle_inactive_ctx:
                for name, vals in argnames_ctx_vars:
                    prefix.append(create_instruction("LOAD_FAST", argval=name))
                    prefix.extend(_load_tuple_and_call(vals))
                    prefix.append(create_instruction("STORE_FAST", argval=name))

            # 3.12+: store NULL into variables that were NULL
            if argnames_null:
                if sys.version_info < (3, 12):
                    raise AssertionError(
                        f"argnames_null requires Python 3.12+, got {sys.version_info}"
                    )
                for v in argnames_null:
                    if v in args:
                        raise AssertionError(
                            f"argnames_null variable {v!r} should not be in args"
                        )
                    prefix.extend(
                        [
                            create_instruction("PUSH_NULL"),
                            create_instruction("STORE_FAST", argval=v),
                        ]
                    )

            # Call nested resume function
            if nested_code_objs:
                prefix.extend(
                    [
                        # set up __nested_resume_fns[-1] call
                        *add_push_null(
                            [
                                create_instruction(
                                    "LOAD_FAST", argval="__nested_resume_fns"
                                ),
                                create_instruction("LOAD_CONST", argval=-1),
                                create_binary_subscr(),
                            ]
                        ),
                        # del __nested_resume_fns[-1]
                        create_instruction("LOAD_FAST", argval="__nested_resume_fns"),
                        create_instruction("LOAD_CONST", argval=-1),
                        create_instruction("DELETE_SUBSCR"),
                        # load [__nested_resume_fns, __nested_frame_values]
                        create_instruction("LOAD_FAST", argval="__nested_resume_fns"),
                        create_instruction("LOAD_FAST", argval="__nested_frame_values"),
                        create_instruction("BUILD_LIST", arg=2),
                        # load __nested_frame_values[-1]
                        create_instruction("LOAD_FAST", argval="__nested_frame_values"),
                        create_instruction("LOAD_CONST", argval=-1),
                        create_binary_subscr(),
                        # create [
                        #     __nested_resume_fns,
                        #     __nested_frame_values,
                        #     *__nested_frame_values[-1],
                        # ]
                        create_instruction("LIST_EXTEND", arg=1),
                        # del __nested_frame_values[-1]
                        create_instruction("LOAD_FAST", argval="__nested_frame_values"),
                        create_instruction("LOAD_CONST", argval=-1),
                        create_instruction("DELETE_SUBSCR"),
                        # Set is_tracing_resume_prologue back to allow graph breaks
                        # in the nested resume
                        create_instruction("LOAD_CONST", argval=False),
                        create_instruction(
                            "STORE_FAST", argval=IS_TRACING_RESUME_PROLOGUE_VARNAME
                        ),
                    ]
                )
                if _is_boxed_resume_code(nested_code_objs[-1]):
                    prefix.extend(create_call_function(1, False))
                else:
                    if nested_resume_args_varname is None:
                        raise AssertionError("nested_resume_args_varname must be set")
                    prefix.append(
                        create_instruction(
                            "STORE_FAST", argval=nested_resume_args_varname
                        )
                    )
                    for idx in range(nested_code_objs[-1].co_argcount):
                        prefix.extend(
                            [
                                create_instruction(
                                    "LOAD_FAST", argval=nested_resume_args_varname
                                ),
                                create_instruction("LOAD_CONST", argval=idx),
                                create_binary_subscr(),
                            ]
                        )
                    prefix.append(
                        create_instruction(
                            "DELETE_FAST", argval=nested_resume_args_varname
                        )
                    )
                    prefix.extend(
                        create_call_function(nested_code_objs[-1].co_argcount, False)
                    )
                if pop_nested_resume_result:
                    # pop the result of calling the nested resume function
                    prefix.append(create_instruction("POP_TOP"))
            else:
                # Set is_tracing_resume_prologue back to allow graph breaks after the jump
                prefix.extend(
                    [
                        create_instruction("LOAD_CONST", argval=False),
                        create_instruction(
                            "STORE_FAST", argval=IS_TRACING_RESUME_PROLOGUE_VARNAME
                        ),
                        create_instruction("DELETE_FAST", argval="__nested_resume_fns"),
                        create_instruction(
                            "DELETE_FAST", argval="__nested_frame_values"
                        ),
                    ]
                )

            prefix.append(create_jump_absolute(target))

            # because the line number table monotonically increases from co_firstlineno
            # remove starts_line for any instructions before the graph break instruction
            # this will ensure the instructions after the break have the correct line numbers
            for inst in instructions:
                if inst.offset == target.offset:
                    break
                inst.starts_line = None
                if sys.version_info >= (3, 11):
                    inst.positions = None

            if cleanup:
                prefix.extend(cleanup)
                prefix.extend(cls.unreachable_codes(code_options))

            # remap original instructions' exception table entries
            if old_hook_target_remap:
                # pyrefly: ignore [unbound-name]
                if not is_py311_plus:
                    raise AssertionError("old_hook_target_remap requires Python 3.11+")
                for inst in instructions:
                    if (
                        inst.exn_tab_entry
                        and inst.exn_tab_entry.target in old_hook_target_remap
                    ):
                        inst.exn_tab_entry.target = old_hook_target_remap[  # type: ignore[assignment]
                            inst.exn_tab_entry.target
                        ]

            # TODO(jansel): add dead code elimination here
            instructions[:] = prefix + instructions

        new_code, _ = transform_code_object(code, update)
        ContinueExecutionCache.generated_code_metadata[new_code] = meta
        return new_code

    @classmethod
    def uses_boxed_call(cls, code: types.CodeType) -> bool:
        return _is_boxed_resume_code(code)

    @staticmethod
    def unreachable_codes(code_options: dict[str, Any]) -> list[Instruction]:
        """Codegen a `raise None` to make analysis work for unreachable code"""
        return [
            create_load_const(None),
            create_instruction("RAISE_VARARGS", arg=1),
        ]

    @classmethod
    def generate_based_on_original_code_object(
        cls,
        code: types.CodeType,
        lineno: int,
        init_offset: int,
        resume_offset: int,
        setup_fn_target_offsets: tuple[int, ...],
        *args: Any,
    ) -> types.CodeType:
        """
        This handles the case of generating a resume into code generated
        to resume something else.  We want to always generate starting
        from the original code object so that if control flow paths
        converge we only generated 1 resume function (rather than 2^n
        resume functions).
        """

        meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[
            code
        ]

        def find_orig_offset(cur_offset: int) -> int:
            orig_offset = -1

            def find_orig_offset_transform(
                instructions: list[Instruction], code_options: dict[str, Any]
            ) -> None:
                nonlocal orig_offset
                (target,) = (i for i in instructions if i.offset == cur_offset)
                # match the functions starting at the last instruction as we have added a prefix
                new_target_tuple = tuple(
                    i2
                    for i1, i2 in zip(
                        reversed(instructions), reversed(meta.instructions)
                    )
                    if i1 is target
                )

                if not new_target_tuple:
                    # Instruction with cur_offset in instructions was not found
                    # in the original code - orig_offset left as -1.
                    # Caller expected to handle this case.
                    return

                if len(new_target_tuple) != 1:
                    raise AssertionError(
                        f"Expected exactly 1 matching target, got {len(new_target_tuple)}"
                    )
                new_target = new_target_tuple[0]

                if target.opcode != new_target.opcode:
                    raise AssertionError(
                        f"Opcode mismatch: target has {target.opcode}, "
                        f"new_target has {new_target.opcode}"
                    )
                if new_target.offset is None:
                    raise AssertionError("new_target.offset must not be None")
                orig_offset = new_target.offset

            transform_code_object(code, find_orig_offset_transform)
            return orig_offset

        orig_init_offset = find_orig_offset(init_offset)
        # It is fine if the initial instruction is not found in the original code;
        # this means we graph broke in the prefix, which only happens with nested graph breaks.
        # We should not be running into ambiguous graph break issues here.
        orig_resume_offset = find_orig_offset(resume_offset)
        if orig_resume_offset <= -1:
            raise AssertionError(
                "resume instruction not found in original code - this is a bug."
            )

        if sys.version_info >= (3, 11):
            # setup_fn_target_offsets currently contains the target offset of
            # each setup_fn, based on `code`. When we codegen the resume function
            # based on the original code object, `meta.code`, the offsets in
            # setup_fn_target_offsets must be based on `meta.code` instead.
            offset_key = (orig_init_offset, orig_resume_offset)
            # NOTE: we key by offset_key since the same resume function may graph
            # break in multiple places and we need different block_target_offset_remap's
            # for each graph break location. Keying by orig_resume_offset may not be enough
            # if 2 graph breaks on different initial offsets resume on the same instruction
            # (although this is rare and not tested anywhere).
            if offset_key not in meta.block_target_offset_remap:
                block_target_offset_remap = meta.block_target_offset_remap[
                    offset_key
                    # pyrefly: ignore [implicit-any]
                ] = {}

                def remap_block_offsets(
                    instructions: list[Instruction], code_options: dict[str, Any]
                ) -> None:
                    # NOTE: each prefix block generates exactly one PUSH_EXC_INFO,
                    # so we can tell which block a prefix PUSH_EXC_INFO belongs to,
                    # by counting. Then we can use meta.prefix_block_target_offset_remap
                    # to determine where in the original code the PUSH_EXC_INFO offset
                    # replaced.
                    prefix_blocks: list[Instruction] = []
                    for inst in instructions:
                        # NOTE meta.prefix_block_target_offset_remap is based off of how we codegen'd
                        # context managers at the prefix/prologue of the resume function. It is the same for
                        # every graph break in the same resume function, so we do not need to recompute
                        # for each graph break (unlike for meta.block_target_offset_remap)
                        if len(prefix_blocks) == len(
                            meta.prefix_block_target_offset_remap
                        ):
                            break
                        if inst.opname == "PUSH_EXC_INFO":
                            prefix_blocks.append(inst)

                    # remap block target offsets for blocks generated in the resume prefix
                    for inst, o in zip(
                        prefix_blocks, meta.prefix_block_target_offset_remap
                    ):
                        block_target_offset_remap[cast(int, inst.offset)] = o

                    # current bytecode targets are after the prefix PUSH_EXC_INFO's
                    cur_start_offset = (
                        cast(int, prefix_blocks[-1].offset) if prefix_blocks else -1
                    )
                    # get the remaining block target offsets of the current bytecode
                    cur_inst_offsets = sorted(
                        n for n in setup_fn_target_offsets if n > cur_start_offset
                    )
                    targets = _filter_iter(
                        instructions, cur_inst_offsets, lambda inst, o: inst.offset == o
                    )
                    # The original code and resume code should have matching suffixes.
                    # Match the post-prefix block target offsets of the current resume code
                    # and the original code.
                    orig_targets = reversed(
                        _filter_iter(
                            zip(reversed(instructions), reversed(meta.instructions)),
                            reversed(targets),
                            lambda v1, v2: v1[0] is v2,
                        )
                    )
                    for orig, cur in zip(orig_targets, targets):
                        block_target_offset_remap[cur.offset] = orig[1].offset

                transform_code_object(code, remap_block_offsets)

            # if offset_key or offset is not in setup_fn_target_offsets, it is an error
            # that needs to be fixed
            setup_fn_target_offsets = tuple(
                meta.block_target_offset_remap[offset_key][n]
                for n in setup_fn_target_offsets
            )
        return ContinueExecutionCache.lookup(
            meta.code,
            lineno,
            orig_init_offset,
            orig_resume_offset,
            setup_fn_target_offsets,
            *args,
        )
