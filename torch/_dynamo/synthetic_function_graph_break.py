from __future__ import annotations

import copy
import dataclasses
import dis
import functools
import logging
import sys
import types
from typing import Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from .symbolic_convert import InstructionTranslatorBase

from .bytecode_transformation import (
    create_copy,
    create_dup_top,
    create_instruction,
    create_swap,
    Instruction,
    unique_id,
)
from .codegen import PyCodegen
from .exc import unimplemented
from .output_graph import GraphCompileReason, StackLocalsMetadata
from .variables.misc import NullVariable, UnknownVariable


log = logging.getLogger(__name__)


@functools.cache
def _get_comprehension_bytecode_prefix() -> list[str]:
    """Get the bytecode instructions that precede BUILD_LIST in a list comprehension."""

    assert sys.version_info >= (3, 12)

    def fn() -> list[int]:
        return [i for i in range(1)]  # noqa: C416

    insts = [inst.opname for inst in dis.get_instructions(fn)]

    start_idx = len(insts) - 1 - insts[::-1].index("LOAD_FAST_AND_CLEAR")
    end_idx = insts.index("BUILD_LIST")

    return insts[start_idx:end_idx]


@functools.cache
def _get_comprehension_result_patterns() -> dict[str, dict[str, Any]]:
    """Discover bytecode patterns for comprehension result handling.

    Analyzes sample functions to extract the opcode sequences that appear
    after END_FOR for each result disposition (stored, discarded, returned, consumed).

    Returns patterns with:
        - pre_store_ops: opcodes between END_FOR and first STORE_FAST
        - post_store_op: first opcode after all STORE_FASTs (for disambiguation)
    """
    assert sys.version_info >= (3, 12)

    def fn_stored() -> list[int]:
        result = [i for i in range(1)]  # noqa: C416
        return result

    def fn_discarded() -> int:
        [i for i in range(1)]  # noqa: C416
        return 1

    def fn_returned() -> list[int]:
        return [i for i in range(1)]  # noqa: C416

    def fn_consumed() -> int:
        return sum([i for i in range(1)])  # noqa: C416

    def extract_pattern(fn: Callable[..., Any]) -> tuple[list[str], Optional[str]]:
        """Extract (pre_store_ops, post_store_op) from comprehension bytecode."""
        target_line = list(dis.findlinestarts(fn.__code__))[1][1]
        insts: list[str] = []
        started = False
        for instr in dis.get_instructions(fn):
            if started and instr.starts_line:
                break
            pos = instr.positions
            if pos and pos.lineno == target_line:
                started = started or bool(instr.starts_line)
                insts.append(instr.opname)

        ops = insts[insts.index("END_FOR") + 1 :]
        idx = 0

        pre_store_ops = []
        while idx < len(ops) and ops[idx] != "STORE_FAST":
            pre_store_ops.append(ops[idx])
            idx += 1

        while idx < len(ops) and ops[idx] == "STORE_FAST":
            idx += 1

        return pre_store_ops, ops[idx] if idx < len(ops) else None

    stored = extract_pattern(fn_stored)
    discarded = extract_pattern(fn_discarded)
    returned = extract_pattern(fn_returned)
    consumed = extract_pattern(fn_consumed)

    return {
        "stored": {"pre_store_ops": stored[0], "post_store_op": stored[1]},
        "discarded": {"pre_store_ops": discarded[0], "post_store_op": discarded[1]},
        "returned": {"pre_store_ops": returned[0], "post_store_op": returned[1]},
        "consumed": {"pre_store_ops": consumed[0], "post_store_op": []},
    }


@dataclasses.dataclass
class ComprehensionAnalysis:
    """Metadata about a comprehension's bytecode structure.

    Attributes:
        end_ip: Instruction pointer after all comprehension bytecode
        result_var: Name of result variable, or None if result stays on stack
        result_on_stack: True if result stays on stack (discarded, returned, or in expression)
        iterator_vars: Variables from LOAD_FAST_AND_CLEAR (need restoration)
        walrus_vars: Variables assigned via walrus operator (:=) inside comprehension
        captured_vars: Variables read from outer scope via LOAD_FAST inside comprehension
    """

    end_ip: int
    result_var: Optional[str]
    result_on_stack: bool
    iterator_vars: list[str]
    walrus_vars: list[str]
    captured_vars: list[str]


@dataclasses.dataclass
class ForLoopAnalysis:
    """Metadata about a for loop's bytecode structure.

    Attributes:
        for_iter_ip: The FOR_ITER instruction pointer
        end_ip: First instruction after the for/else construct
        has_break: Whether the loop body contains break
    """

    for_iter_ip: int
    end_ip: int
    has_break: bool


@functools.cache
def _get_for_loop_cleanup_count() -> int:
    """Count cleanup instructions after END_FOR for the current Python version.

    In Python 3.13+, there is a POP_TOP after END_FOR. In 3.12, there are none.
    Returns 0 for versions < 3.12.
    """
    if sys.version_info < (3, 12):
        return 0

    def fn() -> None:
        for i in range(1):  # noqa: B007
            pass

    insts = list(dis.get_instructions(fn))
    end_for_idx = next(i for i, inst in enumerate(insts) if inst.opname == "END_FOR")

    # Count non-loop instructions between END_FOR and the next meaningful instruction
    count = 0
    for inst in insts[end_for_idx + 1 :]:
        if inst.opname in ("POP_TOP", "POP_ITER"):
            count += 1
        else:
            break
    return count


def _find_for_loop_end_ip(tx: InstructionTranslatorBase, for_iter_ip: int) -> int:
    """Find the first instruction after the for loop construct.

    For 3.12+: finds the matching END_FOR by tracking nesting,
    returns end_for_ip + 1 + cleanup_count.
    For < 3.12: uses FOR_ITER.target (which points past the loop).
    """
    for_iter_inst = tx.instructions[for_iter_ip]
    assert for_iter_inst.opname == "FOR_ITER"

    if sys.version_info >= (3, 12):
        nesting_depth = 1
        for search_ip in range(for_iter_ip + 1, len(tx.instructions)):
            inst = tx.instructions[search_ip]
            if inst.opname == "FOR_ITER":
                nesting_depth += 1
            elif inst.opname == "END_FOR":
                nesting_depth -= 1
                if nesting_depth == 0:
                    return search_ip + 1 + _get_for_loop_cleanup_count()
        return -1
    else:
        # For < 3.12, FOR_ITER.target points past the loop
        assert for_iter_inst.target is not None
        return tx.indexof[for_iter_inst.target]


def _analyze_for_loop(
    tx: InstructionTranslatorBase, for_iter_ip: int
) -> ForLoopAnalysis:
    """Analyze for loop bytecode to determine structure."""
    assert tx.instruction_pointer is not None

    end_ip = _find_for_loop_end_ip(tx, for_iter_ip)
    assert end_ip > 0, "Could not find end of for loop"

    for_iter_inst = tx.instructions[for_iter_ip]

    # Detect break: look for POP_TOP followed by a forward jump past end
    has_break = False
    break_target_ip = end_ip
    # In 3.12+, FOR_ITER.target points to END_FOR
    if sys.version_info >= (3, 12):
        assert for_iter_inst.target is not None
        end_for_ip = tx.indexof[for_iter_inst.target]
    else:
        end_for_ip = end_ip

    for body_ip in range(for_iter_ip + 1, end_for_ip):
        inst = tx.instructions[body_ip]
        if inst.opname == "POP_TOP":
            # Check if the next instruction jumps past the loop end
            if body_ip + 1 < len(tx.instructions):
                next_inst = tx.instructions[body_ip + 1]
                if next_inst.target is not None:
                    target_ip = tx.indexof[next_inst.target]
                    if target_ip >= end_ip:
                        has_break = True
                        break_target_ip = max(break_target_ip, target_ip)

    # When break exists, extend end_ip to include the else body so
    # break vs normal exit is handled inside the synthetic function.
    if has_break:
        end_ip = break_target_ip

    return ForLoopAnalysis(
        for_iter_ip=for_iter_ip,
        end_ip=end_ip,
        has_break=has_break,
    )


def _is_comprehension_start(tx: InstructionTranslatorBase) -> bool:
    """Detect if we're at the start of a list/dict comprehension in 3.12+.

    In Python 3.12+, comprehensions are inlined with a bytecode pattern that
    precedes BUILD_LIST/BUILD_MAP.
    """
    assert sys.version_info >= (3, 12)

    assert tx.instruction_pointer is not None
    ip = tx.instruction_pointer - 1

    pattern = _get_comprehension_bytecode_prefix()
    prefix = [inst.opname for inst in tx.instructions[ip - len(pattern) : ip]]

    return prefix == pattern


def _find_comprehension_end_for_ip(tx: InstructionTranslatorBase) -> int:
    """Find the instruction pointer of the outermost END_FOR for current comprehension."""
    assert sys.version_info >= (3, 12)
    assert tx.instruction_pointer is not None

    nesting_depth = 0
    for search_ip in range(tx.instruction_pointer, len(tx.instructions)):
        inst = tx.instructions[search_ip]
        if inst.opname == "FOR_ITER":
            nesting_depth += 1
        elif inst.opname == "END_FOR":
            nesting_depth -= 1
            if nesting_depth == 0:
                return search_ip
    return -1


def _analyze_comprehension(tx: InstructionTranslatorBase) -> ComprehensionAnalysis:
    """Analyze comprehension bytecode to determine result handling pattern."""
    assert sys.version_info >= (3, 12)
    assert tx.instruction_pointer is not None

    patterns = _get_comprehension_result_patterns()
    start_ip = tx.instruction_pointer - 1  # BUILD_LIST/BUILD_MAP

    iterator_vars: list[str] = []
    walrus_vars: list[str] = []
    captured_vars: list[str] = []
    defined_inside: set[str] = set()

    # Collect iterator variables from LOAD_FAST_AND_CLEAR before BUILD_LIST/BUILD_MAP
    iter_scan_ip = start_ip - 1
    while iter_scan_ip >= 0:
        inst = tx.instructions[iter_scan_ip]
        if inst.opname == "LOAD_FAST_AND_CLEAR":
            iterator_vars.insert(0, inst.argval)
            iter_scan_ip -= 1
        elif inst.opname in ("SWAP", "GET_ITER"):
            iter_scan_ip -= 1
        else:
            break
    defined_inside.update(iterator_vars)

    end_for_ip = _find_comprehension_end_for_ip(tx)
    if end_for_ip == -1:
        unimplemented(
            gb_type="Comprehension analysis failed: No END_FOR",
            context="",
            explanation="Could not find END_FOR instruction in comprehension bytecode.",
            hints=[],
        )

    # Find first FOR_ITER to know where loop body starts
    for_iter_ip = next(
        i
        for i in range(start_ip, end_for_ip)
        if tx.instructions[i].opname == "FOR_ITER"
    )

    # Single pass through loop body to detect walrus vars and captured vars
    for body_ip in range(for_iter_ip + 1, end_for_ip):
        inst = tx.instructions[body_ip]

        # Detect walrus pattern: COPY 1 followed by STORE_FAST
        if inst.opname == "COPY" and inst.arg == 1 and body_ip + 1 < end_for_ip:
            next_inst = tx.instructions[body_ip + 1]
            if next_inst.opname == "STORE_FAST":
                var_name = next_inst.argval
                if var_name not in iterator_vars and var_name not in walrus_vars:
                    walrus_vars.append(var_name)
                    defined_inside.add(var_name)

        # Track variables defined inside the loop
        if inst.opname == "STORE_FAST":
            defined_inside.add(inst.argval)

        # Detect LOAD_FAST referencing outer variables
        elif inst.opname.startswith("LOAD_FAST"):
            var_names = (
                inst.argval if isinstance(inst.argval, tuple) else (inst.argval,)
            )
            for var_name in var_names:
                if var_name not in defined_inside and var_name not in captured_vars:
                    captured_vars.append(var_name)

    # Extract pre_store_ops: all opcodes from END_FOR+1 until first STORE_FAST
    pre_store_ops: list[str] = []
    scan_ip = end_for_ip + 1
    while (
        scan_ip < len(tx.instructions)
        and tx.instructions[scan_ip].opname != "STORE_FAST"
    ):
        pre_store_ops.append(tx.instructions[scan_ip].opname)
        scan_ip += 1

    store_fast_ip = scan_ip

    # Skip all STORE_FASTs to find post_store_op
    while (
        scan_ip < len(tx.instructions)
        and tx.instructions[scan_ip].opname == "STORE_FAST"
    ):
        scan_ip += 1

    post_store_op = (
        tx.instructions[scan_ip].opname if scan_ip < len(tx.instructions) else None
    )

    def matches(name: str) -> bool:
        pat = patterns[name]
        return pre_store_ops == pat["pre_store_ops"] and (
            post_store_op == pat["post_store_op"] or not pat["post_store_op"]
        )

    result_var: Optional[str] = None
    if matches("stored"):
        result_var = tx.instructions[store_fast_ip].argval
        result_on_stack = False
    elif matches("discarded"):
        result_var = None
        result_on_stack = False
        scan_ip = scan_ip + 1 if patterns["discarded"]["post_store_op"] else scan_ip
    elif matches("returned") or pre_store_ops == patterns["consumed"]["pre_store_ops"]:
        result_var = None
        result_on_stack = True
    else:
        unimplemented(
            gb_type="Comprehension analysis failed: No matches",
            context=f"pre_store_ops={pre_store_ops}, post_store_op={post_store_op}",
            explanation="Comprehension does not match any known bytecode pattern.",
            hints=[],
        )

    return ComprehensionAnalysis(
        end_ip=scan_ip,
        result_var=result_var,
        # pyrefly: ignore [unbound-name]
        result_on_stack=result_on_stack,
        iterator_vars=iterator_vars,
        walrus_vars=walrus_vars,
        captured_vars=captured_vars,
    )


def _handle_comprehension_graph_break(
    tx: InstructionTranslatorBase, inst: Instruction
) -> None:
    """Handle list/dict comprehension graph break.

    Builds a synthetic function wrapping the comprehension bytecode,
    calls it via codegen_call_resume, then chains into the resume
    function for the post-comprehension code.
    """
    assert sys.version_info >= (3, 12)
    assert tx.instruction_pointer is not None

    start_ip = tx.instruction_pointer - 1  # BUILD_LIST/BUILD_MAP
    analysis = _analyze_comprehension(tx)
    stack_pops = 1 + len(analysis.iterator_vars)
    reason = GraphCompileReason("comprehension_graph_break", [tx.frame_summary()])
    log.debug("comprehension triggered compile")

    # --- Step 1: Compile the graph up to the comprehension ---

    all_stack_locals_metadata = tx.output.compile_subgraph(
        tx,
        reason=reason,
        stack_pops=stack_pops,
    )
    # Record which stack_pops items are NULL before popn loses the info.
    # NULLs on the CPython stack can't be passed as function arguments.
    stack_pops_null_mask = [
        isinstance(tx.stack[len(tx.stack) - stack_pops + i], NullVariable)
        for i in range(stack_pops)
    ]

    tx.popn(stack_pops)
    meta = all_stack_locals_metadata[0]
    cg = PyCodegen(tx.output.root_tx)

    # Runtime stack after compile_subgraph:
    #   cells, [frame_values], *(non-popped items), *(stack_pops items w/ NULLs)
    # frame_values[0] = [frame N locals] (no stack items yet)

    nonnull_count = sum(1 for m in stack_pops_null_mask if not m)

    # live_stack_depth: stack items above cells/frame_values excluding NULLs
    # that compile_subgraph didn't codegen (tracked in stack_null_idxes).
    live_stack_depth = len(tx.stack) - len(meta.stack_null_idxes)

    # --- Step 2: Pop stack_pops items and append non-nulls to frame_values[0] ---
    # SWAP each item to TOS then LIST_APPEND or pop_null; fv_list stays at
    # TOS throughout. Items append in TOS-first (reversed) order;
    # _build_comprehension_fn compensates by loading in reverse.
    cg.extend_output(
        [
            # frame_values[0] to TOS
            *create_copy(live_stack_depth + stack_pops + 1),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
        ]
    )
    for i in reversed(range(stack_pops)):
        cg.extend_output(create_swap(2))
        if stack_pops_null_mask[i]:
            cg.extend_output(cg.pop_null())
        else:
            cg.extend_output([create_instruction("LIST_APPEND", arg=1)])
    cg.extend_output([create_instruction("POP_TOP")])

    # Stack: cells, [frame_values], *(non-popped items)

    # --- Step 3: Build comprehension function ---
    new_code, fn_name = _build_comprehension_fn(
        tx,
        analysis,
        start_ip,
        stack_pops,
        stack_pops_null_mask,
        nonnull_count,
        meta,
    )

    # --- Step 4: Extract [cells[0]] and [frame_values[0]] for codegen_call_resume ---
    cg.extend_output(
        [
            *create_copy(live_stack_depth + 2),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
            create_instruction("BUILD_LIST", arg=1),
            *create_copy(live_stack_depth + 2),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
            create_instruction("BUILD_LIST", arg=1),
        ]
    )

    # Stack: ..., *(non-popped), [cells[0]], [frame_values[0]]

    # --- Step 5: Call comprehension function via codegen_call_resume ---
    tx.codegen_call_resume([new_code], [fn_name], cg)

    # Stack: ..., *(non-popped), comp_result

    # --- Step 6: Remove appended stack_pops items from frame_values[0] ---
    if nonnull_count > 0:
        frame_values_pos = live_stack_depth + 1 + 1  # +1 result, +1 frame_values
        cg.extend_output(
            [
                *create_copy(frame_values_pos),
                cg.create_load_const(0),
                cg.create_binary_subscr(),
                # frame_values[0] on TOS
                create_dup_top(),
                # frame_values[0], frame_values[0]
                cg.create_load_const(-nonnull_count),
                cg.create_load_const(None),
                create_instruction("BUILD_SLICE", arg=2),
                create_instruction("DELETE_SUBSCR"),
                # del frame_values[0][-nonnull_count:]
                create_instruction("POP_TOP"),
            ]
        )

    # --- Step 7: Pass comprehension outputs to frame_values[0] ---
    # Walrus vars first, then result_var.
    vars_to_pass = analysis.walrus_vars + (
        [analysis.result_var] if analysis.result_var else []
    )

    existing_vars: dict[str, int] = {}
    for var_name in vars_to_pass:
        tx.symbolic_locals[var_name] = UnknownVariable()
        if var_name in meta.locals_names:
            existing_vars[var_name] = meta.locals_names[var_name]
        else:
            meta.locals_names[var_name] = len(meta.locals_names)

    fv_depth = live_stack_depth + 2  # comp_result + frame_values

    # --- Walrus vars: extract from comp_result tuple ---
    if analysis.walrus_vars:
        # comp_result is (result, *walrus_vars).
        cg.extend_output(
            [
                *create_copy(fv_depth),
                cg.create_load_const(0),
                cg.create_binary_subscr(),
            ]
        )
        # Stack: ..., comp_tuple, fv0
        for j, walrus_var in enumerate(analysis.walrus_vars):
            cg.extend_output(
                [
                    *create_copy(2),
                    cg.create_load_const(j + 1),
                    cg.create_binary_subscr(),
                ]
            )
            # Stack: ..., comp_tuple, fv0, walrus_value
            if walrus_var in existing_vars:
                # fv0[idx] = walrus_value
                cg.extend_output(
                    [
                        *create_copy(2),  # copy fv0
                        cg.create_load_const(existing_vars[walrus_var]),
                        create_instruction("STORE_SUBSCR"),
                    ]
                )
            else:
                cg.extend_output([create_instruction("LIST_APPEND", arg=1)])
            # Stack: ..., comp_tuple, fv0
        cg.extend_output(
            [
                create_instruction("POP_TOP"),  # pop fv0
                # Extract the result from the tuple.
                cg.create_load_const(0),
                cg.create_binary_subscr(),
            ]
        )
        # Stack: ..., result

    # --- Result: keep on stack, overwrite/append to fv[0], or discard ---
    if analysis.result_on_stack:
        tx.push(UnknownVariable())
    elif analysis.result_var:
        cg.extend_output(
            [
                *create_copy(fv_depth),
                cg.create_load_const(0),
                cg.create_binary_subscr(),
                # Stack: ..., result, fv0
            ]
        )
        if analysis.result_var in existing_vars:
            cg.extend_output(
                [
                    cg.create_load_const(existing_vars[analysis.result_var]),
                    create_instruction("STORE_SUBSCR"),
                    # fv0[idx] = result
                ]
            )
        else:
            cg.extend_output(
                [
                    *create_swap(2),
                    create_instruction("LIST_APPEND", arg=1),
                    create_instruction("POP_TOP"),
                ]
            )
    else:
        cg.extend_output([create_instruction("POP_TOP")])

    # Stack: cells, [frame_values], *(non-popped stack)
    tx.output.add_output_instructions(cg.get_instructions())

    # --- Step 8: Create resume function chain ---
    resume_inst = tx.instructions[analysis.end_ip]
    tx.output.add_output_instructions(
        tx.create_call_resume_at(resume_inst, all_stack_locals_metadata)
    )

    tx.instruction_pointer = None


def _build_comprehension_fn(
    tx: InstructionTranslatorBase,
    analysis: ComprehensionAnalysis,
    start_ip: int,
    stack_pops: int,
    stack_pops_null_mask: list[bool],
    nonnull_count: int,
    meta: StackLocalsMetadata,
) -> tuple[types.CodeType, str]:
    """Build a synthetic function wrapping comprehension bytecode.

    Uses the same calling convention as resume functions created by
    create_resume / ContinueExecutionCache.generate: the first two args
    are __nested_resume_fns and __nested_frame_values (ignored here),
    followed by stack items and live locals.

    Returns (code, name) where name is the global name for the function.
    """
    from .bytecode_transformation import transform_code_object
    from .eval_frame import skip_code
    from .resume_execution import CO_VARARGS, CO_VARKEYWORDS

    # Args follow frame_values layout: locals first, then stack_pops items
    # (appended to end of frame_values[0] by the caller).
    # codegen_call_resume unpacks frame_values[0] as positional args.
    argnames = tuple(k for k in meta.locals_names if k not in tx.cell_and_freevars())
    args = (
        ["__nested_resume_fns", "__nested_frame_values"]
        + list(argnames)
        + [f"___stack{i}" for i in range(nonnull_count)]
    )

    freevars = tuple(
        sorted(list(tx.f_code.co_cellvars or []) + list(tx.f_code.co_freevars or []))
    )

    lineno = tx.lineno if tx.lineno is not None else tx.f_code.co_firstlineno
    fn_name = unique_id(f"__comprehension_{tx.f_code.co_name}_at_{lineno}")

    comprehension_body_vars = (
        analysis.iterator_vars
        + analysis.walrus_vars
        + ([analysis.result_var] if analysis.result_var else [])
        + analysis.captured_vars
    )

    def update(instructions: list[Instruction], code_options: dict[str, Any]) -> None:
        code_options["co_name"] = fn_name
        if sys.version_info >= (3, 11):
            code_options["co_qualname"] = fn_name
        code_options["co_firstlineno"] = lineno
        code_options["co_cellvars"] = ()
        code_options["co_freevars"] = freevars
        code_options["co_argcount"] = len(args)
        code_options["co_posonlyargcount"] = 0
        code_options["co_kwonlyargcount"] = 0
        code_options["co_varnames"] = tuple(
            args + [v for v in comprehension_body_vars if v not in args]
        )
        code_options["co_flags"] = code_options["co_flags"] & ~(
            CO_VARARGS | CO_VARKEYWORDS
        )

        prefix: list[Instruction] = []
        if freevars:
            prefix.append(create_instruction("COPY_FREE_VARS", arg=len(freevars)))
        prefix.append(create_instruction("RESUME", arg=0))

        # Push stack_pops items onto operand stack so the comprehension
        # bytecode finds them where it expects (iterator + saved vars).
        # NULL positions get PUSH_NULL, non-null get LOAD_FAST.
        # Items were appended to frame_values[0] in TOS-first order,
        # so load in reverse to reconstruct the original stack layout.
        nonnull_i = nonnull_count - 1
        for i in range(stack_pops):
            if stack_pops_null_mask[i]:
                prefix.append(create_instruction("PUSH_NULL"))
            else:
                prefix.append(
                    create_instruction("LOAD_FAST", argval=f"___stack{nonnull_i}")
                )
                nonnull_i -= 1

        comp_insts = _copy_bytecode_range(tx, start_ip, analysis.end_ip)

        # Epilogue: ensure result is on stack, pack walrus vars, return.
        epilogue: list[Instruction] = []
        if not analysis.result_on_stack:
            if analysis.result_var:
                epilogue.append(
                    create_instruction("LOAD_FAST", argval=analysis.result_var)
                )
            else:
                epilogue.append(create_instruction("LOAD_CONST", argval=None))
        if analysis.walrus_vars:
            for var_name in analysis.walrus_vars:
                epilogue.append(create_instruction("LOAD_FAST", argval=var_name))
            epilogue.append(
                create_instruction(
                    "BUILD_TUPLE",
                    arg=1 + len(analysis.walrus_vars),
                )
            )
        epilogue.append(create_instruction("RETURN_VALUE"))

        instructions[:] = prefix + comp_insts + epilogue

    new_code, _ = transform_code_object(tx.f_code, update)
    skip_code(new_code)

    # Install as global
    if new_code.co_freevars:
        tx.output.install_global_unsafe(fn_name, new_code)
    else:
        tx.output.install_global_unsafe(
            fn_name,
            types.FunctionType(new_code, tx.f_globals, fn_name),
        )

    return new_code, fn_name


def _copy_bytecode_range(
    tx: InstructionTranslatorBase, start_ip: int, end_ip: int
) -> list[Instruction]:
    """Copy bytecode instructions in [start_ip, end_ip), remapping jump targets."""
    inst_map: dict[Instruction, Instruction] = {}
    copied_insts: list[Instruction] = []

    for ip in range(start_ip, end_ip):
        original_inst = tx.instructions[ip]
        copied_inst = copy.copy(original_inst)
        copied_inst.exn_tab_entry = None
        inst_map[original_inst] = copied_inst
        copied_insts.append(copied_inst)

    for copied_inst in copied_insts:
        if copied_inst.target is not None and copied_inst.target in inst_map:
            copied_inst.target = inst_map[copied_inst.target]

    return copied_insts


def maybe_setup_comprehension_speculation(
    tx: InstructionTranslatorBase, inst: Instruction
) -> bool:
    """
    Handle comprehension start for Python 3.12+ BUILD_LIST/BUILD_MAP with argval 0.
    Returns True if a graph break was triggered and the caller should return early.
    """
    if not (sys.version_info >= (3, 12) and inst.argval == 0):
        return False

    if not _is_comprehension_start(tx):
        return False

    can_speculate = (
        all(b.can_restore() for b in tx.block_stack)
        and not tx.one_graph
        and not tx.error_on_graph_break
        and not tx.is_tracing_resume_prologue
        and not tx.active_generic_context_managers
        and tx.output.current_tracer.parent is None
    )

    if can_speculate and tx.parent is not None:
        can_speculate = tx._can_speculate_comprehension_nested()
    # Only set up speculation at depth 0 (outermost comprehension)
    if can_speculate and tx._comprehension_depth == 0:
        speculation = tx.speculate()
        if speculation.failed(tx):
            _handle_comprehension_graph_break(tx, inst)
            return True
        tx.current_speculation = speculation
    end_for_ip = _find_comprehension_end_for_ip(tx)
    assert end_for_ip >= 0
    tx._comprehension_end_for_ips.add(end_for_ip)
    tx._comprehension_depth += 1
    return False


def _build_for_loop_fn(
    tx: InstructionTranslatorBase,
    analysis: ForLoopAnalysis,
    meta: StackLocalsMetadata,
) -> tuple[types.CodeType, str]:
    """Build a synthetic function wrapping for loop bytecode.

    The function receives all locals + the iterator as arguments, runs the
    loop eagerly, then returns a tuple of ALL locals so the caller can
    update its frame state.
    """
    from .bytecode_transformation import transform_code_object
    from .eval_frame import skip_code
    from .resume_execution import CO_VARARGS, CO_VARKEYWORDS

    argnames = tuple(k for k in meta.locals_names if k not in tx.cell_and_freevars())
    args = (
        ["__nested_resume_fns", "__nested_frame_values"]
        + list(argnames)
        + ["___stack0"]
    )

    freevars = tuple(
        sorted(list(tx.f_code.co_cellvars or []) + list(tx.f_code.co_freevars or []))
    )

    lineno = tx.lineno if tx.lineno is not None else tx.f_code.co_firstlineno
    fn_name = unique_id(f"__for_loop_{tx.f_code.co_name}_at_{lineno}")

    def update(instructions: list[Instruction], code_options: dict[str, Any]) -> None:
        code_options["co_name"] = fn_name
        if sys.version_info >= (3, 11):
            code_options["co_qualname"] = fn_name
        code_options["co_firstlineno"] = lineno
        code_options["co_cellvars"] = ()
        code_options["co_freevars"] = freevars
        code_options["co_argcount"] = len(args)
        code_options["co_posonlyargcount"] = 0
        code_options["co_kwonlyargcount"] = 0

        extra_vars = [v for v in tx.f_code.co_varnames if v not in args]
        code_options["co_varnames"] = tuple(args + extra_vars)
        code_options["co_flags"] = code_options["co_flags"] & ~(
            CO_VARARGS | CO_VARKEYWORDS
        )

        prefix: list[Instruction] = []
        if freevars:
            prefix.append(create_instruction("COPY_FREE_VARS", arg=len(freevars)))
        prefix.append(create_instruction("RESUME", arg=0))
        # Load the iterator onto the operand stack
        prefix.append(create_instruction("LOAD_FAST", argval="___stack0"))

        loop_insts = _copy_bytecode_range(tx, analysis.for_iter_ip, analysis.end_ip)

        # Epilogue: load ALL locals in order and return as list
        epilogue: list[Instruction] = []
        for var_name in argnames:
            epilogue.append(create_instruction("LOAD_FAST", argval=var_name))
        epilogue.append(create_instruction("BUILD_LIST", arg=len(argnames)))
        epilogue.append(create_instruction("RETURN_VALUE"))

        # Redirect jumps targeting outside the copied range to the epilogue.
        # This handles break statements whose jump targets are past the loop.
        if epilogue:
            epilogue_start = epilogue[0]
            for copied_inst in loop_insts:
                if (
                    copied_inst.target is not None
                    and copied_inst.target not in loop_insts
                    and copied_inst.target not in epilogue
                ):
                    copied_inst.target = epilogue_start

        instructions[:] = prefix + loop_insts + epilogue

    new_code, _ = transform_code_object(tx.f_code, update)
    skip_code(new_code)

    if new_code.co_freevars:
        tx.output.install_global_unsafe(fn_name, new_code)
    else:
        tx.output.install_global_unsafe(
            fn_name,
            types.FunctionType(new_code, tx.f_globals, fn_name),
        )

    return new_code, fn_name


def _handle_for_loop_graph_break(
    tx: InstructionTranslatorBase, inst: Instruction
) -> None:
    """Handle for loop graph break.

    Builds a synthetic function wrapping the for loop bytecode,
    calls it via codegen_call_resume, then replaces frame_values[0]
    with the returned locals tuple and chains into the resume function
    for the post-loop code.
    """
    assert tx.instruction_pointer is not None

    for_iter_ip = tx.instruction_pointer - 1
    analysis = _analyze_for_loop(tx, for_iter_ip)
    reason = GraphCompileReason("for_loop_graph_break", [tx.frame_summary()])
    log.debug("for loop triggered compile")

    # --- Step 1: Compile the graph up to the for loop ---
    # stack_pops=1: the iterator on top of the stack
    all_stack_locals_metadata = tx.output.compile_subgraph(
        tx,
        reason=reason,
        stack_pops=1,
    )

    tx.pop()  # pop the iterator
    meta = all_stack_locals_metadata[0]
    cg = PyCodegen(tx.output.root_tx)

    # Runtime stack after compile_subgraph:
    #   cells, [frame_values], *(non-popped items), iterator
    # frame_values[0] = [frame N locals]

    live_stack_depth = len(tx.stack) - len(meta.stack_null_idxes)

    # --- Step 2: Append iterator to frame_values[0] ---
    # Get frame_values[0], swap with iterator, LIST_APPEND, pop fv0
    cg.extend_output(
        [
            *create_copy(live_stack_depth + 1 + 1),  # +1 iterator, +1 frame_values
            cg.create_load_const(0),
            cg.create_binary_subscr(),
            # Stack: ..., iterator, fv0
            *create_swap(2),
            # Stack: ..., fv0, iterator
            create_instruction("LIST_APPEND", arg=1),
            # Stack: ..., fv0 (iterator appended)
            create_instruction("POP_TOP"),
        ]
    )

    # Stack: cells, [frame_values], *(non-popped items)

    # --- Step 3: Build for loop function ---
    new_code, fn_name = _build_for_loop_fn(tx, analysis, meta)

    # --- Step 4: Extract [cells[0]] and [frame_values[0]] for codegen_call_resume ---
    cg.extend_output(
        [
            *create_copy(live_stack_depth + 2),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
            create_instruction("BUILD_LIST", arg=1),
            *create_copy(live_stack_depth + 2),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
            create_instruction("BUILD_LIST", arg=1),
        ]
    )

    # Stack: ..., *(non-popped), [cells[0]], [frame_values[0]]

    # --- Step 5: Call for loop function via codegen_call_resume ---
    tx.codegen_call_resume([new_code], [fn_name], cg)

    # Stack: ..., *(non-popped), result_list

    # --- Step 6: Replace frame_values[0] with the returned locals list ---
    # result_list is on TOS. Store it as frame_values[0].
    # STORE_SUBSCR does TOS1[TOS] = TOS2, so we need:
    # Stack: ..., result_list, frame_values, 0
    frame_values_pos = live_stack_depth + 1 + 1  # +1 result, +1 frame_values
    cg.extend_output(
        [
            *create_copy(frame_values_pos),
            # Stack: ..., result_list, frame_values
            cg.create_load_const(0),
            # Stack: ..., result_list, frame_values, 0
            create_instruction("STORE_SUBSCR"),
            # frame_values[0] = result_list
        ]
    )

    # --- Step 7: Mark all locals as unknown ---
    for name in meta.locals_names:
        tx.symbolic_locals[name] = UnknownVariable()

    # Stack: cells, [frame_values], *(non-popped stack)
    tx.output.add_output_instructions(cg.get_instructions())

    # --- Step 8: Create resume function chain ---
    resume_inst = tx.instructions[analysis.end_ip]
    tx.output.add_output_instructions(
        tx.create_call_resume_at(resume_inst, all_stack_locals_metadata)
    )

    tx.instruction_pointer = None


def maybe_setup_for_loop_speculation(
    tx: InstructionTranslatorBase, inst: Instruction
) -> bool:
    """Set up for loop speculation at FOR_ITER.

    Returns True if a graph break was triggered and the caller should return early.
    """
    # Skip if inside a comprehension
    if tx._comprehension_depth > 0:
        return False

    can_speculate = (
        all(b.can_restore() for b in tx.block_stack)
        and not tx.one_graph
        and not tx.error_on_graph_break
        and not tx.is_tracing_resume_prologue
        and not tx.active_generic_context_managers
        and tx.output.current_tracer.parent is None
    )

    if can_speculate and tx.parent is not None:
        can_speculate = tx._can_speculate_for_loop_nested()

    if can_speculate and tx._for_loop_depth == 0:
        speculation = tx.speculate()
        if speculation.failed(tx):
            _handle_for_loop_graph_break(tx, inst)
            return True
        tx.current_speculation = speculation

    # Track the end IP for depth management
    assert tx.instruction_pointer is not None
    for_iter_ip = tx.instruction_pointer - 1
    end_ip = _find_for_loop_end_ip(tx, for_iter_ip)
    assert end_ip > 0

    if sys.version_info >= (3, 12):
        end_for_ip = end_ip - 1 - _get_for_loop_cleanup_count()
        tx._for_loop_end_ips.add(end_for_ip)
    else:
        tx._for_loop_end_ips.add(end_ip)

    tx._for_loop_depth += 1
    return False
