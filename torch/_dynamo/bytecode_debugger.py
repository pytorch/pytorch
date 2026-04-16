"""
Bytecode debugger for Dynamo-optimized code.

This module provides a pdb-like debugger for stepping through Python bytecode
one instruction at a time, with the ability to inspect the value stack,
locals, and globals.

Usage:
    >>> # xdoctest: +SKIP
    >>> import torch
    >>>
    >>> @torch.compile
    >>> def my_fn(x):
    ...     return x + 1
    >>>
    >>> with torch._dynamo.bytecode_debugger.debug():
    ...     my_fn(torch.randn(3))  # Debugger activates on Dynamo-generated code

Programmatic breakpoints (for Dynamo developers):
    In PyCodegen, use create_breakpoint() to insert a debugger stop:

        from torch._dynamo.bytecode_transformation import create_breakpoint
        codegen.extend_output(create_breakpoint())
"""

from __future__ import annotations

import dis
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from typing_extensions import Self

from .bytecode_transformation import convert_instruction, Instruction, instruction_size


# Python 3.12+ has sys.monitoring for efficient instruction-level tracing
_HAS_SYS_MONITORING = hasattr(sys, "monitoring")


# Sentinel for breakpoints inserted by PyCodegen via create_breakpoint()
class _BreakpointMarker:
    """Sentinel constant that signals the debugger to break.

    Usage in PyCodegen:
        from torch._dynamo.bytecode_transformation import create_breakpoint
        codegen.extend_output(create_breakpoint())
    """

    __slots__ = ()
    _instance: _BreakpointMarker | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = cast("_BreakpointMarker", object.__new__(cls))
        return cls._instance

    def __repr__(self) -> str:
        return "<BREAKPOINT>"


BREAKPOINT_MARKER = _BreakpointMarker()


# Import NULL_STACK_VALUE sentinel from C++ module
# This is returned by _get_frame_value_stack_with_depth for NULL stack slots
from torch._C._dynamo.eval_frame import NULL_STACK_VALUE  # noqa: F401


@dataclass
class CodeInfo:
    """Per-code-object data shared across all frames executing the same code."""

    code: types.CodeType
    instructions: list[Instruction]
    offset_to_inst: dict[int, Instruction]
    offset_to_index: dict[int, int]
    index_width: int
    offset_width: int
    breakpoints: set[int] = field(default_factory=set)


@dataclass
class DebuggerState:
    """Per-frame debugging state."""

    code_info: CodeInfo
    frame: types.FrameType
    current_offset: int = -1  # -1 indicates first instruction not yet seen
    current_stack_depth: int = 0  # Tracked dynamically
    first_instruction_seen: bool = False
    step_mode: bool = True
    step_count: int = 0  # Remaining steps for "s [n]" command (0 = prompt each time)
    user_locals: dict[str, Any] = field(
        default_factory=dict
    )  # User-defined variables from debugger
    last_command: str = "s"  # Last command for repeat on empty input
    list_index: int | None = (
        None  # Current position for 'l' command (None = use current)
    )

    @property
    def code(self) -> types.CodeType:
        return self.code_info.code

    @property
    def instructions(self) -> list[Instruction]:
        return self.code_info.instructions

    @property
    def offset_to_inst(self) -> dict[int, Instruction]:
        return self.code_info.offset_to_inst

    @property
    def offset_to_index(self) -> dict[int, int]:
        return self.code_info.offset_to_index

    @property
    def index_width(self) -> int:
        return self.code_info.index_width

    @property
    def offset_width(self) -> int:
        return self.code_info.offset_width

    @property
    def breakpoints(self) -> set[int]:
        return self.code_info.breakpoints


class _DebugContext:
    """Internal debug context that manages the debugging session."""

    def __init__(self) -> None:
        self._code_info: dict[types.CodeType, CodeInfo] = {}
        self._frame_states: dict[types.FrameType, DebuggerState] = {}
        self._active = False
        self._tracked_codes: set[types.CodeType] = set()
        self._old_trace: Callable[..., Any] | None = None
        self._prev_callback: Callable[..., Any] | None = None
        self._return_from_frame: types.FrameType | None = None
        self._stop_after_return: bool = False
        self._next_in_frame: types.FrameType | None = None
        self._next_count: int = 0
        self._verbose: bool = False
        self._stop_at_new_code: bool = True
        self._quitting: bool = False
        if _HAS_SYS_MONITORING:
            self._tool_id = sys.monitoring.DEBUGGER_ID

    def get_instructions(self, code: types.CodeType | None = None) -> list[Instruction]:
        """Get the list of instructions for a tracked code object.

        Args:
            code: The code object to get instructions for. If None, returns
                  instructions from the most recently tracked code object.

        Returns:
            List of instructions, or empty list if the code is not tracked.
        """
        if not self._code_info:
            return []
        if code is None:
            info = next(reversed(self._code_info.values()))
        else:
            info = self._code_info.get(code)
            if info is None:
                return []
        return info.instructions

    def get_tracked_codes(self) -> list[types.CodeType]:
        """Get all code objects that have been tracked by the debugger."""
        return list(self._code_info.keys())

    @staticmethod
    def is_programmatic_breakpoint(inst: Instruction) -> bool:
        return inst.opname == "LOAD_CONST" and inst.argval is BREAKPOINT_MARKER

    def _get_or_create_code_info(self, code: types.CodeType) -> CodeInfo:
        """Get or create CodeInfo for a code object."""
        if code not in self._code_info:
            # Use dis.get_instructions directly to preserve original offsets.
            # cleaned_instructions strips EXTENDED_ARG and recalculates offsets,
            # which would cause mismatches with offsets reported by callbacks.
            instructions = [convert_instruction(i) for i in dis.get_instructions(code)]

            # In 3.11+, instructions have inline cache entries that occupy
            # bytecode space (e.g. BINARY_OP is 2 bytes + 2 bytes cache).
            # dis.get_instructions only reports the instruction start offset,
            # but sys.monitoring RAISE reports offsets within cache entries.
            # Map the full byte range of each instruction so lookups succeed.
            offset_to_inst: dict[int, Instruction] = {}
            offset_to_index: dict[int, int] = {}
            for i, inst in enumerate(instructions):
                if inst.offset is not None:
                    inst_size = instruction_size(inst)
                    for off in range(inst.offset, inst.offset + inst_size, 2):
                        offset_to_inst[off] = inst
                        offset_to_index[off] = i

            max_index = len(instructions) - 1 if instructions else 0
            max_offset = max(
                (inst.offset for inst in instructions if inst.offset is not None),
                default=0,
            )

            # Pre-populate breakpoints from BREAKPOINT_MARKER instructions
            programmatic_breakpoints: set[int] = {
                i
                for i, inst in enumerate(instructions)
                if self.is_programmatic_breakpoint(inst)
            }

            self._code_info[code] = CodeInfo(
                code=code,
                instructions=instructions,
                offset_to_inst=offset_to_inst,
                offset_to_index=offset_to_index,
                index_width=max(1, len(str(max_index))),
                offset_width=max(1, len(str(max_offset))),
                breakpoints=programmatic_breakpoints,
            )
        return self._code_info[code]

    def _get_or_create_frame_state(
        self, code: types.CodeType, frame: types.FrameType
    ) -> DebuggerState:
        """Get or create per-frame DebuggerState."""
        if frame not in self._frame_states:
            code_info = self._get_or_create_code_info(code)
            self._frame_states[frame] = DebuggerState(code_info=code_info, frame=frame)
        return self._frame_states[frame]

    def _find_frame_for_code(self, code: types.CodeType) -> types.FrameType | None:
        """Walk the stack to find the frame executing the given code."""
        f: types.FrameType | None = sys._getframe()
        while f is not None:
            if f.f_code is code:
                return f
            f = f.f_back
        return None

    def _build_tracked_frame_stack(self, state: DebuggerState) -> list[DebuggerState]:
        """Build list of tracked ancestor frames, outermost-first.

        Walks f_back from the current execution frame to find all ancestor
        frames whose code is tracked by the debugger. Returns the corresponding
        DebuggerState objects with the execution state (state) last.
        """
        ancestors: list[DebuggerState] = []
        f: types.FrameType | None = state.frame.f_back
        while f is not None:
            ancestor_state = self._frame_states.get(f)
            if ancestor_state is not None:
                ancestors.append(ancestor_state)
            f = f.f_back
        ancestors.reverse()
        ancestors.append(state)
        return ancestors

    def _format_instruction(
        self, state: DebuggerState, offset: int, mark_current: bool = True
    ) -> str:
        """Format an instruction for display."""
        inst = state.offset_to_inst.get(offset)
        iw, ow = state.index_width, state.offset_width
        if inst is None:
            return f"    [{offset:{ow}d}]: <unknown>"

        index = state.offset_to_index.get(offset, -1)
        marker = ">>>" if mark_current and offset == state.current_offset else "   "
        bp_marker = (
            "*" if state.offset_to_index.get(offset, -1) in state.breakpoints else " "
        )
        arg_str = f" {inst.argval}" if inst.arg is not None else ""
        return f"{marker}{bp_marker} {index:{iw}d} [{offset:{ow}d}]: {inst.opname}{arg_str}"

    def _format_header(self, state: DebuggerState) -> str:
        """Format the header line for instruction listings."""
        iw, ow = state.index_width, state.offset_width
        return f"     {'#':>{iw}} [{'offset':>{ow}}]"

    def _print_context(
        self, state: DebuggerState, before: int = 3, after: int = 3
    ) -> None:
        """Print instructions around the current position."""
        current_idx = state.offset_to_index.get(state.current_offset, 0)
        start = max(0, current_idx - before)
        end = min(len(state.instructions), current_idx + after + 1)

        print(f"\nInstruction {current_idx} at offset {state.current_offset}:")
        print(self._format_header(state))
        for i in range(start, end):
            inst = state.instructions[i]
            if inst.offset is not None:
                print(self._format_instruction(state, inst.offset))
        print()

    def _safe_stack_depth(self, state: DebuggerState, is_current_frame: bool) -> int:
        """Compute a safe stack depth for reading frame values.

        For parent frames suspended at a CALL instruction, CPython has already
        popped (and DECREF'd) the call arguments.  The tracked depth still
        includes those consumed entries, so reading them would dereference
        dangling pointers.  Adjust by the current instruction's stack_effect.
        """
        depth = state.current_stack_depth
        if not is_current_frame and state.current_offset >= 0:
            inst = state.offset_to_inst.get(state.current_offset)
            if inst is not None:
                try:
                    effect = dis.stack_effect(inst.opcode, inst.arg)
                    if effect < 0:
                        depth = max(0, depth + effect)
                except (ValueError, TypeError):
                    pass
        return depth

    def _print_stack(self, state: DebuggerState, is_current_frame: bool = True) -> None:
        """Print the current value stack."""
        from torch._C._dynamo.eval_frame import _get_frame_value_stack_with_depth

        depth = self._safe_stack_depth(state, is_current_frame)
        try:
            stack = _get_frame_value_stack_with_depth(state.frame, depth)
        except Exception as e:
            print(f"\nStack: (error reading stack: {e})")
            return
        print("\nStack (TOS at end):")
        if not stack:
            print("  (empty)")
        else:
            for i, value in enumerate(stack):
                if value is NULL_STACK_VALUE:
                    addr = "0x0"
                else:
                    addr = f"0x{id(value):x}"
                print(f"  [{i}] {addr} {value!r}")
        print()

    def _print_locals(self, state: DebuggerState) -> None:
        """Print local variables."""
        print("\nLocals:")
        locals_dict = state.frame.f_locals
        if not locals_dict:
            print("  (none)")
        else:
            for name, value in locals_dict.items():
                print(f"  {name} = {value!r}")
        print()

    def _print_globals(self, state: DebuggerState, pattern: str | None = None) -> None:
        """Print global variables."""
        print("\nGlobals:")
        for name, value in state.frame.f_globals.items():
            if pattern and pattern not in name:
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, types.ModuleType):
                continue
            print(f"  {name} = {value!r}")
        print()

    def _disassemble(self, state: DebuggerState) -> None:
        """Print the full disassembly."""
        print(f"\nDisassembly ({len(state.instructions)} instructions):")
        print(self._format_header(state))
        for inst in state.instructions:
            if inst.offset is not None:
                print(self._format_instruction(state, inst.offset, mark_current=False))
        print()

    def _print_help(self) -> None:
        """Print help message."""
        print("\nCommands:")
        print("  s [n]       - Step n instructions (default 1), stepping into calls")
        print("  n [n]       - Step n instructions (default 1), stepping over calls")
        print("  c, cont     - Continue until breakpoint or exception")
        print("  r           - Continue until current function's return instruction")
        print(
            "  v, verbose  - Toggle verbose mode (print each instruction before executing)"
        )
        print("                Use with 'c' to find segfaults - last printed = culprit")
        print("  u [count]   - Move up the call stack (toward caller) for inspection")
        print("  d [count]   - Move down the call stack (toward callee)")
        print("  bt, w       - Print tracked frame backtrace")
        print("  p <expr>    - Print expression")
        print("  l [first[, last]]  - List instructions (like pdb)")
        print("  l .         - List around current instruction")
        print("  ll          - Disassemble all bytecode")
        print("  stack       - Print value stack")
        print("  locals      - Print local variables")
        print("  globals     - Print global variables")
        print("  b <n>       - Set breakpoint at instruction n (see # column)")
        print("  b           - List breakpoints")
        print("  cl <n>      - Clear breakpoint at instruction n")
        print("  q, quit     - Exit debugger")
        print("  <expr>      - Evaluate Python expression (like pdb)")
        print()
        print("Special variables: __stack__ (list of stack values, TOS at end)")
        print("Note: Debugger stops on exceptions and shows the failing instruction.")
        print()

    def _get_stack_for_eval(
        self, state: DebuggerState, is_current_frame: bool = True
    ) -> list[Any]:
        """Get the current stack for use in expression evaluation."""
        from torch._C._dynamo.eval_frame import _get_frame_value_stack_with_depth

        depth = self._safe_stack_depth(state, is_current_frame)
        try:
            return _get_frame_value_stack_with_depth(state.frame, depth)
        except Exception:
            return []

    def _build_eval_locals(
        self, state: DebuggerState, is_current_frame: bool = True
    ) -> dict[str, Any]:
        """Build locals dict for expression evaluation, including user-defined variables."""
        eval_locals = dict(state.frame.f_locals)
        eval_locals.update(state.user_locals)
        eval_locals["__stack__"] = self._get_stack_for_eval(state, is_current_frame)
        return eval_locals

    def _interactive_prompt(self, state: DebuggerState) -> None:
        """Interactive prompt during debugging."""
        frame_stack = self._build_tracked_frame_stack(state)
        view_index = len(frame_stack) - 1
        view_state = frame_stack[view_index]

        self._print_context(view_state)

        while True:
            try:
                cmd = input("(bdb) ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting debugger.")
                self._quitting = True
                raise KeyboardInterrupt from None

            if not cmd:
                cmd = state.last_command
            else:
                state.last_command = cmd

            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if action in ("s", "step"):
                state.step_mode = True
                self._stop_at_new_code = True
                # Parse optional count argument (e.g., "s 3" to step 3 times)
                if arg:
                    try:
                        count = int(arg)
                        state.step_count = (
                            count - 1
                        )  # -1 because we're about to execute one
                    except ValueError:
                        print(f"Invalid count: {arg}")
                        continue
                else:
                    state.step_count = 0
                return

            elif action in ("n", "next"):
                self._next_in_frame = view_state.frame
                if arg:
                    try:
                        count = int(arg)
                        self._next_count = count - 1
                    except ValueError:
                        print(f"Invalid count: {arg}")
                        continue
                else:
                    self._next_count = 0
                state.step_mode = False
                self._stop_at_new_code = False
                return

            elif action in ("c", "cont", "continue"):
                state.step_mode = False
                self._stop_at_new_code = False
                self._stop_after_return = False
                return

            elif action in ("r", "return"):
                self._return_from_frame = state.frame
                state.step_mode = False
                self._stop_at_new_code = False
                return

            elif action in ("u", "up"):
                count = 1
                if arg:
                    try:
                        count = int(arg)
                    except ValueError:
                        print(f"Invalid count: {arg}")
                        continue
                new_index = view_index - count
                if new_index < 0:
                    new_index = 0
                    print("Oldest tracked frame")
                view_index = new_index
                view_state = frame_stack[view_index]
                print(f"> {view_state.code.co_name}")
                self._print_context(view_state)

            elif action in ("d", "down"):
                count = 1
                if arg:
                    try:
                        count = int(arg)
                    except ValueError:
                        print(f"Invalid count: {arg}")
                        continue
                max_index = len(frame_stack) - 1
                new_index = view_index + count
                if new_index > max_index:
                    new_index = max_index
                    print("Newest tracked frame")
                view_index = new_index
                view_state = frame_stack[view_index]
                print(f"> {view_state.code.co_name}")
                self._print_context(view_state)

            elif action in ("bt", "w", "where"):
                for i, fs in enumerate(frame_stack):
                    marker = ">" if i == view_index else " "
                    idx = fs.offset_to_index.get(fs.current_offset, -1)
                    inst = fs.offset_to_inst.get(fs.current_offset)
                    inst_str = inst.opname if inst else "<unknown>"
                    print(
                        f"  {marker} [{i}] {fs.code.co_name} instruction {idx} ({inst_str})"
                    )

            elif action in ("v", "verbose"):
                self._verbose = not self._verbose
                status = "enabled" if self._verbose else "disabled"
                print(f"Verbose mode {status}.")

            elif action in ("l", "list"):
                # Like pdb:
                # - 'l' continues from last position (or centers on current if first time)
                # - 'l .' centers on current instruction
                # - 'l N' lists 11 instructions starting at N
                # - 'l first, last' lists range (if last < first, last is a count)
                num_lines = 11  # pdb default
                start: int | None = None
                end: int | None = None

                if arg == ".":
                    # Reset to current instruction
                    view_state.list_index = None
                elif arg:
                    # Check for range syntax: "first, last" or "first,last"
                    if "," in arg:
                        parts = arg.split(",", 1)
                        try:
                            first = int(parts[0].strip())
                            second = int(parts[1].strip())
                            if second < first:
                                # second is a count
                                start = first
                                end = first + second
                            else:
                                start = first
                                end = second + 1  # inclusive
                        except ValueError:
                            print(f"Invalid range: {arg}")
                            continue
                    else:
                        # Single number: start at that instruction
                        try:
                            start = int(arg)
                            end = start + num_lines
                        except ValueError:
                            print(f"Invalid argument: {arg}")
                            continue

                if start is None:
                    if view_state.list_index is None:
                        # Center on current instruction
                        current_idx = view_state.offset_to_index.get(
                            view_state.current_offset, 0
                        )
                        start = max(0, current_idx - num_lines // 2)
                    else:
                        start = view_state.list_index
                    end = start + num_lines

                assert end is not None
                end = min(len(view_state.instructions), end)
                if start >= len(view_state.instructions):
                    print("(End of bytecode)")
                else:
                    print(self._format_header(view_state))
                    for i in range(start, end):
                        inst = view_state.instructions[i]
                        if inst.offset is not None:
                            print(self._format_instruction(view_state, inst.offset))
                    print()
                    view_state.list_index = end

            elif action == "ll":
                self._disassemble(view_state)

            elif action == "locals":
                self._print_locals(view_state)

            elif action == "globals":
                self._print_globals(view_state, arg if arg else None)

            elif action == "stack":
                is_current = view_state is frame_stack[-1]
                self._print_stack(view_state, is_current_frame=is_current)

            elif action == "b":
                if arg:
                    try:
                        index = int(arg)
                        num_instructions = len(view_state.instructions)
                        if index < 0 or index >= num_instructions:
                            print(
                                f"Invalid instruction number: {index} "
                                f"(must be 0-{num_instructions - 1})"
                            )
                        else:
                            view_state.breakpoints.add(index)
                            print(f"Breakpoint set at instruction {index}")
                    except ValueError:
                        print(f"Invalid instruction number: {arg}")
                else:
                    print("\nBreakpoints:")
                    if not view_state.breakpoints:
                        print("  (none)")
                    else:
                        for bp_index in sorted(view_state.breakpoints):
                            if bp_index < len(view_state.instructions):
                                inst = view_state.instructions[bp_index]
                                if inst.offset is not None:
                                    print(
                                        f"  {self._format_instruction(view_state, inst.offset, mark_current=False)}"
                                    )
                    print()

            elif action == "cl":
                if arg:
                    try:
                        index = int(arg)
                        num_instructions = len(view_state.instructions)
                        if index < 0 or index >= num_instructions:
                            print(
                                f"Invalid instruction number: {index} "
                                f"(must be 0-{num_instructions - 1})"
                            )
                        elif index in view_state.breakpoints:
                            view_state.breakpoints.discard(index)
                            print(f"Breakpoint cleared at instruction {index}")
                        else:
                            print(f"No breakpoint at instruction {index}")
                    except ValueError:
                        print(f"Invalid instruction number: {arg}")

            elif action == "p":
                if arg:
                    try:
                        is_current = view_state is frame_stack[-1]
                        frame_globals = view_state.frame.f_globals
                        eval_locals = self._build_eval_locals(
                            view_state, is_current_frame=is_current
                        )
                        result = eval(arg, frame_globals, eval_locals)
                        print(f"{arg} = {result!r}")
                    except Exception as e:
                        print(f"Error evaluating '{arg}': {e}")
                else:
                    print("Usage: p <expression>")

            elif action in ("h", "help", "?"):
                self._print_help()

            elif action in ("q", "quit", "exit"):
                print("Exiting debugger.")
                self._quitting = True
                raise KeyboardInterrupt

            else:
                # Try to execute as Python code (like pdb)
                is_current = view_state is frame_stack[-1]
                frame_globals = view_state.frame.f_globals
                eval_locals = self._build_eval_locals(
                    view_state, is_current_frame=is_current
                )

                try:
                    # First try as expression (eval)
                    result = eval(cmd, frame_globals, eval_locals)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    # If not a valid expression, try as statement (exec)
                    try:
                        keys_before = set(eval_locals.keys())
                        ids_before = {k: id(v) for k, v in eval_locals.items()}
                        exec(cmd, frame_globals, eval_locals)
                        # Capture any new or modified variables into user_locals
                        for key in eval_locals:
                            is_new = key not in keys_before
                            is_user_var = key in view_state.user_locals
                            is_modified = id(eval_locals[key]) != ids_before.get(key)
                            if (
                                is_new or is_user_var or is_modified
                            ) and key != "__stack__":
                                view_state.user_locals[key] = eval_locals[key]
                    except Exception as e:
                        print(f"*** {type(e).__name__}: {e}")
                except Exception as e:
                    print(f"*** {type(e).__name__}: {e}")

    def _handle_instruction(
        self, code: types.CodeType, offset: int, frame: types.FrameType | None = None
    ) -> None:
        """Common instruction handling logic for both tracing backends."""
        if frame is None:
            frame = self._find_frame_for_code(code)
            assert frame is not None
        state = self._get_or_create_frame_state(code, frame)

        # Update stack depth based on the previous instruction's effect.
        # The callback is called BEFORE instruction at 'offset' executes,
        # so we need to apply the effect of the previous instruction first.
        previous_offset = state.current_offset
        if previous_offset >= 0:
            prev_inst = state.offset_to_inst.get(previous_offset)
            if prev_inst is not None:
                # Detect if a jump was taken: current offset != sequential next
                prev_index = state.offset_to_index.get(previous_offset, -1)
                expected_next_index = prev_index + 1
                current_index = state.offset_to_index.get(offset, -1)
                did_jump = current_index != expected_next_index

                try:
                    effect = dis.stack_effect(
                        prev_inst.opcode, prev_inst.arg, jump=did_jump
                    )
                    state.current_stack_depth += effect
                    if state.current_stack_depth < 0:
                        raise RuntimeError(
                            f"Stack depth went negative after {prev_inst.opname} "
                            f"at offset {previous_offset}: depth={state.current_stack_depth}"
                        )
                except (ValueError, TypeError):
                    pass

        state.current_offset = offset
        state.list_index = None  # Reset list position when stepping

        # After 'r' command completes, resume stepping at the next instruction
        if self._stop_after_return:
            self._stop_after_return = False
            state.step_mode = True
            state.step_count = 0
            self._stop_at_new_code = True

        # After 'n', resume stepping when we reach that frame
        if self._next_in_frame is not None and frame is self._next_in_frame:
            if self._next_count > 0:
                # More steps remaining — re-arm for the next instruction in this frame
                self._next_count -= 1
            else:
                self._next_in_frame = None
                state.step_mode = True
                state.step_count = 0
                self._stop_at_new_code = True
            if _HAS_SYS_MONITORING:
                sys.monitoring.restart_events()

        # First instruction - print header and enter step mode (unless suppressed)
        if not state.first_instruction_seen:
            state.first_instruction_seen = True
            if not self._stop_at_new_code:
                state.step_mode = False
            if state.step_mode:
                print(f"\n=== Entering Dynamo-generated code: {code.co_name} ===")
                self._print_help()
            elif self._verbose:
                print(f"\n=== Entering Dynamo-generated code: {code.co_name} ===")

        # Verbose mode: print each instruction before executing (for segfault debugging)
        if self._verbose:
            inst = state.offset_to_inst.get(offset)
            if inst:
                idx = state.offset_to_index.get(offset, -1)
                arg_str = f" {inst.argval}" if inst.arg is not None else ""
                print(f"Running [{idx}] {inst.opname}{arg_str}", flush=True)

        inst = state.offset_to_inst.get(offset)

        # Check if current instruction has a breakpoint (by index)
        current_index = state.offset_to_index.get(offset, -1)
        hit_breakpoint = current_index in state.breakpoints

        # 'r' command: stop at RETURN_VALUE/RETURN_CONST in the target frame
        # so the user can inspect the return value on the stack before returning.
        hit_return_target = False
        if self._return_from_frame is not None and frame is self._return_from_frame:
            if inst is not None and inst.opname in ("RETURN_VALUE", "RETURN_CONST"):
                self._return_from_frame = None
                # After stepping past the return, stop in the caller
                self._stop_after_return = True
                hit_return_target = True

        # Check if we should stop
        # If step_count > 0, we're in the middle of "N s" and should continue
        if state.step_count > 0:
            state.step_count -= 1
            # But still stop for breakpoints and return targets
            if hit_breakpoint or hit_return_target:
                state.step_count = 0  # Cancel remaining steps
            else:
                return  # Continue without prompting

        should_stop = state.step_mode or hit_breakpoint or hit_return_target
        if should_stop:
            if hit_breakpoint:
                if inst is not None and self.is_programmatic_breakpoint(inst):
                    print("Breakpoint hit (programmatic)")
                else:
                    print(f"Breakpoint hit at instruction {current_index}")
            elif hit_return_target:
                print(f"About to return from {code.co_name}")
            self._interactive_prompt(state)

    def _handle_return(
        self, code: types.CodeType, retval: object, frame: types.FrameType | None = None
    ) -> None:
        """Common return handling logic."""
        if self._quitting:
            return
        print(f"\n=== {code.co_name} returned: {retval!r} ===")

        # For sys.monitoring, we don't get the frame directly, but since
        # the frame that's currently returning is the one whose PY_RETURN just
        # fired, code identity is sufficient to match.
        if self._return_from_frame is not None and (
            frame is self._return_from_frame
            or (frame is None and code is self._return_from_frame.f_code)
        ):
            self._return_from_frame = None
            self._stop_after_return = True
            if _HAS_SYS_MONITORING:
                sys.monitoring.restart_events()
        if self._next_in_frame is not None and (
            frame is self._next_in_frame
            or (frame is None and code is self._next_in_frame.f_code)
        ):
            self._next_in_frame = None
            self._stop_after_return = True
            if _HAS_SYS_MONITORING:
                sys.monitoring.restart_events()

        # Clean up frame state to avoid holding dead frame references
        if frame is not None:
            self._frame_states.pop(frame, None)
        else:
            # sys.monitoring path: find and remove matching frame state by code
            for f, s in list(self._frame_states.items()):
                if s.code is code:
                    self._frame_states.pop(f, None)
                    break

    def _handle_exception(
        self, code: types.CodeType, offset: int, exception: BaseException
    ) -> None:
        """Common exception handling logic."""
        if self._quitting:
            return
        frame = self._find_frame_for_code(code)
        if frame is None:
            return
        state = self._frame_states.get(frame)
        if state is None:
            return

        # If we're waiting for this frame to return, the return was interrupted
        if self._return_from_frame is not None and (
            frame is self._return_from_frame or code is self._return_from_frame.f_code
        ):
            self._return_from_frame = None

        # If we're waiting for next-in-frame, the execution was interrupted
        if self._next_in_frame is not None and (
            frame is self._next_in_frame or code is self._next_in_frame.f_code
        ):
            self._next_in_frame = None
            self._stop_at_new_code = True

        state.current_offset = offset

        inst = state.offset_to_inst.get(offset)
        inst_str = inst.opname if inst else "<unknown>"
        current_index = state.offset_to_index.get(offset, -1)

        print(f"\n=== Exception raised at instruction {current_index}: {inst_str} ===")
        print(f"=== {type(exception).__name__}: {exception} ===")
        self._interactive_prompt(state)

    # =========================================================================
    # Python 3.12+ implementation using sys.monitoring
    # =========================================================================

    def _dynamo_code_callback(self, code: types.CodeType) -> None:
        """Called by C++ before executing Dynamo-generated code."""
        if code in self._tracked_codes:
            return
        self._tracked_codes.add(code)

        if _HAS_SYS_MONITORING:
            # Enable INSTRUCTION and PY_RETURN events for this specific code object
            # RAISE must be a global event (cannot be set locally)
            sys.monitoring.set_local_events(
                self._tool_id,
                code,
                sys.monitoring.events.INSTRUCTION | sys.monitoring.events.PY_RETURN,
            )
        # For settrace, we enable opcode tracing when we see the frame

        # Pre-create code info for this code
        self._get_or_create_code_info(code)

    def _monitoring_return_callback(
        self, code: types.CodeType, instruction_offset: int, retval: object
    ) -> None:
        """Callback for PY_RETURN events (sys.monitoring)."""
        self._handle_return(code, retval)

    def _monitoring_instruction_callback(
        self, code: types.CodeType, offset: int
    ) -> object:
        """Callback for INSTRUCTION events (sys.monitoring)."""
        self._handle_instruction(code, offset)
        return sys.monitoring.DISABLE

    def _monitoring_raise_callback(
        self, code: types.CodeType, offset: int, exception: BaseException
    ) -> object:
        """Callback for RAISE events (sys.monitoring)."""
        # Only handle exceptions from tracked Dynamo-generated code
        if code not in self._tracked_codes:
            return None
        self._handle_exception(code, offset, exception)
        # Cannot return DISABLE for global events like RAISE
        return None

    # =========================================================================
    # Python 3.11 and below implementation using sys.settrace
    # =========================================================================

    def _settrace_callback(
        self, frame: types.FrameType, event: str, arg: Any
    ) -> Callable[..., Any] | None:
        """Trace function for sys.settrace."""
        code = frame.f_code

        # Only trace Dynamo-generated code
        if code not in self._tracked_codes:
            return self._settrace_callback

        if event == "call":
            # Enable opcode tracing for this frame
            frame.f_trace_opcodes = True
            return self._settrace_callback

        elif event == "opcode":
            # Get the current instruction offset
            offset = frame.f_lasti
            self._handle_instruction(code, offset, frame)
            return self._settrace_callback

        elif event == "return":
            self._handle_return(code, arg, frame)
            return self._settrace_callback

        elif event == "exception":
            # arg is (exception_type, exception_value, traceback)
            offset = frame.f_lasti
            exc_type, exc_value, exc_tb = arg
            self._handle_exception(code, offset, exc_value)
            return self._settrace_callback

        return self._settrace_callback

    # =========================================================================
    # Context manager implementation
    # =========================================================================

    def __enter__(self) -> Self:
        """Start the debug context."""
        from torch._C._dynamo.eval_frame import (
            get_bytecode_debugger_callback,
            set_bytecode_debugger_callback,
        )

        self._active = True
        self._prev_callback = get_bytecode_debugger_callback()
        self._code_info.clear()
        self._frame_states.clear()
        self._tracked_codes.clear()

        if _HAS_SYS_MONITORING:
            # Python 3.12+: Use sys.monitoring
            try:
                sys.monitoring.use_tool_id(self._tool_id, "bytecode_debugger")
            except ValueError:
                pass

            # Register callbacks (events enabled per-code via set_local_events)
            sys.monitoring.register_callback(
                self._tool_id,
                sys.monitoring.events.INSTRUCTION,
                self._monitoring_instruction_callback,
            )
            sys.monitoring.register_callback(
                self._tool_id,
                sys.monitoring.events.PY_RETURN,
                self._monitoring_return_callback,
            )
            sys.monitoring.register_callback(
                self._tool_id,
                sys.monitoring.events.RAISE,
                self._monitoring_raise_callback,
            )
            # RAISE must be a global event (cannot be local)
            sys.monitoring.set_events(self._tool_id, sys.monitoring.events.RAISE)
        else:
            # Python 3.11 and below: Use sys.settrace
            self._old_trace = sys.gettrace()
            sys.settrace(self._settrace_callback)

        # Set the C callback that will be called before executing Dynamo code
        set_bytecode_debugger_callback(self._dynamo_code_callback)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        """End the debug context."""
        if not self._active:
            return False
        from torch._C._dynamo.eval_frame import set_bytecode_debugger_callback

        self._active = False
        prev = self._prev_callback
        set_bytecode_debugger_callback(prev)

        if _HAS_SYS_MONITORING:
            if prev is None:
                sys.monitoring.set_events(self._tool_id, 0)
                try:
                    sys.monitoring.free_tool_id(self._tool_id)
                except ValueError:
                    pass
            else:
                # Restore outer context's monitoring callbacks and events
                assert isinstance(prev, types.MethodType)
                outer = cast("_DebugContext", prev.__self__)
                sys.monitoring.register_callback(
                    self._tool_id,
                    sys.monitoring.events.INSTRUCTION,
                    outer._monitoring_instruction_callback,
                )
                sys.monitoring.register_callback(
                    self._tool_id,
                    sys.monitoring.events.PY_RETURN,
                    outer._monitoring_return_callback,
                )
                sys.monitoring.register_callback(
                    self._tool_id,
                    sys.monitoring.events.RAISE,
                    outer._monitoring_raise_callback,
                )
                sys.monitoring.set_events(self._tool_id, sys.monitoring.events.RAISE)
                for code in outer._tracked_codes:
                    sys.monitoring.set_local_events(
                        self._tool_id,
                        code,
                        sys.monitoring.events.INSTRUCTION
                        | sys.monitoring.events.PY_RETURN,
                    )
        else:
            # Python 3.11 and below: Restore old trace
            sys.settrace(self._old_trace)

        return False  # Don't suppress exceptions


@contextmanager
def debug() -> Generator[_DebugContext, None, None]:
    """
    Context manager for debugging Dynamo-generated bytecode.

    Any Dynamo-generated code executed within this context will trigger
    the interactive bytecode debugger.

    Example:
        >>> # xdoctest: +SKIP
        >>> import torch
        >>>
        >>> @torch.compile
        >>> def my_fn(x):
        ...     return x + 1
        >>>
        >>> with torch._dynamo.bytecode_debugger.debug():
        ...     my_fn(torch.randn(3))
    """
    ctx = _DebugContext()
    try:
        with ctx:
            yield ctx
    except KeyboardInterrupt:
        print("\n=== Debug session ended ===")


def breakpoint() -> None:
    """Programmatic breakpoint for user code.

    Place this in code compiled by Dynamo. During tracing, Dynamo inserts a
    BREAKPOINT_MARKER into the compiled bytecode. When the bytecode debugger
    is active, execution pauses at this marker.
    """
