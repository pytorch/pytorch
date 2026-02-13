"""
Bytecode debugger for Dynamo-optimized code.

This module provides a pdb-like debugger for stepping through Python bytecode
one instruction at a time, with the ability to inspect the value stack,
locals, and globals.

Usage:
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
class DebuggerState:
    """State for a code object being debugged."""

    code: types.CodeType
    instructions: list[Instruction]
    offset_to_inst: dict[int, Instruction]
    offset_to_index: dict[int, int]
    index_width: int
    offset_width: int
    breakpoints: set[int] = field(default_factory=set)
    step_mode: bool = True
    verbose_mode: bool = (
        False  # Print each instruction before executing (for segfault debugging)
    )
    current_frame: types.FrameType | None = None
    current_offset: int = -1  # -1 indicates first instruction not yet seen
    current_stack_depth: int = 0  # Tracked dynamically
    first_instruction_seen: bool = False
    user_locals: dict[str, Any] = field(
        default_factory=dict
    )  # User-defined variables from debugger
    last_command: str = "s"  # Last command for repeat on empty input


class _DebugContext:
    """Internal debug context that manages the debugging session."""

    def __init__(self) -> None:
        self._code_states: dict[types.CodeType, DebuggerState] = {}
        self._active = False
        self._tracked_codes: set[types.CodeType] = set()
        self._old_trace: Callable[..., Any] | None = None
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
        if not self._code_states:
            return []
        if code is None:
            # Return instructions from the most recently tracked code
            state = next(reversed(self._code_states.values()))
        else:
            state = self._code_states.get(code)
            if state is None:
                return []
        return state.instructions

    def get_tracked_codes(self) -> list[types.CodeType]:
        """Get all code objects that have been tracked by the debugger."""
        return list(self._code_states.keys())

    def _get_or_create_state(self, code: types.CodeType) -> DebuggerState:
        """Get or create debugger state for a code object."""
        if code not in self._code_states:
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

            self._code_states[code] = DebuggerState(
                code=code,
                instructions=instructions,
                offset_to_inst=offset_to_inst,
                offset_to_index=offset_to_index,
                index_width=max(1, len(str(max_index))),
                offset_width=max(1, len(str(max_offset))),
            )
        return self._code_states[code]

    def _find_frame_for_code(self, code: types.CodeType) -> types.FrameType | None:
        """Walk the stack to find the frame executing the given code."""
        f: types.FrameType | None = sys._getframe()
        while f is not None:
            if f.f_code is code:
                return f
            f = f.f_back
        return None

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

    def _print_stack(self, state: DebuggerState) -> None:
        """Print the current value stack."""
        if state.current_frame is None:
            print("\nStack: (no frame available)")
            return

        from torch._C._dynamo.eval_frame import _get_frame_value_stack_with_depth

        depth = state.current_stack_depth
        try:
            stack = _get_frame_value_stack_with_depth(state.current_frame, depth)
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
        if state.current_frame is None:
            print("\nLocals: (no frame available)")
            return

        print("\nLocals:")
        locals_dict = state.current_frame.f_locals
        if not locals_dict:
            print("  (none)")
        else:
            for name, value in locals_dict.items():
                print(f"  {name} = {value!r}")
        print()

    def _print_globals(self, state: DebuggerState, pattern: str | None = None) -> None:
        """Print global variables."""
        if state.current_frame is None:
            print("\nGlobals: (no frame available)")
            return

        print("\nGlobals:")
        for name, value in state.current_frame.f_globals.items():
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
        print("  s, step     - Execute one instruction")
        print(
            "  c, cont     - Continue until breakpoint, exception, or next Dynamo code"
        )
        print(
            "  v, verbose  - Toggle verbose mode (print each instruction before executing)"
        )
        print("                Use with 'c' to find segfaults - last printed = culprit")
        print("  p <expr>    - Print expression")
        print("  l, list     - Show context around current instruction")
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

    def _get_stack_for_eval(self, state: DebuggerState) -> list[Any]:
        """Get the current stack for use in expression evaluation."""
        if state.current_frame is None:
            return []
        from torch._C._dynamo.eval_frame import _get_frame_value_stack_with_depth

        depth = state.current_stack_depth
        try:
            return _get_frame_value_stack_with_depth(state.current_frame, depth)
        except Exception:
            return []

    def _build_eval_locals(self, state: DebuggerState) -> dict[str, Any]:
        """Build locals dict for expression evaluation, including user-defined variables."""
        frame_locals = state.current_frame.f_locals if state.current_frame else {}
        # Start with frame locals, overlay with user-defined variables
        eval_locals = dict(frame_locals)
        eval_locals.update(state.user_locals)
        eval_locals["__stack__"] = self._get_stack_for_eval(state)
        return eval_locals

    def _interactive_prompt(self, state: DebuggerState) -> None:
        """Interactive prompt during debugging."""
        self._print_context(state)

        while True:
            try:
                cmd = input("(bdb) ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting debugger.")
                raise KeyboardInterrupt from None

            if not cmd:
                cmd = state.last_command
            else:
                state.last_command = cmd

            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if action in ("s", "step", "n", "next"):
                state.step_mode = True
                return

            elif action in ("c", "cont", "continue"):
                state.step_mode = False
                return

            elif action in ("v", "verbose"):
                state.verbose_mode = not state.verbose_mode
                status = "enabled" if state.verbose_mode else "disabled"
                print(f"Verbose mode {status}.")

            elif action in ("l", "list"):
                self._print_context(state)

            elif action == "ll":
                self._disassemble(state)

            elif action == "locals":
                self._print_locals(state)

            elif action == "globals":
                self._print_globals(state, arg if arg else None)

            elif action == "stack":
                self._print_stack(state)

            elif action == "b":
                if arg:
                    try:
                        index = int(arg)
                        num_instructions = len(state.instructions)
                        if index < 0 or index >= num_instructions:
                            print(
                                f"Invalid instruction number: {index} "
                                f"(must be 0-{num_instructions - 1})"
                            )
                        else:
                            state.breakpoints.add(index)
                            print(f"Breakpoint set at instruction {index}")
                    except ValueError:
                        print(f"Invalid instruction number: {arg}")
                else:
                    print("\nBreakpoints:")
                    if not state.breakpoints:
                        print("  (none)")
                    else:
                        for bp_index in sorted(state.breakpoints):
                            if bp_index < len(state.instructions):
                                inst = state.instructions[bp_index]
                                if inst.offset is not None:
                                    print(
                                        f"  {self._format_instruction(state, inst.offset, mark_current=False)}"
                                    )
                    print()

            elif action == "cl":
                if arg:
                    try:
                        index = int(arg)
                        num_instructions = len(state.instructions)
                        if index < 0 or index >= num_instructions:
                            print(
                                f"Invalid instruction number: {index} "
                                f"(must be 0-{num_instructions - 1})"
                            )
                        elif index in state.breakpoints:
                            state.breakpoints.discard(index)
                            print(f"Breakpoint cleared at instruction {index}")
                        else:
                            print(f"No breakpoint at instruction {index}")
                    except ValueError:
                        print(f"Invalid instruction number: {arg}")

            elif action == "p":
                if arg:
                    try:
                        frame_globals = (
                            state.current_frame.f_globals if state.current_frame else {}
                        )
                        eval_locals = self._build_eval_locals(state)
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
                raise KeyboardInterrupt

            else:
                # Try to execute as Python code (like pdb)
                frame_globals = (
                    state.current_frame.f_globals if state.current_frame else {}
                )
                eval_locals = self._build_eval_locals(state)

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
                            is_user_var = key in state.user_locals
                            is_modified = id(eval_locals[key]) != ids_before.get(key)
                            if (
                                is_new or is_user_var or is_modified
                            ) and key != "__stack__":
                                state.user_locals[key] = eval_locals[key]
                    except Exception as e:
                        print(f"*** {type(e).__name__}: {e}")
                except Exception as e:
                    print(f"*** {type(e).__name__}: {e}")

    def _handle_instruction(
        self, code: types.CodeType, offset: int, frame: types.FrameType | None = None
    ) -> None:
        """Common instruction handling logic for both tracing backends."""
        state = self._get_or_create_state(code)

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
        state.current_frame = (
            frame if frame is not None else self._find_frame_for_code(code)
        )

        # First instruction - print header
        if not state.first_instruction_seen:
            state.first_instruction_seen = True
            print(f"\n=== Entering Dynamo-generated code: {code.co_name} ===")
            self._print_help()

        # Verbose mode: print each instruction before executing (for segfault debugging)
        if state.verbose_mode:
            inst = state.offset_to_inst.get(offset)
            if inst:
                idx = state.offset_to_index.get(offset, -1)
                arg_str = f" {inst.argval}" if inst.arg is not None else ""
                print(f"Running [{idx}] {inst.opname}{arg_str}", flush=True)

        # Check for BREAKPOINT_MARKER (inserted by PyCodegen)
        inst = state.offset_to_inst.get(offset)
        hit_breakpoint_marker = (
            inst is not None
            and inst.opname == "LOAD_CONST"
            and inst.argval is BREAKPOINT_MARKER
        )

        # Check if current instruction has a breakpoint (by index)
        current_index = state.offset_to_index.get(offset, -1)
        hit_breakpoint = current_index in state.breakpoints
        should_stop = state.step_mode or hit_breakpoint or hit_breakpoint_marker
        if should_stop:
            if hit_breakpoint:
                print(f"Breakpoint hit at instruction {current_index}")
            elif hit_breakpoint_marker:
                print("Breakpoint hit (programmatic)")
            self._interactive_prompt(state)

    def _handle_return(self, code: types.CodeType, retval: object) -> None:
        """Common return handling logic."""
        print(f"\n=== {code.co_name} returned: {retval!r} ===")

    def _handle_exception(
        self, code: types.CodeType, offset: int, exception: BaseException
    ) -> None:
        """Common exception handling logic."""
        state = self._code_states.get(code)
        if state is None:
            return

        state.current_offset = offset
        state.current_frame = self._find_frame_for_code(code)

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

        # Pre-create state for this code
        self._get_or_create_state(code)

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
            self._handle_return(code, arg)
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
        from torch._C._dynamo.eval_frame import set_bytecode_debugger_callback

        self._active = True
        self._code_states.clear()
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
        from torch._C._dynamo.eval_frame import set_bytecode_debugger_callback

        self._active = False
        # Clear the C callback
        set_bytecode_debugger_callback(None)

        if _HAS_SYS_MONITORING:
            # Python 3.12+: Clean up sys.monitoring
            sys.monitoring.set_events(self._tool_id, 0)
            try:
                sys.monitoring.free_tool_id(self._tool_id)
            except ValueError:
                pass
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
