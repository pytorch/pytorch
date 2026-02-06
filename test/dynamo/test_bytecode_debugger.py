# Owner(s): ["module: dynamo"]

"""
Tests for torch._dynamo.bytecode_debugger
"""

import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch
import torch._dynamo
from torch._dynamo.bytecode_debugger import debug
from torch._dynamo.test_case import run_tests, TestCase


class InteractiveDebugSession:
    """Allows reactive interaction with the bytecode debugger in tests.

    Uses a generator for cooperative multitasking between test and debugger.
    The test_logic function receives initial output as its argument, then
    yields commands and receives subsequent output via send().
    """

    def __init__(self, fn, args, test_logic):
        self.test_logic = test_logic
        self.output_buffer = StringIO()

        with patch("builtins.input", self._fake_input):
            with redirect_stdout(self.output_buffer):
                with debug() as self.ctx:
                    self.result = fn(*args)

        # Ensure test_logic completed by sending final output
        if hasattr(self.test_logic, "send"):
            final_output = self.output_buffer.getvalue()
            try:
                cmd = self.test_logic.send(final_output)
                raise AssertionError(
                    f"test_logic did not complete - yielded {cmd!r} after session ended"
                )
            except StopIteration:
                pass

    def _fake_input(self, prompt=""):
        self.output_buffer.write(prompt)

        output = self.output_buffer.getvalue()
        self.output_buffer.seek(0)
        self.output_buffer.truncate(0)

        try:
            if not hasattr(self.test_logic, "send"):
                self.test_logic = self.test_logic(self, output)
                return next(self.test_logic)
            return self.test_logic.send(output)
        except StopIteration:
            return "q"


class TestBytecodeDebugger(TestCase):
    def test_debug_context_manager_basic(self):
        """Test that the debug context manager works without errors."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            self.assertIn("Commands:", initial)
            output = yield "c"
            self.assertIn("returned:", output)

        inp = torch.randn(3)
        sess = InteractiveDebugSession(fn, (inp,), test_logic)
        self.assertEqual(sess.result, inp + 1)

    def test_step_mode(self):
        """Test that step mode steps through each instruction."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            step_count = 0
            output = initial
            while "returned:" not in output:
                self.assertIn("Instruction", output)
                step_count += 1
                output = yield "s"
            expected = len(sess.ctx.get_instructions())
            self.assertGreaterEqual(step_count, expected - 1)
            self.assertLessEqual(step_count, expected + 1)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_verbose_mode(self):
        """Test that verbose mode prints each instruction in order."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            output = yield "v"  # Enable verbose mode
            self.assertIn("Verbose mode enabled", output)
            output = yield "c"  # Continue to end

            # Parse verbose output to extract printed instruction opnames
            printed_opnames = []
            for line in output.split("\n"):
                match = re.match(r"Running \[(\d+)\] (\w+)", line)
                if match:
                    printed_opnames.append(match.group(2))

            self.assertGreater(len(printed_opnames), 0)

            # Printed instructions should be a suffix of the full instruction list
            expected_opnames = [inst.opname for inst in sess.ctx.get_instructions()]
            start_idx = len(expected_opnames) - len(printed_opnames)
            self.assertGreaterEqual(start_idx, 0)
            self.assertEqual(printed_opnames, expected_opnames[start_idx:])

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_return_value_printed(self):
        """Test that return values are printed."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        expected_result = input_tensor + 1

        def test_logic(sess, initial):
            output = yield "c"
            expected_line = f"=== fn returned: {expected_result!r} ==="
            self.assertIn(expected_line, output)

        sess = InteractiveDebugSession(fn, (input_tensor,), test_logic)
        self.assertEqual(sess.result, expected_result)

    def test_stack_command(self):
        """Test that the stack command shows correct values including NULL.

        Uses a function with a method call to exercise NULL stack values
        that appear in Python 3.11+ during call setup.
        """

        @torch.compile(backend="eager")
        def fn(x):
            return torch.sin(x)

        def test_logic(sess, initial):
            output = initial
            # Step until we hit a CALL instruction (or PRECALL on Python 3.11)
            if sys.version_info >= (3, 12):
                call_pattern = r">>>.*\[\d+\]:\s*CALL\b"
            else:
                call_pattern = r">>>.*\[\d+\]:\s*PRECALL\b"
            while not re.search(call_pattern, output):
                output = yield "s"
            # Get the stack at CALL/PRECALL
            stack_output = yield "stack"

            # Extract just the stack lines (skip prompt)
            lines = [
                l for l in stack_output.split("\n") if l.strip() and "(bdb)" not in l
            ]
            stack_at_call = "\n".join(lines)

            # Normalize addresses and tensor values
            stack_at_call = re.sub(r"0x[0-9a-f]+(?! <NULL>)", "0xADDR", stack_at_call)
            stack_at_call = re.sub(r"tensor\([^)]+\)", "tensor(...)", stack_at_call)

            if sys.version_info >= (3, 13):
                self.assertExpectedInline(
                    stack_at_call,
                    """\
Stack (TOS at end):
  [0] 0xADDR <function forward at 0xADDR>
  [1] 0x0 <NULL>
  [2] 0xADDR tensor(...)""",
                )
            elif sys.version_info >= (3, 11):
                self.assertExpectedInline(
                    stack_at_call,
                    """\
Stack (TOS at end):
  [0] 0x0 <NULL>
  [1] 0xADDR <function forward at 0xADDR>
  [2] 0xADDR tensor(...)""",
                )
            else:
                # Python 3.10
                self.assertExpectedInline(
                    stack_at_call,
                    """\
Stack (TOS at end):
  [0] 0xADDR <function forward at 0xADDR>
  [1] 0xADDR tensor(...)""",
                )

        InteractiveDebugSession(fn, (torch.ones(3),), test_logic)

    def test_locals_command(self):
        """Test that the locals command shows local variables with values.

        Checks that:
        1. Initially, locals shows x = tensor([...])
        2. After a STORE_FAST instruction, the new local appears
        """

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            return y

        input_tensor = torch.tensor([1.0, 2.0, 3.0])

        def test_logic(sess, initial):
            # Check initial locals shows x
            locals_output = yield "locals"
            self.assertIn("x = tensor([1., 2., 3.])", locals_output)

            # Step until we find a STORE_FAST for a variable other than x
            output = initial
            stored_var_name = None
            while stored_var_name is None:
                output = yield "s"
                match = re.search(r">>>.*STORE_FAST\s+(\w+)", output)
                if match and match.group(1) != "x":
                    stored_var_name = match.group(1)

            # Step once more so STORE_FAST executes
            yield "s"

            # Check that the new variable appears in locals
            locals_output = yield "locals"
            self.assertIn(f"{stored_var_name} = ", locals_output)

        InteractiveDebugSession(fn, (input_tensor,), test_logic)

    def test_list_command(self):
        """Test that the list command shows context around current instruction."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # Step a few times to move past the beginning
            for _ in range(5):
                yield "s"

            # Now list to see context around current instruction
            list_output = yield "l"

            self.assertIn(">>>", list_output)  # Current instruction marker
            self.assertIn("# [offset]", list_output)  # Header
            self.assertRegex(
                list_output, r"\d+\s*\[\s*\d+\]:\s*\w+"
            )  # Instruction format

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_disassemble_command(self):
        """Test that the ll (disassemble) command shows all bytecode."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            ll_output = yield "ll"

            # Check for disassembly header with instruction count
            self.assertRegex(ll_output, r"Disassembly \(\d+ instructions\):")
            self.assertIn("# [offset]", ll_output)

            # Check for stable bytecode instructions
            if sys.version_info >= (3, 11):
                self.assertIn("RESUME", ll_output)
            self.assertIn("LOAD_FAST x", ll_output)
            self.assertIn("RETURN_VALUE", ll_output)

            # Verify instruction count matches
            instruction_matches = re.findall(
                r"^\s*\d+\s*\[\s*\d+\]:", ll_output, re.MULTILINE
            )
            expected = len(sess.ctx.get_instructions())
            self.assertEqual(len(instruction_matches), expected)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_breakpoint_commands(self):
        """Test breakpoint set, list, hit, and clear commands."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # Set breakpoint at instruction 5
            output = yield "b 5"
            self.assertIn("Breakpoint set at instruction 5", output)

            # List breakpoints
            output = yield "b"
            self.assertIn("Breakpoints:", output)
            self.assertIn("5", output)

            # Continue to breakpoint - should stop at instruction 5
            output = yield "c"
            self.assertIn("Breakpoint hit at instruction 5", output)

            # Clear breakpoint and set a new one further ahead
            output = yield "cl 5"
            self.assertIn("Breakpoint cleared at instruction 5", output)

            output = yield "b 10"
            self.assertIn("Breakpoint set at instruction 10", output)

            # Continue - should skip instruction 5 (cleared) and hit instruction 10
            output = yield "cl 10"
            self.assertIn("Breakpoint cleared at instruction 10", output)

            output = yield "c"
            self.assertNotIn("Breakpoint hit at instruction 10", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_breakpoint_validation(self):
        """Test that invalid instruction numbers are rejected."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            output = yield "b 9999"
            self.assertIn("Invalid instruction number: 9999", output)

            output = yield "b -1"
            self.assertIn("Invalid instruction number: -1", output)

            output = yield "b abc"
            self.assertIn("Invalid instruction number: abc", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_expression_evaluation(self):
        """Test that arbitrary expressions can be evaluated."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # Step a few times
            for _ in range(5):
                yield "s"

            # Evaluate expression
            output = yield "2 + 2"
            self.assertIn("4", output)

        InteractiveDebugSession(fn, (torch.tensor([1.0, 2.0, 3.0]),), test_logic)

    def test_stack_variable_access(self):
        """Test that __stack__ variable is accessible in expressions."""

        @torch.compile(backend="eager")
        def fn(x):
            return torch.sin(x)

        def test_logic(sess, initial):
            output = initial
            # Step until we hit a CALL instruction (or PRECALL on Python 3.11)
            if sys.version_info >= (3, 12):
                call_pattern = r">>>.*\[\d+\]:\s*CALL\b"
            else:
                call_pattern = r">>>.*\[\d+\]:\s*PRECALL\b"
            while not re.search(call_pattern, output):
                output = yield "s"

            # Access __stack__ - should have function, NULL, and tensor
            output = yield "__stack__"

            # Normalize addresses and tensor values
            normalized = re.sub(r"0x[0-9a-f]+", "0xADDR", output)
            normalized = re.sub(r"tensor\([^)]+\)", "tensor(...)", normalized)
            # Extract just the list part
            match = re.search(r"\[.*\]", normalized, re.DOTALL)
            self.assertIsNotNone(match)
            stack_list = match.group(0)

            if sys.version_info >= (3, 13):
                self.assertExpectedInline(
                    stack_list,
                    """[<function forward at 0xADDR>, <NULL>, tensor(...)]""",
                )
            elif sys.version_info >= (3, 11):
                self.assertExpectedInline(
                    stack_list,
                    """[<NULL>, <function forward at 0xADDR>, tensor(...)]""",
                )
            else:
                # Python 3.10
                self.assertExpectedInline(
                    stack_list,
                    """[<function forward at 0xADDR>, tensor(...)]""",
                )

        InteractiveDebugSession(fn, (torch.ones(3),), test_logic)

    def test_print_command(self):
        """Test the p (print) command."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # Step a few times to ensure x is in scope
            for _ in range(5):
                yield "s"

            output = yield "p x"
            self.assertIn("x = tensor([1., 2.])", output)

        InteractiveDebugSession(fn, (torch.tensor([1.0, 2.0]),), test_logic)

    def test_help_command(self):
        """Test that the help command works."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            output = yield "h"
            self.assertIn("Commands:", output)
            self.assertIn("step", output)
            self.assertIn("cont", output)
            self.assertIn("verbose", output)
            self.assertIn("__stack__", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_multiple_calls(self):
        """Test debugger works across multiple function calls."""

        @torch.compile(backend="eager")
        def fn1(x):
            return x + 1

        @torch.compile(backend="eager")
        def fn2(x):
            return x * 2

        def test_logic(sess, initial):
            # First function - continue
            output = yield "c"
            self.assertIn("returned:", output)

            # Second function - continue
            output = yield "c"
            self.assertIn("returned:", output)

        inp1 = torch.randn(3)
        inp2 = torch.randn(3)

        def multi_call_fn(x1, x2):
            r1 = fn1(x1)
            r2 = fn2(x2)
            return r1, r2

        sess = InteractiveDebugSession(multi_call_fn, (inp1, inp2), test_logic)
        self.assertEqual(sess.result, (inp1 + 1, inp2 * 2))

    def test_graph_break(self):
        """Test stepping through compiled resume function"""

        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            output = yield "c"
            self.assertIn(
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX,
                output,
            )
            output = yield "c"
            self.assertRegex(output, rf"{TORCH_DYNAMO_RESUME_IN_PREFIX}\w+ returned:")
            self.assertIn("fn returned: ", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_stack_at_return_value(self):
        """Test that stack has exactly 1 element at RETURN_VALUE instruction.

        This tests that stack depth calculation handles opcodes without arguments
        correctly (passing None to dis.stack_effect instead of 0).
        """

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            return y

        def test_logic(sess, initial):
            output = initial
            # Step until we hit RETURN_VALUE
            while not re.search(r">>>.*RETURN_VALUE", output):
                output = yield "s"

            # Check stack at RETURN_VALUE
            stack_output = yield "stack"
            self.assertIn("Stack (TOS at end):", stack_output)

            # Count stack entries - should be exactly 1
            stack_entries = re.findall(r"^\s*\[\d+\]", stack_output, re.MULTILINE)
            self.assertEqual(len(stack_entries), 1)

            # Should contain a tensor (the return value)
            self.assertIn("tensor(", stack_output)

        InteractiveDebugSession(fn, (torch.ones(3),), test_logic)

    def test_breakpoint_marker(self):
        """Test that BREAKPOINT_MARKER detection logic works.

        This verifies that the debugger correctly identifies LOAD_CONST
        instructions with BREAKPOINT_MARKER as breakpoints, and that the
        breakpoint is hit at the expected location (before side effect replay).
        """
        from torch._dynamo.bytecode_transformation import create_breakpoint
        from torch._dynamo.side_effects import SideEffects

        original_codegen_update_mutated = SideEffects.codegen_update_mutated

        def patched_codegen_update_mutated(self, cg, log_side_effects=False):
            cg.extend_output(create_breakpoint())
            return original_codegen_update_mutated(self, cg, log_side_effects)

        global mylist

        mylist = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x):
            mylist.append(4)
            return x + 1

        def test_logic(sess, initial):
            # Continue - should hit the programmatic breakpoint
            output = yield "c"
            self.assertIn("Breakpoint hit (programmatic)", output)

            # Verify mylist still unchanged (side effect not yet applied)
            output = yield "mylist"
            self.assertIn("[1, 2, 3]", output)

            # Step until we find LOAD_GLOBAL mylist
            found_load_mylist = False
            for _ in range(10):
                output = yield "s"
                if "LOAD_GLOBAL mylist" in output:
                    found_load_mylist = True
                    break
            self.assertTrue(
                found_load_mylist, "Expected to find LOAD_GLOBAL mylist instruction"
            )

            # Step and check mylist until we see the mutation
            for _ in range(30):
                output = yield "s"
                if "returned:" in output:
                    self.fail("Function returned before seeing mutation")
                output = yield "mylist"
                if "[1, 2, 3, 4]" in output:
                    break
            else:
                self.fail("Never saw mylist mutated to [1, 2, 3, 4]")

        with patch.object(
            SideEffects,
            "codegen_update_mutated",
            patched_codegen_update_mutated,
        ):
            InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

        # Verify the mutation happened after the session completed
        self.assertEqual(mylist, [1, 2, 3, 4])

    def test_user_defined_variables_persist(self):
        """Test that user-defined variables persist across commands."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # Define a variable
            yield "tmp = 42"
            # Access it in a later command - should evaluate to 84
            output = yield "tmp * 2"
            self.assertIn("84", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_user_defined_variables_shadow_locals(self):
        """Test that user-defined variables can shadow local variables."""

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            return y

        input_tensor = torch.tensor([1.0, 2.0, 3.0])

        def test_logic(sess, initial):
            # Step a few times to get into the execution
            for _ in range(5):
                yield "s"

            # Check original value of x
            output = yield "x"
            self.assertIn("tensor([1., 2., 3.])", output)

            # Shadow x with a new value
            yield "x = 999"

            # Now x should show the shadowed value
            output = yield "x"
            self.assertIn("999", output)

            # Continue to let the function complete
            yield "c"

        sess = InteractiveDebugSession(fn, (input_tensor,), test_logic)

        # Verify the original computation was unaffected by shadowing
        self.assertEqual(sess.result, input_tensor + 1)

    def test_inplace_mutation_affects_execution(self):
        """Test that in-place mutations in debugger affect actual execution."""

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            return y

        input_tensor = torch.tensor([1.0, 2.0, 3.0])

        def test_logic(sess, initial):
            # Step a few times to get into the execution
            for _ in range(5):
                yield "s"

            # In-place mutation - this DOES affect execution
            yield "x += 1"

            # Continue to let the function complete
            yield "c"

        sess = InteractiveDebugSession(fn, (input_tensor,), test_logic)

        # In-place mutation affected execution: x was [1,2,3], became [2,3,4],
        # then y = x + 1 = [3,4,5]
        self.assertEqual(sess.result, torch.tensor([3.0, 4.0, 5.0]))

    def test_exception_in_bytecode(self):
        """Test debugger stops when bytecode raises an exception.

        Injects a division by zero into the bytecode and verifies:
        1. The debugger stops at the exception
        2. Shows the instruction that caused the exception
        3. Allows inspection before propagating
        """
        from torch._dynamo.bytecode_transformation import create_instruction
        from torch._dynamo.side_effects import SideEffects

        original_codegen_update_mutated = SideEffects.codegen_update_mutated

        def patched_codegen_update_mutated(self, cg, log_side_effects=False):
            # Inject 31415/0 to cause ZeroDivisionError
            cg.extend_output(
                [
                    create_instruction("LOAD_CONST", argval=31415),
                    create_instruction("LOAD_CONST", argval=0),
                    create_instruction("BINARY_OP", arg=11),  # 11 = TRUEDIV
                    create_instruction("POP_TOP"),
                ]
            )
            return original_codegen_update_mutated(self, cg, log_side_effects)

        global mylist
        mylist = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x):
            mylist.append(4)
            return x + 1

        def test_logic(sess, initial):
            # Continue - this will run until the exception
            output = yield "c"

            # Debugger should have stopped at the exception
            self.assertIn("Exception raised at instruction", output)
            self.assertIn("BINARY_OP", output)
            self.assertIn("ZeroDivisionError", output)
            self.assertIn("division by zero", output)

            # Verify we can inspect locals at the exception point
            locals_output = yield "locals"
            self.assertIn("x =", locals_output)  # Input tensor should be in locals

            # Verify we can inspect the stack at the exception point
            # Note: On Python 3.11 with settrace, when an exception is raised,
            # CPython has already popped the operands from the stack before the
            # division fails, so we can't see them.
            stack_output = yield "stack"
            self.assertIn("Stack", stack_output)
            if sys.version_info >= (3, 12):
                self.assertIn("31415", stack_output)
                self.assertRegex(stack_output, r"\[\d+\].*\b0\b")  # constant 0

            # Verify we can evaluate expressions
            eval_output = yield "1 + 1"
            self.assertIn("2", eval_output)

            # next step should terminate and exception should be propagated
            yield "n"

        with patch.object(
            SideEffects,
            "codegen_update_mutated",
            patched_codegen_update_mutated,
        ):
            with self.assertRaises(ZeroDivisionError):
                InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_empty_input_repeats_last_command(self):
        """Test that empty input repeats the last command."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # First command: step
            output = yield ""
            self.assertIn("Instruction", output)

            # Empty input should repeat 's' (step)
            output = yield ""
            self.assertIn("Instruction", output)

            # Now use verbose
            output = yield "v"
            self.assertIn("Verbose mode enabled", output)

            # Empty input should repeat 'v' (toggle verbose off)
            output = yield ""
            self.assertIn("Verbose mode disabled", output)

            # Continue to end
            yield "c"

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)


if __name__ == "__main__":
    run_tests()
