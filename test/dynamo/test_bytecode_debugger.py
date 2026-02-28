# Owner(s): ["module: dynamo"]

"""
Tests for torch._dynamo.bytecode_debugger
"""

import re
import sys
from contextlib import nullcontext, redirect_stdout
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

    Set use_debug=False to test auto-activation (no debug() wrapper).
    """

    def __init__(self, fn, args, test_logic, use_debug=True):
        self.test_logic = test_logic
        self.output_buffer = StringIO()
        self._test_error: BaseException | None = None

        with patch("builtins.input", self._fake_input):
            with redirect_stdout(self.output_buffer):
                with debug() if use_debug else nullcontext() as self.ctx:
                    self.result = fn(*args)

        # Re-raise any assertion error captured from the test generator
        if self._test_error is not None:
            raise self._test_error

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
        except Exception as e:
            # Capture test failures (e.g. AssertionError) and exit the
            # debugger cleanly.  Without this, the exception propagates
            # through sys.monitoring callbacks, gets converted to
            # KeyboardInterrupt by the "q" handler, and is silently
            # swallowed by the debug() context manager.
            self._test_error = e
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
            # Step until we hit the CALL that invokes the compiled graph
            # function (forward), skipping instrumentation calls
            # (record_pregraph_bytecode_enter/exit).
            if sys.version_info >= (3, 12):
                call_pattern = r">>>.*\[\s*\d+\]:\s*CALL\b"
            elif sys.version_info >= (3, 11):
                call_pattern = r">>>.*\[\s*\d+\]:\s*PRECALL\b"
            else:
                call_pattern = r">>>.*\[\s*\d+\]:\s*CALL_FUNCTION\b"
            while True:
                while not re.search(call_pattern, output):
                    output = yield "s"
                stack_output = yield "stack"
                if "forward" in stack_output and "record_pregraph" not in stack_output:
                    break
                output = yield "s"

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
        """Test that the list command behaves like pdb."""

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            z = y * 2
            w = z - 1
            return w + x

        def test_logic(sess, initial):
            # First list shows around current instruction
            list_output1 = yield "l"
            self.assertIn(">>>", list_output1)  # Current instruction marker
            self.assertIn("# [offset]", list_output1)  # Header

            # Second list continues (may or may not have >>> depending on position)
            list_output2 = yield "l"
            # Should show different instructions (or end of bytecode message)
            if "(End of bytecode)" not in list_output2:
                self.assertIn("# [offset]", list_output2)

            # 'l .' resets to current instruction
            list_dot_output = yield "l ."
            self.assertIn(">>>", list_dot_output)  # Current instruction marker again

            # 'l N' lists 11 instructions starting at N
            list_from_0 = yield "l 0"
            # Should show instruction 0
            self.assertRegex(list_from_0, r"\s+0\s+\[")

            # 'l first, last' lists range (inclusive)
            list_range = yield "l 0, 2"
            # Should show exactly 3 instructions (0, 1, 2)
            instruction_lines = [
                line
                for line in list_range.split("\n")
                if re.match(r"\s*(?:>>>)?\s*\*?\s*\d+\s+\[", line)
            ]
            self.assertEqual(len(instruction_lines), 3)

            # 'l first, count' when count < first, count is number of lines
            list_count = yield "l 5, 2"
            # Should show 2 instructions starting at 5
            instruction_lines = [
                line
                for line in list_count.split("\n")
                if re.match(r"\s*(?:>>>)?\s*\*?\s*\d+\s+\[", line)
            ]
            self.assertEqual(len(instruction_lines), 2)

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
            # Step until we hit the CALL that invokes the compiled graph
            # function. See test_stack_command for details.
            if sys.version_info >= (3, 12):
                call_pattern = r">>>.*\[\s*\d+\]:\s*CALL\b"
            elif sys.version_info >= (3, 11):
                call_pattern = r">>>.*\[\s*\d+\]:\s*PRECALL\b"
            else:
                call_pattern = r">>>.*\[\s*\d+\]:\s*CALL_FUNCTION\b"
            while True:
                while not re.search(call_pattern, output):
                    output = yield "s"
                stack_output = yield "__stack__"
                if "forward" in stack_output and "record_pregraph" not in stack_output:
                    break
                output = yield "s"

            # Normalize addresses and tensor values
            normalized = re.sub(r"0x[0-9a-f]+", "0xADDR", stack_output)
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
            self.assertIn("s [n]", output)
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
            self.assertIn("Entering Dynamo-generated code: fn1", initial)
            # Enable verbose so we can see both functions' execution
            output = yield "v"
            self.assertIn("Verbose mode enabled", output)
            output = yield "c"
            # fn1 should have continued and returned
            self.assertIn("fn1 returned:", output)
            # fn2 should have entered (verbose), run, and returned
            self.assertIn("Entering Dynamo-generated code: fn2", output)
            self.assertIn("fn2 returned:", output)
            # Verify we saw instructions from both functions
            self.assertIn("LOAD_FAST", output)
            self.assertIn("RETURN_VALUE", output)

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
            # Enable verbose and continue — should trace through both
            # the first compiled code and the resume function
            output = yield "v"
            self.assertIn("Verbose mode enabled", output)
            output = yield "c"
            # Should see resume function entered and both returns
            self.assertIn(
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX,
                output,
            )
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
        """Test that create_breakpoint() auto-activates the debugger.

        Manually injects a BREAKPOINT_MARKER via create_breakpoint() (without
        bdb_breakpoint()) and verifies the debugger activates automatically,
        hitting the breakpoint before side effect replay.
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
            # Auto-breakpoint skips to the breakpoint without pausing at start
            self.assertIn("Breakpoint hit (programmatic)", initial)

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

            # Continue to end
            yield "c"

        with patch.object(
            SideEffects,
            "codegen_update_mutated",
            patched_codegen_update_mutated,
        ):
            inp = torch.randn(3)
            sess = InteractiveDebugSession(fn, (inp,), test_logic, use_debug=False)

        self.assertEqual(sess.result, inp + 1)
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
            if sys.version_info >= (3, 11):
                div_inst = create_instruction("BINARY_OP", arg=11)  # 11 = TRUEDIV
            else:
                div_inst = create_instruction("BINARY_TRUE_DIVIDE")
            cg.extend_output(
                [
                    create_instruction("LOAD_CONST", argval=31415),
                    create_instruction("LOAD_CONST", argval=0),
                    div_inst,
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
            if sys.version_info >= (3, 11):
                self.assertIn("BINARY_OP", output)
            else:
                self.assertIn("BINARY_TRUE_DIVIDE", output)
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

    def test_step_with_count_argument(self):
        """Test that step command accepts count argument"""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # Get starting instruction number
            match = re.search(r"Instruction (\d+)", initial)
            self.assertIsNotNone(match)
            start_inst = int(match.group(1))

            # Step 3 times with "s 3"
            output = yield "s 3"
            match = re.search(r"Instruction (\d+)", output)
            self.assertIsNotNone(match)
            end_inst = int(match.group(1))

            # Should have advanced by 3 instructions
            self.assertEqual(end_inst, start_inst + 3)

            # Also test with "step 2"
            output = yield "step 2"
            match = re.search(r"Instruction (\d+)", output)
            self.assertIsNotNone(match)
            new_inst = int(match.group(1))

            # Should have advanced by 2 more
            self.assertEqual(new_inst, end_inst + 2)

            # Continue to end
            yield "c"

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_next_with_count_argument(self):
        """Test that 'n [count]' steps over calls without entering them."""
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Use 'n' to step through fn's code, collecting all output
            output = initial
            all_output = output
            while "fn returned:" not in output:
                output = yield "n 3"
                all_output += output

            # 'n' should never have entered the resume function interactively
            self.assertNotIn(
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX,
                all_output,
            )

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_return_command(self):
        """Test that 'r' stops at the return instruction."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            # Step a few times, then use 'r' to run to return instruction
            for _ in range(3):
                yield "s"
            output = yield "r"
            # Should stop at the return instruction, not after it
            self.assertIn("About to return from", output)
            self.assertRegex(output, r"RETURN_VALUE|RETURN_CONST")
            # Step past to complete the return
            output = yield "s"
            self.assertIn("returned:", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_return_stops_at_return_instruction(self):
        """Test that 'r' runs through inner frames and stops at return."""
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Use 'r' — should skip the resume function and stop at fn's return
            output = yield "r"
            self.assertRegex(output, rf"{TORCH_DYNAMO_RESUME_IN_PREFIX}\w+ returned:")
            self.assertIn("About to return from fn", output)
            self.assertRegex(output, r"RETURN_VALUE|RETURN_CONST")
            # Step past to complete
            output = yield "s"
            self.assertIn("fn returned:", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_up_down_single_frame(self):
        """Test u/d with only one tracked frame."""

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        def test_logic(sess, initial):
            output = yield "u"
            self.assertIn("Oldest tracked frame", output)
            output = yield "d"
            self.assertIn("Newest tracked frame", output)

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_up_down_commands(self):
        """Test u/d with multiple tracked frames (graph break creates resume fn)."""
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Step into the resume function
            output = initial
            while (
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX
                not in output
            ):
                output = yield "s"

            # Lower frame (resume fn) just entered — stack should be empty
            lower_stack = yield "stack"
            self.assertIn("(empty)", lower_stack)

            # Navigate up to the parent frame
            output = yield "u"
            self.assertIn("> fn", output)

            # Upper frame stack may or may not be empty
            upper_stack = yield "stack"
            self.assertIn("Stack (TOS at end):", upper_stack)

            # Check locals in upper frame — should have 'x'
            locals_output = yield "locals"
            self.assertIn("x =", locals_output)

            # Check locals in lower frame — should NOT have 'x', but should have 'y'
            # (resume function receives Dynamo-renamed parameters)
            yield "d"
            lower_locals_output = yield "locals"
            self.assertNotIn("x =", lower_locals_output)
            self.assertIn("y =", lower_locals_output)

            # Go back up and check globals
            yield "u"
            globals_output = yield "globals"
            self.assertIn("Globals:", globals_output)

            # 'p' from upper frame should see fn's locals
            p_output = yield "p x"
            self.assertIn("x =", p_output)

            # Set a breakpoint on fn's RETURN_VALUE from the upper frame.
            # When we continue, the resume function will complete first,
            # then fn will hit this breakpoint.
            fn_code = sess.ctx.get_tracked_codes()[0]
            fn_instructions = sess.ctx.get_instructions(fn_code)
            return_idx = None
            for i, inst in enumerate(fn_instructions):
                if inst.opname in ("RETURN_VALUE", "RETURN_CONST"):
                    return_idx = i
            self.assertIsNotNone(return_idx)
            output = yield f"b {return_idx}"
            self.assertIn(f"Breakpoint set at instruction {return_idx}", output)

            # Continue — should run through the resume function and hit
            # the breakpoint in fn
            output = yield "c"
            self.assertIn(f"Breakpoint hit at instruction {return_idx}", output)
            self.assertRegex(output, r"RETURN_VALUE|RETURN_CONST")
            # The resume function should have returned by now
            self.assertRegex(output, rf"{TORCH_DYNAMO_RESUME_IN_PREFIX}\w+ returned:")

            # Continue to end
            yield "c"

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_bt_command(self):
        """Test that 'bt' shows the tracked frame backtrace."""
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Step into the resume function
            output = initial
            while (
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX
                not in output
            ):
                output = yield "s"

            # bt should show both frames, with the resume function selected
            bt_output = yield "bt"
            lines = [l.strip() for l in bt_output.split("\n") if "[" in l and "]" in l]
            self.assertEqual(len(lines), 2)
            # First frame is fn (no > marker since we're viewing the callee)
            self.assertIn("fn", lines[0])
            self.assertNotIn(">", lines[0])
            # Second frame is resume function (> marker since it's current)
            self.assertIn(">", lines[1])
            self.assertIn(TORCH_DYNAMO_RESUME_IN_PREFIX, lines[1])

            # Move up and bt again — marker should move to fn
            yield "u"
            bt_output = yield "bt"
            lines = [l.strip() for l in bt_output.split("\n") if "[" in l and "]" in l]
            self.assertIn(">", lines[0])
            self.assertNotIn(">", lines[1])

            # Continue to end
            yield "c"

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_step_from_upper_frame(self):
        """Test that 's' from an upper frame steps the inner (execution) frame."""
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Step into the resume function
            output = initial
            while (
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX
                not in output
            ):
                output = yield "s"

            # Move up to the parent frame (fn)
            output = yield "u"
            self.assertIn("> fn", output)

            # 's' from the upper frame should step the inner (resume) frame,
            # not the viewed frame
            output = yield "s"
            # Should show the resume function's next instruction, not fn's
            self.assertIn("Instruction", output)

            # Keep stepping — should complete the resume function and
            # return to fn
            all_output = output
            while "fn returned:" not in output:
                output = yield "s"
                all_output += output
            self.assertRegex(
                all_output, rf"{TORCH_DYNAMO_RESUME_IN_PREFIX}\w+ returned:"
            )

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_next_from_upper_frame(self):
        """Test that 'n' from an upper frame steps over the callee."""
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Step into the resume function
            output = initial
            while (
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX
                not in output
            ):
                output = yield "s"

            # Move up to the parent frame (fn)
            output = yield "u"
            self.assertIn("> fn", output)

            # 'n' from the parent frame should step over the resume function
            # and stop at the next instruction in fn's frame
            output = yield "n"
            # The resume function should have returned
            self.assertRegex(output, rf"{TORCH_DYNAMO_RESUME_IN_PREFIX}\w+ returned:")
            # We should now be stopped at an instruction in the fn frame
            self.assertIn("Instruction", output)

            # Continue to end
            yield "c"

        InteractiveDebugSession(fn, (torch.randn(3),), test_logic)

    def test_breakpoint_function(self):
        """Test that breakpoint() inserts BREAKPOINT_MARKER without a graph break."""
        from torch._dynamo.bytecode_debugger import breakpoint as bdb_breakpoint
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            bdb_breakpoint()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Breakpoint is at the first instruction, so it fires immediately
            self.assertIn("Breakpoint hit (programmatic)", initial)
            # No graph break: should NOT see a resume function
            self.assertNotIn(TORCH_DYNAMO_RESUME_IN_PREFIX, initial)
            # Continue to end
            output = yield "c"
            self.assertIn("returned:", output)

        inp = torch.randn(3)
        sess = InteractiveDebugSession(fn, (inp,), test_logic)
        self.assertEqual(sess.result, inp + 3)

    def test_programmatic_breakpoint_listed_and_clearable(self):
        """Test that programmatic breakpoints appear in 'b' and can be cleared with 'cl'."""
        from torch._dynamo.bytecode_debugger import breakpoint as bdb_breakpoint

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            bdb_breakpoint()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Breakpoint is at the first instruction, so it fires immediately
            self.assertIn("Breakpoint hit (programmatic)", initial)

            # List breakpoints — should show the programmatic breakpoint
            output = yield "b"
            self.assertIn("Breakpoints:", output)
            self.assertIn("LOAD_CONST <BREAKPOINT>", output)

            # Extract the instruction index and clear it
            bp_match = re.search(r"(\d+)\s+\[.*LOAD_CONST <BREAKPOINT>", output)
            self.assertIsNotNone(bp_match)
            bp_index = bp_match.group(1)
            output = yield f"cl {bp_index}"
            self.assertIn(f"Breakpoint cleared at instruction {bp_index}", output)

            # Verify it's gone
            output = yield "b"
            self.assertIn("(none)", output)

            # Continue to end
            output = yield "c"
            self.assertIn("returned:", output)

        inp = torch.randn(3)
        sess = InteractiveDebugSession(fn, (inp,), test_logic)
        self.assertEqual(sess.result, inp + 3)

    def test_breakpoint_after_graph_break(self):
        """Test that breakpoint() after a graph break pauses in the resume function."""
        from torch._dynamo.bytecode_debugger import breakpoint as bdb_breakpoint
        from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            bdb_breakpoint()
            return x + 2

        def test_logic(sess, initial):
            self.assertIn("Entering Dynamo-generated code: fn", initial)
            # Enable verbose so we can see the resume function entering
            yield "v"
            # Continue — should enter the resume function and hit the breakpoint
            output = yield "c"
            self.assertIn(
                "Entering Dynamo-generated code: " + TORCH_DYNAMO_RESUME_IN_PREFIX,
                output,
            )
            self.assertIn("Breakpoint hit (programmatic)", output)
            # Continue to end
            output = yield "c"
            self.assertIn("returned:", output)

        inp = torch.randn(3)
        sess = InteractiveDebugSession(fn, (inp,), test_logic)
        self.assertEqual(sess.result, inp + 3)

    def test_breakpoint_without_debug_context(self):
        """Test that breakpoint() auto-activates the debugger without with debug()."""
        from torch._dynamo.bytecode_debugger import breakpoint as bdb_breakpoint

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            bdb_breakpoint()
            return x + 2

        def test_logic(sess, initial):
            # Auto-breakpoint runs to the breakpoint without pausing at start
            self.assertIn("Breakpoint hit (programmatic)", initial)
            yield "c"

        inp = torch.randn(3)
        sess = InteractiveDebugSession(fn, (inp,), test_logic, use_debug=False)
        self.assertEqual(sess.result, inp + 3)

    def test_breakpoint_without_debug_context_after_graph_break(self):
        """Test auto-activate only fires for the resume function, not the first graph."""
        from torch._dynamo.bytecode_debugger import breakpoint as bdb_breakpoint

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            bdb_breakpoint()
            return x + 2

        def test_logic(sess, initial):
            # Auto-breakpoint skips to the breakpoint without pausing at start
            self.assertIn("Breakpoint hit (programmatic)", initial)
            # Should not have entered fn's first graph interactively
            self.assertNotIn("Entering Dynamo-generated code: fn", initial)
            yield "c"

        inp = torch.randn(3)
        sess = InteractiveDebugSession(fn, (inp,), test_logic, use_debug=False)
        self.assertEqual(sess.result, inp + 3)

    def test_nested_debug_contexts(self):
        """Test that nested debug() contexts restore the outer correctly.

        On Python 3.12+, compiled functions called during a sys.monitoring
        callback skip Dynamo (tstate->tracing > 0), so nested debug contexts
        must be tested outside monitoring callbacks.
        """

        @torch.compile(backend="eager")
        def fn1(x):
            return x + 1

        @torch.compile(backend="eager")
        def fn2(x):
            return x * 2

        @torch.compile(backend="eager")
        def fn3(x):
            return x + 3

        outer_buf = StringIO()
        with patch("builtins.input", return_value="c"):
            with redirect_stdout(outer_buf):
                with debug():
                    fn1(torch.randn(3))

                    # After fn1 returns, we're outside any monitoring callback.
                    # Create a nested debug context for fn2.
                    inner_buf = StringIO()
                    with redirect_stdout(inner_buf):
                        with debug():
                            fn2(torch.randn(3))
                    inner_output = inner_buf.getvalue()

                    # After inner debug exits, outer should still work.
                    fn3(torch.randn(3))

        outer_output = outer_buf.getvalue()

        self.assertIn("Entering Dynamo-generated code: fn1", outer_output)
        self.assertIn("Entering Dynamo-generated code: fn2", inner_output)
        self.assertIn("fn1 returned:", outer_output)
        self.assertIn("fn2 returned:", inner_output)
        # fn3 should still be tracked by the restored outer context.
        # The "c" command disables stop-at-new-code, so we won't see the
        # "Entering" message, but the return callback should still fire.
        self.assertIn("fn3 returned:", outer_output)

    def test_nested_auto_breakpoints(self):
        """Test multiple functions with breakpoint() called sequentially."""
        from torch._dynamo.bytecode_debugger import breakpoint as bdb_breakpoint

        @torch.compile(backend="eager")
        def fn1(x):
            bdb_breakpoint()
            return x + 1

        @torch.compile(backend="eager")
        def fn2(x):
            bdb_breakpoint()
            return x * 2

        output_buf = StringIO()
        with patch("builtins.input", return_value="c"):
            with redirect_stdout(output_buf):
                inp = torch.randn(3)
                r1 = fn1(inp)
                r2 = fn2(inp)

        output = output_buf.getvalue()
        # Both functions should have hit their breakpoints
        self.assertEqual(output.count("Breakpoint hit (programmatic)"), 2)
        self.assertEqual(r1, inp + 1)
        self.assertEqual(r2, inp * 2)


if __name__ == "__main__":
    run_tests()
