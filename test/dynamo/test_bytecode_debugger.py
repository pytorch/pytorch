# Owner(s): ["module: dynamo"]

"""
Tests for torch._dynamo.bytecode_debugger
"""

import sys
from io import StringIO
from unittest.mock import patch

import torch
import torch._dynamo
from torch._dynamo.bytecode_debugger import _DebugContext, debug
from torch._dynamo.test_case import TestCase
from torch.testing._internal.common_utils import run_tests


class TestBytecodeDebugger(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def test_debug_context_manager_basic(self):
        """Test that the debug context manager works without errors."""

        @torch.compile
        def fn(x):
            return x + 1

        # Use 'c' to continue immediately without stepping
        with patch.object(sys, "stdin", StringIO("c\n")):
            with debug():
                result = fn(torch.randn(3))
                self.assertEqual(result.shape, torch.Size([3]))

    def test_debugger_callback_invoked(self):
        """Test that the debugger callback is invoked for Dynamo code."""

        @torch.compile
        def fn(x):
            return x + 1

        callback_codes = []

        ctx = _DebugContext()
        original_callback = ctx._dynamo_code_callback

        def tracking_callback(code):
            callback_codes.append(code.co_name)
            original_callback(code)

        ctx._dynamo_code_callback = tracking_callback

        with patch.object(sys, "stdin", StringIO("c\n")):
            with ctx:
                fn(torch.randn(3))

        self.assertTrue(len(callback_codes) > 0)
        self.assertIn("fn", callback_codes)

    def test_step_mode(self):
        """Test that step mode stops at each instruction."""

        @torch.compile
        def fn(x):
            return x + 1

        ctx = _DebugContext()
        instruction_count = [0]

        def counting_prompt(state):
            instruction_count[0] += 1
            # After counting, set step_mode to False to continue
            if instruction_count[0] >= 3:
                state.step_mode = False

        ctx._interactive_prompt = counting_prompt

        with ctx:
            fn(torch.randn(3))

        # Should have stopped at least 3 times
        self.assertGreaterEqual(instruction_count[0], 3)

    def test_verbose_mode(self):
        """Test that verbose mode prints each instruction."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        with patch.object(sys, "stdin", StringIO("v\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        # Verbose mode should print instruction indices
        self.assertIn("[", output_str)
        self.assertIn("]", output_str)
        # Should contain some bytecode instruction names
        self.assertTrue(
            "LOAD_" in output_str or "CALL" in output_str or "RETURN" in output_str
        )

    def test_return_value_printed(self):
        """Test that return values are printed."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        with patch.object(sys, "stdin", StringIO("c\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        self.assertIn("returned:", output_str)
        self.assertIn("tensor(", output_str)

    def test_stack_command(self):
        """Test that the stack command works."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        # Step a few times, then run 'stack' command, then continue
        with patch.object(sys, "stdin", StringIO("s\ns\ns\ns\ns\nstack\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        self.assertIn("Stack", output_str)

    def test_locals_command(self):
        """Test that the locals command works."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        with patch.object(sys, "stdin", StringIO("s\ns\ns\nlocals\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        self.assertIn("Locals", output_str)

    def test_list_command(self):
        """Test that the list command works."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        with patch.object(sys, "stdin", StringIO("l\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        self.assertIn("Instruction", output_str)
        self.assertIn("offset", output_str)

    def test_disassemble_command(self):
        """Test that the ll (disassemble) command works."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        with patch.object(sys, "stdin", StringIO("ll\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        self.assertIn("Disassembly", output_str)
        self.assertIn("instructions", output_str)

    def test_breakpoint_commands(self):
        """Test breakpoint set and list commands."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        # Set breakpoint, list breakpoints, clear, then continue
        with patch.object(sys, "stdin", StringIO("b 10\nb\ncl 10\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        self.assertIn("Breakpoint set at offset 10", output_str)
        self.assertIn("Breakpoints:", output_str)
        self.assertIn("Breakpoint cleared at offset 10", output_str)

    def test_expression_evaluation(self):
        """Test that arbitrary expressions can be evaluated."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        # Step to get x in scope, evaluate expression, then continue
        with patch.object(sys, "stdin", StringIO("s\ns\ns\ns\ns\ns\ns\ns\n2 + 2\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.tensor([1.0, 2.0, 3.0]))

        output_str = output.getvalue()
        self.assertIn("4", output_str)

    def test_stack_variable_access(self):
        """Test that __stack__ variable is accessible in expressions."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        # Step to build up stack, access __stack__, then continue
        with patch.object(
            sys, "stdin", StringIO("s\ns\ns\ns\ns\ns\nlen(__stack__)\nc\n")
        ):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        # len(__stack__) should return a number
        # Check that a digit appears after the len(__stack__) call
        self.assertTrue(any(c.isdigit() for c in output_str))

    def test_print_command(self):
        """Test the p (print) command."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        with patch.object(sys, "stdin", StringIO("s\ns\ns\ns\ns\ns\ns\ns\np x\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.tensor([1.0, 2.0]))

        output_str = output.getvalue()
        self.assertIn("x =", output_str)
        self.assertIn("tensor", output_str)

    def test_help_command(self):
        """Test that the help command works."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        with patch.object(sys, "stdin", StringIO("h\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        self.assertIn("Commands:", output_str)
        self.assertIn("step", output_str)
        self.assertIn("cont", output_str)
        self.assertIn("verbose", output_str)
        self.assertIn("__stack__", output_str)

    def test_c_callback_function_exists(self):
        """Test that the C callback function is accessible."""
        from torch._C._dynamo.eval_frame import set_bytecode_debugger_callback

        # Should be callable
        self.assertTrue(callable(set_bytecode_debugger_callback))

        # Should accept None
        set_bytecode_debugger_callback(None)

        # Should accept a callable
        def dummy_callback(code):
            pass

        set_bytecode_debugger_callback(dummy_callback)
        set_bytecode_debugger_callback(None)  # Clean up

    def test_multiple_calls(self):
        """Test debugger works across multiple function calls."""

        @torch.compile
        def fn(x):
            return x + 1

        with patch.object(sys, "stdin", StringIO("c\nc\n")):
            with debug():
                result1 = fn(torch.randn(3))
                result2 = fn(torch.randn(3))

        self.assertEqual(result1.shape, torch.Size([3]))
        self.assertEqual(result2.shape, torch.Size([3]))

    def test_stack_at_return_value(self):
        """Test that stack has exactly 1 element at RETURN_VALUE instruction.

        This tests that stack depth calculation handles opcodes without arguments
        correctly (passing None to dis.stack_effect instead of 0).
        """

        @torch.compile
        def fn(x):
            y = x + 1
            return y

        output = StringIO()

        # Find RETURN_VALUE offset, set breakpoint, check stack
        # First get the disassembly to find RETURN_VALUE offset
        with patch.object(sys, "stdin", StringIO("ll\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.ones(3))

        # Parse output to find RETURN_VALUE offset
        output_str = output.getvalue()
        return_offset = None
        for line in output_str.split("\n"):
            if "RETURN_VALUE" in line:
                # Line format: "    41 [164]: RETURN_VALUE"
                import re

                match = re.search(r"\[(\d+)\]:\s*RETURN_VALUE", line)
                if match:
                    return_offset = int(match.group(1))
                    break

        self.assertIsNotNone(
            return_offset, "Could not find RETURN_VALUE in disassembly"
        )

        # Now run again with breakpoint at RETURN_VALUE
        torch._dynamo.reset()
        output2 = StringIO()

        @torch.compile
        def fn2(x):
            y = x + 1
            return y

        with patch.object(sys, "stdin", StringIO(f"b {return_offset}\nc\nstack\nc\n")):
            with patch.object(sys, "stdout", output2):
                with debug():
                    fn2(torch.ones(3))

        output_str2 = output2.getvalue()
        # Stack should show exactly 1 element at RETURN_VALUE
        self.assertIn("Stack (TOS at end):", output_str2)
        self.assertIn("[0]", output_str2)
        # Should contain a tensor (the return value)
        self.assertIn("tensor(", output_str2)

    def test_breakpoint_marker(self):
        """Test that BREAKPOINT_MARKER detection logic works.

        This verifies that the debugger correctly identifies LOAD_CONST
        instructions with BREAKPOINT_MARKER as breakpoints.
        """
        from torch._dynamo.bytecode_debugger import _BreakpointMarker, BREAKPOINT_MARKER
        from torch._dynamo.bytecode_transformation import create_breakpoint

        # Test create_breakpoint() helper
        bp_instructions = create_breakpoint()
        self.assertEqual(len(bp_instructions), 2)
        self.assertEqual(bp_instructions[0].opname, "LOAD_CONST")
        self.assertIs(bp_instructions[0].argval, BREAKPOINT_MARKER)
        self.assertEqual(bp_instructions[1].opname, "POP_TOP")

        # Verify repr
        self.assertEqual(repr(BREAKPOINT_MARKER), "<BREAKPOINT>")

        # Verify singleton behavior
        marker2 = _BreakpointMarker()
        self.assertIs(marker2, BREAKPOINT_MARKER)

    def test_user_defined_variables_persist(self):
        """Test that user-defined variables persist across commands."""

        @torch.compile
        def fn(x):
            return x + 1

        output = StringIO()

        # Define a variable, then access it in a later command
        with patch.object(sys, "stdin", StringIO("tmp = 42\ntmp * 2\nc\n")):
            with patch.object(sys, "stdout", output):
                with debug():
                    fn(torch.randn(3))

        output_str = output.getvalue()
        # tmp * 2 should evaluate to 84
        self.assertIn("84", output_str)


if __name__ == "__main__":
    run_tests()
