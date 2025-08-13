# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import os
import sys
from typing import Any, List

import torch
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase, make_global


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestWith(JitTestCase):
    """
    A suite of tests for with statements.
    """

    def test_with_as(self):
        """
        Check that with statements that use the 'as' keyword to bind expressions
        to targets work as expected.
        """

        @torch.jit.script
        class Context:
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any) -> bool:
                self.count.sub_(0.3)
                return True

        make_global(Context)

        def test_basic(x: torch.Tensor) -> torch.Tensor:
            """Basic test with one with-statement."""

            c = Context(1)

            with c as mult:
                y = x + mult

            y *= c.count
            return y

        def test_pass(x: torch.Tensor) -> torch.Tensor:
            """
            Test with a pass statement inside a with-statement. Although
            the body of the with is empty, __enter__ and __exit__ should
            still be called.
            """
            c = Context(1)

            with c as mult:
                pass

            x *= c.count
            return x

        def test_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test that returning early from inside a with-statement works
            as expected.
            """
            with c as mult:
                y = x + mult
                return y

            x = y + y
            return x

        def test_conditional_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test that conditionally returning early from inside a with-statement works
            as expected.
            """
            with c as mult:
                y = x + mult
                if mult > 0:
                    return y

            x = y + y
            return x

        def test_break(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            """
            Test that breaking early from inside a with-statement works
            as expected.
            """
            with c as mult:
                for a in l:
                    if a == 0:
                        break
                    x += a * mult

            return x

        def test_continue(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            """
            Test that using continue inside a with-statement works
            as expected.
            """
            with c as mult:
                for a in l:
                    if a == 0:
                        continue
                    x += a * mult

            return x

        def test_serial(x: torch.Tensor) -> torch.Tensor:
            """
            Test two with-statements in a row.
            """
            c = Context(1)

            with c as mult:
                y = x + mult

            with c as mult:
                y *= mult

            return y

        def test_nested(x: torch.Tensor) -> torch.Tensor:
            """
            Test nested with-statements.
            """
            c = Context(1)

            with c as m:
                with c as n:
                    y = x + n

                y *= m

            return y

        def test_combined(x: torch.Tensor) -> torch.Tensor:
            """
            Test a with-statement with multiple with items.
            """
            c = Context(1)
            d = Context(2)

            with c as m, d as n:
                y = x + (m + n)

            return y

        test_input = torch.randn(2, 2)
        test_context = Context(2)
        test_list = [2, 0, 1, 3, 0, 2]

        self.checkScript(test_basic, (test_input,))
        self.checkScript(test_pass, (test_input,))
        self.checkScript(test_early_return, (test_input, test_context))
        self.checkScript(test_break, (test_input, test_context, test_list))
        self.checkScript(test_continue, (test_input, test_context, test_list))
        self.assertEqual(test_context.count, 2)
        self.checkScript(test_serial, (test_input,))
        self.checkScript(test_nested, (test_input,))
        self.checkScript(test_combined, (test_input,))

    def test_with_no_as(self):
        """
        Check that with statements that do not use the 'as' keyword to bind expressions
        to targets work as expected.
        """

        @torch.jit.script
        class Context:
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any):
                self.count.sub_(0.3)

        make_global(Context)

        def test_basic(x: torch.Tensor) -> torch.Tensor:
            """Basic test with one with-statement."""

            c = Context(1)

            with c:
                y = x + c.count

            y *= c.count
            return y

        def test_pass(x: torch.Tensor) -> torch.Tensor:
            """
            Test with a pass statement inside a with-statement. Although
            the body of the with is empty, __enter__ and __exit__ should
            still be called.
            """
            c = Context(1)

            with c:
                pass

            x *= c.count
            return x

        def test_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test that returning early from inside a with-statement works
            as expected.
            """
            with c:
                y = x + c.count
                return y

            x = y + y
            return x

        def test_conditional_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test that conditionally returning early from inside a with-statement works
            as expected.
            """
            with c:
                y = x + c.count
                if c.count > 0:
                    return y

            x = y + y
            return x

        def test_break(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            """
            Test that breaking early from inside a with-statement works
            as expected.
            """
            with c:
                for a in l:
                    if a == 0:
                        break
                    x += a * c.count

            return x

        def test_continue(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            """
            Test that using continue inside a with-statement works
            as expected.
            """
            with c:
                for a in l:
                    if a == 0:
                        continue
                    x += a * c.count

            return x

        def test_serial(x: torch.Tensor) -> torch.Tensor:
            """
            Test two with-statements in a row.
            """
            c = Context(1)

            with c:
                y = x + c.count

            with c:
                y *= c.count

            return y

        def test_nested(x: torch.Tensor) -> torch.Tensor:
            """
            Test nested with-statements.
            """
            c = Context(1)

            with c:
                with c:
                    y = x + c.count

                y *= c.count

            return y

        def test_combined(x: torch.Tensor) -> torch.Tensor:
            """
            Test a with-statement with multiple with items.
            """
            c = Context(1)
            d = Context(2)

            with c, d:
                y = x + (c.count + d.count)

            return y

        test_input = torch.randn(2, 2)
        test_context = Context(2)
        test_list = [2, 0, 1, 3, 0, 2]

        self.checkScript(test_basic, (test_input,))
        self.checkScript(test_pass, (test_input,))
        self.checkScript(test_early_return, (test_input, test_context))
        self.checkScript(test_break, (test_input, test_context, test_list))
        self.checkScript(test_continue, (test_input, test_context, test_list))
        self.assertEqual(test_context.count, 2)
        self.checkScript(test_serial, (test_input,))
        self.checkScript(test_nested, (test_input,))
        self.checkScript(test_combined, (test_input,))

    def test_with_exceptions(self):
        """
        Check that exceptions thrown in the bodies of with-statements are
        handled correctly.
        """

        @torch.jit.script
        class Context:
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any):
                self.count.sub_(0.3)

        make_global(Context)

        @torch.jit.script
        def method_that_raises() -> torch.Tensor:
            raise Exception("raised exception")  # noqa: TRY002

        @torch.jit.script
        def test_exception(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test the case in which an exception is thrown while executing the body of a with-statement.
            """
            with c as _:
                x += method_that_raises()

            return x

        @torch.jit.script
        def test_exception_nested(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test the case in which an exception is thrown while executing the body of a nested with-statement.
            """
            with c as _:
                with c as _:
                    x += method_that_raises()

            return x

        @torch.jit.script
        def with_that_raises(c: Context) -> torch.Tensor:
            a = torch.tensor([1])

            with c as _:
                a += method_that_raises()

            return a

        @torch.jit.script
        def test_exception_fn_call(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test the case in which an exception is thrown while there are active with-statements in two different
            frames.
            """
            with c as _:
                x += with_that_raises(c)

            return x

        c = Context(1)

        # checkScript and checkScriptRaisesRegex cannot be used because the string frontend will
        # not compile class types (of which Context, the context manager being used for this test
        # is one).
        with self.assertRaisesRegexWithHighlight(
            Exception, r"raised exception", 'raise Exception("raised exception'
        ):
            test_exception(torch.randn(2), c)
        self.assertEqual(c.count, 1)

        with self.assertRaisesRegexWithHighlight(
            Exception, r"raised exception", 'raise Exception("raised exception'
        ):
            test_exception_nested(torch.randn(2), c)
        self.assertEqual(c.count, 1)

        with self.assertRaisesRegexWithHighlight(
            Exception, r"raised exception", 'raise Exception("raised exception'
        ):
            test_exception_fn_call(torch.randn(2), c)
        self.assertEqual(c.count, 1)

    def test_with_errors(self):
        """
        Check that errors related to with-statements are detected and reported correctly.
        """

        @torch.jit.script
        class NoEnterNoExit:
            """
            This class is missing __enter__ and __exit__ methods.
            """

            def __init__(self) -> None:
                self.count = 1

        @torch.jit.script
        class BadEnter:
            """
            This class has an __enter__ method with an incorrect signature.
            """

            def __init__(self) -> None:
                self.count = 1

            def __enter__(self, incr: int):  # noqa: PLE0302
                self.count += incr

            def __exit__(self, type: Any, value: Any, tb: Any):
                pass

        @torch.jit.script
        class BadExit:
            """
            This class has an __exit__ method with an incorrect signature.
            """

            def __init__(self) -> None:
                self.count = 1

            def __enter__(self):
                self.count += 1

            def __exit__(self, type: Any, value: Any):  # noqa: PLE0302
                pass

        @torch.jit.script
        class ExitIncorrectTypes:
            """
            This class has an __exit__ method with unsupported argument types.
            """

            def __init__(self) -> None:
                self.count = 1

            def __enter__(self):
                self.count += 1

            def __exit__(self, type: Any, value: int, tb: int):
                pass

        def test_no_enter_no_exit(x: torch.Tensor, cm: NoEnterNoExit) -> torch.Tensor:
            with cm as _:
                pass

            return x

        def test_bad_enter(x: torch.Tensor, cm: BadEnter) -> torch.Tensor:
            with cm as _:
                pass

            return x

        def test_bad_exit(x: torch.Tensor, cm: BadExit) -> torch.Tensor:
            with cm as _:
                pass

            return x

        def test_exit_incorrect_types(
            x: torch.Tensor, cm: ExitIncorrectTypes
        ) -> torch.Tensor:
            with cm as _:
                pass

            return x

        def test_enter_without_object():
            with "not_object" as obj:
                pass

        test_tensor = torch.randn(5, dtype=torch.double)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"does not define __enter__ and __exit__ methods", "cm"
        ):
            self.checkScript(test_no_enter_no_exit, (test_tensor, NoEnterNoExit()))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"__enter__ must have only one argument and one return value",
            "cm",
        ):
            self.checkScript(test_bad_enter, (test_tensor, BadEnter()))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"__exit__ must have four arguments", "cm"
        ):
            self.checkScript(test_bad_exit, (test_tensor, BadExit()))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"argument 2 of __exit__ must have Any type", "cm"
        ):
            self.checkScript(
                test_exit_incorrect_types, (test_tensor, ExitIncorrectTypes())
            )

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"must return an object", '"not_object"'
        ):
            self.checkScript(test_enter_without_object, ())

    def test_with_no_grad(self):
        """
        Check that torch.no_grad() works. Most of these are adapted from
        corresponding tests for eager-mode no_grad.
        """

        # Basic no_grad test.
        def test_no_grad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                w = x + y

            return w

        s = torch.jit.script(test_no_grad)
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        w = s(x, y)

        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.grad_fn)

        # Test assignment of a grad-less Tensor to a Tensor with gradients
        # in a no_grad block.
        def test_no_grad_assignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                x[0] = y

            return x

        s = torch.jit.script(test_no_grad_assignment)
        z = torch.randn(5)
        w = s(x, z)
        self.assertTrue(w.requires_grad)
        self.assertIsNone(w.grad_fn)

        # Check that @torch.jit.ignored functions respect no_grad when it is
        # called in JIT mode.
        class NoGradModule(torch.nn.Module):
            @torch.jit.ignore
            def adder(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                w = x + y
                return w

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    w = self.adder(x, y)

                return w

        s = torch.jit.script(NoGradModule())
        w = s(x, y)

        self.assertFalse(w.requires_grad)

    @skipIfTorchDynamo("Torchdynamo cannot correctly handle profiler.profile calls")
    def test_with_record_function(self):
        """
        Check that torch.autograd.profiler.record_function context manager is
        torchscriptable.
        """

        def with_rf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.autograd.profiler.record_function("foo"):
                # Nested record_function.
                with torch.autograd.profiler.record_function("nested"):
                    a = x + y
            return a

        scripted = torch.jit.script(with_rf)
        x, y = torch.ones(2), torch.ones(2)
        with torch.autograd.profiler.profile() as p:
            scripted(x, y)

        # Need to call below to populate CPU children.
        p.key_averages()
        function_events = p.function_events
        # Event with name "foo" should be recorded.
        rf_events = [evt for evt in function_events if evt.name == "foo"]
        self.assertEqual(len(rf_events), 1)
        rf_event = rf_events[0]
        child_events = rf_event.cpu_children
        # Ensure we find nested record_function event
        self.assertTrue("nested" in (child.name for child in child_events))
        nested_function_event = [
            evt for evt in function_events if evt.name == "nested"
        ][0]
        # Nested record function should have child "aten::add"
        nested_child_events = nested_function_event.cpu_children
        self.assertTrue("aten::add" in (child.name for child in nested_child_events))
