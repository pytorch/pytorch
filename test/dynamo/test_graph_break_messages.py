# Owner(s): ["module: dynamo"]

import logging
import unittest

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_utils import munge_exc
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


def _helper(fn, args, backend="eager"):
    return lambda: torch.compile(fn, backend=backend, fullgraph=True)(*args)


"""
NOTE Adding tests to this file:

It is good practice to add a minimal repro for each graph break site (i.e. `unimplemented()` call
to make sure that there aren't any errors that occur when generating graph break messages.

If a graph break message test fails because the graph break no longer repros,
it is good practice to find a new minimal repro that causes the graph break.
If this is too much work, it is likely safe to skip/remove the test, assuming
it was previously passing and the graph break message is not changed.
However, if you add a new graph break or modify a graph break message, you should
make sure that there is a test for it.
"""


class GraphBreakMessagesTest(torch._dynamo.test_case.TestCase):
    maxDiff = None

    def test_dynamic_shape_operator(self):
        def fn():
            return torch.nonzero(torch.rand([10, 10]))

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Dynamic shape operator
  Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: aten.nonzero.default


from user code:
   File "test_graph_break_messages.py", line N, in fn
    return torch.nonzero(torch.rand([10, 10]))""",
        )

    def test_dynamic_shape_operator_no_meta_kernel(self):
        def fn():
            return torch.bincount(torch.randint(0, 10, (10,)))

        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
                """\
Dynamic shape operator (no meta kernel)
  Explanation: Operator `aten.bincount.default` does not have a meta kernel that supports dynamic output shapes
  Hint: Please report an issue to PyTorch

  Developer debug context: aten.bincount.default


from user code:
   File "test_graph_break_messages.py", line N, in fn
    return torch.bincount(torch.randint(0, 10, (10,)))""",
            )

    def test_data_dependent_operator(self):
        def fn(x):
            return x.item()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.Tensor([1])),
            """\
Tensor.item

from user code:
   File "test_graph_break_messages.py", line N, in fn
    return x.item()""",
        )

    def test_data_dependent_operator2(self):
        def fn(x):
            return torch.equal(x, x)

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3)),
                """\
Data dependent operator
  Explanation: Operator `aten.equal.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Consider wrapping the operator into a PyTorch-understood custom operator (see https:/pytorch.org/tutorials/advanced/custom_ops_landing_page.html)

  Developer debug context: aten.equal.default


from user code:
   File "test_graph_break_messages.py", line N, in fn
    _helper(fn, (torch.ones(3),)),""",
            )

    def test_super_call_method(self):
        def fn(it):
            return [x for x in it]

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(zip(range(5), range(10))),
            """\
Unsupported method call
  Explanation: Dynamo does not know how to trace method `__iter__` of class `zip`
  Hint: Avoid calling `zip.__iter__` in your code.
  Hint: Please report an issue to PyTorch.

  Developer debug context: call_method UserDefinedObjectVariable(zip) __iter__ () {}


from user code:
   File "test_graph_break_messages.py", line N, in fn
    return [x for x in it]""",
        )

    def test_super_call_function(self):
        def fn(it):
            return [x for x in it()]

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(zip(range(5), range(10))),
            """\
Unsupported function call
  Explanation: Dynamo does not know how to trace the function `UserDefinedObjectVariable(zip)`
  Hint: Avoid calling `UserDefinedObjectVariable(zip)` in your code.
  Hint: Please report an issue to PyTorch.

  Developer debug context: call_function UserDefinedObjectVariable(zip) [] {}


from user code:
   File "test_graph_break_messages.py", line N, in fn
    return [x for x in it()]""",
        )

    def test_unsupported_context(self):
        def fn(obj):
            with obj:
                return x

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(3),
            """\
Unsupported context manager
  Explanation: Dynamo does not know how to enter a `int` context manager.
  Hint: Avoid using the unsupported context manager.
  Hint: File an issue to PyTorch. Simple context managers can potentially be supported, but note that context managers can't be supported in general

  Developer debug context: Attempted SETUP_WITH/BEFORE_WITH on ConstantVariable(int: 3)


from user code:
   File "test_graph_break_messages.py", line N, in fn
    with obj:""",
        )

    def test_backend_fake_tensor_exc(self):
        def bad_backend(gm, ex):
            raise torch._subclasses.fake_tensor.UnsupportedFakeTensorException("test")

        def fn(x):
            return x + 1

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend=bad_backend, fullgraph=True)(torch.ones(3, 3))
            """\
Backend compiler fake tensor exception
  Explanation: Backend compiler `bad_backend` failed with a fake tensor exception
  Hint: Report an issue to PyTorch

  Developer debug context: Backend: bad_backend
Traceback:
  File "test_graph_break_messages.py", line N, in fn
    return x + 1""",
        )

    def test_unsupported_builtin(self):
        import operator

        def fn():
            print("abc")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print` with argument types ['str'] (has_kwargs False)
  Hint: Avoid calling builtin `print` with argument types ['str']. Consider using an equivalent alternative function/method to `print`.
  Hint: If you are attempting to call a logging function (e.g. `print`), you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.
  Hint: Please report an issue to PyTorch.

  Developer debug context: builtin print [<class 'torch._dynamo.variables.constant.ConstantVariable'>] False


from user code:
   File "test_graph_break_messages.py", line N, in fn
    print("abc")"""",
        )
