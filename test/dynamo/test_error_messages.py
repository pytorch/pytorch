# Owner(s): ["module: dynamo"]

import logging
import re
import traceback
import unittest
import unittest.mock
import warnings
from functools import lru_cache

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch.utils._pytree as python_pytree
from torch._dynamo.exc import ResumePrologueTracingError, Unsupported
from torch._dynamo.testing import skipIfNotPy312
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    munge_exc,
    scoped_load_inline,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


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


class GenericCtxMgr:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class GraphBreakMessagesTest(LoggingTestCase):
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

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0036

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.nonzero(torch.rand([10, 10]))""",
        )

    def test_dynamic_shape_operator_no_meta_kernel(self):
        def fn():
            return torch.linalg.lstsq(torch.rand(10, 10), torch.rand(10, 10))

        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
                """\
Dynamic shape operator (no meta kernel)
  Explanation: Operator `aten.linalg_lstsq.default` does not have a meta kernel that supports dynamic output shapes
  Hint: Please report an issue to PyTorch

  Developer debug context: aten.linalg_lstsq.default

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0037

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.linalg.lstsq(torch.rand(10, 10), torch.rand(10, 10))""",
            )

    def test_data_dependent_operator(self):
        def fn(x):
            return x.item()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                torch.Tensor([1])
            ),
            """\
Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0124

from user code:
   File "test_error_messages.py", line N, in fn
    return x.item()""",
        )

    def test_data_dependent_operator2(self):
        def fn(x):
            return torch.equal(x, x)

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(3)
                ),
                """\
Data dependent operator
  Explanation: Operator `aten.equal.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Consider wrapping the operator into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)

  Developer debug context: aten.equal.default

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0033

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.equal(x, x)""",
            )

    def test_sort_with_nonconstant_keys(self):
        lst = [
            torch.tensor(4),
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
        ]

        def fn(lst):
            return sorted(lst)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(lst),
            """\
sort with non-constant keys
  Explanation: Cannot perform sort with non-constant key. First non-constant key type: <class 'torch.Tensor'>. Most notably, we cannot sort with Tensor or SymInt keys, but we can sort ints.
  Hint: Use something else as the key.

  Developer debug context: TensorVariable()

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0207

from user code:
   File "test_error_messages.py", line N, in fn
    return sorted(lst)""",
        )

    def test_super_call_method(self):
        def fn(it):
            return [x + 1 for x in it]

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                zip(range(5), range(10))
            ),
            """\
Unsupported method call
  Explanation: Dynamo does not know how to trace method `__iter__` of class `zip`
  Hint: Avoid calling `zip.__iter__` in your code.
  Hint: Please report an issue to PyTorch.
  Hint: Dynamo does not fully support tracing builtin iterators (e.g. `map`, `zip`, `enumerate`) passed in from uncompiled to compiled regions (e.g. `torch.compile(fn)(enumerate(...))`). This can happen unintentionally if a previous graph break happens with a builtin iterator in the local scope.
  Hint: List/dict comprehensions in Python <= 3.11 result in implicit function calls, which Dynamo cannot trace as a top level frame. Possible workarounds are (1) use a loop instead of a comprehension, (2) fix any graph breaks in the function above the comprehension, (3) wrap the comprehension in a function, or (4) use Python 3.12+.

  Developer debug context: call_method UserDefinedObjectVariable(zip) __iter__ () {}

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0156

from user code:
   File "test_error_messages.py", line N, in fn
    return [x + 1 for x in it]""",
        )

    def test_dict_items_input(self):
        def fn(x, items):
            it = iter(items)
            return next(it), x.sin()

        x = torch.randn(3)
        dct = {"a": 3, "b": 3}

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(x, dct.items()),
            """\
Unsupported method call
  Explanation: Dynamo does not know how to trace method `__iter__` of class `dict_items`
  Hint: Avoid calling `dict_items.__iter__` in your code.
  Hint: Please report an issue to PyTorch.
  Hint: Consider moving the creation of dict view object (e.g. `dict.keys()`, `dict.items()`,) to the compiled region, instead of passing it as an input to the compiled region.
  Hint: Dynamo does not fully support tracing builtin iterators (e.g. `map`, `zip`, `enumerate`) passed in from uncompiled to compiled regions (e.g. `torch.compile(fn)(enumerate(...))`). This can happen unintentionally if a previous graph break happens with a builtin iterator in the local scope.
  Hint: List/dict comprehensions in Python <= 3.11 result in implicit function calls, which Dynamo cannot trace as a top level frame. Possible workarounds are (1) use a loop instead of a comprehension, (2) fix any graph breaks in the function above the comprehension, (3) wrap the comprehension in a function, or (4) use Python 3.12+.

  Developer debug context: call_method UserDefinedObjectVariable(dict_items) __iter__ () {}

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0156

from user code:
   File "test_error_messages.py", line N, in fn
    it = iter(items)""",
        )

    def test_super_call_function(self):
        def fn(it):
            return [x + 1 for x in it()]

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                zip(range(5), range(10))
            ),
            """\
Unsupported function call
  Explanation: Dynamo does not know how to trace the function `UserDefinedObjectVariable(zip)`
  Hint: Avoid calling `UserDefinedObjectVariable(zip)` in your code.
  Hint: Please report an issue to PyTorch.

  Developer debug context: call_function UserDefinedObjectVariable(zip) [] {}

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0147

from user code:
   File "test_error_messages.py", line N, in fn
    return [x + 1 for x in it()]""",
        )

    def test_unsupported_context(self):
        def fn(obj):
            with obj:
                return 1

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(3),
            """\
Unsupported context manager
  Explanation: Dynamo does not know how to enter a `int` context manager.
  Hint: Avoid using the unsupported context manager.
  Hint: If the context manager seems like it should be supported (e.g. torch.set_grad_enabled), then it may be the case that it was created outside the compiled region, which Dynamo does not support. Supported context managers can cross graph break boundaries only if they are local non-closure variables, or are intermediate values.
  Hint: File an issue to PyTorch. Simple context managers can potentially be supported, but note that context managers can't be supported in general

  Developer debug context: Attempted SETUP_WITH/BEFORE_WITH on ConstantVariable(int: 3)

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0142

from user code:
   File "test_error_messages.py", line N, in fn
    with obj:""",
        )

    def test_backend_fake_tensor_exc(self):
        def bad_backend(gm, ex):
            raise torch._subclasses.fake_tensor.UnsupportedFakeTensorException("test")

        def fn(x):
            return x + 1

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend=bad_backend, fullgraph=True)(
                torch.ones(3, 3)
            ),
            """\
Backend compiler exception
  Explanation: Backend compiler `bad_backend` failed with test. Adding a graph break.
  Hint: Report an issue to the backend compiler repo.

  Developer debug context: Backend: bad_backend
    Exception:test
    Traceback:
      File "test_error_messages.py", line N, in fn
        return x + 1


 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0219""",
        )

    def test_unsupported_builtin(self):
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

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0059

from user code:
   File "test_error_messages.py", line N, in fn
    print("abc")""",
        )

    def test_skipfile_call(self):
        def fn():
            return unittest.skip("test")

        def post_munge(s):
            return re.sub(r"file `.*case\.py`", "file `case.py`", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `skip` in file `case.py` should not be traced.
  Hint: Avoid calling the function `skip`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `skip` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: module: unittest.case, qualname: skip, skip reason: <missing reason>

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0007

from user code:
   File "test_error_messages.py", line N, in fn
    return unittest.skip("test")""",
            post_munge=post_munge,
        )

    def test_skipfile_dynamo_call(self):
        def fn():
            torch._dynamo.disable()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `disable` in file `_dynamo/decorators.py` should not be traced.
  Hint: Avoid calling the function `disable`.

  Developer debug context: module: torch._dynamo.decorators, qualname: disable, skip reason: <missing reason>

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0007

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.disable()""",
        )

    def test_skipfile_inline(self):
        class Foo:
            fn = unittest.skip

        def fn():
            Foo().fn()

        def post_munge(s):
            return re.sub(r"`.*case\.py`", "`case.py`", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `skip` should not be traced.
  Hint: Avoid calling the function `skip`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `skip` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: skip, name: skip, filename: `case.py`, skip reason: skipped according trace_rules.lookup unittest

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0008

from user code:
   File "test_error_messages.py", line N, in fn
    Foo().fn()""",
            post_munge=post_munge,
        )

    def test_dynamo_graph_break_fn(self):
        def fn():
            torch._dynamo.graph_break()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()""",
        )

    def test_dynamo_graph_break_fn_with_msg(self):
        def fn():
            torch._dynamo.graph_break(msg="test graph break")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: test graph break
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{'msg': ConstantVariable(str: 'test graph break')}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break(msg="test graph break")""",
        )

    def test_warnings(self):
        def fn():
            warnings.warn("test")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the Python builtin `_warnings.warn`.
  Hint: If you are attempting to call a logging function (e.g. `_warnings.warn`), you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.
  Hint: Please file an issue on GitHub so the PyTorch team can add support for it.

  Developer debug context: module: _warnings, qualname: warn, skip reason: <missing reason>

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0007

from user code:
   File "test_error_messages.py", line N, in fn
    warnings.warn("test")""",
        )

    @unittest.skipIf(not python_pytree._cxx_pytree_exists, "missing optree package")
    def test_optree_graph_break_message(self):
        import optree

        @torch.compile(backend="eager")
        def fn(x):
            d = {"a": 1}
            optree.tree_flatten(d)
            return torch.sin(x)

        fn(torch.randn(4))
        self.assertEqual(len(counters["graph_break"]), 1)
        first_graph_break = next(iter(counters["graph_break"].keys()))
        self.assertExpectedInline(
            first_graph_break,
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo cannot trace optree C/C++ function optree._C.PyCapsule.flatten.
  Hint: Consider using torch.utils._pytree - https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py

  Developer debug context: module: optree._C, qualname: PyCapsule.flatten, skip reason: <missing reason>

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0007""",
        )

    @scoped_load_inline
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    @unittest.skipIf(IS_FBCODE, "inline cpp_extension doesn't work in fbcode")
    def test_cpp_extension_recommends_custom_ops(self, load_inline):
        cpp_source = """
        #include <torch/extension.h>
        at::Tensor foobar(const at::Tensor& x) {
            return x.clone();
        }
        """
        module = load_inline(
            name="mylib",
            cpp_sources=cpp_source,
            functions="foobar",
            verbose=True,
        )

        x = torch.ones(2, 2, requires_grad=True)
        counters.clear()

        @torch.compile(backend="eager")
        def f(x):
            return module.foobar(x)

        with self.assertWarnsOnceRegex(
            UserWarning,
            "(?s).*https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html.*",
        ):
            f(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        first_graph_break = next(iter(counters["graph_break"].keys()))

        first_graph_break = re.sub(r"mylib(_v\d+)?", "mylib", first_graph_break)

        self.assertExpectedInline(
            first_graph_break,
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `mylib.PyCapsule.foobar.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: mylib, qualname: PyCapsule.foobar, skip reason: <missing reason>

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0007""",
        )

        cpp_source = """
        #include <torch/extension.h>
        at::Tensor baz(const at::Tensor& x) {
            return x.clone();
        }
        """
        module2 = load_inline(
            name="mylib2",
            cpp_sources=cpp_source,
            functions="baz",
            verbose=True,
        )

        torch._dynamo.reset()

        # Test that each warning only happens once
        @torch.compile(backend="eager")
        def f(x):
            module2.baz(x)
            module.foobar(x)
            module.foobar(x)
            module2.baz(x)
            module.foobar(x)
            module2.baz(x)
            return x.clone()

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            f(x)
            f(x)
        self.assertEqual(len(ws), 2)

    def test_slice_with_tensor(self):
        def fn(x, y):
            return x[:y]

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                torch.randn(10),
                torch.tensor([3]),
            ),
            """\
Dynamic slicing with Tensor arguments
  Explanation: Creating slices with Tensor arguments is not supported. e.g. `l[:x]`, where `x` is a 1-element tensor.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: SliceVariable start: ConstantVariable(NoneType: None), stop: TensorVariable(), step: ConstantVariable(NoneType: None)

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0038

from user code:
   File "test_error_messages.py", line N, in fn
    return x[:y]""",
        )

    def test_observed_exception(self):
        def fn():
            raise RuntimeError("test")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Observed exception
  Explanation: Dynamo found no exception handler at the top-level compiled function when encountering an exception. Exception will propagate outside the compiled region.
  Hint: Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: raised exception ExceptionVariable(<class 'RuntimeError'>)

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0088

from user code:
   File "test_error_messages.py", line N, in fn
    raise RuntimeError("test")""",
        )

    def test_uninitialized_module(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                pass

        def fn(mod):
            return mod(1)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(Foo()),
            """\
Uninitialized nn.Module
  Explanation: Attempted to trace an uninitialized nn.Module of type Foo.
  Hint: Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled.
  Hint: Ensure your nn.Module instance has called `super().__init__()`.

  Developer debug context: Foo

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0119

from user code:
   File "test_error_messages.py", line N, in fn
    return mod(1)""",
        )

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_class_property(self):
        class Foo(torch.nn.Module):
            attr = unittest

        def fn(mod, x):
            return mod.attr

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                Foo(), torch.randn(3)
            ),
            """\
Unsupported nn.Module attribute type
  Explanation: Dynamo does not support tracing nn.Module attributes of type `module`
  Hint: Refactor your code so that `attr` (type `module`) is not an attribute of `Foo`
  Hint: Currently supported attribute types are methods, classmethods, staticmethods, properties, constants, and tensors.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: nn.Module subclass: Foo, name: attr, attribute type: module

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0161

from user code:
   File "test_error_messages.py", line N, in fn
    return mod.attr""",
        )

    def test_generic_ctx_mgr_graph_break(self):
        def fn():
            with GenericCtxMgr():
                with GenericCtxMgr():
                    pass
                with GenericCtxMgr():
                    with GenericCtxMgr():
                        pass
                    torch._dynamo.graph_break()

        with self.assertRaises(Unsupported) as cm:
            torch.compile(fn, backend="eager", fullgraph=True)()

        self.assertExpectedInline(
            munge_exc(cm.exception, suppress_suffix=True, skip=0),
            """\
Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0066

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()""",
        )

        self.assertExpectedInline(
            munge_exc(cm.exception.__cause__, suppress_suffix=True, skip=0),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025""",
        )

    def test_load_build_class(self):
        def fn():
            class Foo:
                pass

            return Foo

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
LOAD_BUILD_CLASS bytecode not supported
  Explanation: Dynamo does not support tracing classes that are defined in the compiled region.
  Hint: Move the class definition out of the compiled region.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context:

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0075

from user code:
   File "test_error_messages.py", line N, in fn
    class Foo:""",
        )

    @skipIfNotPy312
    def test_unsupported_bytecode(self):
        async def fn():
            async for i in range(3):
                print(i)
            return 1

        def post_munge(s):
            s = re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)
            s = re.sub(
                r"Instruction\(.*opname='GET_AITER'.*\)\n",
                "Instruction(GET_AITER)",
                s,
            )
            return s

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Missing bytecode handler
  Explanation: Dynamo does not know how to handle the bytecode instruction `GET_AITER`.
  Hint: Do not trace code that produces the `GET_AITER` bytecode instruction (see https://docs.python.org/3/library/dis.html for bytecode semantics).
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: GET_AITER with args (<torch._dynamo.symbolic_convert.InstructionTranslator object at 0xmem_addr>, Instruction(GET_AITER)
 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0082

from user code:
   File "test_error_messages.py", line N, in fn
    async for i in range(3):""",
            post_munge=post_munge,
        )

    def test_reconstruction_failure(self):
        class Foo:
            def meth(self):
                return 0

        def fn():
            return Foo().meth

        def post_munge(s):
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function GraphBreakMessagesTest.test_reconstruction_failure.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function GraphBreakMessagesTest.test_reconstruction_failure.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo))

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0092

from user code:
   File "test_error_messages.py", line N, in fn
    return Foo().meth""",
            post_munge=post_munge,
        )

    @make_logging_test(graph_breaks=True)
    def test_reconstruction_failure_gb(self, records):
        class Foo:
            def meth(self):
                return 0

        def fn():
            f = Foo().meth
            torch._dynamo.graph_break()
            return f

        def post_munge(s):
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        torch.compile(fn, backend="eager")()

        self.assertExpectedInline(
            post_munge(
                munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025
User code traceback:
  File "test_error_messages.py", line N, in test_reconstruction_failure_gb
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

        self.assertExpectedInline(
            post_munge(munge_exc(records[1].exc_info[1], suppress_suffix=True, skip=0)),
            """\
Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function GraphBreakMessagesTest.test_reconstruction_failure_gb.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function GraphBreakMessagesTest.test_reconstruction_failure_gb.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo))

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0092

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()""",
        )

    def test_faketensor_nyi(self):
        @torch.library.custom_op("mylib::foo", mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        @foo.register_fake
        def _(x):
            raise NotImplementedError

        def fn(x):
            return torch.ops.mylib.foo(x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
NotImplementedError/UnsupportedFakeTensorException when running FX node
  Explanation: Dynamo failed to run FX node with fake tensors: call_function mylib.foo(*(FakeTensor(..., size=(3,)),), **{}): got NotImplementedError()
  Hint: If the op is a PyTorch op, please file an issue to PyTorch.

  Developer debug context:

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0087

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.ops.mylib.foo(x)""",
        )

    def test_data_dependent_branching_fullgraph(self):
        def fn(x):
            if x.sum() > 0:
                return x.sin()
            return x.cos()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0170

from user code:
   File "test_error_messages.py", line N, in fn
    if x.sum() > 0:""",
        )

    @make_logging_test(graph_breaks=True)
    def test_data_dependent_branching_gb(self, records):
        def fn(x):
            if x.sum() > 0:
                return x.sin()
            return x.cos()

        torch.compile(fn, backend="eager")(torch.randn(3))

        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

User code traceback:
  File "test_error_messages.py", line N, in test_data_dependent_branching_gb
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    if x.sum() > 0:
""",
        )

    @unittest.skipIf(IS_FBCODE, "assert gets patched in internal pytest")
    @make_logging_test(graph_breaks=True)
    def test_assert_failure_in_generic_ctx_mgr(self, records):
        def fn(x):
            with GenericCtxMgr():
                assert x is None

        with self.assertRaises(AssertionError):
            torch.compile(fn, backend="eager")(torch.randn(3))

        # only 1 graph break message
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break: skip: from user code at:
  File "test_error_messages.py", line N, in fn
    assert x is None
""",
        )
        self.assertExpectedInline(
            munge_exc(records[0].exc_info[1], suppress_suffix=True, skip=0),
            """\
Data-dependent assertion failed (cannot compile partial graph)
  Explanation: Dynamo has determined when encountering a data-dependent assert failure that it should not compile the partial graph.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch._assert()` to raise a hard AssertionError when the check fails. This error will propagate back the user code that called the compiled function (i.e. Dynamo will not trace any exception handling).
  Hint: Remove the assert statement.
  Hint: Move the assert statement outside of any context managers in order to graph break with partial graph compilation (if fullgraph=False).

  Developer debug context: value: ConstantVariable(bool: False)

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0034

from user code:
   File "test_error_messages.py", line N, in fn
    assert x is None""",
        )

    def test_no_internal_compiler_stacktrace(self):
        def fn():
            gn()

        def gn():
            torch._dynamo.graph_break()

        # assertRaises suppresses the traceback, so manually catch
        e = None
        try:
            torch.compile(fn, backend="eager", fullgraph=True)()
        except Exception as exn:
            e = exn

        self.assertIsNotNone(e)

        msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # only keep the filenames in the traceback
        msg = re.sub(r'File ".*\W(\w+\.py)"', 'File "\\1"', msg)
        # remove line numbers
        msg = re.sub(r"line (\d+)", "line N", msg)
        # remove carets
        msg = re.sub(r"\n\s*~*\^+\n", "\n", msg)
        self.assertExpectedInline(
            msg,
            """\
Traceback (most recent call last):
  File "test_error_messages.py", line N, in test_no_internal_compiler_stacktrace
    torch.compile(fn, backend="eager", fullgraph=True)()
  File "eval_frame.py", line N, in compile_wrapper
    raise e.with_traceback(None) from e.__cause__  # User compiler error
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025

from user code:
   File "test_error_messages.py", line N, in fn
    gn()
  File "test_error_messages.py", line N, in gn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

""",
        )

    @torch._dynamo.config.patch(verbose=True)
    def test_internal_compiler_stacktrace_verbose(self):
        def fn():
            gn()

        def gn():
            torch._dynamo.graph_break()

        # assertRaises suppresses the traceback, so manually catch
        e = None
        try:
            torch.compile(fn, backend="eager", fullgraph=True)()
        except Exception as exn:
            e = exn

        self.assertIsNotNone(e)

        msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # only keep the filenames in the traceback
        msg = re.sub(r'File ".*\W(\w+\.py)"', 'File "\\1"', msg)
        # remove line numbers
        msg = re.sub(r"line (\d+)", "line N", msg)
        msg = re.sub(
            r"""(?s)Traceback \(most recent call last\):.*
  File "exc.py", line N, in unimplemented_v2
    raise Unsupported\(msg\)""",
            "<Internal traceback>\n",
            msg,
        )
        self.assertExpectedInline(
            msg,
            """\
<Internal traceback>

torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025

from user code:
   File "test_error_messages.py", line N, in fn
    gn()
  File "test_error_messages.py", line N, in gn
    torch._dynamo.graph_break()

""",
        )

    @make_logging_test(graph_breaks=True)
    def test_nested_compile_user_frames(self, records):
        def fn(x):
            gn(x + 1)

        def gn(x):
            hn(x + 1)

        def hn(x):
            torch._dynamo.graph_break()  # 0
            torch._dynamo.graph_break()  # 1

        torch.compile(fn, backend="eager")(torch.randn(3))

        # check the log for the 2nd torch._dynamo.graph_break()
        self.assertExpectedInline(
            munge_exc(records[-1].getMessage(), skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025
User code traceback:
  File "test_error_messages.py", line N, in test_nested_compile_user_frames
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    gn(x + 1)
  File "test_error_messages.py", line N, in gn
    hn(x + 1)
  File "test_error_messages.py", line N, in hn
    torch._dynamo.graph_break()  # 1
""",
        )

    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_graph_break_traceback_above_dynamo_shows_user_code(self, records):
        @torch.compile(backend="eager")
        # NOTE: comments in this test are used to differentiate lines!
        def f1(x):
            torch._dynamo.graph_break()  # 0
            torch._dynamo.graph_break()  # 1
            torch._dynamo.graph_break()

        @torch.compile(backend="eager")
        def f2(x):
            if x.sum() > 0:  # 0
                x = x + 1
            if x.sum() > 0:  # 1
                x = x + 1
            if x.sum() > 0:
                x = x + 1

        class Foo:
            def __setattr__(self, name, value):
                torch._dynamo.graph_break()

        @torch.compile(backend="eager")
        def f3(x):
            Foo().attr = x  # 0
            Foo().attr = x  # 1
            Foo().attr = x

        f1(torch.randn(3))
        self.assertIn("torch._dynamo.graph_break()  # 0", records[-1].getMessage())
        self.assertIn("torch._dynamo.graph_break()  # 1", records[-1].getMessage())
        f2(torch.ones(3))
        self.assertIn("if x.sum() > 0:  # 0", records[-1].getMessage())
        self.assertIn("if x.sum() > 0:  # 1", records[-1].getMessage())
        f3(torch.randn(3))
        self.assertIn("Foo().attr = x  # 0", records[-1].getMessage())
        self.assertIn("Foo().attr = x  # 1", records[-1].getMessage())

        def post_munge(s):
            return re.sub(
                r"torch_dynamo_resume_in_f(\d)_at_(\d+)",
                r"torch_dynamo_resume_in_f\1_at_N",
                s,
            )

        self.assertExpectedInline(
            post_munge(munge_exc(records[-1].getMessage(), skip=0)),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: STORE_ATTR-caused graph break
User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_traceback_above_dynamo_shows_user_code
    f3(torch.randn(3))
  File "test_error_messages.py", line N, in f3
    Foo().attr = x  # 0
  File "test_error_messages.py", line N, in torch_dynamo_resume_in_f3_at_N
    Foo().attr = x  # 1

========== most recent `torch.compile` tracing attempt started here ==========

  File "test_error_messages.py", line N, in torch_dynamo_resume_in_f3_at_N
    Foo().attr = x

NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python function, which Dynamo intercepts as a top-level frame.
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_traceback_collapsed_resume_frames(self, records):
        @torch.compile(backend="eager")
        def f1(x):
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            f2(x)

        def f2(x):
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            f3(x)

        def f3(x):
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()  # correct
            return x + 1

        f1(torch.randn(3))

        self.assertExpectedInline(
            munge_exc(records[-1].getMessage(), skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0025
User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_traceback_collapsed_resume_frames
    f1(torch.randn(3))
  File "test_error_messages.py", line N, in f1
    f2(x)
  File "test_error_messages.py", line N, in f2
    f3(x)
  File "test_error_messages.py", line N, in f3
    torch._dynamo.graph_break()  # correct
""",
        )

    @make_logging_test(dynamo=logging.DEBUG)
    def test_lru_cache_warning_logs_user_stack_trace(self, records):
        @lru_cache
        def foo(x):
            return x + 1

        torch.compile(foo, backend="eager")(torch.randn(4))

        lru_cache_log = None
        for record in records:
            if "call to a lru_cache wrapped function at:" in record.getMessage():
                lru_cache_log = record.getMessage()
                break

        self.assertIsNotNone(lru_cache_log, "No lru_cache warning was logged")

        self.assertExpectedInline(
            munge_exc(lru_cache_log),
            """\
call to a lru_cache wrapped function at: _dynamo/external_utils.py:N
  File "test_error_messages.py", line N, in test_lru_cache_warning_logs_user_stack_trace
    torch.compile(foo, backend="eager")(torch.randn(4))
""",
        )

    @make_logging_test(dynamo=logging.DEBUG)
    def test_lru_cache_warning_logs_nested_call(self, records):
        @lru_cache
        def foo(x):
            return x + 1

        def nested(x):
            return foo(x)

        torch.compile(nested, backend="eager")(torch.randn(4))

        lru_cache_log = None
        for record in records:
            if "call to a lru_cache wrapped function at:" in record.getMessage():
                lru_cache_log = record.getMessage()
                break

        self.assertIsNotNone(lru_cache_log, "No lru_cache warning was logged")

        self.assertExpectedInline(
            munge_exc(lru_cache_log),
            """\
call to a lru_cache wrapped function at: test_error_messages.py:N
  File "test_error_messages.py", line N, in test_lru_cache_warning_logs_nested_call
    torch.compile(nested, backend="eager")(torch.randn(4))
  File "test_error_messages.py", line N, in nested
    return foo(x)
""",
        )

    def test_disable_message(self):
        @torch.compile(backend="eager", fullgraph=True)
        def outer(fn, x):
            return fn(x)

        @torch.compiler.disable
        def f(x):
            return x + 1

        def post_munge(s):
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: outer(f, torch.randn(3)),
            """\
Skip calling `torch.compiler.disable()`d function
  Explanation: Skip calling function `<function GraphBreakMessagesTest.test_disable_message.<locals>.f at 0xmem_addr>` since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function GraphBreakMessagesTest.test_disable_message.<locals>.f at 0xmem_addr>

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0098

from user code:
   File "test_error_messages.py", line N, in outer
    return fn(x)""",
            post_munge=post_munge,
        )

        @torch.compiler.disable(reason="test message")
        def g(x):
            return x + 2

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: outer(g, torch.randn(3)),
            """\
Skip calling `torch.compiler.disable()`d function
  Explanation: Skip calling function `<function GraphBreakMessagesTest.test_disable_message.<locals>.g at 0xmem_addr>` since it was wrapped with `torch.compiler.disable` (reason: test message)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function GraphBreakMessagesTest.test_disable_message.<locals>.g at 0xmem_addr>

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0098

from user code:
   File "test_error_messages.py", line N, in outer
    return fn(x)""",
            post_munge=post_munge,
        )

        class Mod(torch.nn.Module):
            def forward(self, x):
                return x + 3

        mod = Mod()
        mod.compile()
        mod = torch.compiler.disable(mod, reason="test message 2")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: outer(mod, torch.randn(3)),
            """\
Unsupported function call (delayed)
  Explanation: Dynamo determined that a graph break should occur when calling `L['fn']`. Reason: Optimized `nn.Module` is wrapped with `torch.compiler.disable` (reason: test message 2)


  Developer debug context: source: LocalSource(local_name='fn', is_input=True, dynamism=None, is_derefed_cell_contents=False)

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0148

from user code:
   File "test_error_messages.py", line N, in outer
    return fn(x)""",
            post_munge=post_munge,
        )

    # Test that errors while tracing resume function prologues do not get suppressed
    def test_graph_break_in_buggy_resume_prologue(self):
        import torch._dynamo.bytecode_transformation as bt
        import torch._dynamo.resume_execution as rex

        # NOTE: do not define non_global as a global in this file!
        @torch.compile(backend="eager")
        def fn(non_global):
            non_global = non_global + 1
            torch._dynamo.graph_break()
            return non_global + 1

        orig_clean_and_assemble_instructions = bt.clean_and_assemble_instructions

        def bad_clean_and_assemble_instructions(instructions, *args):
            # Inject an invalid LOAD_GLOBAL after the first STORE_FAST IS_TRACING_RESUME_PROLOGUE_VARNAME
            for i, inst in enumerate(instructions):
                if (
                    inst.opname == "STORE_FAST"
                    and inst.argval == rex.IS_TRACING_RESUME_PROLOGUE_VARNAME
                ):
                    instructions[:] = (
                        instructions[: i + 1]
                        + [
                            # this should cause a graph break
                            bt.create_instruction("LOAD_GLOBAL", argval="non_global"),
                        ]
                        + instructions[i + 1 :]
                    )
                    break
            return orig_clean_and_assemble_instructions(instructions, *args)

        with unittest.mock.patch(
            "torch._dynamo.bytecode_transformation.clean_and_assemble_instructions",
            bad_clean_and_assemble_instructions,
        ):
            with self.assertRaisesRegex(
                ResumePrologueTracingError,
                "Error while tracing through a Dynamo-generated resume function prologue.",
            ):
                fn(torch.randn(3))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
