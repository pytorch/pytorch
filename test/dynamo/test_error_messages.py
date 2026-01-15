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
from torch._dynamo.testing import skipIfNotPy312, skipIfOnlyNotPy312
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


class ErrorMessagesTest(LoggingTestCase):
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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0037.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.linalg.lstsq(torch.rand(10, 10), torch.rand(10, 10))""",
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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

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

  Developer debug context: LazyVariableTracker(realized: TensorVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0207.html

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

  Developer debug context: call_method UserDefinedObjectVariable(zip) __iter__ [] {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0156.html

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

  Developer debug context: call_method UserDefinedObjectVariable(dict_items) __iter__ [] {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0156.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0147.html

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

  Developer debug context: Attempted SETUP_WITH/BEFORE_WITH/LOAD_SPECIAL on LazyVariableTracker(realized: ConstantVariable(int: 3))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0142.html

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


 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0219.html""",
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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0059.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "test_error_messages.py", line N, in fn
    warnings.warn("test")""",
        )

    @unittest.skipIf(not python_pytree._cxx_pytree_exists, "missing optree package")
    def test_optree_graph_break_message(self):
        import optree

        @torch.compile(backend="eager")
        def fn1(x):
            tree = {"a": x, "b": (x - 1, 2 * x)}
            sin, cos = optree.tree_transpose_map(
                lambda t: (torch.sin(t), torch.cos(t)),
                tree,
            )
            return sin, cos

        fn1(torch.randn(4))
        self.assertEqual(len(counters["graph_break"]), 0)

        @torch.compile(backend="eager")
        def fn2(x):
            spec = optree.treespec_deque([])
            return spec, x

        fn2(torch.randn(4))
        self.assertGreaterEqual(len(counters["graph_break"]), 1)
        first_graph_break = next(iter(counters["graph_break"].keys()))

        def post_munge(string):
            return re.sub(
                r"(optree\.|qualname: )\S*(\.make_from_collection)",
                r"\1<path>\2",
                string,
            )

        self.assertExpectedInline(
            post_munge(first_graph_break),
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo cannot trace optree C/C++ function optree.<path>.make_from_collection.
  Hint: Consider using torch.utils._pytree - https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py

  Developer debug context: module: optree._C, qualname: <path>.make_from_collection, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html""",
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
        # HACK: this patches around the fact that PyBind11 improperly sets the
        # __qualname__ attribute on functions and methods; see
        # https://github.com/pybind/pybind11/issues/5774.  This should be removed if
        # that issue is fixed.
        first_graph_break = re.sub(
            r"pybind11_detail_function_record_v[^ .]+", "PyCapsule", first_graph_break
        )

        self.assertExpectedInline(
            first_graph_break,
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `mylib.PyCapsule.foobar.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: mylib, qualname: PyCapsule.foobar, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html""",
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

  Developer debug context: raised exception RuntimeError([ConstantVariable(str: 'test')])

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0088.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0119.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0161.html

from user code:
   File "test_error_messages.py", line N, in fn
    return mod.attr""",
        )

    def test_generic_ctx_mgr_graph_break_fullgraph_true(self):
        def fn():
            with GenericCtxMgr():
                torch._dynamo.graph_break()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()""",
        )

    @make_logging_test(graph_breaks=True)
    def test_generic_ctx_mgr_graph_break_fullgraph_false(self, records):
        def fn():
            with GenericCtxMgr():
                with GenericCtxMgr():
                    pass
                with GenericCtxMgr():
                    with GenericCtxMgr():
                        pass
                    torch._dynamo.graph_break()

        torch.compile(fn, backend="eager")()
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

User code traceback:
  File "test_error_messages.py", line N, in test_generic_ctx_mgr_graph_break_fullgraph_false
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
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
Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `builtins.__build_class__.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: builtins, qualname: __build_class__, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

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
 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0082.html

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
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

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
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_reconstruction_failure_gb
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

        self.assertExpectedInline(
            post_munge(
                munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure_gb.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure_gb.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

User code traceback:
  File "test_error_messages.py", line N, in test_reconstruction_failure_gb
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

    def test_faketensor_nyi(self):
        op_name = "mylib::error_messages_faketensor"

        @torch.library.custom_op(op_name, mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        @foo.register_fake
        def _(x):
            raise NotImplementedError

        def fn(x):
            return torch.ops.mylib.error_messages_faketensor(x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
NotImplementedError/UnsupportedFakeTensorException when running FX node
  Explanation: Dynamo failed to run FX node with fake tensors: call_function mylib.error_messages_faketensor(*(FakeTensor(..., size=(3,)),), **{}): got NotImplementedError()
  Hint: If the op is a PyTorch op, please file an issue to PyTorch.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0087.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.ops.mylib.error_messages_faketensor(x)""",
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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

from user code:
   File "test_error_messages.py", line N, in fn
    if x.sum() > 0:""",
        )

    # Test that the bytecode source attribution is correct with VariableTracker
    @make_logging_test(trace_bytecode=True)
    def test_variable_tracker_source_attribution(self, records):
        def inner(x):
            return x + 1

        @torch.compile(backend="eager")
        def fn(x):
            x = inner(x)
            return inner(x)

        fn(torch.ones(3))

        def find_trace_bytecode_lines(long_string):
            # Split the string into lines
            lines = long_string.split("\n")
            # More comprehensive pattern to capture LazyVariableTracker info
            pattern = r"LazyVariableTracker\([^)]*\)"
            # Find all lines containing the pattern
            result = [line for line in lines if re.search(pattern, line)]
            return result

        # Get all log messages, not just the last one
        all_messages = []
        for record in records:
            msg = munge_exc(record.getMessage(), skip=0)

            all_messages.append(msg)

        # Combine all messages to search through
        combined_msg = "\n".join(all_messages)
        all_lines = find_trace_bytecode_lines(combined_msg)

        # For now, just check that we found some lines with LazyVariableTracker
        self.assertGreater(
            len(all_lines), 0, "Should find at least one LazyVariableTracker line"
        )

        self.assertIn(
            "LazyVariableTracker(unrealized: <class 'function'>)", all_lines[0]
        )
        self.assertIn(
            "LazyVariableTracker(realized: UserFunctionVariable())", all_lines[3]
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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

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
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent assertion failed (cannot compile partial graph)
  Explanation: Dynamo has determined when encountering a data-dependent assert failure that it should not compile the partial graph.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch._assert()` to raise a hard AssertionError when the check fails. This error will propagate back the user code that called the compiled function (i.e. Dynamo will not trace any exception handling).
  Hint: Remove the assert statement.
  Hint: Move the assert statement outside of any context managers in order to graph break with partial graph compilation (if fullgraph=False).

  Developer debug context: value: ConstantVariable(bool: False)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0034.html

User code traceback:
  File "test_error_messages.py", line N, in test_assert_failure_in_generic_ctx_mgr
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    assert x is None
""",
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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

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
  File "exc.py", line N, in unimplemented
    raise Unsupported\([^\)]+\)""",
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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    gn()
  File "test_error_messages.py", line N, in gn
    torch._dynamo.graph_break()

""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_in_loop(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            for i in range(2):
                torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(3))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: fn (test_error_messages.py line N)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_in_loop
    fn(torch.ones(3))
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

        @torch.compile(backend="eager")
        def gn(x):
            for i in range(2):
                if x.sum() > 0:
                    x = x + 1
                else:
                    x = x + 2
            return x + 4

        gn(torch.ones(3))
        self.assertEqual(len(records), 2)
        self.assertExpectedInline(
            munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: gn (test_error_messages.py line N)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_in_loop
    gn(torch.ones(3))
  File "test_error_messages.py", line N, in gn
    if x.sum() > 0:
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_skip_frame_in_loop_message(self, records):
        def fn(x):
            for i in range(2):
                with GenericCtxMgr():
                    if x.sum() > 0:
                        x = x + 1
            return x

        torch.compile(fn, backend="eager")(torch.randn(3))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

User code traceback:
  File "test_error_messages.py", line N, in test_skip_frame_in_loop_message
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    if x.sum() > 0:
""",
        )

    @make_logging_test(dynamo=logging.DEBUG)
    def test_skip_frame_empty_function_message(self, records):
        def empty_fn(x):
            pass

        torch.compile(empty_fn, backend="eager")(torch.randn(3))
        skip_messages = [
            r for r in records if "No ops traced for the FX graph." in r.getMessage()
        ]
        self.assertEqual(len(skip_messages), 1)
        msg = munge_exc(skip_messages[0].getMessage(), suppress_suffix=True, skip=0)
        msg = re.sub(r" (\d+)$", r" N", msg, flags=re.MULTILINE)

        self.assertExpectedInline(
            msg,
            """\
Received signal to skip frame (without graph break): No ops traced for the FX graph. `torch.compile` will skip the frame and fall back to eager.
Frame info: empty_fn (test_error_messages.py line N) empty_fn                 test_error_messages.py N""",
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

        self.assertExpectedInline(
            munge_exc(records[-1].getMessage(), skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

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
    def test_latest_bytecode_to_graph_break_fullgraph(self, records):
        def fn(x):
            y = x + 1
            z = x + y
            torch._dynamo.graph_break()
            return z

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

    @skipIfOnlyNotPy312
    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_latest_bytecode_to_graph_break_python_versioning(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            z = x + y
            torch._dynamo.graph_break()
            return z

        fn(torch.ones(3))

        s = munge_exc(records[0].getMessage(), skip=0)

        self.assertExpectedInline(
            s,
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_latest_bytecode_to_graph_break_python_versioning
    fn(torch.ones(3))

========== most recent `torch.compile` tracing attempt started here ==========

  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()

NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python function, which Dynamo intercepts as a top-level frame.

Most recent bytecode instructions traced (max 20):
TRACE RESUME 0 []
TRACE LOAD_FAST 'x' []
TRACE LOAD_CONST 1 [LazyVariableTracker(unrealized: <class 'torch.Tensor'>)]
TRACE BINARY_OP 0 [LazyVariableTracker(unrealized: <class 'torch.Tensor'>), ConstantVariable(int: 1)]
TRACE STORE_FAST 'y' [TensorVariable()]
TRACE LOAD_FAST 'x' []
TRACE LOAD_FAST 'y' [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), TensorVariable()]
TRACE STORE_FAST 'z' [TensorVariable()]
TRACE LOAD_GLOBAL 'torch' []
TRACE LOAD_ATTR '_dynamo' [LazyVariableTracker(unrealized: <class 'module'>)]
TRACE LOAD_ATTR 'graph_break' [LazyVariableTracker(unrealized: <class 'module'>)]
TRACE CALL 0 [NullVariable, LazyVariableTracker(unrealized: <class 'function'>)]
""",
        )

    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_latest_bytecode_to_graph_break(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            z = x + y
            torch._dynamo.graph_break()
            return z

        fn(torch.ones(3))

        pattern = r"TRACE.*"
        s = munge_exc(records[0].getMessage(), skip=0)
        matches = re.findall(pattern, s)
        self.assertEqual((len(matches) > 10), True)
        self.assertEqual((len(matches) <= 20), True)
        self.assertIn("Most recent bytecode instructions traced (max 20):", s)

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
            s = re.sub(
                r"torch_dynamo_resume_in_f(\d)_at_(\d+)",
                r"torch_dynamo_resume_in_f\1_at_N",
                s,
            )
            # remove most recent bytecode instructions
            # DOTALL is needed to entirely remove TRACE ... lines (including the newline)
            return re.sub(r"TRACE.*$", "", s, flags=re.DOTALL)

        self.assertExpectedInline(
            post_munge(munge_exc(records[-1].getMessage(), skip=0)),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace STORE_ATTR: storing an object's attribute, e.g. x.attr = y:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

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
  File "test_error_messages.py", line N, in __setattr__
    torch._dynamo.graph_break()

NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python function, which Dynamo intercepts as a top-level frame.

Most recent bytecode instructions traced (max 20):
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
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

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

    def test_lru_cache_warning(self):
        # test only the warning message itself
        @lru_cache
        def bax(x):
            return x + 1

        def bar(x):
            return bax(x)

        @torch.compile(backend="eager", fullgraph=True)
        def foo(x):
            return bar(x)

        x = torch.randn(2)
        with self.assertWarnsOnceRegex(
            UserWarning,
            r"(?s).*This call originates from:\n.*File .*, line (\d+), in bar",
        ):
            foo(x)

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
  Explanation: Skip calling function `<function ErrorMessagesTest.test_disable_message.<locals>.f at 0xmem_addr>` since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function ErrorMessagesTest.test_disable_message.<locals>.f at 0xmem_addr>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0098.html

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
  Explanation: Skip calling function `<function ErrorMessagesTest.test_disable_message.<locals>.g at 0xmem_addr>` since it was wrapped with `torch.compiler.disable` (reason: test message)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function ErrorMessagesTest.test_disable_message.<locals>.g at 0xmem_addr>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0098.html

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

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0148.html

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

    @make_logging_test(graph_breaks=True)
    def test_step_graph_break(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            x = x + 2
            torch._dynamo.step_unsupported()
            return x + 4

        fn(torch.ones(3))

        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Call to `torch._dynamo.step_unsupported()`
  Explanation: User-inserted step_unsupported.
  Hint: Remove the `torch._dynamo.step_unsupported()` call.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4636.html

User code traceback:
  File "test_error_messages.py", line N, in test_step_graph_break
    fn(torch.ones(3))
  File "test_error_messages.py", line N, in fn
    torch._dynamo.step_unsupported()
""",
        )

        torch._dynamo.reset()

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: fn(torch.ones(3)),
                """\
cannot resume from torch._dynamo.step_unsupported()
  Explanation: traced torch._dynamo.step_unsupported(), but Dynamo is instructed to error on graph break. This graph break is used for debugging only.
  Hint: Remove the torch._dynamo.step_unsupported() call.
  Hint: Make sure fullgraph=False and error_on_graph_break=False.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0284.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.step_unsupported()""",
            )

    @make_logging_test(graph_breaks=True)
    def test_store_attr_graph_break(self, records):
        class Foo:
            def __setattr__(self, name, value):
                torch._dynamo.graph_break()

        @torch.compile(backend="eager")
        def fn(x):
            Foo().attr = x

        fn(torch.ones(3))

        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace STORE_ATTR: storing an object's attribute, e.g. x.attr = y:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_store_attr_graph_break
    fn(torch.ones(3))
  File "test_error_messages.py", line N, in fn
    Foo().attr = x
  File "test_error_messages.py", line N, in __setattr__
    torch._dynamo.graph_break()
""",
        )

        torch._dynamo.reset()

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: fn(torch.ones(3)),
                """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    Foo().attr = x
  File "test_error_messages.py", line N, in __setattr__
    torch._dynamo.graph_break()""",
            )

    def test_runtime_error_readable_shape_mismatch(self):
        def fn(x, y):
            return x + y

        x = torch.randn(4, 4)
        y = torch.randn(10, 10)
        torch._dynamo.mark_dynamic(x, 2)
        torch._dynamo.mark_dynamic(y, 1)

        from torch._dynamo.exc import TorchRuntimeError

        def post_munge(s):
            s = re.sub(r"s\d+: hint = 10", "s94: hint = 10", s)
            return s

        self.assertExpectedInlineMunged(
            TorchRuntimeError,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(x, y),
            """\
Dynamo failed to run FX node with fake tensors: call_function <built-in function add>(*(FakeTensor(..., size=(4, 4)), FakeTensor(..., size=(10, s94))), **{}): got RuntimeError('The size of tensor a (4) must match the size of tensor b (s94: hint = 10) at non-singleton dimension 1)')

from user code:
   File "test_error_messages.py", line N, in fn
    return x + y""",
            post_munge=post_munge,
        )

    def test_hop_side_effect_error_includes_hop_context(self):
        # Test that graph breaks inside HOPs include the HOP context
        stack = []

        def fn(x):
            stack.append(1)  # Mutation outside checkpoint scope
            return x.sin()

        def model(x):
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(model, backend="eager", fullgraph=True)(
                torch.rand(4)
            ),
            """\
HOP: Unsafe side effect
  Higher Order Operator: torch.utils.checkpoint.checkpoint
  Explanation: Mutating a variable from outside the scope of this HOP is not supported.
  Hint: If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a side effect in forward method. Eager activation checkpointing replays that side-effect while recomputing the forward in the backward. If you are ok with side-effect not replayed in the backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`

  Developer debug context: Attempted to mutate ListVariable(length=0)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0067.html

from user code:
   File "test_error_messages.py", line N, in model
    return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
  File "test_error_messages.py", line N, in fn
    stack.append(1)  # Mutation outside checkpoint scope""",
        )

    @make_logging_test(graph_breaks=True)
    def test_hop_side_effect_error_includes_hop_context_fullgraph_false(self, records):
        # Test that graph breaks inside HOPs include the HOP context
        # even with fullgraph=False (checkpoint allows graph breaks)
        stack = []

        def fn(x):
            stack.append(1)  # Mutation outside checkpoint scope
            return x.sin()

        def model(x):
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=True)

        def post_munge(s):
            # deal with CALL_FUNCTION_KW/CALL_KW
            return re.sub(
                r"attempting to trace (CALL_FUNCTION_KW|CALL_KW):.*$",
                "attempting to trace CALL: a function call, e.g. f(x, y):",
                s,
                flags=re.MULTILINE,
            )

        torch.compile(model, backend="eager", fullgraph=False)(torch.rand(4))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            post_munge(
                munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

HOP: Unsafe side effect
  Higher Order Operator: torch.utils.checkpoint.checkpoint
  Explanation: Mutating a variable from outside the scope of this HOP is not supported.
  Hint: If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a side effect in forward method. Eager activation checkpointing replays that side-effect while recomputing the forward in the backward. If you are ok with side-effect not replayed in the backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`

  Developer debug context: Attempted to mutate ListVariable(length=0)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0067.html

User code traceback:
  File "test_error_messages.py", line N, in test_hop_side_effect_error_includes_hop_context_fullgraph_false
    torch.compile(model, backend="eager", fullgraph=False)(torch.rand(4))
  File "test_error_messages.py", line N, in model
    return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=True)
  File "test_error_messages.py", line N, in fn
    stack.append(1)  # Mutation outside checkpoint scope
""",
        )

    def test_nested_hop_side_effect_error(self):
        # Test that only the innermost HOP context is shown (checkpoint, not wrap)
        stack = []

        def fn(x):
            stack.append(1)  # Mutation outside checkpoint scope
            return x.sin()

        def inner(x):
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)

        def model(x):
            return torch.ops.higher_order.wrap(inner, x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(model, backend="eager", fullgraph=True)(
                torch.rand(4)
            ),
            """\
HOP: Unsafe side effect
  Higher Order Operator: torch.utils.checkpoint.checkpoint
  Explanation: Mutating a variable from outside the scope of this HOP is not supported.
  Hint: If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a side effect in forward method. Eager activation checkpointing replays that side-effect while recomputing the forward in the backward. If you are ok with side-effect not replayed in the backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`

  Developer debug context: Attempted to mutate ListVariable(length=0)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0067.html

from user code:
   File "test_error_messages.py", line N, in model
    return torch.ops.higher_order.wrap(inner, x)
  File "test_error_messages.py", line N, in inner
    return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
  File "test_error_messages.py", line N, in fn
    stack.append(1)  # Mutation outside checkpoint scope""",
        )

    def test_cond_with_graph_break_shows_hop_context(self):
        # Test that torch.cond graph breaks include HOP context
        def true_fn(x):
            torch._dynamo.graph_break()
            return x.sin()

        def false_fn(x):
            return x.cos()

        def fn(x):
            return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        from torch._dynamo.exc import UncapturedHigherOrderOpError

        self.assertExpectedInlineMunged(
            UncapturedHigherOrderOpError,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

Call to `torch._dynamo.graph_break()`
  Higher Order Operator: torch.cond
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.cond(x.sum() > 0, true_fn, false_fn, [x])
  File "test_error_messages.py", line N, in true_fn
    torch._dynamo.graph_break()""",
        )

    @make_logging_test(graph_breaks=True)
    def test_cond_with_generator_gb_skip(self, records):
        def generator():
            yield 1
            yield 2
            torch._dynamo.graph_break()
            yield 4

        def true_fn(x):
            with GenericCtxMgr():
                l = list(generator())
            return x.sin() + sum(l)

        def false_fn(x):
            return x.cos()

        @torch.compile(backend="eager")
        def fn(x):
            return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        from torch._dynamo.exc import UncapturedHigherOrderOpError

        self.assertExpectedInlineMunged(
            UncapturedHigherOrderOpError,
            lambda: fn(torch.ones(3)),
            """\
This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

Call to `torch._dynamo.graph_break()`
  Higher Order Operator: torch.cond
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

Skipping frame due to graph break in a generator's next() call.

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.cond(x.sum() > 0, true_fn, false_fn, [x])
  File "test_error_messages.py", line N, in true_fn
    l = list(generator())
  File "test_error_messages.py", line N, in generator
    torch._dynamo.graph_break()""",
        )

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                UncapturedHigherOrderOpError,
                lambda: fn(torch.ones(3)),
                """\
This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

Call to `torch._dynamo.graph_break()`
  Higher Order Operator: torch.cond
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.cond(x.sum() > 0, true_fn, false_fn, [x])
  File "test_error_messages.py", line N, in true_fn
    l = list(generator())
  File "test_error_messages.py", line N, in generator
    torch._dynamo.graph_break()""",
            )

    @make_logging_test()
    @torch._dynamo.config.patch(recompile_limit=1)
    def test_recompile_limit_hit_message(self, records):
        @torch.compile(backend="eager", dynamic=False)
        def fn(x, n):
            return x + n

        def outer(x):
            for i in range(2):
                x = fn(x, i)
            return x

        outer(torch.ones(3))

        def post_munge(s):
            # Remove user stack trace section that appears in recompile limit messages
            s = re.sub(
                r"\nUser stack trace:.*?(?=\nTo log all)", "", s, flags=re.DOTALL
            )
            return re.sub(
                r"# \S+/test_error_messages\.py", "# test_error_messages.py", s
            )

        self.assertExpectedInline(
            post_munge(
                munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
torch._dynamo hit config.recompile_limit (1)
   function: 'fn' (test_error_messages.py:N)
   last reason: 0/0: n == 0                                                   # return x + n  # test_error_messages.py:N in fn
To log all recompilation reasons, use TORCH_LOGS="recompiles".
To diagnose recompilation issues, see https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html""",
        )

        torch.compiler.reset()

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: outer(torch.ones(3)),
                """\
Dynamo recompile limit exceeded
  Explanation: Dynamo attempted to recompile the code object too many times, exceeding the recompile_limit cache size limit (currently set to 1). Excessive recompilations can degrade performance due to the compilation overhead of each recompilation.
  Hint: To monitor recompilations, enable TORCH_LOGS=recompiles. If recompilations are expected, consider increasing torch._dynamo.config.recompile_limit to an appropriate value.
  Hint: See https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html for tips on dealing with recompilations.

  Developer debug context: Limit type: recompile_limit

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0039.html""",
            )


class NestedGraphBreakLoggingTests(
    LoggingTestCase, torch._dynamo.test_case.TestCaseWithNestedGraphBreaks
):
    @make_logging_test(graph_breaks=True)
    def test_nested_generic_ctx_mgr(self, records):
        def inner():
            with GenericCtxMgr():
                with GenericCtxMgr():
                    with GenericCtxMgr():
                        torch._dynamo.graph_break()

        def fn():
            with GenericCtxMgr():
                with GenericCtxMgr():
                    inner()

        torch.compile(fn, backend="eager")()
        self.assertEqual(len(records), 2)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

User code traceback:
  File "test_error_messages.py", line N, in test_nested_generic_ctx_mgr
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    inner()
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )
        self.assertExpectedInline(
            munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break (user stack suppressed due to duplicate graph break) in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html""",
        )

    @make_logging_test(graph_breaks=True)
    def test_skipped_frame_with_verbose_traceback_nested(self, records):
        global f1, f2, f3

        class GenericCtxMgr:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

        def f1(x):
            with GenericCtxMgr():
                torch._dynamo.graph_break()
                return x + 1

        def f2(x):
            return f1(x + 2)

        def f3(x):
            return f2(x + 3)

        torch.compile(f3, backend="eager")(torch.randn(3))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

User code traceback:
  File "test_error_messages.py", line N, in test_skipped_frame_with_verbose_traceback_nested
    torch.compile(f3, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in f3
    return f2(x + 3)
  File "test_error_messages.py", line N, in f2
    return f1(x + 2)
  File "test_error_messages.py", line N, in f1
    torch._dynamo.graph_break()
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_skip_frame_in_loop_message_nested(self, records):
        global f1, f2, f3

        class GenericCtxMgr:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

        def f1(x):
            for i in range(2):
                with GenericCtxMgr():
                    if x.sum() > 0:
                        x = x + 1
            return x

        def f2(x):
            return f1(x + 4)

        def f3(x):
            return f2(x + 5)

        result = torch.compile(f3, backend="eager")(torch.randn(3))  # noqa: F841
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

User code traceback:
  File "test_error_messages.py", line N, in test_skip_frame_in_loop_message_nested
    result = torch.compile(f3, backend="eager")(torch.randn(3))  # noqa: F841
  File "test_error_messages.py", line N, in f3
    return f2(x + 5)
  File "test_error_messages.py", line N, in f2
    return f1(x + 4)
  File "test_error_messages.py", line N, in f1
    if x.sum() > 0:
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_try_block_with_graph_break_suppression(self, records):
        global inner, middle_with_try, outer

        def inner(x):
            result = x + 1
            torch._dynamo.graph_break()
            return result + 1

        def middle_with_try(x):
            try:
                return inner(x)
            except Exception:
                pass
            return x

        def outer(x):
            return middle_with_try(x)

        with torch._dynamo.config.patch(nested_graph_breaks=True, verbose=False):
            torch.compile(outer, backend="eager")(torch.ones(3))

        full_messages = [
            r for r in records if "Graph break in user code" in r.getMessage()
        ]
        suppressed_messages = [
            r
            for r in records
            if "user stack suppressed due to duplicate" in r.getMessage()
        ]

        self.assertEqual(
            len(full_messages),
            1,
            f"Expected 1 full graph break message, got {len(full_messages)}",
        )
        self.assertEqual(
            len(suppressed_messages),
            2,
            f"Expected 2 suppressed messages, got {len(suppressed_messages)}",
        )

        self.assertExpectedInline(
            munge_exc(full_messages[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_try_block_with_graph_break_suppression
    torch.compile(outer, backend="eager")(torch.ones(3))
  File "test_error_messages.py", line N, in outer
    return middle_with_try(x)
  File "test_error_messages.py", line N, in middle_with_try
    return inner(x)
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )

        # graph break in middle_with_try as top frame
        self.assertExpectedInline(
            munge_exc(
                suppressed_messages[0].getMessage(), suppress_suffix=True, skip=0
            ),
            """\
Graph break (user stack suppressed due to duplicate graph break) in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html""",
        )

        # graph break in inner as top frame
        self.assertExpectedInline(
            munge_exc(
                suppressed_messages[1].getMessage(), suppress_suffix=True, skip=0
            ),
            """\
Graph break (user stack suppressed due to duplicate graph break) in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html""",
        )

    @make_logging_test(graph_breaks=True)
    def test_nested_graph_break_different_call_sites_not_suppressed(self, records):
        global inner, outer

        def inner(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        @torch.compile(backend="eager")
        def outer(x):
            x = inner(x + 4) + 8
            return inner(x) + 16

        with torch._dynamo.config.patch(nested_graph_breaks=True, verbose=False):
            outer(torch.ones(3))

        self.assertEqual(
            len(records),
            2,
            f"Expected 2 graph break messages (one per call site), got {len(records)}",
        )

        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_nested_graph_break_different_call_sites_not_suppressed
    outer(torch.ones(3))
  File "test_error_messages.py", line N, in outer
    x = inner(x + 4) + 8
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )

        self.assertExpectedInline(
            munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_nested_graph_break_different_call_sites_not_suppressed
    outer(torch.ones(3))
  File "test_error_messages.py", line N, in outer
    return inner(x) + 16
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
