# Owner(s): ["module: dynamo"]
# flake8: noqa
import copy
import dataclasses
import unittest
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch._dynamo as torchdynamo
from functorch.experimental.control_flow import cond, map
from torch import Tensor
from torch.export import (
    Constraint,
    Dim,
    dynamic_dim,
    export,
    unflatten,
)
from torch.export._trace import (
    _export_to_torch_ir,
    DEFAULT_EXPORT_DYNAMO_CONFIG,
)
from torch._export import capture_pre_autograd_graph
from torch._export.pass_base import _ExportPassBase
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
from torch._subclasses import FakeTensorMode
from torch.export import Constraint, Dim
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests
from torch._dynamo.test_case import TestCase
from torch.utils._pytree import (
    LeafSpec,
    tree_flatten,
    tree_map,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_loads,
)
try:
    from . import testing
except ImportError:
    import testing
# The following import pattern matters as `test_export.export` is patched
# in other files (like test_export_nonstrict.py). `torch.export.export`
# will invalidate the patch.
from torch.export import export


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    @testing.expectedFailureNonStrict
    def test_export_inline_constraints(self):

        def f(x):
            b = x.item()
            torch._constrain_as_size(b)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = export(f, inp)
        res = gm(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    def test_export_constraints_error(self):
        def invalid_input_conflict_with_input_constraints(x):
            return x + 1

        inp = torch.zeros([3])
        dim_x = torch.export.Dim("dim_x", min=6)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "not in range"):
            torch.export.export(
                invalid_input_conflict_with_input_constraints,
                (inp,),
                dynamic_shapes={"x": {0: dim_x}},
            )

        def conflicting_constraints(x):
            b = x.item()
            torch._constrain_as_size(b)
            torch._constrain_as_value(b, min=4, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ep = torch.export.export(conflicting_constraints, inp)

        with self.assertRaisesRegex(
            RuntimeError, r"is outside of inline constraint \[4, 5\]"
        ):
            ep(torch.tensor([3]))

    @testing.expectedFailureNonStrict
    def test_export_assume_static_by_default(self):
        def branch_on_shape(x: torch.Tensor):
            if x.shape[0] == 4:
                return x + 1
            else:
                return x

        inp = (torch.rand(4, 5),)

        # Being able to export means shape is preserved as static
        export(branch_on_shape, inp)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):

    def _test_export_same_as_eager(self, f, args, kwargs=None):
        kwargs = kwargs or {}
        exported_program = export(f, args, kwargs)
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        self.assertEqual(exported_program(*args, **kwargs), f(*args, **kwargs))
        self.assertEqual(exported_program(*args, **reversed_kwargs), f(*args, **reversed_kwargs))

    @testing.expectedFailureNonStrict
    def test_basic(self):
        def f(x, y):
            return x[0] + y

        inp = ([torch.ones(1, 3)], torch.ones(1, 3))
        self._test_export_same_as_eager(f, inp)

    @testing.expectedFailureRetraceability
    def test_external_call_non_strict_real_tensor(self):
        class ExternalMethod:
            def add(self, x):
                return x + x

        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.external_add = ExternalMethod().add

            def forward(self, x):
                return self.external_add(x)

        f = Basic()
        args = (torch.randn(1, 3), )
        ep = export(f, args, strict=False)
        self.assertEqual(ep(*args), f(*args))

    @testing.expectedFailureRetraceability
    def test_basic_non_strict_real_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(1, 3))

            def forward(self, x, y):
                return x[0] + y - self.param

        f = Basic()
        args = ([torch.randn(1, 3)], torch.randn(1, 3))
        ep = export(f, args, strict=False)
        self.assertEqual(ep(*args), f(*args))

    @testing.expectedFailureRetraceability
    def test_basic_non_strict_fake_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x, y):
                return x[0] + y - self.param

        fake_mode = FakeTensorMode()
        f = Basic()
        with fake_mode:
            args = ([torch.empty(3, 2)], torch.empty(3, 2))
        ep = export(f, args, strict=False)
        inputs = ([torch.randn(3, 2)], torch.randn(3, 2))
        self.assertEqual(ep(*inputs), f(*inputs))

    def test_non_strict_dynamic_shapes(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("u", torch.ones(1))
                self.register_buffer("v", torch.ones(1))

            def forward(self, x, ys, zs, c):
                y = ys[0] + ys[1] + zs["a"] + zs["b"]
                self.v.add_(3)
                w = self.u - self.v
                if x.shape[0] < 3 and c.shape[0] != 4:
                    return x + w, x + y
                else:
                    return x - w, x - y

        foo = Foo()

        inp = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            [{0: dim}, {0: dim}],
            {"a": {0: dim}, "b": {0: dim}},
            None,
        )

        ep_ns = torch.export.export(foo, inp, dynamic_shapes=dynamic_shapes, strict=False)

        bad_runtime_inp1 = (
            torch.ones(6),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected input arg3_1.shape\[0\] to be equal to 6, but got 5"
        ):
            ep_ns(*bad_runtime_inp1)

        bad_runtime_inp2 = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(6),
        )
        with self.assertRaisesRegex(RuntimeError, "Expected input arg7_1.shape\[0\] to be equal to 4, but got 6"):
            ep_ns(*bad_runtime_inp2)

        good_runtime_inp = (
            torch.ones(7),
            [torch.zeros(7), torch.ones(7)],
            {"a": torch.zeros(7), "b": torch.ones(7)},
            torch.ones(4),
        )
        ep_ns(*good_runtime_inp)

        bad_example_inp = (
            torch.ones(2),
            [torch.zeros(2), torch.ones(2)],
            {"a": torch.zeros(2), "b": torch.ones(2)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            "2 not in range.*3,",
        ):
            ep_ns = torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_non_strict_dynamic_shapes_suggested_fixes(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, c):
                if x.shape[0] <= 6:
                    return x + 1, c + 2
                else:
                    return x - 1, c - 2

        foo = Foo()

        bad_example_inp = (
            torch.ones(5),
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            None,
        )

        with self.assertRaisesRegex(
            torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            "Constraints violated \\(dim\\)!(.*\n)*.*"
            "Not all values of dim.*satisfy the generated guard(.*\n)*.*"
            "Suggested fixes:(.*\n)*.*"
            "dim = Dim\\('dim', min=3, max=6\\)",
        ):
            torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.shape[0]
                if z > 2:
                    return x.cos()
                else:
                    return x.sin()

        with self.assertRaisesRegex(
            (torchdynamo.exc.UserError, torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode),
            "trying to get a value out of symbolic int"
        ):
            _ = export(M(), (torch.tensor([2, 3, 5]),))

    @testing.expectedFailureNonStrict
    def test_if_functional(self):
        def foo(x):
            z = x + 4
            z.add_(4)
            y = z.view(x.shape)
            return x.cos() + y.cos()

        gm = export(foo, (torch.tensor([2, 3, 5]),))

        view_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # No more inplace mutation
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph."
                )
            if node.op == "call_function" and node.target == torch.ops.aten.view.default:
                view_count += 1

        # There should be nonzero view nodes in the graph
        self.assertTrue(view_count > 0)

    def test_export_mod_constraints(self):
        class BasicDynamiShapeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.shape[0] - 1, -1)

        m = BasicDynamiShapeModel()
        a = torch.randn(3, 4)
        dim0_x = torch.export.Dim("dim0_x", min=3)
        dim1_x = torch.export.Dim("dim1_x")
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Specializations unexpectedly required"
                ".*\n.*\\[0\\] must be specialized to 3.*guards.*too complex"
                ".*\n.*\\[1\\] must be specialized to 4.*guards.*too complex"
            ),
        ):
            torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)
        em = torch.export.export(m, (a,))
        x = torch.randn(3, 5)
        with self.assertRaisesRegex(RuntimeError, "shape\[1\] to be equal to 4, but got 5"):
            em(x)

    def test_not_correct_dim(self):
        def f(x):
            return x.cos()

        def g(x):
            return x + 4

        inp_for_f = torch.tensor(5)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Cannot mark 0-dimension tensors to be dynamic"):
            constraints = [dynamic_dim(inp_for_f, 0)]

        inp_for_f_mul_dim = torch.ones(5, 5)
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "Expected the dimension passed to dynamic_dim to be in the range \\[0:1\\]"
        ):
            constraints = [dynamic_dim(inp_for_f_mul_dim, 2)]

        inp_for_g = 4
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Expected tensor as input to dynamic_dim"):
            constraints = [dynamic_dim(inp_for_g, 0)]

    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_map(self):
        def list_tensor_map(xs, y, z):
            def body(x, y, z):
                return x + y + z

            return map(body, xs, y, z)

        inps = (torch.ones(6, 4), torch.tensor(5), torch.tensor(4))
        self._test_export_same_as_eager(list_tensor_map, inps)

    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_export_func_with_kwargs(self):
        def kw_func(arg1, arg2, kw1, kw2):
            return arg1 + arg2, kw1 + kw2

        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs = {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_export_func_with_pytree_kwargs(self):
        def kw_func(arg1, arg2, a, b):
            return arg1 + a["kw1"] + b[0], arg2 + a["kw2"] + b[1]

        args = (torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"a": {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}, "b": [torch.ones(2, 3), torch.ones(3, 4)]}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureSerDer
    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_export_func_with_default_kwargs(self):
        def kw_func(arg1, arg2, a, b=1):
            return arg1 + arg2, a["kw1"] + a["kw2"] + b

        def kw_func2(arg1, arg2, a=1, b=2):
            return arg1 + a, arg2 + b


        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs1 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}}
        kwargs2 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}, "b": 2}
        self._test_export_same_as_eager(kw_func, args, kwargs1)
        self._test_export_same_as_eager(kw_func, args, kwargs2)
        kwargs3 = {"b": 1}
        self._test_export_same_as_eager(kw_func2, args, kwargs3)

    @testing.expectedFailureNonStrict
    def test_export_func_with_var_postional_args(self):
        def kw_func(arg1, arg2, *args):
            return arg1 + args[0], arg2 + args[1]

        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        self._test_export_same_as_eager(kw_func, args)

    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_export_func_with_keyword_only_args(self):
        def kw_func(arg1, arg2, *args, kw1, kw2):
            return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_export_func_with_var_keyword_args(self):
        def kw_func(arg1, arg2, *args, kw1, kw2, **kwargs):
            return arg1 + args[0] + kw1 + kwargs["kw3"], arg2 + args[1] + kw2 + kwargs["kw4"]

        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4), "kw3": torch.ones(2, 3), "kw4": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_export_func_with_var_keyword_pytree_args(self):
        def kw_func(arg1, arg2, *args, kw1, kw2, **kwargs):
            return arg1 + arg2[0][0] + args[0] + kw1[0] + kwargs["kw3"][0], arg2[1] + args[1] + kw2 + kwargs["kw4"]

        args = (torch.ones(2, 3), [(torch.ones(2, 3), ), torch.ones(3, 4)], torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": (torch.ones(2, 3), ), "kw2": torch.ones(3, 4),
                  "kw3": (torch.ones(2, 3), torch.ones(3, 4)), "kw4": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureSerDer
    @testing.expectedFailureNonStrict
    def test_linear_conv(self):

        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                x_conv = self.conv(x)
                x_linear = self.linear(x_conv)
                return x_linear.cos()

        ep = export(Foo(), (torch.randn(20, 16, 50, 100),))
        for node in ep.graph.nodes:
            if (
                node.op == "placeholder" and
                node.name in ep.graph_signature.inputs_to_buffers or
                node.name in ep.graph_signature.inputs_to_parameters
            ):
                self.assertTrue("source_fn_stack" in node.meta)
                self.assertTrue("nn_module_stack" in node.meta)

    def test_export_api_with_dynamic_shapes(self):
        from torch.export import Dim, dims, export

        # pass dynamic shapes of inputs [args]
        def foo(x, y):
            return torch.matmul(x, y)

        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(foo, inputs, dynamic_shapes={k: {0: batch} for k in ["x", "y"]})
        self.assertEqual(efoo(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [kwargs]
        def foo(x, y):
            return torch.matmul(x, y)

        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        efoo = export(
            foo, inputs, kwinputs, dynamic_shapes={k: {0: batch} for k in ["x", "y"]}
        )
        self.assertEqual(efoo(*inputs, **kwinputs).shape, foo(*inputs, **kwinputs).shape)

        # pass dynamic shapes of inputs [partial, error]
        def foo(x, y):
            return torch.matmul(x, y)

        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(batch\\)!(.*\n)*.*"
                "batch was inferred to be a constant(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = None  # 10"
            ),
        ):
            export(foo, inputs, kwinputs, dynamic_shapes={"x": {0: batch}, "y": None})

        # pass dynamic shapes of inputs [module]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(foo, inputs, dynamic_shapes={"x": {0: batch}, "y": {0: batch}})
        self.assertEqual(efoo(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [bounds, mostly shared]
        def foo(x, y):
            return torch.matmul(x, y)

        inputs = (torch.randn(10, 3, 3), torch.randn(10, 3, 3))
        batch = Dim("batch", min=8, max=64)
        size = Dim("size")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={
                "x": (batch, size, size),
                "y": (batch, size, size),
            },
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, s1, s1])", "torch.Size([s0, s1, s1])"],
        )
        self.assertEqual(efoo(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [multiple, mostly distinct]
        def foo(x, y):
            return torch.matmul(x, y)

        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, s1, s2])", "torch.Size([s0, s2, s5])"],
        )
        self.assertEqual(efoo(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [dict]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs["x"], inputs["y"])

        foo = Foo()
        inputs = ({"x": torch.randn(10, 2, 3), "y": torch.randn(10, 3, 4)},)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": {k: {0: batch} for k in ["x", "y"]}}
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )
        self.assertEqual(efoo(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [list]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs[0], inputs[1])

        foo = Foo()
        inputs = ((torch.randn(10, 2, 3), torch.randn(10, 3, 4)),)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": [{0: batch} for _ in range(2)]}
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )
        self.assertEqual(efoo(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [dataclass]
        @dataclass
        class DataClass:
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(DataClass)

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs.a, inputs.b)

        foo = Foo()
        inputs = (DataClass(a=torch.randn(10, 2, 3), b=torch.randn(10, 3, 4)),)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": DataClass(a={0: batch}, b={0: batch})}
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )

        # pass dynamic shapes of inputs [distinct, error]
        def foo(x, y):
            return torch.matmul(x, y)

        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, K2, N = dims("batch", "M", "K1", "K2", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(K2\\)!(.*\n)*.*"
                "K2.*and.*K1.*must always be equal(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "K2 = K1"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, K2, N)},
            )

        # pass dynamic shapes of inputs [specialized, error]
        def foo(x, y):
            return torch.matmul(x, y)

        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, N = dims("batch", "M", "K1", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(K1\\)!(.*\n)*.*"
                "K1 was inferred to be a constant(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "K1 = None  # 3"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, None, N)},
            )

        # pass dynamic shapes of inputs [guards, error]
        def foo(x, y):
            if x.shape[0] < 16 and y.shape[1] % 3 == 0:
                return torch.matmul(x, y)
            else:
                return x + y

        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(batch\\)!(.*\n)*.*"
                "Not all values of batch.*satisfy the generated guard(.*\n)*.*"
                "Specializations unexpectedly required \\(K\\)!(.*\n)*.*"
                "K.*specialized.*because the guards generated for it are too complex(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = Dim\\('batch', max=15\\)(.*\n)*.*"
                "K = None  # 3"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
            )

    def test_dynamic_shapes_spec_with_pytree(self):
        from torch.export import Dim, export
        from torch.utils._pytree import tree_map

        inputs = {
            "tensor": torch.randn(3),
            "dict_of_tensors": {k: torch.randn(3) for k in ["A", "B", "C", "D"]},
            "list_of_tensors": [torch.randn(3) for _ in range(4)],
        }

        batch = Dim("batch")
        # uniformly specify dynamic shapes for all inputs
        spec = tree_map(lambda x: {0: batch}, inputs)

        def foo(inputs):
            return (
                inputs["tensor"]
                + inputs["dict_of_tensors"]["A"]
                + inputs["list_of_tensors"][0]
            )

        ep = export(foo, (inputs,), dynamic_shapes={"inputs": spec})
        input_shapes = [
            str(node.meta["val"].shape)
            for node in ep.graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(len(input_shapes), 9)
        self.assertTrue(all(shape == "torch.Size([s0])" for shape in input_shapes))

    @testing.expectedFailureNonStrict
    def test_error_does_not_reference_eager_fallback(self):
        def fn_ddo(x):
            y = x.nonzero()
            z = y.shape[0]
            if z > 2:
                return x.cos()
            else:
                return x.sin()

        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            r"^(?!.*fall back to eager).*"
        ):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),))

    def test_pytree_register_data_class(self):

        @dataclass
        class MyDataClass:
            x: int
            y: int
            z: int = None

        dt = MyDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertTrue(spec, LeafSpec())
        self.assertTrue(len(flat) == 1)

        register_dataclass_as_pytree_node(MyDataClass, serialized_type_name="test_pytree_register_data_class.MyDataClass")

        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyDataClass,
                (
                    MyDataClass,
                    ['x', 'y'],
                    ['z']
                ),
                [LeafSpec(), LeafSpec()]
            )
        )
        self.assertEqual(flat, [3, 4])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

        @dataclass
        class MyOtherDataClass:  # the pytree registration don't allow registering the same class twice
            x: int
            y: int
            z: int = None

        # Override the registration with keep none fields
        register_dataclass_as_pytree_node(MyOtherDataClass, return_none_fields=True, serialized_type_name="test_pytree_regster_data_class.MyOtherDataClass")

        dt = MyOtherDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyOtherDataClass,
                (
                    MyOtherDataClass,
                    ['x', 'y', 'z'],
                    [],
                ),
                [LeafSpec(), LeafSpec(), LeafSpec()]
            )
        )
        self.assertEqual(flat, [3, 4, None])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyOtherDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_register_nested_data_class(self):

        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            xy: Inner
            ab: Inner

        xy = Inner(1, 2)
        ab = Inner(3, 4)
        dt = Outer(xy, ab)
        inp = {"dt1": (dt, ({},)), "dt2": ((torch.ones(1),), dt)}

        register_dataclass_as_pytree_node(Inner, serialized_type_name="test_pytree_register_nested_data_class.Inner")
        register_dataclass_as_pytree_node(Outer, serialized_type_name="test_pytree_register_nested_data_class.Outer")

        flat, spec = tree_flatten(inp)
        self.assertEqual(flat, [1, 2, 3, 4, torch.ones(1), 1, 2, 3, 4])

        unflat = tree_unflatten(flat, spec)
        self.assertEqual(unflat, inp)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_param_util(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        ep = export(Basic(), (torch.randn(5, 10),))
        num_params = 0
        params = []
        for node in ep.graph.nodes:
            if is_param(ep, node):
                num_params += 1
                params.append(get_param(ep, node))
        self.assertEqual(num_params, 2)
        self.assertEqual(params[0].shape, [1, 10])  # weight
        self.assertEqual(params[1].shape, [1])  # bias

    def test_buffer_util(self):
        ep = export(torch.nn.BatchNorm2d(100, affine=False), (torch.ones(20, 100, 35, 45), ))
        num_buffer = 0
        buffer = []

        for node in ep.graph.nodes:
            if is_buffer(ep, node):
                num_buffer += 1
                buffer.append(get_buffer(ep, node))
        self.assertEqual(num_buffer, 3)

        self.assertEqual(buffer[0].shape, torch.Size([100]))  # running_mean
        self.assertEqual(buffer[1].shape, torch.Size([100]))  # running_var
        self.assertEqual(buffer[2].shape, torch.Size([]))  # num_batches_tracked


    @testing.expectedFailureNonStrict
    def test_export_dynamo_config(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.lstm(inputs)


        config = DEFAULT_EXPORT_DYNAMO_CONFIG
        mod = MyModule()

        @contextmanager
        def _patch_config(kwargs):
            orig_config_dict = dataclasses.asdict(config)

            try:
                for k, v in kwargs.items():
                    setattr(config, k, v)
                yield
            finally:
                for k, v in orig_config_dict.items():
                    setattr(config, k, v)

        inp = (torch.rand(5, 4), )
        exported_program = export(mod, inp)

        with _patch_config({"allow_rnn": False}):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "TorchDynamo purposely graph breaks on RNN, GRU, LSTMs"
            ):
                _ = export(mod, inp)

    @testing.expectedFailureSerDer
    @testing.expectedFailureNonStrict
    def test_module(self):

        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a, b = x
                a_conv = self.conv(a)
                a_linear = self.linear(a_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return (a_linear.cos() + b_linear.sin(), a_linear.sin() + b_linear.cos())

        inp_container = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        self.assertTrue(torch.allclose(ep(*inp_test)[0], ep_rexported(*inp_test)[0]))
        self.assertTrue(torch.allclose(ep(*inp_test)[1], ep_rexported(*inp_test)[1]))

    @testing.expectedFailureSerDer
    @testing.expectedFailureNonStrict
    def test_module_with_dict_container_inp_out(self):

        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a1, a2 = x["a"]
                b = x["b"]
                a1_conv = self.conv(a1)
                a1_linear = self.linear(a1_conv)
                a2_conv = self.conv(a2)
                a2_linear = self.linear(a2_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return {"a": a1_linear.cos() + b_linear.sin(), "b": a2_linear.sin() + b_linear.cos()}

        inp_container = ({"a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)), "b": torch.randn(20, 16, 50, 100)},)

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = ({"a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)), "b": torch.randn(20, 16, 50, 100)},)

        self.assertTrue(torch.allclose(ep(*inp_test)["a"], ep_rexported(*inp_test)["a"]))
        self.assertTrue(torch.allclose(ep(*inp_test)["b"], ep_rexported(*inp_test)["b"]))

    @testing.expectedFailureNonStrict
    def test_args_type_checked(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.rand(2, 2)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "to be a tuple"):
            # Intentionally not wrapping `inp` in a tuple to trigger the error
            _ = export(M(), inp)

    @testing.expectedFailureNonStrict
    def test_constrain_value_with_no_default(self):
        def fn(x, y):
            n = x.max().item()
            torch._constrain_as_value(n)
            return y + n

        ep = export(fn, (torch.randint(3, 5, (2, 2)), torch.randint(3, 5, (2, 3))))
        test_inp = (torch.randint(3, 5, (2, 2)), torch.randint(3, 5, (2, 3)))
        self.assertTrue(torch.allclose(ep(*test_inp), fn(*test_inp)))

    def test_decomp_batch_norm_functional_predispatch(self):
        class ConvBatchnorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return (x,)

        mod = ConvBatchnorm()
        mod.eval()
        inp = torch.randn(1, 1, 3, 3)

        gm = capture_pre_autograd_graph(mod, (inp,), _functional_pre_dispatch_IR=True)
        self.assertExpectedInline(str(gm.code).strip(), """\
def forward(self, arg_0):
    l_x_, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    conv_weight = self.conv_weight
    conv_bias = self.conv_bias
    bn_weight = self.bn_weight
    bn_bias = self.bn_bias
    bn_running_mean = self.bn_running_mean
    bn_running_var = self.bn_running_var
    conv2d = torch.ops.aten.conv2d.default(l_x_, conv_weight, conv_bias);  l_x_ = conv_weight = conv_bias = None
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, 0.1, 1e-05);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    getitem = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
    return pytree.tree_unflatten((getitem,), self._out_spec)""")

        mod.train()
        gm_train = capture_pre_autograd_graph(mod, (inp,), _functional_pre_dispatch_IR=True)
        self.assertExpectedInline(str(gm_train.code).strip(), """\
def forward(self, arg_0):
    l_x_, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    conv_weight = self.conv_weight
    conv_bias = self.conv_bias
    bn_weight = self.bn_weight
    bn_bias = self.bn_bias
    bn_running_mean = self.bn_running_mean
    bn_running_var = self.bn_running_var
    bn_num_batches_tracked = self.bn_num_batches_tracked
    conv2d = torch.ops.aten.conv2d.default(l_x_, conv_weight, conv_bias);  l_x_ = conv_weight = conv_bias = None
    add = torch.ops.aten.add.Tensor(bn_num_batches_tracked, 1)
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, True, 0.1, 1e-05);  conv2d = bn_weight = bn_bias = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    copy__default = torch.ops.aten.copy_.default(bn_running_mean, getitem_3);  bn_running_mean = getitem_3 = None
    copy__default_1 = torch.ops.aten.copy_.default(bn_running_var, getitem_4);  bn_running_var = getitem_4 = None
    copy__default_2 = torch.ops.aten.copy_.default(bn_num_batches_tracked, add);  bn_num_batches_tracked = add = None
    return pytree.tree_unflatten((getitem,), self._out_spec)""")



    @testing.expectedFailureNonStrict
    def test_constrain_value_with_symfloat(self):
        def fn(x, y):
            n = x.max().item()
            torch._constrain_as_value(n)
            return y + n

        with self.assertRaisesRegex(torch._dynamo.exc.TorchRuntimeError, "Constraining SymFloat or Symbool is nyi"):
            _ = export(fn, (torch.rand(2, 2), torch.rand(2, 3)))

    @testing.expectedFailureNonStrict
    def test_constrain_size_in_eager(self):
        def fn(x, y):
            n = x.max().item()
            torch._constrain_as_size(n)
            return y + n

        ep = export(fn, (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3))))
        test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
        self.assertTrue(torch.allclose(ep(*test_inp), fn(*test_inp)))

    @testing.expectedFailureNonStrict
    def test_constrain_size_with_constrain_value(self):
        def fn(x, y):
            n = x.max().item()
            torch._constrain_as_value(n, 2, 10)
            torch._constrain_as_size(n)
            return y + n

        with self.assertRaisesRegex(RuntimeError, r"Invalid value range for 1 between \[2, 10\]."):
            _ = fn(torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))

        ep = export(fn, (torch.randint(3, 4, (2, 2)), torch.randint(3, 5, (2, 3))))
        with self.assertRaisesRegex(RuntimeError, "is outside of inline constraint"):
            test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
            _ = ep(*test_inp)

    @testing.expectedFailureNonStrict
    def test_constrain_size_with_various_cases(self):

        def case_1(x, y):
            n = x.item()
            torch._constrain_as_size(n, min=0)
            return y.sum() + torch.ones(n, 5).sum()

        def case_2(x, y):
            n = x.item()
            torch._constrain_as_size(n, min=0, max=6)
            return y.sum() + torch.ones(n, 5).sum()

        def case_3(x, y):
            n = x.item()
            torch._constrain_as_size(n, min=0, max=1)
            return y.sum() + torch.ones(n, 5).sum()

        def case_4(x, y):
            n = x.item()
            torch._constrain_as_size(n, min=2)
            return y.sum() + torch.ones(n, 5).sum()

        def case_5(x, y):
            n = x.item()
            torch._constrain_as_size(n, min=1)
            return y.sum() + torch.ones(n, 5).sum()

        ep = export(case_1, (torch.tensor(1), torch.ones(4, 5)))

        with self.assertRaisesRegex(RuntimeError, r"Invalid value range for -1 between"):
            _ = case_1(torch.tensor(-1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep(torch.tensor(1), torch.ones(4, 5)),
                case_1(torch.tensor(1), torch.ones(4, 5)),
            )
        )

        ep = export(case_2, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(RuntimeError, r"Invalid value range for 7 between"):
            _ = case_2(torch.tensor(7), torch.randn(4, 5))

        with self.assertRaisesRegex(RuntimeError, r"Invalid value range for 9 between"):
            _ = case_2(torch.tensor(9), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep(torch.tensor(5), torch.ones(4, 5)),
                case_2(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        with self.assertRaisesRegex(RuntimeError, "Max value to constrain_range_for_size must be greater than 2. got: 1"):
            _ = case_3(torch.tensor(1), torch.randn(4, 5))

        with self.assertRaisesRegex(RuntimeError, r"Invalid value range for 1 between \[2, 9223372036854775807\]."):
            _ = case_4(torch.tensor(1), torch.randn(4, 5))

        ep = export(case_4, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(RuntimeError, r"Invalid value range for 1"):
            _ = case_4(torch.tensor(1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep(torch.tensor(5), torch.ones(4, 5)),
                case_4(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        ep = export(case_5, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(RuntimeError, r"Invalid value range for 0"):
            _ = case_5(torch.tensor(0), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep(torch.tensor(5), torch.ones(4, 5)),
                case_5(torch.tensor(5), torch.ones(4, 5)),
            )
        )

    @testing.expectedFailureSerDer
    @testing.expectedFailureNonStrict
    def test_mixed_input(self):
        def func(a, b, alpha: int):
            return torch.add(a, b, alpha=alpha)

        a = torch.rand(1, 2)
        b = torch.rand(1, 2)
        alpha = 10

        exported = export(func, (a, b, alpha))
        for node in exported.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(isinstance(node.meta["val"], (Tensor, int)))

    @testing.expectedFailureNonStrict
    def test_export_with_inline_constraints(self):
        def f(x):
            a = x.item()
            torch._constrain_as_value(a, 4, 7)
            return torch.empty((a, 4))

        ep = export(f, (torch.tensor([5]),))
        self.assertEqual(ep(torch.tensor([6])).shape, (6, 4))

        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 1, exactly=True
        ).run(ep.graph_module.code)

        with self.assertRaisesRegex(
            RuntimeError,
            r"_local_scalar_dense is outside of inline constraint \[4, 7\]",
        ) as cm:
            ep(torch.tensor([30]))

    def test_constrain_decomp(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.freq = torch.ones(5, 5)

            def forward(self, start_pos: torch.Tensor):
                pos = start_pos.item()
                torch._constrain_as_size(pos, min=0, max=4)
                return self.freq[pos] * self.freq[pos]

        ep = torch.export.export(M(), (torch.tensor(1),))
        FileCheck().check_count(
            "torch.ops.aten._assert_async.msg", 2, exactly=True
        ).run(ep.graph_module.code)
        decompose_ep = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.aten._assert_async.msg", 2, exactly=True
        ).run(decompose_ep.graph_module.code)

    @testing.expectedFailureNonStrict
    def test_export_with_inline_constraints_complex(self):
        def f(x):
            a = x.item()
            torch._constrain_as_value(a, 4, 7)
            empty = torch.empty((a, 4))

            return torch.cat((empty.transpose(0, 1), torch.zeros(6, a)), 0)

        ep = export(f, (torch.tensor([6]),))
        self.assertEqual(ep(torch.tensor([5])).shape, (10, 5))
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_to_module_with_mutated_buffer(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.ones(1), buffer))

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5))))

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.tensor(2, dtype=torch.float), buffer))

    def test_to_module_with_mutated_buffer_multiple(self):

        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                self.bar.buf.add_(2)
                bar = self.bar(x)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(torch.allclose(torch.tensor(4, dtype=torch.float), buffer))

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5))))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.tensor(2, dtype=torch.float), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(torch.allclose(torch.tensor(7, dtype=torch.float), buffer))

    def test_runtime_assert_for_prim(self):
        def f(x, y):
            return x + y

        tensor_inp = torch.ones(7, 5)
        dim0_x = torch.export.Dim("dim0_x", min=6)
        dynamic_shapes = {"x": {0: dim0_x}, "y": None}
        exported = torch.export.export(f, (tensor_inp, 5), dynamic_shapes=dynamic_shapes)
        self.assertTrue(
            torch.allclose(exported(torch.ones(8, 5), 5), f(torch.ones(8, 5), 5))
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected input arg1 to be equal to 5, but got 6"
        ):
            _ = exported(torch.ones(8, 5), 6)

        exported = torch.export.export(f, (tensor_inp, 5.0), dynamic_shapes=dynamic_shapes)
        with self.assertRaisesRegex(
            RuntimeError, "Expected input arg1 to be equal to 5.0, but got 6.0"
        ):
            _ = exported(torch.ones(7, 5), 6.0)

    @testing.expectedFailureSerDer
    @testing.expectedFailureNonStrict
    def test_runtime_assert_for_prm_str(self):

        def g(a, b, mode):
            return torch.div(a, b, rounding_mode=mode)

        inps = (torch.randn(4, 4), torch.randn(4), "trunc")
        exported = export(g, inps)
        with self.assertRaisesRegex(RuntimeError, "to be equal to trunc, but got floor"):
            _ = exported(torch.randn(4, 4), torch.randn(4), "floor")
        self.assertTrue(torch.allclose(exported(*inps), g(*inps)))

    def test_to_module_with_mutated_buffer_multiple_update_sub_later(self):

        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(torch.allclose(torch.tensor(4, dtype=torch.float), buffer))

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5))))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.tensor(2, dtype=torch.float), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(torch.allclose(torch.tensor(7, dtype=torch.float), buffer))

    def test_retracable_ep(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        inp = torch.ones(5, 5)
        exported = torch.export.export(Foo(), (inp,))
        reexported = torch.export.export(exported, (inp,))

        self.assertTrue(torch.allclose(exported(inp), reexported(inp)))

        dim0_x = torch.export.Dim("dim0_x")
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x}})
        reexported = torch.export.export(exported, (inp,))
        with self.assertRaisesRegex(RuntimeError, "shape\[0\] to be equal to 5, but got 7"):
            reexported(torch.ones(7, 5))

        reexported = torch.export.export(exported, (inp,), dynamic_shapes=({0: dim0_x},))
        self.assertTrue(torch.allclose(exported(torch.ones(7, 5)), reexported(torch.ones(7, 5))))

        # can't retrace with invalid inputs with respect to the original ExportedProgram
        dim0_x_v2 = torch.export.Dim("dim0_x_v2", min=3)
        exported_v2 = torch.export.export(Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x_v2}})
        with self.assertRaisesRegex(RuntimeError, "Expected input l_x_.shape\[0\] to be >= 3, but got 2"):
            torch.export.export(exported_v2, (torch.randn(2, 2),))

    @testing.expectedFailureSerDer
    @testing.expectedFailureNonStrict
    def test_retrace_graph_level_meta_preservation(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                if x.shape[0] > 4:
                    return x.cos()
                return x.sin()

        inp = torch.ones(7, 5)
        dim0_x = torch.export.Dim("dim0_x", min=6)
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x}})
        stateful_module = exported.module()
        self.assertTrue(len(stateful_module.meta["input_shape_constraints"]), 1)

        re_exported = export(stateful_module, (inp,), constraints=[dynamic_dim(inp, 0) > 5])
        self.assertTrue(len(re_exported.graph_module.meta["input_shape_constraints"]) == 1)
        self.assertTrue(
            torch.allclose(exported(torch.ones(7, 5)), re_exported(torch.ones(7, 5)))
        )

        re_exported_v2 = export(exported, (inp,))
        self.assertTrue(len(re_exported_v2.graph_module.meta["input_shape_constraints"]) == 0)
        self.assertTrue(
            torch.allclose(exported(torch.ones(7, 5)), re_exported_v2(torch.ones(7, 5)))
        )

    @testing.expectedFailureNonStrict
    def test_constrain_as_size_error(self):

        def f(x):
            a = x.item()
            # We cannot automatically infer a is a size here because view
            # accepts -1
            return torch.randn(24).view(a, 4)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Tried to use data-dependent value in the subsequent computation"
        ):
            _ = export(f, (torch.tensor(6),))

    def test_constraint_directly_construct(self):
        with self.assertRaisesRegex(
            TypeError,
            "Constraint has no public constructor. Please use torch.export.dynamic_dim"
        ):
            _ = Constraint()

    def test_train_eval_on_exported_preautograd_module(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                if x.shape[0] > 4:
                    return x.cos()
                return x.sin()

        graph_module = capture_pre_autograd_graph(Foo(), (torch.ones(7, 5),), _functional_pre_dispatch_IR=True)
        with self.assertRaisesRegex(NotImplementedError, r"Calling train\(\) is not supported yet."):
            graph_module.train()

        with self.assertRaisesRegex(NotImplementedError, r"Calling eval\(\) is not supported yet."):
            graph_module.eval()

    def test_export_cond_preserve_stack_trace_for_subgraphs(self):
        class MySubModule(torch.nn.Module):
            def foo(self, x):
                return x.cos()

            def forward(self, x):
                return self.foo(x)

        class CondBranchClassMethod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subm = MySubModule()

            def bar(self, x):
                return x.sin()

            def forward(self, x):
                return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])


        from torch._export import capture_pre_autograd_graph

        example_inputs = (torch.randn(1, 3, 3, 3),)
        m = CondBranchClassMethod()
        m.eval()
        # TODO (tmanlaibaatar) Setting functional IR doesn't work on aot_export yet
        # as the branch source_fn is not captured.
        gm = capture_pre_autograd_graph(m, example_inputs, _functional_pre_dispatch_IR=False)

        actual_source_fns = []
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in {"sin", "cos"}:
                    source_fn_st = node.meta.get("source_fn_stack", None)
                    if source_fn_st is not None:
                        source_names = []
                        for source_fn in source_fn_st:
                            source_names.append(source_fn[0])
                        actual_source_fns.append(source_names)
        exp_source_fns = [["cond", "cos"], ["cond", "sin"]]
        self.assertEqual(actual_source_fns, exp_source_fns)

    @testing.expectedFailureRetraceability
    @testing.expectedFailureNonStrict
    def test_lifted_constants(self) -> None:
        def f(x):
            return x + torch.tensor(3)

        ep = export(f, (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 2)
        self.assertEqual(len(ep.tensor_constants), 1)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Foo(), (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertEqual(len(ep.state_dict), 1)
        self.assertEqual(len(ep.tensor_constants), 2)

        inp = (torch.tensor(5),)
        self.assertTrue(torch.allclose(ep(*inp), Foo()(*inp)))

        transform = ep.run_decompositions()
        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertTrue(torch.allclose(ep(*inp), transform(*inp)))

        unlifted = ep.module()
        self.assertTrue(torch.allclose(ep(*inp), unlifted(*inp)))

    def test_preserve_shape_dynamism_for_unused_inputs(self):
        @dataclass
        class Input:
            f: torch.Tensor
            p: torch.Tensor

        torch._export.utils.register_dataclass_as_pytree_node(Input)

        class Module(torch.nn.Module):
            def forward(self, x: Input):
                return x.f + 1

        mod = Module()
        example_inputs = (Input(f=torch.ones(10, 4), p=torch.zeros(10, 4)),)
        ep_static = torch.export.export(mod, example_inputs)
        for node in ep_static.graph.nodes:
            if node.op == "placeholder":
                for s in node.meta["val"].shape:
                    self.assertIsInstance(s, int)

        dim0_x_f, dim0_x_p = torch.export.dims("dim0_x_f", "dim0_x_p")
        dynamic_shapes = {"x": Input(f={0: dim0_x_f}, p={0: dim0_x_p})}
        ep_dynamic = torch.export.export(mod, example_inputs, dynamic_shapes=dynamic_shapes)
        for node in ep_dynamic.graph.nodes:
            if node.op == "placeholder":
                for i, s in enumerate(node.meta["val"].shape):
                    if i == 0:
                        self.assertIsInstance(s, torch.SymInt)
                    else:
                        self.assertIsInstance(s, int)

    def test_multiple_definitions_same_name_dim(self):
        def foo(x, y):
            return torch.matmul(x, y)

        A = torch.export.Dim("C", min=3)
        B = torch.export.Dim("C", max=12)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Found different definitions Dim\\(.*min=3\\) and Dim\\(.*max=12\\) "
            "for the same symbolic dimension",
        ):
            torch.export.export(
                foo,
                (torch.randn(10, 10), torch.randn(10, 10)),
                dynamic_shapes={"x": (A, B), "y": (B, A)},
            )

    def test_export_with_wrong_inputs(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        exported_program = export(MyModule(), (torch.rand(2, 3),), {})
        with self.assertRaisesRegex(
            TypeError, "Trying to flatten user inputs with exported input tree spec"
        ):
            exported_program(torch.rand(2, 3), torch.rand(2, 3))

    def test_export_decomps_simple(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp)
        state_dict = ep.state_dict

        FileCheck().check_count(
            "torch.ops.aten.t.default", 1, exactly=True
        ).run(ep.graph_module.code)
        self.assertTrue(torch.allclose(ep(*inp), m(*inp)))

        core_aten_ep = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.aten.permute.default", 1, exactly=True
        ).run(core_aten_ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.t.default", 0, exactly=True
        ).run(core_aten_ep.graph_module.code)
        self.assertTrue(torch.allclose(core_aten_ep(*inp), m(*inp)))
        self.assertEqual(id(state_dict), id(ep.state_dict))

    @testing.expectedFailureRetraceability
    def test_export_decomps_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp, dynamic_shapes={"x": {0: Dim("batch")}})

        core_aten_ep = ep.run_decompositions()

        input_node = [node for node in core_aten_ep.graph.nodes if node.op == "placeholder"][-1]
        self.assertTrue(isinstance(input_node.meta["val"].shape[0], torch.SymInt))

        FileCheck().check_count(
            "torch.ops.aten.permute.default", 1, exactly=True
        ).run(core_aten_ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.t.default", 0, exactly=True
        ).run(core_aten_ep.graph_module.code)
        self.assertTrue(torch.allclose(core_aten_ep(*inp), m(*inp)))

    @testing.expectedFailureNonStrict
    def test_nonzero_2(self):
        def f(x):
            return torch.nonzero(x)
        ep = export(f, (torch.ones(2),))
        inp = torch.randn(2)
        self.assertTrue(torch.allclose(ep(inp), torch.nonzero(inp)))

    @testing.expectedFailureSerDer
    @testing.expectedFailureNonStrict
    def test_redundant_asserts(self):
        def f(x):
            y = x.item()
            torch._constrain_as_size(y)
            return torch.zeros(y)

        ep = export(f, (torch.tensor([3]),))
        self.assertExpectedInline(str(ep.graph_module.code).strip(), """\
def forward(self, l_x_):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(l_x_);  l_x_ = None
    ge = _local_scalar_dense >= 0
    scalar_tensor = torch.ops.aten.scalar_tensor.default(ge);  ge = None
    _assert_async = torch.ops.aten._assert_async.msg(scalar_tensor, '_local_scalar_dense is outside of inline constraint [0, inf].');  scalar_tensor = None
    sym_constrain_range_for_size = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense)
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return (zeros,)""")

    def test_non_arg_name_dynamic_shapes_api(self):
        def foo(a, b):
            return a.sum() + b.sum()

        dim = torch.export.Dim("dim")
        ep = torch.export.export(foo, (torch.randn(4, 4), torch.randn(4, 4)), dynamic_shapes=(None, {0: dim}))

        test_inp = (torch.randn(4, 4), torch.randn(7, 4))
        self.assertEqual(ep(*test_inp), foo(*test_inp))

        ep_v2 = torch.export.export(foo, (torch.randn(4, 4), torch.randn(4, 4)), dynamic_shapes=(None, None))
        with self.assertRaisesRegex(RuntimeError, "shape\[0\] to be equal to 4, but got 7"):
            ep_v2(*test_inp)

    def test_constant_output(self):
        class ModuleConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.randn(3, 2)

            def forward(self):
                return self.b

        class ModuleNestedConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bff = torch.randn(3, 2)

            def forward(self, x, y):
                return {"prediction": (x + y, self.bff)}

        mod = ModuleConstant()
        ep = torch.export.export(mod, ())
        self.assertEqual(ep(), mod())

        args = (torch.randn(3, 2), torch.randn(3, 2))
        mod = ModuleNestedConstant()
        ep = torch.export.export(mod, args)
        self.assertEqual(ep(*args), mod(*args))

    def test_non_arg_name_dynamic_shapes_api_with_kwarg(self):
        def foo(a, b, kw1, kw2):
            return a.sum() + b.sum() + kw1.sum() - kw2.sum()

        dim = torch.export.Dim("dim")
        dim_for_kw1 = torch.export.Dim("dim_for_kw1")
        ep = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            {"kw2": torch.ones(4, 4), "kw1": torch.zeros(4, 4)},
            # We are specifying dynamism on the first kwarg even though user passed in
            # different order
            dynamic_shapes=(None, {0: dim}, {0: dim_for_kw1}, None))

        test_inp = (torch.randn(4, 4), torch.randn(7, 4))
        test_kwargs = {"kw2": torch.ones(4, 4), "kw1": torch.zeros(9, 4)}
        # This should work even if the kwarg order are flipped.
        self.assertEqual(ep(*test_inp, **test_kwargs), foo(*test_inp, **test_kwargs))

    def test_non_arg_name_dynamic_shapes_api_with_container_type(self):
        def foo(a, b):
            return a[0].sum() + a[1].sum() + b.sum()

        inp_a = (torch.randn(4, 4), torch.randn(4, 4))
        inp_b = torch.randn(4, 4)
        inp = (inp_a, inp_b)

        count = 0
        def dynamify_inp(x):
            # Mark the second input a[1] dynamic
            nonlocal count
            if count == 1:
                dim = torch.export.Dim("dim", min=3)
                count += 1
                return {0: dim}
            count += 1
            return None

        dynamic_shapes = tree_map(dynamify_inp, inp)

        ep = torch.export.export(foo, inp, dynamic_shapes=dynamic_shapes)

        test_inp = ((torch.randn(4, 4), torch.randn(2, 4)), torch.randn(4, 4))
        with self.assertRaisesRegex(
            RuntimeError, "shape\[0\] to be >= 3, but got 2"
        ):
            ep(*test_inp)

    def test_lazy_module_kwargs(self):
        class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def initialize_parameters(self, *args, **kwargs):
                pass

            def forward(self, x, y):
                return x + y

        m = LazyModule()
        ep = torch.export.export(m, (), {'x': torch.randn(3, 3), 'y': torch.randn(3, 3)})
        inputs = {'x': torch.randn(3, 3), 'y': torch.randn(3, 3)}
        self.assertEqual(ep(**inputs), m(**inputs))

    def test_retrace_pre_autograd(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(4, 4))

            def forward(self, x):
                self.buffer.add_(4)
                return x.sum() + self.buffer.sum()

        inp = torch.randn(4, 4)
        gm = capture_pre_autograd_graph(Foo(), (inp,), constraints=[dynamic_dim(inp, 0) >= 3], _functional_pre_dispatch_IR=True)

        with self.assertRaisesRegex(RuntimeError, "Expected input l_x_.shape\[0\]"):
            gm(torch.randn(2, 2))

        with self.assertRaisesRegex(RuntimeError, "Expected input l_x_.shape\[0\]"):
            torch.export.export(gm, (torch.randn(2, 2),))

        ep = torch.export.export(gm, (torch.randn(5, 4),), dynamic_shapes=({0: torch.export.Dim("dim", min=3)},))

        test_inp = torch.ones(8, 4)
        self.assertTrue(torch.allclose(ep(test_inp), Foo().forward(test_inp)))

    @testing.expectedFailureNonStrict
    def test_issue_113041(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor(1.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.a

        def forward_hook(
            module: torch.nn.Module, inputs, output
        ) -> torch.Tensor:
            return 2 * output

        seq = torch.nn.Sequential(TestModule()).eval()
        seq.b = torch.tensor(2)
        handle = seq.register_forward_hook(forward_hook)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = seq

            def forward(self, x):
                return self.seq(x) + self.seq.b

        inp = (torch.randn(2, 8),)
        ep = export(M(), inp)  # This errors because dynamo adds an extra input

    def test_export_with_fake_tensor_inputs(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        # Put the inputs on a device
        with fake_mode, torch.device('meta'):
            x = torch.rand(5, 2, 2)
            model = Model()

            exported_program = torch.export.export(model, (x,))
            export_res = exported_program(x)
            exp_res = model(x)
            all_meta_val = [node.meta["val"] for node in exported_program.graph_module.graph.nodes if 'val' in node.meta]
            self.assertTrue(export_res.size() == exp_res.size())
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            self.assertTrue(all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val))
            decomposed_ep = exported_program.run_decompositions()
            export_res = decomposed_ep(x)
            self.assertTrue(export_res.size() == exp_res.size())

    def test_export_with_fake_tensor_inputs_on_cuda_devices(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        # Put the inputs on a device
        with fake_mode, torch.device('meta'):
            x = torch.rand(5, 2, 2)
            model = Model()

        # Manualy set the fake_device of fake tensors.
        x.fake_device = torch.device('cuda:0')
        for n, p in model.named_parameters():
            p.fake_device = torch.device('cuda:0')

        # Need to set all the requires_grad of tensors to False, because fake_tensor with CUDA device
        # doesn't quite work well with aot_autograd right now due to some logic fails
        # the check in call getDeviceGuardImpl in InputMetadata.
        x.requires_grad = False
        for n, p in model.named_parameters():
            p.requires_grad = False


        def check_device_and_fake_mode():
            exported_program = torch.export.export(model, (x,))
            export_res = exported_program(x)
            exp_res = model(x)
            all_meta_val = [node.meta["val"] for node in exported_program.graph_module.graph.nodes if 'val' in node.meta]
            self.assertTrue(export_res.size() == exp_res.size())
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            self.assertTrue(all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val))

        check_device_and_fake_mode()


    def test_export_graph_with_no_inputs(self):
        # We saw this pattern when users want to export
        # a graph that initlizes the states of a model.
        def f():
            return torch.randn(3, 4), torch.randn(3, 4)

        ep = torch.export.export(f, ())
        a, b = ep()
        self.assertEqual(a.size(), torch.Size([3, 4]))
        self.assertEqual(b.size(), torch.Size([3, 4]))

    def test_export_then_compile_tensor_ctor(self):
        class M(torch.nn.Module):
            def __init__(self,):
                super().__init__()

            def forward(self, scores, mask):
                scores = scores.masked_fill(
                    mask, torch.tensor(torch.finfo(scores.dtype).min)
                )  # (bs, n_heads, q_length, k_length)
                return scores

        tensor_cpu = torch.randn(2, 4)
        mask_cpu = torch.BoolTensor(
            [[False,  True, False, False],
            [False, False, False, False]]
        )

        m = M().eval()
        # res_ref = m(tensor_cpu, mask_cpu)
        # print("res_ref is: {}".format(res_ref), flush=True)

        exported_model = capture_pre_autograd_graph(
            m,
            (tensor_cpu, mask_cpu),
            _functional_pre_dispatch_IR=True,
        )
        optimized_model = torch.compile(exported_model)
        optimized_model(tensor_cpu, mask_cpu)

    def test_export_mkldnn_disabled(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.lstm(inputs)

        inp = (torch.ones(3, 4),)
        torch._C._set_mkldnn_enabled(False)
        ep = torch.export.export(M(), inp)
        FileCheck().check_count(
            "torch.ops.aten.mkldnn_rnn_layer.default", 0, exactly=True
        ).run(ep.graph_module.code)

        torch._C._set_mkldnn_enabled(True)
        ep = torch.export.export(M(), inp)
        FileCheck().check_count(
            "torch.ops.aten.mkldnn_rnn_layer.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_export_input_mutation_static_shape(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x, y):
                x.view(3, 2, -1).add_(y)
                return x
        inputs = (torch.randn(12), 2.0)
        model = MutationModel()
        ep = torch.export.export(model, inputs)
        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        self.assertEqual(ep(*inputs_export), model(*inputs_model))
        self.assertEqual(inputs[0] + 2.0, inputs_model[0])
        self.assertEqual(inputs[0] + 2.0, inputs_export[0])

    def test_export_input_mutation_dynamic_shape(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x, y):
                x[0].mul_(y)
                return x
        inputs = ((torch.randn(12), torch.randn(3, 2)), 2.0)
        model = MutationModel()
        ep = torch.export.export(
            model,
            inputs,
            dynamic_shapes={'x': ({0: torch.export.Dim("dim")}, None), "y": None}
        )
        nodes = list(ep.graph.nodes)
        self.assertEqual(nodes[0].op, "placeholder")
        self.assertIsInstance(nodes[0].meta['val'], torch.Tensor)
        self.assertIsInstance(nodes[0].meta['val'].shape[0], torch.SymInt)

        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        self.assertEqual(ep(*inputs_export), model(*inputs_model))
        self.assertEqual(inputs[0][0] * 2.0, inputs_model[0][0])
        self.assertEqual(inputs[0][0] * 2.0, inputs_export[0][0])

    @testing.expectedFailureSerDer  # symfloat nyi
    @testing.expectedFailureRetraceability
    def test_sym_sqrt(self):
        import math
        class M(torch.nn.Module):
            def forward(self, x):
                return x / torch.sym_sqrt(x.shape[0])

        ep = export(M(), (torch.ones(16, 4),), dynamic_shapes={'x': {0: Dim("dim")}})
        _ExportPassBase()(ep.graph_module)
        FileCheck().check_count(
            "torch.sym_sqrt", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_check_specialized_int(self):
        class SingleOp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.ops.aten.scatter_add

            def forward(self, t, dim, index, src, **kwargs):
                return self.op(t, dim, index, src, **kwargs)


        t = torch.randn(10, 5)
        dim = -1
        index = torch.tensor([[2, 4, 3, 1, 0],[0, 2, 1, 4, 3],[3, 1, 4, 2, 0],[4, 0, 3, 1, 2],[3, 0, 4, 1, 2]])
        src = torch.randn(5, 5)

        model = SingleOp()
        output = model(t, dim, index, src)

        ep = torch.export.export(model, args=(t, dim, index, src))
        ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)
        self.assertEqual(ep(t, dim, index, src), output)

    @testing.expectedFailureRetraceability
    def test_fqn(self):
        class NestedChild(torch.nn.Module):
            def forward(self, x):
                return x / x

        class Child1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("child2buffer", torch.ones(2, 3))

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        test_inp = torch.randn(2, 3)

        torch_gm = _export_to_torch_ir(orig_eager, (torch.rand(2, 3),), {})
        for k, v in orig_eager.state_dict().items():
            normalized_k = k.replace(".", "_")
            self.assertIn(normalized_k, torch_gm.state_dict())
            self.assertEqual(v, torch_gm.state_dict()[normalized_k])
        self.assertTrue(torch.allclose(torch_gm(test_inp), orig_eager(test_inp)))

        pre_autograd_gm = capture_pre_autograd_graph(orig_eager, (torch.rand(2, 3),), {})
        for k, v in orig_eager.state_dict().items():
            normalized_k = k.replace(".", "_")
            self.assertIn(normalized_k, pre_autograd_gm.state_dict())
            self.assertEqual(v, pre_autograd_gm.state_dict()[normalized_k])
        self.assertTrue(torch.allclose(pre_autograd_gm(test_inp), orig_eager(test_inp)))

        ep = export(orig_eager, (torch.rand(2, 3),), {})
        for k, v in orig_eager.state_dict().items():
            # We do not need to normalize the key here because exported
            # program's state dict is able to contain the module information.
            self.assertIn(k, ep.state_dict)
            self.assertEqual(v, ep.state_dict[k])
        self.assertTrue(torch.allclose(ep(test_inp), orig_eager(test_inp)))

    def test_nn_module_stack(self):
        class Leaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()
                self.register_buffer("buffer", torch.randn(4, 4))

            def forward(self, x):
                return self.buffer.sum() + self.leaf(x).sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                y = self.bar.buffer + x
                return (self.bar(x) + y.sum(),)

        inp = (torch.randn(4, 4),)
        mod = Foo()
        ep_strict = torch.export.export(mod, inp)
        ep_non_strict = torch.export.export(mod, inp, strict=False)

        gm_unflat_non_strict = unflatten(ep_non_strict)
        self.assertTrue(hasattr(gm_unflat_non_strict, "bar"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "buffer"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "leaf"))

        gm_unflat_strict = unflatten(ep_strict)

        self.assertEqual(gm_unflat_non_strict(*inp), gm_unflat_strict(*inp))
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar.leaf.linear.graph).strip(), """\
graph():
    %arg3_1 : [num_users=1] = placeholder[target=arg3_1]
    %bias : [num_users=1] = get_attr[target=bias]
    %weight : [num_users=1] = get_attr[target=weight]
    %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%weight,), kwargs = {})
    %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%bias, %arg3_1, %t), kwargs = {})
    return addmm"""
        )

        gm_flat_non_strict = ep_non_strict.module()
        gm_flat_strict = ep_strict.module()

        self.assertEqual(gm_flat_non_strict(*inp), gm_flat_strict(*inp))

    def test_nn_module_stack_shared_submodule(self):
        class Leaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()
                self.register_buffer("buffer", torch.randn(4, 4))

            def forward(self, x):
                return self.buffer.sum() + self.leaf(x).sum()

        class BarDifferent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()

            def forward(self, x):
                a = self.leaf(x).sum()
                b = self.leaf(x).sum()
                return a + b

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()
                self.bar_different = BarDifferent()

            def forward(self, x):
                y = self.bar.buffer + x
                return (self.bar(x) + self.bar_different(x + 2), y.sum(),)

        inp = (torch.randn(4, 4),)
        mod = Foo()
        ep_strict = torch.export.export(mod, inp)
        ep_non_strict = torch.export.export(mod, inp, strict=False)

        gm_unflat_non_strict = unflatten(ep_non_strict)
        self.assertTrue(hasattr(gm_unflat_non_strict, "bar"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "buffer"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "leaf"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar_different, "leaf"))

        gm_unflat_strict = unflatten(ep_strict)

        self.assertEqual(gm_unflat_non_strict(*inp), gm_unflat_strict(*inp))
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar.leaf.linear.graph).strip(), """\
graph():
    %arg5_1 : [num_users=1] = placeholder[target=arg5_1]
    %bias : [num_users=1] = get_attr[target=bias]
    %weight : [num_users=1] = get_attr[target=weight]
    %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%weight,), kwargs = {})
    %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%bias, %arg5_1, %t), kwargs = {})
    return addmm"""
        )
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar_different.leaf.linear.graph).strip(), """\
graph():
    %add_2 : [num_users=1] = placeholder[target=add_2]
    %bias : [num_users=1] = get_attr[target=bias]
    %weight : [num_users=1] = get_attr[target=weight]
    %t_1 : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%weight,), kwargs = {})
    %addmm_1 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%bias, %add_2, %t_1), kwargs = {})
    return addmm_1"""
        )

        gm_flat_non_strict = ep_non_strict.module()
        gm_flat_strict = ep_strict.module()

        self.assertEqual(gm_flat_non_strict(*inp), gm_flat_strict(*inp))

    def test_cond_with_module_stack_export_with(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_fn(x):
                    return self.linear(x).cos()
                def false_fn(x):
                    return self.linear(x).sin()
                return torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])

        class CondExport(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x.cos() + self.bar(x)

        inp = (torch.randn(4, 4),)
        ep = torch.export.export(CondExport(), inp, strict=False)
        self.assertExpectedInline(ep.graph_module.code.strip(), """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg2_1)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [arg1_1, arg0_1, arg2_1]);  true_graph_0 = false_graph_0 = arg1_1 = arg0_1 = arg2_1 = None
    getitem = conditional[0];  conditional = None
    add = torch.ops.aten.add.Tensor(cos, getitem);  cos = getitem = None
    return (add,)""")

        cond_top_level_nn_module_stack = [
            node.meta["nn_module_stack"]
            for node in ep.graph.nodes
            if node.name == "true_graph_0"
        ][0]

        self.assertTrue("test_cond_with_module_stack_export_with.<locals>.Bar" in str(cond_top_level_nn_module_stack))

    # TODO: See https://github.com/pytorch/pytorch/issues/115790
    @unittest.expectedFailure
    def test_cond_with_module_stack_export_with_unflatten(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_fn(x):
                    return self.linear(x).cos()
                def false_fn(x):
                    return self.linear(x).sin()
                return torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])

        class CondExport(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x.cos() + self.bar(x)

        inp = (torch.randn(4, 4),)
        ep = torch.export.export(CondExport(), inp, strict=False)

        cond_top_level_nn_module_stack = [
            node.meta["nn_module_stack"]
            for node in ep.graph.nodes
            if node.name == "true_graph_0"
        ][0]

        # we can't preserve nn_module_stack for the subgraphs for now.
        for node in ep.graph_module.true_graph_0.graph.nodes:
            self.assertEqual(node.meta["nn_module_stack"], cond_top_level_nn_module_stack)

        # this doesn't work today
        gm_unflat_strict = unflatten(ep)


if __name__ == '__main__':
    run_tests()
