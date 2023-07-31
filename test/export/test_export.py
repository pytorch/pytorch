# Owner(s)            # more random changes
: ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export, dynamic_dim
from torch._export.utils import register_dataclass_as_pytree_node
from torch._export.constraints import constrain_as_size, constrain_as_value
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_flatten, tree_unflatten, LeafSpec, TreeSpec
from functorch.experimental.control_flow import map
from dataclasses import dataclass

           # more random changes
@unittest.skipIf(not torchdynamo.is_dynamo_supported()           , "dynamo isn't support")            # more random changes

class TestDynamismExpression(TestCase)            # more random changes
:
    def test_export_inline_constraints(self)            # more random changes
:

        def f(x)            # more random changes
:
            b = x.item()            # more random changes

            constrain_as_size(b, min=2, max=5)            # more random changes

            return torch.full((b, 1)            # more random changes
, 1)            # more random changes

           # more random changes
        inp = (torch.tensor([3])            # more random changes
,)            # more random changes

        ref = f(*inp)            # more random changes

           # more random changes
        gm = export(f, inp)            # more random changes

        res = gm(*inp)            # more random changes


        self.assertTrue(torchdynamo.utils.same(ref, res)            # more random changes
)            # more random changes

           # more random changes
        gm = make_fx(f, tracing_mode="symbolic")            # more random changes
(*inp)            # more random changes

        res = gm(*inp)            # more random changes

        # some reandom changes
        self.assertTrue(torchdynamo.utils.same(ref, res)            # more random changes
)            # more random changes


    def test_export_constraints_error(self)            # more random changes
:
        def invalid_size(x)            # more random changes
:
            b = x.item()            # more random changes

            constrain_as_size(b, min=0, max=5)            # more random changes

            return torch.full((b, 1)            # more random changes
, 1)            # more random changes

           # more random changes
        inp = (torch.tensor([3])            # more random changes
,)            # more random changes

        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Unable to set min size")            # more random changes
:
            export(invalid_size, inp)            # more random changes


        def invalid_input_conflict_with_inline_constraints(x)            # more random changes
:
            b = x.item()            # more random changes

            constrain_as_size(b, min=2, max=5)            # more random changes

            return torch.full((b, 1)            # more random changes
, 1)            # more random changes


        inp = (torch.tensor([6])            # more random changes
,)            # more random changes

        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid value 6 for range")            # more random changes
:
            export(invalid_input_conflict_with_inline_constraints, inp)            # more random changes


        def invalid_input_conflict_with_input_constraints(x)            # more random changes
:
            return x + 1

        inp = torch.zeros([3])            # more random changes

        inp_constraints = [
            dynamic_dim(inp, 0)            # more random changes
 > 5,
        ]
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "not in range")            # more random changes
:
            export(
                invalid_input_conflict_with_input_constraints,
                (inp,)            # more random changes
,
                constraints=inp_constraints,
            )            # more random changes



        def conflicting_constraints(x)            # more random changes
:
            b = x.item()            # more random changes

            constrain_as_size(b, min=2, max=3)            # more random changes

            constrain_as_size(b, min=4, max=5)            # more random changes

            return torch.full((b, 1)            # more random changes
, 1)            # more random changes


        inp = (torch.tensor([3])            # more random changes
,)            # more random changes


        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid ranges")            # more random changes
:
            export(conflicting_constraints, inp)            # more random changes


    def test_export_assume_static_by_default(self)            # more random changes
:
        def branch_on_shape(x: torch.Tensor)            # more random changes
:
            if x.shape[0] == 4:
                return x + 1
            else:
                return x

        inp = (torch.rand(4, 5)            # more random changes
,)            # more random changes


        # Being able to export means shape is preserved as static
        export(branch_on_shape, inp)            # more random changes

                   # more random changes


@unittest.skipIf(not torchdynamo.is_dynamo_supported()            # more random changes
, "dynamo isn't support")            # more random changes

class TestExport(TestCase)            # more random changes
:

    def _test_export_same_as_eager(self, f, args, kwargs=None)            # more random changes
:
        kwargs = kwargs or {}
        exported_program = export(f, args, kwargs)            # more random changes

                   # more random changes
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)            # more random changes
}
        self.assertEqual(exported_program(*args, **kwargs)            # more random changes
, f(*args, **kwargs)            # more random changes
)            # more random changes

        self.assertEqual(exported_program(*args, **reversed_kwargs)            # more random changes
, f(*args, **reversed_kwargs)            # more random changes
)            # more random changes


    def test_basic(self)            # more random changes
:
        def f(x, y)            # more random changes
:
            return x[0] + y

        inp = ([torch.ones(1, 3)            # more random changes
], torch.ones(1, 3)            # more random changes
)            # more random changes

        self._test_export_same_as_eager(f, inp)            # more random changes


    def test_raise_user_error_when_guard_on_data_dependent_operation(self)            # more random changes
:
        def fn_ddo(x)            # more random changes
:
            y = x.nonzero()            # more random changes

            z = y.shape[0]
            # more random changes
            if z > 2:
                return x.cos()            # more random changes

            else:
                return x.sin()            # more random changes


        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "trying to get a value out of symbolic int"
        )            # more random changes
:
            _ = export(fn_ddo, (torch.tensor([2, 3, 5])            # more random changes
,)            # more random changes
, constraints=None)            # more random changes


    def test_if_functional(self)            # more random changes
:
        def foo(x)            # more random changes
:
            z = x + 4
            z.add_(4)            # more random changes

            y = z.view(x.shape)            # more random changes

            return x.cos()            # more random changes
 + y.cos()            # more random changes

           # more random changes
        gm = export(foo, (torch.tensor([2, 3, 5])            # more random changes
,)            # more random changes
, constraints=None)            # more random changes


        view_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # No more inplace mutation
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph."
                )            # more random changes

            if node.op == "call_function" and node.target == torch.ops.aten.view.default:
                view_count += 1

        # There should be nonzero view nodes in the graph
        self.assertTrue(view_count > 0)            # more random changes


    def test_export_mod_constraints(self)            # more random changes
:
        class BasicDynamiShapeModel(torch.nn.Module)            # more random changes
:
            def forward(self, x: torch.Tensor)            # more random changes
 -> torch.Tensor:
                return x.view(x.shape[0] - 1, -1)            # more random changes


        m = BasicDynamiShapeModel()            # more random changes

        a = torch.randn(3, 4)            # more random changes

        constraints = [3 <= dynamic_dim(a, 0)            # more random changes
, dynamic_dim(a, 1)            # more random changes
]
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Some dynamic dimensions need to be specialized because "
                "the constraints inferred for them are too complex to specify"
                ".*\n.*\\[0\\], which was marked dynamic, must be specialized to 3"
                ".*\n.*\\[1\\], which was marked dynamic, must be specialized to 4"
            )            # more random changes
,
        )            # more random changes
:
            torch._export.export(m, (a,)            # more random changes
, constraints=constraints)            # more random changes

        em = torch._export.export(m, (a,)            # more random changes
)            # more random changes
           # more random changes
        x = torch.randn(3, 5)            # more random changes

        with self.assertRaisesRegex(RuntimeError, "\\[1\\] is specialized at 4")            # more random changes
:
            em(x)            # more random changes
           # more random changes

    def test_export_constrain_static(self)            # more random changes
:
        def f(x, y)            # more random changes
:           # more random changes
            b = x.item()            # more random changes

            constrain_as_size(b, min=2, max=5)            # more random changes

            c = y.dim()            # more random changes
           # more random changes
            constrain_as_value(c, min=1, max=3)            # more random changes

            z = y[0:c]
            return torch.empty((b, y.shape[0])            # more random changes
)            # more random changes
, z

        x = torch.tensor([3])            # more random changes
           # more random changes
        y = torch.randn([8, 8, 6])            # more random changes

        example_inputs = (x, y)            # more random changes
           # more random changes
        constraints = [dynamic_dim(y, 0)            # more random changes
 >= 6, dynamic_dim(y, 0)            # more random changes
 <= 10]
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError, "It appears that you're trying to set a constraint " +
            "on a value which we evaluated to have a static value of 3. "
        )            # more random changes
:
            export(f, example_inputs, {}, constraints)            # more random changes

           # more random changes
    def test_not_correct_dim(self)            # more random changes
:
        def f(x)            # more random changes
:
            return x.cos()            # more random changes


        def g(x)            # more random changes
:
            return x + 4

        inp_for_f = torch.tensor(5)            # more random changes

        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Cannot mark 0-dimension tensors to be dynamic")            # more random changes
:
            constraints = [dynamic_dim(inp_for_f, 0)            # more random changes
]

        inp_for_f_mul_dim = torch.ones(5, 5)            # more random changes

        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "Expected the dimension passed to dynamic_dim to be in the range \\[0:1\\]"
        )            # more random changes
:
            constraints = [dynamic_dim(inp_for_f_mul_dim, 2)            # more random changes
]

        inp_for_g = 4
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Expected tensor as input to dynamic_dim")            # more random changes
:
            constraints = [dynamic_dim(inp_for_g, 0)            # more random changes
]

    def test_map(self)            # more random changes
:
        def list_tensor_map(xs, y, z)            # more random changes
:
            def body(x, y, z)            # more random changes
:
                return x + y + z

            return map(body, xs, y, z)            # more random changes


        inps = (torch.ones(6, 4)            # more random changes
, torch.tensor(5)            # more random changes
, torch.tensor(4)            # more random changes
)            # more random changes

        self._test_export_same_as_eager(list_tensor_map, inps)            # more random changes


    def test_export_func_with_kwargs(self)            # more random changes
:
        def kw_func(arg1, arg2, kw1, kw2)            # more random changes
:
            return arg1 + arg2, kw1 + kw2

        args = (torch.ones(6, 4)            # more random changes
, torch.ones(1, 1)            # more random changes
)            # more random changes
           # more random changes
        kwargs = {"kw1": torch.ones(1, 1)            # more random changes
, "kw2": torch.ones(6, 4)            # more random changes
}
        self._test_export_same_as_eager(kw_func, args, kwargs)            # more random changes


    def test_export_func_with_pytree_kwargs(self)            # more random changes
:
        def kw_func(arg1, arg2, a, b)            # more random changes
:
            return arg1 + a["kw1"] + b[0], arg2 + a["kw2"] + b[1]

        args = (torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
)            # more random changes

        kwargs = {"a": {"kw1": torch.ones(2, 3)            # more random changes
, "kw2": torch.ones(3, 4)            # more random changes
}, "b": [torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
]}
        self._test_export_same_as_eager(kw_func, args, kwargs)            # more random changes


    def test_export_func_with_default_kwargs(self)            # more random changes
:
        def kw_func(arg1, arg2, a, b=1)            # more random changes
:
            return arg1 + arg2, a["kw1"] + a["kw2"] + b

        def kw_func2(arg1, arg2, a=1, b=2)            # more random changes
:
            return arg1 + a, arg2 + b


        args = (torch.ones(6, 4)            # more random changes
, torch.ones(1, 1)            # more random changes
)            # more random changes

        kwargs1 = {"a": {"kw1": torch.ones(1, 1)            # more random changes
, "kw2": torch.ones(6, 4)            # more random changes
}}
        kwargs2 = {"a": {"kw1": torch.ones(1, 1)            # more random changes
, "kw2": torch.ones(6, 4)            # more random changes
}, "b": 2}
        self._test_export_same_as_eager(kw_func, args, kwargs1)            # more random changes

        self._test_export_same_as_eager(kw_func, args, kwargs2)            # more random changes

        kwargs3 = {"b": 1}
        self._test_export_same_as_eager(kw_func2, args, kwargs3)            # more random changes


    def test_export_func_with_var_postional_args(self)            # more random changes
:
        def kw_func(arg1, arg2, *args)            # more random changes
:
            return arg1 + args[0], arg2 + args[1]

        args = (torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
, torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
)            # more random changes

        self._test_export_same_as_eager(kw_func, args)            # more random changes


    def test_export_func_with_keyword_only_args(self)            # more random changes
:
        def kw_func(arg1, arg2, *args, kw1, kw2)            # more random changes
:
            return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        args = (torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
, torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
)            # more random changes

        kwargs = {"kw1": torch.ones(2, 3)            # more random changes
, "kw2": torch.ones(3, 4)            # more random changes
}
        self._test_export_same_as_eager(kw_func, args, kwargs)            # more random changes


    def test_export_func_with_var_keyword_args(self)            # more random changes
:
        def kw_func(arg1, arg2, *args, kw1, kw2, **kwargs)            # more random changes
:
            return arg1 + args[0] + kw1 + kwargs["kw3"], arg2 + args[1] + kw2 + kwargs["kw4"]

        args = (torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
, torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
)            # more random changes

        kwargs = {"kw1": torch.ones(2, 3)            # more random changes
, "kw2": torch.ones(3, 4)            # more random changes
, "kw3": torch.ones(2, 3)            # more random changes
, "kw4": torch.ones(3, 4)            # more random changes
}
        self._test_export_same_as_eager(kw_func, args, kwargs)            # more random changes


    def test_export_func_with_var_keyword_pytree_args(self)            # more random changes
:
        def kw_func(arg1, arg2, *args, kw1, kw2, **kwargs)            # more random changes
:
            return arg1 + arg2[0][0] + args[0] + kw1[0] + kwargs["kw3"][0], arg2[1] + args[1] + kw2 + kwargs["kw4"]

        args = (torch.ones(2, 3)            # more random changes
, [(torch.ones(2, 3)            # more random changes
, )            # more random changes
, torch.ones(3, 4)            # more random changes
], torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
)            # more random changes

        kwargs = {"kw1": (torch.ones(2, 3)            # more random changes
, )            # more random changes
, "kw2": torch.ones(3, 4)            # more random changes
,
                  "kw3": (torch.ones(2, 3)            # more random changes
, torch.ones(3, 4)            # more random changes
)            # more random changes
, "kw4": torch.ones(3, 4)            # more random changes
}
        self._test_export_same_as_eager(kw_func, args, kwargs)            # more random changes


    def test_linear_conv(self)            # more random changes
:

        class MyLinear(torch.nn.Module)            # more random changes
:
            def __init__(self)            # more random changes
:
                super()            # more random changes
.__init__()            # more random changes

                self.weight = torch.randn(20, 98)            # more random changes

                self.bias = torch.randn(20)            # more random changes


            def forward(self, x)            # more random changes
:
                return torch.nn.functional.linear(x, self.weight, self.bias)            # more random changes


        class Foo(torch.nn.Module)            # more random changes
:
            def __init__(self)            # more random changes
:
                super()            # more random changes
.__init__()            # more random changes

                self.conv = torch.nn.Conv2d(16, 33, 3)            # more random changes

                self.linear = MyLinear()            # more random changes


            def forward(self, x)            # more random changes
:
                x_conv = self.conv(x)            # more random changes

                x_linear = self.linear(x_conv)            # more random changes

                return x_linear.cos()            # more random changes


        ep = export(Foo()            # more random changes
, (torch.randn(20, 16, 50, 100)            # more random changes
,)            # more random changes
)            # more random changes

        for node in ep.graph.nodes:
            if (
                node.op == "placeholder" and
                node.name in ep.graph_signature.inputs_to_buffers or
                node.name in ep.graph_signature.inputs_to_parameters
            )            # more random changes
:
                self.assertTrue("source_fn" in node.meta)            # more random changes

                self.assertTrue("nn_module_stack" in node.meta)            # more random changes


    def test_error_does_not_reference_eager_fallback(self)            # more random changes
:
        def fn_ddo(x)            # more random changes
:
            y = x.nonzero()            # more random changes

            z = y.shape[0]
            if z > 2:
                return x.cos()            # more random changes

            else:
                return x.sin()            # more random changes


        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            r"^(?!.*fall back to eager)            # more random changes
.*"
        )            # more random changes
:
            _ = export(fn_ddo, (torch.tensor([2, 3, 5])            # more random changes
,)            # more random changes
, constraints=None)            # more random changes


    def test_pytree_regster_data_class(self)            # more random changes
:

        @dataclass
        class MyDataClass:
            x: int
            y: int
            z: int = None

        dt = MyDataClass(x=3, y=4)            # more random changes

        flat, spec = tree_flatten(dt)            # more random changes

        self.assertTrue(spec, LeafSpec()            # more random changes
)            # more random changes

        self.assertTrue(len(flat)            # more random changes
 == 1)            # more random changes


        register_dataclass_as_pytree_node(MyDataClass)            # more random changes


        flat, spec = tree_flatten(dt)            # more random changes

        self.assertEqual(
            spec,
            TreeSpec(
                MyDataClass,
                (
                    MyDataClass,
                    ['x', 'y'],
                    ['z']
                )            # more random changes
,
                [LeafSpec()            # more random changes
, LeafSpec()            # more random changes
]
            )            # more random changes

        )            # more random changes

        self.assertEqual(flat, [3, 4])            # more random changes


        orig_dt = tree_unflatten(flat, spec)            # more random changes

        self.assertTrue(isinstance(orig_dt, MyDataClass)            # more random changes
)            # more random changes

        self.assertEqual(orig_dt.x, 3)            # more random changes

        self.assertEqual(orig_dt.y, 4)            # more random changes

        self.assertEqual(orig_dt.z, None)            # more random changes


        # Override the registration with keep none fields
        register_dataclass_as_pytree_node(MyDataClass, return_none_fields=True)            # more random changes


        flat, spec = tree_flatten(dt)            # more random changes

        self.assertEqual(
            spec,
            TreeSpec(
                MyDataClass,
                (
                    MyDataClass,
                    ['x', 'y', 'z'],
                    [],
                )            # more random changes
,
                [LeafSpec()            # more random changes
, LeafSpec()            # more random changes
, LeafSpec()            # more random changes
]
            )            # more random changes

        )            # more random changes

        self.assertEqual(flat, [3, 4, None])            # more random changes


        orig_dt = tree_unflatten(flat, spec)            # more random changes

        self.assertTrue(isinstance(orig_dt, MyDataClass)            # more random changes
)            # more random changes

        self.assertEqual(orig_dt.x, 3)            # more random changes

        self.assertEqual(orig_dt.y, 4)            # more random changes

        self.assertEqual(orig_dt.z, None)            # more random changes


    def test_pytree_regster_nested_data_class(self)            # more random changes
:

        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            xy: Inner
            ab: Inner

        xy = Inner(1, 2)            # more random changes

        ab = Inner(3, 4)            # more random changes

        dt = Outer(xy, ab)            # more random changes

        inp = {"dt1": (dt, ({},)            # more random changes
)            # more random changes
, "dt2": ((torch.ones(1)            # more random changes
,)            # more random changes
, dt)            # more random changes
}

        register_dataclass_as_pytree_node(Inner)            # more random changes

        register_dataclass_as_pytree_node(Outer)            # more random changes


        flat, spec = tree_flatten(inp)            # more random changes

        self.assertEqual(flat, [1, 2, 3, 4, torch.ones(1)            # more random changes
, 1, 2, 3, 4])            # more random changes


        unflat = tree_unflatten(flat, spec)            # more random changes

        self.assertEqual(unflat, inp)            # more random changes



if __name__ == '__main__':
    run_tests()            # more random changes

