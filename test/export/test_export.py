# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import _export, export, dynamic_dim
from torch._export.trace import do_not_use_experimental_export
from torch._export.constraints import constrain_as_size
from torch._export.graph_module import get_export_meta
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestExperimentalExport(TestCase):
    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
    def test_export_simple_model_with_attr(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                y = x + self.float_val
                return y.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    @unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
    def test_export_simple_model(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                return x.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
    def test_export_simple_model_buffer_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 1))

            def forward(self, x):
                self.buffer1.add_(2)
                return x.cos() + self.buffer1.sin()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        mutated_buffer, output = exported_program.fw_module(*inp)
        # TODO (tmanlaibaatar) enable this once we figure out
        # how to do buffer mutation
        # self.assertEqual(mutated_buffer.sum().item(), 30)
        self.assertEqual(output, mod(*inp))


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    def test_export_constraints(self):

        def f(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = _export(f, inp)
        res = gm(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    def test_export_constraints_error(self):
        def invalid_size(x):
            b = x.item()
            constrain_as_size(b, min=0, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Unable to set min size"):
            _export(invalid_size, inp)

        def invalid_input_conflict_with_inline_constraints(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([6]),)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid value 6 for range"):
            _export(invalid_input_conflict_with_inline_constraints, inp)

        def invalid_input_conflict_with_input_constraints(x):
            return x + 1

        inp = torch.zeros([3])
        inp_constraints = [
            dynamic_dim(inp, 0) > 5,
        ]
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "not in range"):
            _export(
                invalid_input_conflict_with_input_constraints,
                (inp,),
                constraints=inp_constraints,
            )


        def conflicting_constraints(x):
            b = x.item()
            constrain_as_size(b, min=2, max=3)
            constrain_as_size(b, min=4, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)

        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid ranges"):
            _export(conflicting_constraints, inp)

    def test_export_assume_static_by_default(self):
        def branch_on_shape(x: torch.Tensor):
            if x.shape[0] == 4:
                return x + 1
            else:
                return x

        inp = (torch.rand(4, 5),)

        # Being able to export means shape is preserved as static
        _export(branch_on_shape, inp)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):
    def test_capture_multiple(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

            def method2(
                self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                return x + y - z

        module = MultipleMethodModule()
        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
            "method2": (torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)),
        }

        mmep = export(module, method_name_to_args)

        for method_name, args in method_name_to_args.items():
            eager_method = getattr(module, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_merge(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        class AnotherMultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method3(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module1 = MultipleMethodModule()
        method_name_to_args1 = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
        }

        module2 = AnotherMultipleMethodModule()
        method_name_to_args2 = {
            "method2": (torch.rand(2, 2), torch.rand(2, 2)),
            "method3": (torch.rand(2, 2),),
        }

        mmep1 = export(module1, method_name_to_args1)
        mmep2 = export(module2, method_name_to_args2)

        mmep1.merge(mmep2)
        self.assertEqual(
            len(mmep1.methods()), len(method_name_to_args1) + len(method_name_to_args2)
        )

        for method_name, args in method_name_to_args1.items():
            eager_method = getattr(module1, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep1.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

        for method_name, args in method_name_to_args2.items():
            eager_method = getattr(module2, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep1.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_merge_failure(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        class AnotherMultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method1(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method2(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module1 = MultipleMethodModule()
        method_name_to_args1 = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
        }

        module2 = AnotherMultipleMethodModule()
        method_name_to_args2 = {
            "method1": (torch.rand(2, 2), torch.rand(2, 2)),
            "method2": (torch.rand(2, 2),),
        }

        mmep1 = export(module1, method_name_to_args1)
        mmep2 = export(module2, method_name_to_args2)

        with self.assertRaisesRegex(
            AssertionError, "There already is a method named method1"
        ):
            mmep1.merge(mmep2)

    def test_capture_multiple_part_of_method(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

            def method2(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        module = MultipleMethodModule()
        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
            # Intentionally do not capture method2
        }

        mmep = export(module, method_name_to_args)

        # Check that only `forward` and `method1` are captured.
        self.assertEqual(len(mmep.methods()), 2)

        for method_name, args in method_name_to_args.items():
            eager_method = getattr(module, method_name)
            eager_results = eager_method(*args)

            exported_method = mmep.find_method(method_name)
            self.assertIsNotNone(exported_method)
            exported_results = exported_method(*args)

            self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_no_method_specified(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        method_name_to_args = {}

        with self.assertRaisesRegex(AssertionError, "Expected at least 1 graph module"):
            _ = export(module, method_name_to_args)

    def test_capture_multiple_program_property_access_success_forward(self) -> None:
        """
        A MultiMethodExportedProgram should allow property access even if
        it contains multiple methods as long as one of the method is named
        `forward`
        """

        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
            "method1": (torch.rand(2, 2),),
        }

        mmep = export(module, method_name_to_args)
        self.assertEqual(len(mmep.methods()), 2)

        forward_method = mmep.find_method("forward")
        self.assertEqual(mmep.module, forward_method)
        self.assertEqual(mmep.meta, forward_method.meta)
        meta = get_export_meta(forward_method)
        self.assertEqual(mmep.in_spec, meta.in_spec)
        self.assertEqual(mmep.out_spec, meta.out_spec)
        self.assertEqual(mmep.graph, forward_method.graph)
        self.assertEqual(mmep.code, forward_method.code)

    def test_capture_multiple_program_property_access_success_non_forward(self) -> None:
        """
        A MultiMethodExportedProgram should allow property access if it only
        contains a single method even if the method isn't named `forward`
        """

        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        method_name_to_args = {
            "method1": (torch.rand(2, 2),),
        }

        mmep = export(module, method_name_to_args)
        self.assertEqual(len(mmep.methods()), 1)

        method1_gm = mmep.find_method("method1")
        self.assertEqual(mmep.module, method1_gm)
        self.assertEqual(mmep.meta, method1_gm.meta)
        meta = get_export_meta(method1_gm)
        self.assertEqual(mmep.in_spec, meta.in_spec)
        self.assertEqual(mmep.out_spec, meta.out_spec)
        self.assertEqual(mmep.graph, method1_gm.graph)
        self.assertEqual(mmep.code, method1_gm.code)

    def test_capture_multiple_program_property_access_failure(self) -> None:
        """
        A MultiMethodExportedProgram should NOT allow property access when
        there are multiple methods captured and none of them is named `forward`
        """

        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

            def method2(self, x: torch.Tensor) -> torch.Tensor:
                return x + x + 1

        module = MultipleMethodModule()
        method_name_to_args = {
            "method1": (torch.rand(2, 2),),
            "method2": (torch.rand(2, 2),),
        }

        mmep = export(module, method_name_to_args)
        self.assertEqual(len(mmep.methods()), 2)

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.module

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.meta

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.in_spec

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.out_spec

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.graph

        with self.assertRaisesRegex(
            AssertionError, "impossible to identify the default method"
        ):
            _ = mmep.code

    def test_capture_multiple_non_module_callable(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        args = (torch.rand(2, 2), torch.rand(2, 2))
        mmep = export(fn, args)
        self.assertEqual(len(mmep.methods()), 1)

        eager_results = fn(*args)

        exported_method = mmep.find_method("forward")
        self.assertIsNotNone(exported_method)
        exported_results = exported_method(*args)

        self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_capture_multiple_non_module_callable_dict_args(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        method_name_to_args = {
            "forward": (torch.rand(2, 2), torch.rand(2, 2)),
        }

        with self.assertRaisesRegex(
            AssertionError, "must be a tuple of tracing inputs"
        ):
            _ = export(fn, method_name_to_args)

    def test_capture_multiple_capture_default_forward(self) -> None:
        class MultipleMethodModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def method1(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        module = MultipleMethodModule()
        args = (torch.rand(2, 2), torch.rand(2, 2))

        mmep = export(module, args)

        self.assertEqual(len(mmep.methods()), 1)

        eager_results = module(*args)

        exported_method = mmep.find_method("forward")
        self.assertIsNotNone(exported_method)
        exported_results = exported_method(*args)

        self.assertTrue(torch.allclose(eager_results, exported_results))

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        def fn_ddo(x):
            y = x.nonzero()
            z = y.shape[0]
            if z > 2:
                return x.cos()
            else:
                return x.sin()

        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "trying to get a value out of symbolic int"
        ):
            _ = _export(fn_ddo, (torch.tensor([2, 3, 5]),), constraints=None)

    def test_if_functional(self):
        def foo(x):
            x.add_(4)
            y = x.view(x.shape)
            return x.cos() + y.cos()

        gm = _export(foo, (torch.tensor([2, 3, 5]),), constraints=None)

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


if __name__ == '__main__':
    run_tests()
