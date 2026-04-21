# Owner(s): ["module: dynamo"]
import unittest
from unittest import mock

import torch
import torch._dynamo.testing
import torch._functorch.config
import torch._inductor.test_case
from torch._dynamo.testing import same
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests


requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


class MiscTestsDevice(torch._inductor.test_case.TestCase):
    def test_rand(self, device):
        cnts = torch._dynamo.testing.CompileCounter()
        device = device  # noqa: PLW0127

        def fn():
            return torch.randn(10, device=device)

        torch.manual_seed(10)
        ref_run1 = fn()

        torch.manual_seed(10)
        ref_run2 = fn()
        self.assertTrue(same(ref_run1, ref_run2))

        torch.manual_seed(10)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn()

        self.assertTrue(same(res, ref_run1))

    def test_torch_device_is_available(self, device):
        def fn(x):
            if torch.accelerator.is_available():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable(self, device):
        def fn(x):
            if torch.backends.cudnn.is_acceptable(tensor=x):
                return x + 1
            return x

        x = torch.rand(4).to(device)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable_bad_inputs(self, device):
        def fn1(x):
            if torch.backends.cudnn.is_acceptable("invalid"):
                return x + 1
            return x

        def fn2(x):
            if torch.backends.cudnn.is_acceptable(x, 3.14):
                return x + 1
            return x

        with self.assertRaisesRegex(
            AssertionError, "Expect input to cudnn.is_acceptable to be a tensor"
        ):
            x1 = torch.rand(4).to(device)
            opt_fn1 = torch.compile(fn1, backend="eager", fullgraph=True)
            res1 = opt_fn1(x1)  # noqa: F841

        with self.assertRaisesRegex(
            AssertionError, "Expect 1 input to cudnn.is_acceptable"
        ):
            x2 = torch.rand(4).to(device)
            opt_fn2 = torch.compile(fn2, backend="eager", fullgraph=True)
            res = opt_fn2(x2)  # noqa: F841

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @torch._dynamo.config.patch(recompile_limit=999)
    def test_legacy_cuda_tensor(self):
        typs = [
            torch.cuda.FloatTensor,
            torch.cuda.DoubleTensor,
            torch.cuda.HalfTensor,
            torch.cuda.BFloat16Tensor,
            torch.cuda.ByteTensor,
            torch.cuda.CharTensor,
            torch.cuda.IntTensor,
            torch.cuda.ShortTensor,
            torch.cuda.LongTensor,
        ]

        def f2(typ):
            return typ([1, 2, 3])

        compiled_f2 = torch.compile(f2, backend="eager", fullgraph=True)
        for typ in typs:
            output = compiled_f2(typ)
            expected = f2(typ)
            self.assertEqual(output, expected)

    def test_get_device(self, device):
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x.get_device(), y.get_device()

        x = torch.rand(4, device=device)
        y = torch.rand(4, device="cpu")
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_symint_as_device_kwarg(self, device):
        def f(rank):
            # -2 to make device id 0 for easier testing on CI
            return torch.ones(10, device=rank.size(0) - 2)

        x = torch.randn(2)
        out = f(torch.randn(2))
        opt_out = torch.compile(backend="eager", dynamic=True, fullgraph=True)(f)(x)
        self.assertEqual(out, opt_out)

    def test_torch_device_python_type(self, device):
        device_type = torch.device(device).type
        for device, device_type, index in [  # noqa: B020
            ("cpu", "cpu", None),
            (device, device_type, 0),
        ]:

            def fn(target):
                target_device = target.device
                a = torch.zeros(2, 3, device=target_device)
                # Constant assert at trace time
                assert isinstance(target_device, torch.device)  # noqa: S101
                assert target_device.type == device_type  # noqa: S101
                assert target_device.index == index  # noqa: S101
                b = torch.zeros(2, 3, device=target_device)
                c = torch.zeros(2, 3, device=target_device)
                return a + b + c

            from torch._dynamo.variables import ConstantVariable

            device = torch.device(device)
            expected_variable = ConstantVariable(device)
            self.assertEqual(expected_variable.python_type(), type(device))

            opt_func = torch.compile(fn, backend="eager", fullgraph=True)
            a = torch.tensor([2, 3], device=device)
            res = opt_func(a)
            self.assertIsInstance(res, torch.Tensor)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_interpolate_propagate_real_tensors(self, device):
        @torch.compile(backend="eager", fullgraph=True)
        def f(mask, box):
            # u0, u1 = mask.tolist()
            mask = torch.randn(1, 1, 30, 30, device=device)
            h, w = box.tolist()
            return torch.nn.functional.interpolate(
                mask, (h, w), mode="bilinear", align_corners=False
            )

        f(torch.tensor([30, 30], device=device), torch.tensor([68, 32], device=device))

    def test_scalar_isin_decomposition(self):
        def f():
            x = torch.tensor(0)
            return torch.isin(x, x)

        opt_f = torch.compile(f, backend="inductor", fullgraph=True)
        ref = f()
        res = opt_f()
        self.assertEqual(ref, res)

    def test_randint_no_graphbreak(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(actions, n_act, epsilon=0.1):
            actions_random = torch.randint_like(actions, n_act)

            return actions_random

        x = torch.ones([1], dtype=torch.int64)
        y = torch.tensor(5)
        f(x, y)

    def test_full_graph_capture_scalar_outputs(self):
        @torch.compile(fullgraph=True, backend="eager")
        def foo(a):
            return torch.randn(5) * a.item()

        # We expect to no longer raise here
        foo(torch.tensor(2.0))

    def test_full_graph_capture_dynamic_output_shape_ops(self):
        def fn(x):
            nz = torch.nonzero(x)
            squared = nz * nz
            sliced = torch.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            view = sliced.unsqueeze(dim=0)
            return view.squeeze(dim=0)

        example_inputs = (torch.randn(1, 1, 1, 1),)
        # we expect to no longer raise here
        torch.compile(fn, fullgraph=True, backend="eager")(*example_inputs)

    def test_dynamic_fill_diagonal_(self):
        @torch.compile(dynamic=True, backend="eager")
        def f(x):
            x.fill_diagonal_(True)

        x = torch.zeros(4, 4)
        f(x)

    def test_dynamic_float_scalar_tensor_coersion(self):
        # Minified version of https://github.com/pytorch/pytorch/issues/158376#issuecomment-3079591367
        class Foo:
            def __init__(self):
                self.config = type(
                    "Config", (), {"pad_val": 1123581321.0, "tolerance": 1e-6}
                )

            @torch.compile(fullgraph=True, backend="eager")
            def forward(self, input):
                outputs = torch.where(
                    torch.abs(input - self.config.pad_val) < self.config.tolerance,
                    torch.tensor(
                        self.config.pad_val, dtype=input.dtype, device=input.device
                    ),
                    torch.tensor(
                        self.config.pad_val + 1, dtype=input.dtype, device=input.device
                    ),
                )
                return outputs

        foo = Foo()
        inputs = torch.randn(3, 4)
        result = foo.forward(inputs)  # noqa: F841

        original_pad_val = foo.config.pad_val  # noqa: F841
        foo.config.pad_val += 1.0
        result2 = foo.forward(inputs)  # noqa: F841


class ReproDeviceRuntimeTests(torch._inductor.test_case.TestCase):
    def test_guard_default_device(self, device):
        try:
            torch.set_default_device(device)

            counter = torch._dynamo.testing.CompileCounter()

            @torch._dynamo.optimize(counter)
            def f():
                x = torch.randn(3)
                return x * 2

            self.assertEqual(f().device.type + ":0", device)
            self.assertEqual(counter.frame_count, 1)

            torch.set_default_device("cpu")

            self.assertEqual(f().device.type, "cpu")
            self.assertEqual(counter.frame_count, 2)

        finally:
            torch.set_default_device(None)

    def test_torch_cuda_is_initialized(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            if torch.cuda.is_initialized():
                return x + 1
            return x + 2

        inp = torch.randn(3)
        self.assertEqual(f(inp), inp + 1)

        with mock.patch("torch.cuda.is_initialized", lambda: False):
            self.assertEqual(f(inp), inp + 2)

    @requires_cuda
    def test_zero_dim_param_mixed_device_grad(self):
        class RegressionModel(torch.nn.Module):
            def __init__(self, a=0, b=0):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(a).float())
                self.b = torch.nn.Parameter(torch.tensor(b).float())

            def forward(self, x):
                return x * self.a + self.b

        model = RegressionModel()
        model.forward = torch.compile(
            model.forward, backend="aot_eager", fullgraph=True
        )
        inputs = torch.randn(4, 10).to("cuda")
        out = model(inputs)
        out.sum().backward()
        self.assertIsNotNone(model.a.grad)
        self.assertIsNotNone(model.b.grad)
        self.assertEqual(model.a.grad.device, torch.device("cpu"))
        self.assertEqual(model.b.grad.device, torch.device("cpu"))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_cuda_sync(self):
        def fn(x):
            y = x + 1
            torch.cuda.synchronize()
            return y * 2

        x = torch.ones(2, device="cuda")
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_current_accelerator(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            torch.accelerator.current_accelerator()
            return x + 1

        self.assertEqual(fn(torch.ones(3)), torch.ones(3) + 1)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    MiscTestsDevice, globals(), only_for=devices, allow_xpu=True
)
instantiate_device_type_tests(
    ReproDeviceRuntimeTests, globals(), only_for=("cuda", "hpu")
)
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
