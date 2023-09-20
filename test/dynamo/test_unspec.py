# Owner(s): ["module: dynamo"]
import math
import random
import unittest

import numpy as np
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn.functional as F

from torch._dynamo.comptime import comptime
from torch._dynamo.testing import same


# The intention of this test file is you should put test cases specifically
# for assume_static_by_default=False, aka you want to YOLO make everything as
# dynamic as possible.  If you want to test the more normal situation where
# you assume static by default, put it in a regular test file and
# test_dynamic_shapes will cover both the YOLO and non-YOLO cases.


@torch._dynamo.config.patch(assume_static_by_default=False)
class UnspecTests(torch._dynamo.test_case.TestCase):
    def test_numpy_correctness(self):
        def fn(x, y, z):
            xy = [x + y, y, False]
            np_x = x.numpy()
            np_y = y.numpy()
            return {
                "x": x,
                "z": z,
                "a": np_y.sum(),
                "b": xy,
                "c": np_y[0][0] / 68,
                "d": np_x.sum(),
                "e": np_x + np_y,
            }, x + np_y.sum() + z

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.ones([2, 2], dtype=torch.int64)
        z = np.int64(12)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x, y, z)
        self.assertEqual(res1, res2)

    def test_no_recompilations(self):
        # no recompilations if passing on different numpy int values
        def fn(x, y):
            return {"a": x + 1, "b": y / 2}

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        for i in range(10):
            opt_fn(x, np.int64(i))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_builtin_max_min(self):
        # test unspecialized primitive max/min
        def fn(x, y, z):
            return z + 1, max(x, y), min(x - 4, y)

        x = np.int64(12)
        y = 10
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x, y, z)
        self.assertTrue(same(res1, res2, relax_numpy_equality=True))

    def test_feed_random_values_into_graph_only(self):
        def fn(shape):
            torch.manual_seed(123)
            x = torch.randn(shape, device="cpu") * random.randint(30, 100)
            return x

        shape = [2, 3]
        random.seed(1)
        res1 = fn(shape)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        random.seed(1)
        res2 = opt_fn(shape)

        self.assertTrue(same(res1, res2))

    def test_random_values_with_graph_break(self):
        def fn(x):
            r1 = random.random()
            y = x + random.uniform(10, 20)
            y.sum().item()
            r2 = random.randint(2, 18)  # no graph output in this frame
            y.sum().item()
            return y + r1, r2

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    # Really annoying intersection of specialization and RandomValueSource
    # If we get a RandomValueSource with a single element tensor, we should return a ConstantVariable like other
    # unspects... but if we do, we break the bytecode assumptions and guards will not work as we will be reffering
    # to a name from a source that is not there. If we call .item() and take the wrapped_value out, where we do
    # wrapped_value = wrapped_value.item() where we send unspec down to wrap_fx_proxy, this test passes and then
    # some models fail on missing codegen.tx.output.random_values_var. If we let the tensor value go into wrap as
    # it is, this test fails.
    # The real solution here is to rewrite RandomValueSource and all the codegen it does from the ground up.
    def test_multiple_consecutive_random_calls_before_graph(self):
        def fn(x):
            dim1 = random.randrange(start=0, stop=5)
            dim2 = random.randrange(start=0, stop=5)
            dim3 = random.randrange(start=0, stop=5)
            y = torch.rand(dim1, dim2, dim3)
            return x + 2, y

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_compiled_random_calls_are_random(self):
        # For compiled functions with random calls,
        # it should return different values for every iteration.
        # https://github.com/pytorch/pytorch/issues/95425
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return (x + 1) * random.uniform(0, 1)

        res = []
        for _ in range(5):
            res.append(fn(torch.ones(2)))
        for i in range(1, 5):
            self.assertFalse(same(res[i - 1], res[i]))

    def test_random_call_with_while_loop(self):
        def fn(x):
            dim1 = random.randrange(start=0, stop=3)
            dim2 = dim1
            while dim1 == dim2:
                dim2 = random.randrange(start=0, stop=3)
            return x * 2

        x = torch.randn(4)
        random.seed(1)
        res1 = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_builtin_getitem(self):
        # builtin getitem args[0] is python list and args[1] is unspec
        def fn(x, idx):
            return (torch.zeros(idx), x[idx], x[idx:])

        x = list(range(50))
        ref = fn(x, 48)  # 48 is unspecialized
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x, 48)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_builtin_functions_on_cuda(self):
        def fn(x, scaler):
            m = torch.nn.ReLU()
            y = m(x) * scaler
            return y

        x = torch.randn([3, 6], device="cuda")
        scaler = 0.23  # 0.23 is unspecialized
        ref = fn(x, scaler)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x, scaler)
        self.assertTrue(same(ref, res))
        self.assertEqual(ref.device, res.device)

    def test_unspec_float_precision(self):
        def fn(image, scale_factor):
            image = torch.nn.functional.interpolate(
                image[None],
                size=None,
                scale_factor=scale_factor,
                mode="bilinear",
                recompute_scale_factor=True,
                align_corners=False,
            )[0]

            return image.shape

        x = torch.rand([3, 427, 640])
        scale_factor = 1.873536229133606
        ref = fn(x, scale_factor)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x, scale_factor)
        self.assertTrue(same(ref, res))

    def test_specializing_numpy_float_in_control_flow(self):
        # np.float is unspecialized by default,
        # but it should be specialized when used in control flow.
        def fn(x, y):
            if y > 1.0:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        for t in [np.float16, np.float32, np.float64]:
            y = t(1.23)
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))

    def test_shape_graph_break(self):
        from torch._dynamo.comptime import comptime

        def fn(x):
            x_shape = x.size()
            comptime.graph_break()
            return x + torch.randn(x_shape)

        x = torch.randn(20)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn(x)

    def test_isinstance_symint(self):
        def fn(x):
            assert isinstance(x.size(0), int)
            return x * 2

        x = torch.randn(20)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn(x)
        y = torch.randn(30)
        torch._dynamo.mark_dynamic(y, 0)
        opt_fn(y)

    def test_mark_01_dynamic(self):
        def fn(x):
            return x * 2

        x = torch.randn(1)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # This will fail to compile a generic kernel, but we should not
        # complain about it (mark dynamic will try its best but 0/1
        # specialization is allowed)
        opt_fn(x)

    @unittest.expectedFailure
    def test_conv1d_symint_padding(self):
        kernel = torch.randn(1, 1, 4)

        def func(x):
            padding = math.ceil((kernel.shape[-1] + x.shape[-1] % 2) / 2) - 1
            out = F.conv1d(x, kernel, padding=padding, stride=2)
            return out

        # TODO: NameError: name 's1' is not defined when dynamic=True
        opt_func = torch.compile(func)

        x = torch.randn(1, 1, 175)
        opt_func(x)  # passes
        x = torch.randn(1, 1, 249)
        opt_func(x)  # crashes

    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_propagate_dynamic_dim(self):
        x = torch.randn(20)
        torch._dynamo.mark_dynamic(x, 0)

        @torch.compile()
        def fn(x):
            y = x * 2
            comptime.graph_break()
            z = y * 2
            return z

        z = fn(x)
        self.assertEqual(z._dynamo_weak_dynamic_indices, {0})

    def test_rshift_dynamic(self):
        def shift_right(tensor: torch.Tensor) -> torch.Tensor:
            return (tensor >> 2).to(torch.long)

        opt_fn = torch.compile(shift_right, fullgraph=True, dynamic=True)
        sample_input = torch.tensor([4, 4, 16, 32], dtype=torch.uint8)
        opt_fn(sample_input)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symfloat_to_tensor(self):
        def f1(v):
            return torch.tensor([v.item()])

        def f2(v):
            return torch.tensor([[v.item()], [2.0]])

        def f3(v):
            return torch.tensor(v.item())

        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        r = torch.randn(1)

        self.assertEqual(f1(r), optimize(f1)(r))
        self.assertEqual(f2(r), optimize(f2)(r))
        self.assertEqual(f3(r), optimize(f3)(r))

    def test_sym_int_conversion(self):
        def f(x):
            y = x.size(0)
            return x * int(y == 0)

        opt_fn = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(2, 3)
        opt_fn(x)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
