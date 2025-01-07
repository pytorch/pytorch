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
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend, same
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.logging_utils import logs_to_string


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
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x, y, z)
        self.assertEqual(res1, res2)

    def test_no_recompilations(self):
        # no recompilations if passing on different numpy int values
        def fn(x, y):
            return {"a": x + 1, "b": y / 2}

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for i in range(10):
            opt_fn(x, np.int64(i))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    @unittest.expectedFailure  # array scalars decay to 0D arrays
    def test_builtin_max_min(self):
        # test unspecialized primitive max/min
        def fn(x, y, z):
            return z + 1, max(x, y), min(x - 4, y)

        x = np.int64(12)
        y = 10
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    # Really annoying intersection of specialization and RandomValueSource
    # If we get a RandomValueSource with a single element tensor, we should return a ConstantVariable like other
    # unspects... but if we do, we break the bytecode assumptions and guards will not work as we will be referring
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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend="eager")
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

        random.seed(10)
        res1 = fn(x)
        random.seed(10)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_random_object(self):
        # test argument passing, mutation, reconstruction, state correctness
        def fn(x, rand2):
            r1 = random.randint(1, 9)
            r2 = rand2.randint(1, 9)
            rand3 = random.Random(42)
            r3 = rand3.randint(1, 9)

            y = x + r1 + r2 + r3
            return y, rand2, rand3

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        random.seed(0)
        y_1, rand2_1, rand3_1 = fn(inp, random.Random(12))
        state_1 = random.getstate()
        random.seed(0)
        y_2, rand2_2, rand3_2 = opt_fn(inp, random.Random(12))
        state_2 = random.getstate()
        self.assertEqual(y_1, y_2)
        self.assertEqual(state_1, state_2)
        self.assertEqual(rand2_1.getstate(), rand2_2.getstate())
        self.assertEqual(rand3_1.getstate(), rand3_2.getstate())

    def test_random_object_methods(self):
        def fn(x, rand1, rand2, rand3):
            rand1.seed(42)
            rand4 = random.Random(9002)
            rand2.setstate(rand4.getstate())
            r1 = rand1.random()
            r2 = rand2.randint(1, 10)
            r3 = rand3.randrange(10)
            r4 = rand4.uniform(0, 1)
            return x + r1 + r2 + r3 + r4

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        rand1_1 = random.Random(1)
        rand2_1 = random.Random(2)
        rand3_1 = random.Random(3)
        rand1_2 = random.Random(1)
        rand2_2 = random.Random(2)
        rand3_2 = random.Random(3)
        y1 = fn(inp, rand1_1, rand2_1, rand3_1)
        y2 = opt_fn(inp, rand1_2, rand2_2, rand3_2)
        self.assertEqual(y1, y2)
        self.assertEqual(rand1_1.getstate(), rand1_2.getstate())
        self.assertEqual(rand2_1.getstate(), rand2_2.getstate())
        self.assertEqual(rand3_1.getstate(), rand3_2.getstate())

    def test_random_object_overriden_methods(self):
        # these will result in graph breaks, but we shouldn't crash
        def get_rng():
            rand1 = random.Random(1)
            rand2 = random.Random(2)

            orig_random = rand1.random

            def custom_random():
                return orig_random()

            orig_getstate = rand2.getstate

            def custom_getstate():
                return orig_getstate()

            rand1.random = custom_random
            rand2.getstate = custom_getstate
            return rand1, rand2

        def fn(x, rand1, rand2):
            r1 = rand1.random()
            rand3 = random.Random()
            rand3.setstate(rand2.getstate())
            r2 = rand3.random()
            return x + r1 + r2

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        y1 = fn(inp, *get_rng())
        y2 = opt_fn(inp, *get_rng())
        self.assertEqual(y1, y2)

    def test_builtin_getitem(self):
        # builtin getitem args[0] is python list and args[1] is unspec
        def fn(x, idx):
            return (torch.zeros(idx), x[idx], x[idx:])

        x = list(range(50))
        ref = fn(x, 48)  # 48 is unspecialized
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, 48)
        self.assertTrue(same(ref, res))

    def test_use_and_specialize(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            x = x + y
            if y == 2:
                return x - 1
            else:
                return x + 1

        self.assertTrue(same(fn(torch.tensor([5]), 2), 6))
        self.assertTrue(same(fn(torch.tensor([6]), 2), 7))
        self.assertTrue(same(fn(torch.tensor([5]), 3), 9))
        self.assertTrue(same(fn(torch.tensor([4]), 3), 8))
        self.assertEqual(cnt.frame_count, 2)

    def test_no_recompiles(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            return x + y

        self.assertTrue(same(fn(torch.tensor([5]), 100), 105))
        self.assertTrue(same(fn(torch.tensor([4]), 200), 204))
        self.assertTrue(same(fn(torch.tensor([3]), 300), 303))
        self.assertTrue(same(fn(torch.tensor([2]), 400), 402))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_no_recompiles_prod_backward(self):
        # https://github.com/pytorch/pytorch/issues/120608
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(t):
            return torch.prod(t, 3, keepdim=True)

        input_shapes = [(8, 10, 3, 2), (8, 3, 5, 2), (8, 4, 8, 2)]
        for s in input_shapes:
            t1 = torch.randn(s, requires_grad=True)
            h_result = fn(t1)
            grad = torch.ones_like(h_result)
            h_result.backward(grad)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, scale_factor)
        self.assertTrue(same(ref, res))

    @unittest.expectedFailure  # fails as long as numpy scalars are 0D arrays
    def test_specializing_numpy_float_in_control_flow(self):
        # np.float64 is unspecialized by default,
        # but it should be specialized when used in control flow.
        def fn(x, y):
            if y > 1.0:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        for t in [np.float16, np.float32, np.float64]:
            y = t(1.23)
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))

    def test_mark_static_inside(self):
        def fn(x):
            torch._dynamo.mark_static(x, 0)
            comptime.assert_static(x.size(0))
            return x + 1

        opt_fn = torch.compile(fn, dynamic=True, fullgraph=True)
        opt_fn(torch.randn(12, 23))

    def test_shape_graph_break(self):
        from torch._dynamo.comptime import comptime

        def fn(x):
            x_shape = x.size()
            comptime.graph_break()
            return x + torch.randn(x_shape)

        x = torch.randn(20)
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)

    def test_isinstance_symint(self):
        def fn(x):
            assert isinstance(x.size(0), int)
            return x * 2

        x = torch.randn(20)
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)
        y = torch.randn(30)
        torch._dynamo.mark_dynamic(y, 0)
        opt_fn(y)

    def test_mark_01_dynamic(self):
        def fn(x):
            return x * 2

        x = torch.randn(1)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager")
        # This will fail to compile a generic kernel, but we should not
        # complain about it (mark dynamic will try its best but 0/1
        # specialization is allowed)
        opt_fn(x)

    def test_conv1d_symint_padding(self):
        kernel = torch.randn(1, 1, 4)

        def func(x):
            padding = math.ceil((kernel.shape[-1] + x.shape[-1] % 2) / 2) - 1
            out = F.conv1d(x, kernel, padding=padding, stride=2)
            return out

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

        def f4(v):
            return torch.tensor((v.item(),))

        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        r = torch.randn(1)

        self.assertEqual(f1(r), optimize(f1)(r))
        self.assertEqual(f2(r), optimize(f2)(r))
        self.assertEqual(f3(r), optimize(f3)(r))
        self.assertEqual(f4(r), optimize(f4)(r))

    @skipIfWindows(
        msg="AssertionError: The values for attribute 'dtype' do not match: torch.int32 != torch.int64."
    )
    def test_to_tensor(self):
        def f1():
            a = np.random.uniform(low=-1, high=1, size=(20, 1))
            return torch.tensor([a, a, a, a], dtype=torch.float64, device="cpu")

        def f2():
            a = torch.tensor([[[123]]])
            return torch.tensor([a, a])

        def f3():
            a = torch.tensor(123)
            return torch.tensor([a, a])

        def f4():
            a = torch.tensor(123)
            b = torch.tensor([[[456]]])
            return torch.tensor([a, b])

        def f5():
            a = np.array([1, 2])
            return torch.tensor([a, a])

        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        self.assertEqual(f1().shape, optimize(f1)().shape)
        self.assertEqual(f2(), optimize(f2)())
        self.assertEqual(f3(), optimize(f3)())
        self.assertEqual(f4(), optimize(f4)())
        self.assertEqual(f5(), optimize(f5)())

    def test_sym_int_conversion(self):
        def f(x):
            y = x.size(0)
            return x * int(y == 0)

        opt_fn = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(2, 3)
        opt_fn(x)

    def test_sum_dimlist_spec(self):
        def fn(inputs, dim):
            return torch.sum(inputs, dim)

        inputs = torch.randn(128, 5, 24, 24)
        dim = (-1, 1, 0, 2)
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, dim), fn(inputs, dim))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_max(self):
        def fn(x):
            return torch.ones(max(x.item(), 1024))

        x = torch.tensor([1000])
        y = torch.tensor([2000])
        compl_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compl_fn(x))
        self.assertEqual(fn(y), compl_fn(y))

    # https://github.com/pytorch/pytorch/issues/104812
    def test_argmin_coerces_symint_to_intlist_spec(self):
        def fn(x, dim):
            # the python arg parser coerces dim into a vector<int>
            return torch.amin(x, dim=dim, keepdim=True)

        x = torch.randn(4, 4, 4)
        dim = 2
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(x, dim), fn(x, dim))

    def test_exponential(self):
        def fn(inputs, op_inputs_dict):
            res = inputs.exponential_(**op_inputs_dict)
            return res

        inputs = torch.randn(2, 3, 4)
        op_inputs_dict = {"lambd": 10, "generator": None}
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, op_inputs_dict), fn(inputs, op_inputs_dict))

    def test_symbol_guard_limit_before_specialize(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True)
        def fn(x):
            torch._check(x.size(0) != 3)
            torch._check(x.size(0) != 4)
            torch._check(x.size(0) != 5)
            torch._check(x.size(0) != 6)
            return x + 2

        # Control test
        fn(torch.randn(12))
        fn(torch.randn(13))
        fn(torch.randn(14))

        self.assertExpectedInline(cnts.frame_count, """1""")
        cnts.frame_count = 0

        torch._dynamo.reset()

        with torch.fx.experimental._config.patch(
            symbol_guard_limit_before_specialize=3
        ):
            fn(torch.randn(12))
            fn(torch.randn(13))
            fn(torch.randn(14))

            self.assertExpectedInline(cnts.frame_count, """3""")

    def test_defaults(self):
        def g(x, i=8):
            comptime.assert_static(i)
            return x * i

        def fn(x):
            return g(x)

        inputs = torch.randn(2, 3, 4)
        compl_fn = torch.compile(fn, dynamic=True, backend="eager")
        self.assertEqual(compl_fn(inputs), fn(inputs))

    @torch._dynamo.config.patch(specialize_float=False)
    def test_symfloat_no_replacement(self):
        # See https://github.com/pytorch/pytorch/pull/139250 for more context
        # The high level idea is if we don't want to set a replacement where a
        # symbol is on both the right and left side, otherwise we'll end up
        # in an infinite self._find recursion.
        def fn(t, m):
            return 2 * t if m.is_integer() else t

        t = torch.tensor([1])
        compl_fn = torch.compile(fn, dynamic=True, backend="eager")
        self.assertEqual(fn(t, 1.0), compl_fn(t, 1.0))

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspec_roundtrip_float_input(self):
        def f(x, y):
            if y == 5.0:
                return x + 2
            else:
                return x + y
            return (x, y)

        cf = torch.compile(backend="eager", fullgraph=True)(f)
        x = 1.1234567891234568
        y = 1.1234567891234569
        self.assertAlmostEqual(f(x, y), cf(x, y))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_input(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            if y == 5.0:
                return x + 2
            else:
                return x + y

        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        x = torch.randn(3)
        self.assertEqual(f(x, 2.0), cf(x, 2.0))
        self.assertEqual(f(x, 3.0), cf(x, 3.0))  # automatic dynamic kicks in here
        self.assertEqual(f(x, 4.0), cf(x, 4.0))
        self.assertExpectedInline(cnts.frame_count, """2""")  # no recompile
        self.assertEqual(f(x, 5.0), cf(x, 5.0))
        self.assertExpectedInline(cnts.frame_count, """3""")  # guard worked
        self.assertEqual(f(x, math.nan), cf(x, math.nan))
        self.assertExpectedInline(cnts.frame_count, """4""")  # nan always recompiles

    @torch._dynamo.config.patch(specialize_float=False, capture_scalar_outputs=True)
    def test_unspecialized_float_multiply_precision(self):
        dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
        for i, dtype in enumerate(dtypes):

            def fn(x, y):
                return x * y

            cnt = CompileCounterWithBackend("aot_eager")
            fn_opt = torch.compile(fn, backend=cnt)
            x = torch.randn(5, dtype=dtype, requires_grad=True)
            y1 = 1.00048828125
            y2 = 1.00048828126
            y3 = 1.00048828127

            self.assertEqual(fn_opt(x, y1), fn(x, y1))
            self.assertEqual(fn_opt(x, y2), fn(x, y2))
            self.assertEqual(fn_opt(x, y3), fn(x, y3))
            if i == 0:
                # This is kind of quirky part of automatic dynamic,
                # since it just uses source name + tx.f_code as the key
                # subsequent recompilations will actually reuse the automatic
                # dynamic choices.
                self.assertEqual(cnt.frame_count, 2)
            else:
                self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=False)
    def test_unspec_float_input_f64(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        x = torch.zeros(3, dtype=torch.float64)
        # 17 digits of precision so unrepresentable in float32
        flt = 1.2345678901234567
        self.assertEqual(f(x, flt), cf(x, flt))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_output(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + 1, y * 2

        cf = torch.compile(backend=cnts, fullgraph=True)(f)
        x = torch.randn(3)

        self.assertEqual(f(x, 3.0), cf(x, 3.0))
        self.assertEqual(f(x, 4.0), cf(x, 4.0))
        self.assertEqual(f(x, 5.0), cf(x, 5.0))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_data_dependent_evaluate_expr_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        # To ensure that the continuation frame is compiled,
        # have to write the test function in this funny way.
        # See https://github.com/pytorch/pytorch/issues/111918
        def test(y):
            if y > 2:
                return True
            else:
                return False

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1
            y = x.item()
            if test(y):
                return x * 2
            else:
                return x * 3

        x = torch.tensor([3.0])
        fn(x)

        self.assertExpectedInline(cnts.frame_count, """2""")
        self.assertExpectedInline(cnts.op_count, """4""")

    def test_prune_torch_check(self):
        log_stream, ctx = logs_to_string("torch._dynamo.output_graph", "graph_code")

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def f(x, y):
            torch._check(y + 5 == 85)
            torch._check(x.size(0) == 80)

        with ctx():
            f(torch.randn(80, 100), 80)

        out = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
        self.assertExpectedInline(
            out,
            """\
def forward(self):
        return ()""",
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_aot_autograd(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, i):
            y, z = i.tolist()
            return torch.split(x, [y, z])

        print(f(torch.randn(10, requires_grad=True), torch.tensor([7, 3])))

    def test_bool_tensor_ctor(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True, fullgraph=True)
        def f(x):
            y = torch.empty((x.size(0) // 13) * 13)
            return torch.tensor(y.numel() == 0)

        self.assertTrue(f(torch.empty(8)).item())
        self.assertFalse(f(torch.empty(13)).item())

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked(self):
        class TestModel(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                return x * 2

        main_model = TestModel()
        opt_model = torch.compile(main_model, mode="max-autotune", dynamic=True)

        x1 = torch.rand(3, 5, 4, 8)
        x2 = torch.rand(1, 5, 4, 8)

        torch._dynamo.decorators.mark_unbacked(x1, 0)

        o1_ref = main_model(x1, 2)
        o1 = opt_model(x1, 2)
        self.assertEqual(o1_ref, o1)

        o1_2_ref = main_model(x2, 2)
        o1_2 = opt_model(x2, 2)
        self.assertEqual(o1_2_ref, o1_2)

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked_hint_consistency(self):
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        x = torch.randn(1)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        @torch.compile()
        def f(x):
            if guard_size_oblivious(x.size(0) != 1):
                return x + 3
            else:
                return x + 4

        self.assertEqual(f(x), x + 3)

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked_channels_last(self):
        class TestModel(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                return x * 2

        main_model = TestModel()
        opt_model = torch.compile(main_model, mode="max-autotune", dynamic=True)

        x1 = torch.rand(3, 5, 4, 8).to(memory_format=torch.channels_last)
        x2 = torch.rand(1, 5, 4, 8).to(memory_format=torch.channels_last)

        torch._dynamo.decorators.mark_unbacked(x1, 0)

        o1_ref = main_model(x1, 2)
        o1 = opt_model(x1, 2)
        self.assertEqual(o1_ref, o1)

        o1_2_ref = main_model(x2, 2)
        o1_2 = opt_model(x2, 2)
        self.assertEqual(o1_2_ref, o1_2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
