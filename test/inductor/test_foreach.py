# Owner(s): ["module: inductor"]

import sys
import unittest

import torch
import torch._inductor
from torch._higher_order_ops import foreach_map
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
from torch.testing._internal.triton_utils import requires_cuda


aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_cuda
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_cuda,
        )
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


def foreach_map_wrapper(op):
    def wrapper(*args, **kwargs):
        return foreach_map(op, (args), **kwargs)

    wrapper.__name__ = "foreach_map_" + op.__name__

    return wrapper


def add_op(x, y):
    return torch.add(x, y)


def addrecip_op(x, y):
    return torch.reciprocal(torch.add(x, y))


def addcmul_op(x, y, z):
    return torch.mul(torch.add(x, y), z)


def recipaddmul_op(x, y, z):
    return torch.mul(torch.add(torch.reciprocal(x), y), z)


inplace_bin_ops_under_test = [
    torch._foreach_add_,
    torch._foreach_mul_,
    torch._foreach_sub_,
    torch._foreach_div_,
]
ternary_ops_under_test = [
    foreach_map_wrapper(addcmul_op),
    foreach_map_wrapper(recipaddmul_op),
]

bin_ops_under_test = [
    torch._foreach_add,
    torch._foreach_mul,
    torch._foreach_sub,
    torch._foreach_div,
    foreach_map_wrapper(torch.add),
    foreach_map_wrapper(torch.mul),
    foreach_map_wrapper(torch.sub),
    foreach_map_wrapper(torch.div),
    foreach_map_wrapper(addrecip_op),
    foreach_map_wrapper(add_op),
    torch._foreach_maximum,
    torch._foreach_minimum,
    torch._foreach_clamp_max,
    torch._foreach_clamp_min,
    aten._foreach_copy,
    foreach_map_wrapper(torch.maximum),
    foreach_map_wrapper(torch.minimum),
    foreach_map_wrapper(torch.clamp_max),
    foreach_map_wrapper(torch.clamp_min),
    foreach_map_wrapper(aten.copy),
]

un_ops_under_test = [
    torch._foreach_reciprocal,
    torch._foreach_neg,
    torch._foreach_sign,
    torch._foreach_abs,
    torch._foreach_sqrt,
    torch._foreach_rsqrt,
    foreach_map_wrapper(torch.reciprocal),
    foreach_map_wrapper(torch.neg),
    foreach_map_wrapper(torch.sign),
    foreach_map_wrapper(torch.abs),
]
compose_ops = [torch._foreach_addcdiv, torch._foreach_addcmul]
all_ops = parametrize(
    "op",
    ternary_ops_under_test + bin_ops_under_test + un_ops_under_test,
    name_fn=lambda f: f.__name__,
)
bin_ops = parametrize("op", bin_ops_under_test, name_fn=lambda f: f.__name__)
inplace_bin_ops = parametrize(
    "op", inplace_bin_ops_under_test, name_fn=lambda f: f.__name__
)
scalar_bin_ops = parametrize(
    "op", bin_ops_under_test[:10], name_fn=lambda f: f.__name__
)
scalar_tensor_bin_ops = parametrize(
    "op", bin_ops_under_test[:10], name_fn=lambda f: f.__name__
)
decomp_ops = parametrize("op", compose_ops, name_fn=lambda f: f.__name__)


def gen_args(op):
    if op in un_ops_under_test:
        return (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )
    elif op in bin_ops_under_test:
        return (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )
    else:
        return (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )


@instantiate_parametrized_tests
class ForeachTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()

    def tearDown(self):
        super().tearDown()
        torch._inductor.metrics.reset()

    def _test_single_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                return op([a0, a1])

        elif op in bin_ops_under_test:

            def fn(a0, a1, b0, b1):
                return op([a0, a1], [b0, b1])

        else:

            def fn(a0, a1, b0, b1, c0, c1):
                return op([a0, a1], [b0, b1], [c0, c1])

        self.check_model_cuda(
            fn,
            gen_args(op),
        )

    def _test_single_scalar(self, op):
        def fn(a0, a1):
            return op([a0, a1], 3.3)

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

    def _test_single_scalar_tensor(self, op):
        def fn(a0, a1):
            return op([a0, a1], torch.tensor(3.3, device="cuda:0"))

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

    # called in test_cuda_cpp_wrapper.py
    @requires_cuda
    def test_foreach_cpp_wrapper_cuda(self):
        self._test_single_list(op=torch._foreach_add)

    @requires_cuda
    @all_ops
    def test_single_list(self, op):
        self._test_single_list(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_single_scalar(self, op):
        self._test_single_scalar(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_tensor_bin_ops
    def test_single_scalar_tensor(self, op):
        self._test_single_scalar_tensor(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @all_ops
    def test_scheduler_fusion_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                c = op([a0, a1])
                return torch._foreach_sqrt(c)

        elif op in bin_ops_under_test:

            def fn(a0, a1, b0, b1):
                c = op([a0, a1], [b0, b1])
                return c, torch._foreach_add([a0, a1], c)

        else:

            def fn(a0, a1, b0, b1, c0, c1):
                c = op([a0, a1], [b0, b1], [c0, c1])
                return c, torch._foreach_add([a0, a1], c)

        self.check_model_cuda(
            fn,
            gen_args(op),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_scheduler_fusion_scalar(self, op):
        def fn(a0, a1):
            c = op([a0, a1], 3.4)
            return c, torch._foreach_add([a0, a1], c)

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_broadcasting(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        fn_opt = torch._dynamo.optimize()(fn)

        inputs = (
            torch.rand(10, 1, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(1, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )
        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @all_ops
    def test_singleton_lists(self, op):
        if op in un_ops_under_test:

            def fn(a0):
                return op([a0])

            args = (torch.rand(10, 10, device="cuda:0"),)
        elif op in bin_ops_under_test:

            def fn(a0, b0):
                return op([a0], [b0])

            args = (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
            )

        else:

            def fn(a0, b0, c0):
                return op([a0], [b0], [c0])

            args = (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
            )

        self.check_model_cuda(
            fn,
            args,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @bin_ops
    def test_type_promotion(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        fn_opt = torch._dynamo.optimize()(fn)

        max32 = torch.iinfo(torch.int32).max
        max64 = torch.iinfo(torch.int64).max
        inputs = (
            torch.randint(max32, (10, 10), device="cuda:0", dtype=torch.int32),
            torch.randint(max32, (20, 20), device="cuda:0", dtype=torch.int32),
            torch.randint(max32, (10, 10), device="cuda:0", dtype=torch.int32),
            torch.randint(max64, (20, 20), device="cuda:0", dtype=torch.int64),
        )
        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_kernel_split_arg_limit_list(self, op):
        # NB: foeach_copy won't pass this test because it will dce one set of buffers

        def fn(a, b):
            return op(a, b)

        fn_opt = torch._dynamo.optimize()(fn)

        max_args = 370
        max_list_len = (max_args // 3) + 1
        inputs = (
            [torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],
            [torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],
        )

        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @scalar_bin_ops
    @unittest.skip(
        "Triton recursion depth exceeded: https://github.com/openai/triton/issues/1763"
    )
    def test_kernel_split_arg_limit_scalar(self, op):
        def fn(a):
            return op(a, 3.3)

        fn_opt = torch._dynamo.optimize()(fn)

        max_args = 370
        max_list_len = (max_args // 2) + 1
        inputs = ([torch.rand(10, 10, device="cuda:0") for _ in range(max_list_len)],)

        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @bin_ops
    def test_fusion_duplicate_buffer_list(self, op):
        def fn(a0, a1, b0, b1):
            c = op([a0, a1], [b0, b1])
            return op([a0, b0], [c[0], c[0]])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
            reference_in_float=False,
            check_lowp=False,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @all_ops
    def test_non_foreach_consumer_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                c = op([a0, a1])
                return torch.mul(c[0], a0)

        elif op in bin_ops_under_test:

            def fn(a0, a1, b0, b1):
                c = op([a0, a1], [b0, b1])
                return torch.mul(c[0], a0)

        else:

            def fn(a0, a1, b0, b1, c0, c1):
                c = op([a0, a1], [b0, b1], [c0, c1])
                return torch.mul(c[0], a0)

        self.check_model_cuda(
            fn,
            gen_args(op),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_non_foreach_consumer_scalar(self, op):
        def fn(a0, a1):
            c = op([a0, a1], 4.7)
            return torch.mul(c[0], a0)

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @all_ops
    def test_non_foreach_producer_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                c0 = torch.add(a0, a0)
                c1 = torch.add(a1, a1)
                return op([c0, c1])

        elif op in bin_ops_under_test:

            def fn(a0, a1, b0, b1):
                c0 = torch.add(a0, b0)
                c1 = torch.add(a1, b1)
                return op([a0, a1], [c0, c1])

        else:

            def fn(a0, a1, b0, b1, c0, c1):
                c0 = torch.add(a0, b0)
                c1 = torch.add(a1, b1)
                return op([a0, a1], [b0, b1], [c0, c1])

        self.check_model_cuda(
            fn, gen_args(op), reference_in_float=False, check_lowp=False
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_non_foreach_producer_scalar(self, op):
        def fn(a0, a1, b0, b1):
            c0 = torch.mul(a0, b0)
            c1 = torch.mul(a1, b1)
            return op([c0, c1], 5.6)

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @all_ops
    def test_non_foreach_consumer_producer_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                c0 = torch.add(a0, a0)
                c1 = torch.mul(a1, a1)
                d = op([c0, c1])
                e0 = torch.mul(d[0], a0)
                e1 = torch.mul(d[1], a1)
                return [e0, e1]

        elif op in bin_ops_under_test:

            def fn(a0, a1, b0, b1):
                c0 = torch.add(a0, b0)
                c1 = torch.add(a1, b1)
                d = op([a0, a1], [c0, c1])
                e0 = torch.mul(d[0], a0)
                e1 = torch.mul(d[1], a1)
                return [e0, e1]

        else:

            def fn(a0, a1, b0, b1, c0, c1):
                c0 = torch.add(a0, b0)
                c1 = torch.add(a1, b1)
                d = op([a0, a1], [b0, b1], [c0, c1])
                e0 = torch.mul(d[0], a0)
                e1 = torch.mul(d[1], a1)
                return [e0, e1]

        self.check_model_cuda(
            fn,
            gen_args(op),
            reference_in_float=False,
            check_lowp=False,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @scalar_bin_ops
    def test_non_foreach_consumer_producer_scalar(self, op):
        def fn(a0, a1, b0, b1):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            d = op([c0, c1], 5.8)
            e0 = torch.mul(d[0], a0)
            e1 = torch.mul(d[1], a1)
            return [e0, e1]

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
            reference_in_float=False,
            check_lowp=False,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @bin_ops
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_fallback(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        self.check_model_cuda(fn, inputs)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    @torch._inductor.config.patch("combo_kernel_foreach_dynamic_shapes", True)
    def test_enable_dynamic_shapes_python_wrapper(self, op=torch._foreach_add):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        self.check_model_cuda(fn, inputs)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    @torch._inductor.config.patch("combo_kernel_foreach_dynamic_shapes", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_enable_dynamic_shapes_cpp_wrapper_cuda(self, op=torch._foreach_add):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        self.check_model_cuda(fn, inputs)

    @unittest.skipIf(IS_FBCODE, "cpp compile not supported in fbcode")
    @bin_ops
    def test_cpu_cpp_fallback(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device="cpu"),
            torch.rand(20, 20, device="cpu"),
            torch.rand(10, 10, device="cpu"),
            torch.rand(20, 20, device="cpu"),
        )

        self.check_model_cpu(fn, inputs)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @decomp_ops
    def test_decomp(self, op):
        def fn(a0, a1, b0, b1, c0, c1):
            return op([a0, a1], [b0, b1], [c0, c1], value=0.5)

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(20, 20, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    def test_fuse_concat(self):
        def fn(x1, x2, x3, w1, w2, w3):
            x = torch.stack([x1, x2, x3])
            w = torch.stack([w1, w2, w3])

            y = torch.bmm(x, w)

            return y

        x1 = torch.randn(5, 4).cuda()
        x2 = x1 + 1
        x3 = x1 + 2
        w1 = torch.randn(4, 3).cuda()
        w2 = w1 + 1
        w3 = w1 + 2

        args = (x1, x2, x3, w1, w2, w3)

        self.check_model_cuda(fn, args)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    def test_zero_elems(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        self.check_model_cuda(
            fn,
            (
                torch.rand(0, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(0, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @bin_ops
    def test_2d_blocking(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 40, device="cuda:0"),
                torch.rand(10, 30, device="cuda:0"),
                torch.rand(40, 10, device="cuda:0").t(),
                torch.rand(30, 10, device="cuda:0").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @bin_ops
    def test_2d_blocking_partitioning(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda:0"),
                torch.rand(40, 30, device="cuda:0"),
                torch.rand(30, 20, device="cuda:0"),
                torch.rand(30, 40, device="cuda:0").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @bin_ops
    def test_2d_blocking_partitioning_elems(self, op):
        """2D blocking should be grouped by number of yelems"""

        def fn(a0, a1, a2, b0, b1, b2):
            return op([a0, a1, a2], [b0, b1, b2])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 20, device="cuda:0"),
                torch.rand(30, 20, device="cuda:0"),
                torch.rand(10, 30, device="cuda:0"),
                torch.rand(20, 10, device="cuda:0").t(),
                torch.rand(20, 30, device="cuda:0").t(),
                torch.rand(30, 10, device="cuda:0").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @bin_ops
    @torch._inductor.config.patch("combo_kernel_allow_mixed_sizes", 2)
    def test_2d_blocking_partitioning_mixed_sizes(self, op):
        """2D blocking with mixed sizes should group together"""

        def fn(a0, a1, a2, b0, b1, b2):
            return op([a0, a1, a2], [b0, b1, b2])

        self.check_model_cuda(
            fn,
            (
                torch.rand(10, 20, device="cuda:0"),
                torch.rand(30, 20, device="cuda:0"),
                torch.rand(10, 30, device="cuda:0"),
                torch.rand(20, 10, device="cuda:0").t(),
                torch.rand(20, 30, device="cuda:0").t(),
                torch.rand(30, 10, device="cuda:0").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @inplace_bin_ops
    def test_reinplacing(self, op):
        def fn(a0, a1, b0, b1):
            op([a0, a1], [b0, b1])
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        self.check_model_cuda(fn, inputs, check_lowp=False)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @inplace_bin_ops
    def test_reinplacing_mut_before(self, op):
        def fn(a0, a1, b0, b1):
            a0.add_(torch.ones(10, 10, device="cuda:0"))
            op([a0, a1], [b0, b1])
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        self.check_model_cuda(fn, inputs, check_lowp=False)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    @inplace_bin_ops
    def test_reinplacing_mut_after(self, op):
        def fn(a0, a1, b0, b1):
            op([a0, a1], [b0, b1])
            a0.add_(torch.ones(10, 10, device="cuda:0"))
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )

        self.check_model_cuda(fn, inputs, check_lowp=False)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda
    def test_multi_device(self):
        def test_foreach_add(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        inps = [
            torch.ones(10, 10, device="cuda"),
            torch.ones(20, 20, device="cpu"),
            torch.zeros(10, 10, device="cuda"),
            torch.zeros(20, 20, device="cpu"),
        ]

        out_eager = test_foreach_add(*inps)
        out_compiled = torch.compile(test_foreach_add)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    def test_aliasing(self):
        def test_foreach_add(a0, a1, a2, b0, b1, b2):
            return torch._foreach_add_([a0, a1, a2], [b0, b1, b2])

        input = torch.ones(10, 10, device="cuda")
        input2 = torch.ones(10, 10, device="cuda")
        inps = [
            input,
            input.view(10, 10),
            input.view(10, 10),
            input2,
            input2.view(10, 10),
            input2.view(10, 10),
        ]

        out_eager = test_foreach_add(*inps)
        out_compiled = torch.compile(test_foreach_add)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_cuda
    @torch._inductor.config.patch("combo_kernel_allow_mixed_sizes", 1)
    def test_2d_block_no_mixed_sizes_no_mask(self):
        """2D blocking with no mixed sizes constant mask"""

        def fn(a0, a1, a2, b0, b1, b2):
            return torch._foreach_add([a0, a1, a2], [b0, b1, b2])

        self.check_model_cuda(
            fn,
            (
                torch.rand(1024, 2048, device="cuda:0"),
                torch.rand(2048, 2048, device="cuda:0"),
                torch.rand(1024, 2048, device="cuda:0"),
                torch.rand(2048, 1024, device="cuda:0").t(),
                torch.rand(2048, 2048, device="cuda:0").t(),
                torch.rand(2048, 1024, device="cuda:0").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_cuda
    @torch._inductor.config.patch("combo_kernel_allow_mixed_sizes", 2)
    def test_2d_block_mixed_sizes_with_mask(self):
        """2D blocking with mixed sizes should have mask"""

        def fn(a0, a1, a2, b0, b1, b2):
            return torch._foreach_add([a0, a1, a2], [b0, b1, b2])

        self.check_model_cuda(
            fn,
            (
                torch.rand(1024, 2048, device="cuda:0"),
                torch.rand(2048, 2048, device="cuda:0"),
                torch.rand(1024, 2048, device="cuda:0"),
                torch.rand(2048, 1024, device="cuda:0").t(),
                torch.rand(2048, 2048, device="cuda:0").t(),
                torch.rand(2048, 1024, device="cuda:0").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
