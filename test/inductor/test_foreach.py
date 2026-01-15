# Owner(s): ["module: inductor"]

import sys
import unittest
import unittest.mock as mock

import torch
import torch._inductor
from torch._higher_order_ops import foreach_map
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton, requires_gpu
from torch.utils._pytree import tree_flatten


aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_gpu,
        )
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


def foreach_map_wrapper(op):
    def wrapper(*args, **kwargs):
        return foreach_map(op, *args, **kwargs)

    wrapper.__name__ = "foreach_map_" + op.__name__
    wrapper.original_op = op

    return wrapper


def add_op(x, y):
    return torch.add(x, y)


def add_inplace_op(x, y):
    x.add_(y)
    return x.sin()


def addrecip_op(x, y):
    return torch.reciprocal(torch.add(x, y))


def addcmul_op(x, y, z):
    return torch.mul(torch.add(x, y), z)


def recipaddmul_op(x, y, z):
    return torch.mul(torch.add(torch.reciprocal(x), y), z)


# Foreach map bin op defs which support a scalar arg
foreach_map_add = foreach_map_wrapper(torch.add)
foreach_map_mul = foreach_map_wrapper(torch.mul)
foreach_map_sub = foreach_map_wrapper(torch.sub)
foreach_map_div = foreach_map_wrapper(torch.div)
foreach_map_addrecip = foreach_map_wrapper(addrecip_op)
foreach_map_clamp_max = foreach_map_wrapper(torch.clamp_max)
foreach_map_clamp_min = foreach_map_wrapper(torch.clamp_min)
# No scalar args (due to limitations on the op itself)
foreach_map_max = foreach_map_wrapper(torch.maximum)
foreach_map_min = foreach_map_wrapper(torch.minimum)
foreach_map_copy = foreach_map_wrapper(aten.copy)


# More general functions
foreach_map_add_fn = foreach_map_wrapper(add_op)
foreach_map_add_inplace = foreach_map_wrapper(add_inplace_op)
foreach_map_recipaddmul = foreach_map_wrapper(addrecip_op)
foreach_map_addcmul = foreach_map_wrapper(addcmul_op)
foreach_map_recipaddmul = foreach_map_wrapper(recipaddmul_op)

# Foreach map unary op defs
foreach_map_recip = foreach_map_wrapper(torch.reciprocal)
foreach_map_neg = foreach_map_wrapper(torch.neg)
foreach_map_sign = foreach_map_wrapper(torch.sign)
foreach_map_abs = foreach_map_wrapper(torch.abs)

inplace_bin_ops_under_test = [
    torch._foreach_add_,
    torch._foreach_mul_,
    torch._foreach_sub_,
    torch._foreach_div_,
]

ternary_ops_under_test = [
    foreach_map_addcmul,
    foreach_map_recipaddmul,
]

foreach_map_bin_ops_under_test = [
    foreach_map_add,
    foreach_map_mul,
    foreach_map_sub,
    foreach_map_div,
    foreach_map_addrecip,
    foreach_map_clamp_max,
    foreach_map_clamp_min,
    foreach_map_add_fn,
    foreach_map_max,
    foreach_map_min,
]

foreach_map_un_ops_under_test = [
    foreach_map_recip,
    foreach_map_neg,
    foreach_map_sign,
    foreach_map_abs,
]

bin_ops_under_test = [
    torch._foreach_add,
    torch._foreach_mul,
    torch._foreach_sub,
    torch._foreach_div,
    torch._foreach_maximum,
    torch._foreach_minimum,
    torch._foreach_clamp_max,
    torch._foreach_clamp_min,
    aten._foreach_copy,
    foreach_map_copy,  # aten.copy doesn't support backward
    *foreach_map_bin_ops_under_test,
]

scalar_bin_ops_under_test = [
    op
    for op in bin_ops_under_test
    if op
    not in (foreach_map_max, foreach_map_min, foreach_map_copy, aten._foreach_copy)
]

un_ops_under_test = [
    torch._foreach_reciprocal,
    torch._foreach_neg,
    torch._foreach_sign,
    torch._foreach_abs,
    torch._foreach_sqrt,
    torch._foreach_rsqrt,
    *foreach_map_un_ops_under_test,
]

compose_ops = [torch._foreach_addcdiv]
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
    "op", scalar_bin_ops_under_test, name_fn=lambda f: f.__name__
)
scalar_tensor_bin_ops = parametrize(
    "op", scalar_bin_ops_under_test, name_fn=lambda f: f.__name__
)

foreach_map_bin_ops = parametrize(
    "op", foreach_map_bin_ops_under_test, name_fn=lambda f: f.__name__
)

foreach_map_un_ops = parametrize(
    "op", foreach_map_un_ops_under_test, name_fn=lambda f: f.__name__
)

decomp_ops = parametrize("op", compose_ops, name_fn=lambda f: f.__name__)


def gen_args(op):
    if op in un_ops_under_test:
        return (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )
    elif op in bin_ops_under_test:
        return (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )
    else:
        return (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )


@instantiate_parametrized_tests
class ForeachTests(TestCase):
    check_model_gpu = check_model_gpu
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

        self.check_model_gpu(
            fn,
            gen_args(op),
        )

    def _test_single_scalar(self, op):
        def fn(a0, a1):
            return op([a0, a1], 3.3)

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
        )

    def _test_single_scalar_tensor(self, op):
        def fn(a0, a1):
            return op([a0, a1], torch.tensor(3.3, device=GPU_TYPE))

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
        )

    # called in test_gpu_cpp_wrapper.py
    @requires_gpu
    def test_foreach_cpp_wrapper_cuda(self):
        self._test_single_list(op=torch._foreach_add)

    # called in test_gpu_cpp_wrapper.py
    test_foreach_cpp_wrapper_xpu = test_foreach_cpp_wrapper_cuda

    @requires_gpu
    @all_ops
    def test_single_list(self, op):
        self._test_single_list(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_bin_ops
    def test_single_scalar(self, op):
        self._test_single_scalar(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_tensor_bin_ops
    def test_single_scalar_tensor(self, op):
        self._test_single_scalar_tensor(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
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

        self.check_model_gpu(
            fn,
            gen_args(op),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_bin_ops
    def test_scheduler_fusion_scalar(self, op):
        def fn(a0, a1):
            c = op([a0, a1], 3.4)
            return c, torch._foreach_add([a0, a1], c)

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_bin_ops
    def test_broadcasting(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        fn_opt = torch.compile(fn)

        inputs = (
            torch.rand(10, 1, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(1, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )
        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @all_ops
    def test_singleton_lists(self, op):
        if op in un_ops_under_test:

            def fn(a0):
                return op([a0])

            args = (torch.rand(10, 10, device=GPU_TYPE),)
        elif op in bin_ops_under_test:

            def fn(a0, b0):
                return op([a0], [b0])

            args = (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
            )

        else:

            def fn(a0, b0, c0):
                return op([a0], [b0], [c0])

            args = (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
            )

        self.check_model_gpu(
            fn,
            args,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @bin_ops
    def test_type_promotion(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        fn_opt = torch.compile(fn)

        max32 = torch.iinfo(torch.int32).max
        max64 = torch.iinfo(torch.int64).max
        inputs = (
            torch.randint(max32, (10, 10), device=GPU_TYPE, dtype=torch.int32),
            torch.randint(max32, (20, 20), device=GPU_TYPE, dtype=torch.int32),
            torch.randint(max32, (10, 10), device=GPU_TYPE, dtype=torch.int32),
            torch.randint(max64, (20, 20), device=GPU_TYPE, dtype=torch.int64),
        )
        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_bin_ops
    def test_kernel_split_arg_limit_list(self, op):
        # NB: foeach_copy won't pass this test because it will dce one set of buffers

        def fn(a, b):
            return op(a, b)

        fn_opt = torch.compile(fn)

        max_args = 370
        max_list_len = (max_args // 3) + 1
        inputs = (
            [torch.rand(10, 10, device=GPU_TYPE) for _ in range(max_list_len)],
            [torch.rand(10, 10, device=GPU_TYPE) for _ in range(max_list_len)],
        )

        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    @scalar_bin_ops
    @unittest.skip(
        "Triton recursion depth exceeded: https://github.com/triton-lang/triton/issues/1763"
    )
    def test_kernel_split_arg_limit_scalar(self, op):
        def fn(a):
            return op(a, 3.3)

        fn_opt = torch.compile(fn)

        max_args = 370
        max_list_len = (max_args // 2) + 1
        inputs = ([torch.rand(10, 10, device=GPU_TYPE) for _ in range(max_list_len)],)

        actual = fn_opt(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    @bin_ops
    def test_fusion_duplicate_buffer_list(self, op):
        def fn(a0, a1, b0, b1):
            c = op([a0, a1], [b0, b1])
            return op([a0, b0], [c[0], c[0]])

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
            reference_in_float=False,
            check_lowp=False,
        )

        kernel_count = 1
        if "foreach_map" in op.__name__:
            kernel_count = 2
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, kernel_count)

    @requires_gpu
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

        self.check_model_gpu(
            fn,
            gen_args(op),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_bin_ops
    def test_non_foreach_consumer_scalar(self, op):
        def fn(a0, a1):
            c = op([a0, a1], 4.7)
            return torch.mul(c[0], a0)

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
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

        self.check_model_gpu(
            fn, gen_args(op), reference_in_float=False, check_lowp=False
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_bin_ops
    def test_non_foreach_producer_scalar(self, op):
        def fn(a0, a1, b0, b1):
            c0 = torch.mul(a0, b0)
            c1 = torch.mul(a1, b1)
            return op([c0, c1], 5.6)

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
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

        self.check_model_gpu(
            fn,
            gen_args(op),
            reference_in_float=False,
            check_lowp=False,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @scalar_bin_ops
    def test_non_foreach_consumer_producer_scalar(self, op):
        def fn(a0, a1, b0, b1):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            d = op([c0, c1], 5.8)
            e0 = torch.mul(d[0], a0)
            e1 = torch.mul(d[1], a1)
            return [e0, e1]

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
            reference_in_float=False,
            check_lowp=False,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @bin_ops
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    @torch._inductor.config.patch("combo_kernel_foreach_dynamic_shapes", False)
    def test_dynamic_shapes_fallback(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )

        self.check_model_gpu(fn, inputs)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    @torch._inductor.config.patch("combo_kernel_foreach_dynamic_shapes", True)
    def test_enable_dynamic_shapes_python_wrapper(self, op=torch._foreach_add):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )

        self.check_model_gpu(fn, inputs)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    @torch._dynamo.config.patch("assume_static_by_default", False)
    @torch._inductor.config.patch("combo_kernel_foreach_dynamic_shapes", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_enable_dynamic_shapes_cpp_wrapper_cuda(self, op=torch._foreach_add):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        inputs = (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )

        self.check_model_gpu(fn, inputs)

    # called in test_gpu_cpp_wrapper.py
    test_enable_dynamic_shapes_cpp_wrapper_xpu = (
        test_enable_dynamic_shapes_cpp_wrapper_cuda
    )

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

    @requires_gpu
    @decomp_ops
    def test_decomp(self, op):
        def fn(a0, a1, b0, b1, c0, c1):
            return op([a0, a1], [b0, b1], [c0, c1], value=0.5)

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(20, 20, device=GPU_TYPE),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    def test_fuse_concat(self):
        def fn(x1, x2, x3, w1, w2, w3):
            x = torch.stack([x1, x2, x3])
            w = torch.stack([w1, w2, w3])

            y = torch.bmm(x, w)

            return y

        x1 = torch.randn(5, 4).to(GPU_TYPE)
        x2 = x1 + 1
        x3 = x1 + 2
        w1 = torch.randn(4, 3).to(GPU_TYPE)
        w2 = w1 + 1
        w3 = w1 + 2

        args = (x1, x2, x3, w1, w2, w3)

        self.check_model_gpu(fn, args)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    def test_zero_elems(self):
        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        self.check_model_gpu(
            fn,
            (
                torch.rand(0, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
                torch.rand(0, device=GPU_TYPE),
                torch.rand(10, 10, device=GPU_TYPE),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @bin_ops
    def test_2d_blocking(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 40, device=GPU_TYPE),
                torch.rand(10, 30, device=GPU_TYPE),
                torch.rand(40, 10, device=GPU_TYPE).t(),
                torch.rand(30, 10, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @bin_ops
    def test_2d_blocking_partitioning(self, op):
        def fn(a0, a1, b0, b1):
            return op([a0, a1], [b0, b1])

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    @bin_ops
    def test_2d_blocking_partitioning_elems(self, op):
        """2D blocking should be grouped by number of yelems"""

        def fn(a0, a1, a2, b0, b1, b2):
            return op([a0, a1, a2], [b0, b1, b2])

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 20, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(10, 30, device=GPU_TYPE),
                torch.rand(20, 10, device=GPU_TYPE).t(),
                torch.rand(20, 30, device=GPU_TYPE).t(),
                torch.rand(30, 10, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    @bin_ops
    @torch._inductor.config.patch("combo_kernel_allow_mixed_sizes", 2)
    def test_2d_blocking_partitioning_mixed_sizes(self, op):
        """2D blocking with mixed sizes should group together"""

        def fn(a0, a1, a2, b0, b1, b2):
            return op([a0, a1, a2], [b0, b1, b2])

        self.check_model_gpu(
            fn,
            (
                torch.rand(10, 20, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(10, 30, device=GPU_TYPE),
                torch.rand(20, 10, device=GPU_TYPE).t(),
                torch.rand(20, 30, device=GPU_TYPE).t(),
                torch.rand(30, 10, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @inplace_bin_ops
    def test_reinplacing(self, op):
        def fn(a0, a1, b0, b1):
            op([a0, a1], [b0, b1])
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )

        self.check_model_gpu(fn, inputs, check_lowp=False)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @inplace_bin_ops
    def test_reinplacing_mut_before(self, op):
        def fn(a0, a1, b0, b1):
            a0.add_(torch.ones(10, 10, device=GPU_TYPE))
            op([a0, a1], [b0, b1])
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )

        self.check_model_gpu(fn, inputs, check_lowp=False)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @inplace_bin_ops
    def test_reinplacing_mut_after(self, op):
        def fn(a0, a1, b0, b1):
            op([a0, a1], [b0, b1])
            a0.add_(torch.ones(10, 10, device=GPU_TYPE))
            return [a0, a1]

        inputs = (
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        )

        self.check_model_gpu(fn, inputs, check_lowp=False)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    def test_multi_device(self):
        def test_foreach_add(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        inps = [
            torch.ones(10, 10, device=GPU_TYPE),
            torch.ones(20, 20, device="cpu"),
            torch.zeros(10, 10, device=GPU_TYPE),
            torch.zeros(20, 20, device="cpu"),
        ]

        out_eager = test_foreach_add(*inps)
        out_compiled = torch.compile(test_foreach_add)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    def test_aliasing(self):
        def test_foreach_add(a0, a1, a2, b0, b1, b2):
            return torch._foreach_add_([a0, a1, a2], [b0, b1, b2])

        input = torch.ones(10, 10, device=GPU_TYPE)
        input2 = torch.ones(10, 10, device=GPU_TYPE)
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

    @requires_gpu
    @torch._inductor.config.patch("combo_kernel_allow_mixed_sizes", 1)
    def test_2d_block_no_mixed_sizes_no_mask(self):
        """2D blocking with no mixed sizes constant mask"""

        def fn(a0, a1, a2, b0, b1, b2):
            return torch._foreach_add([a0, a1, a2], [b0, b1, b2])

        self.check_model_gpu(
            fn,
            (
                torch.rand(1024, 2048, device=GPU_TYPE),
                torch.rand(2048, 2048, device=GPU_TYPE),
                torch.rand(1024, 2048, device=GPU_TYPE),
                torch.rand(2048, 1024, device=GPU_TYPE).t(),
                torch.rand(2048, 2048, device=GPU_TYPE).t(),
                torch.rand(2048, 1024, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu
    @torch._inductor.config.patch("combo_kernel_allow_mixed_sizes", 2)
    def test_2d_block_mixed_sizes_with_mask(self):
        """2D blocking with mixed sizes should have mask"""

        def fn(a0, a1, a2, b0, b1, b2):
            return torch._foreach_add([a0, a1, a2], [b0, b1, b2])

        self.check_model_gpu(
            fn,
            (
                torch.rand(1024, 2048, device=GPU_TYPE),
                torch.rand(2048, 2048, device=GPU_TYPE),
                torch.rand(1024, 2048, device=GPU_TYPE),
                torch.rand(2048, 1024, device=GPU_TYPE).t(),
                torch.rand(2048, 2048, device=GPU_TYPE).t(),
                torch.rand(2048, 1024, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu
    @foreach_map_bin_ops
    def test_foreach_map_backward_binary(self, op):
        from torch._dynamo.polyfills import foreach_map_fn

        def fn(xs, ys):
            outs = op(xs, ys)
            return outs[0].sum() + outs[1].sum() + outs[2].sum()

        def ref_fn(xs, ys):
            outs = foreach_map_fn(torch.add, xs, ys)
            return outs[0].sum() + outs[1].sum() + outs[2].sum()

        ref_inps = (
            [
                torch.rand(10, 20, device=GPU_TYPE, requires_grad=True),
                torch.rand(10, 30, device=GPU_TYPE, requires_grad=True),
                torch.rand(30, 30, device=GPU_TYPE, requires_grad=True),
            ],
            [
                torch.rand(10, 20, device=GPU_TYPE, requires_grad=True),
                torch.rand(10, 30, device=GPU_TYPE, requires_grad=True),
                torch.rand(30, 30, device=GPU_TYPE, requires_grad=True),
            ],
        )
        inps = (
            [x.clone().detach().requires_grad_(True) for x in ref_inps[0]],
            [y.clone().detach().requires_grad_(True) for y in ref_inps[1]],
        )

        out_ref = ref_fn(*ref_inps)
        out_ref.backward()

        # unpacking result, (fw_code, bw_code)
        _, (_, _) = run_fw_bw_and_get_code(lambda: torch.compile(fn)(*inps))

        for ref, act in zip(tree_flatten(ref_inps)[0], tree_flatten(inps)[0]):
            torch.allclose(ref.grad, act.grad)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu
    def test_foreach_map_input_mutation(self):
        def fn(xs, ys):
            outs = foreach_map_add_inplace(xs, ys)
            return outs[0].sum() + outs[1].sum() + outs[2].sum()

        ref_inps = (
            [
                torch.rand(10, 20, device=GPU_TYPE, requires_grad=True),
                torch.rand(10, 30, device=GPU_TYPE, requires_grad=True),
                torch.rand(30, 30, device=GPU_TYPE, requires_grad=True),
            ],
            [
                torch.rand(10, 20, device=GPU_TYPE, requires_grad=True),
                torch.rand(10, 30, device=GPU_TYPE, requires_grad=True),
                torch.rand(30, 30, device=GPU_TYPE, requires_grad=True),
            ],
        )
        # Set requires_grad to be False to avoid mutating a leaf variable
        inps = (
            [x.clone().detach().requires_grad_(False) for x in ref_inps[0]],
            [y.clone().detach().requires_grad_(False) for y in ref_inps[1]],
        )

        # TODO: after decomposing auto_functionalized, we're getting
        # a functional subgraph with an inlined epilogue.
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "Buffer mutation detected during lowering of aten.copy_.default",
        ):
            with mock.patch(
                "torch._dynamo.variables.higher_order_ops.BaseHOPVariable.supports_input_mutation",
                True,
            ):
                _ = run_fw_bw_and_get_code(lambda: torch.compile(fn)(*inps))

    @requires_gpu
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("_use_fp64_for_unbacked_floats", True)
    @parametrize(
        "value",
        [
            0.123456789,
            -1.987654321,
            0.0,
            1e-8,
            1e8,
            3.141592653589793,  # pi - more precision than fp32
            -2.718281828459045,  # -e - more precision than fp32
            0.000000001234567890123456,
            # Values that expose fp32/fp64 casting issues:
            1.0000001192092896,  # 1 + 2^-23 (smallest diff from 1.0 in fp32)
            1.00000001,  # Between fp32 representable values
            1e-38,  # Very small but not subnormal
            1.1754944e-38,  # Near fp32 min normal
            0.333333333333333333,  # 1/3 with full fp64 precision
            0.1,  # Cannot be exactly represented in binary
            1.0000000000000002,  # 1 + eps in fp64, rounds to 1.0 in fp32
            0.30000000000000004,  # 0.1 + 0.2 in fp64 (famous precision issue)
        ],
    )
    def test_addcmul_scalar_value_vs_tensor_value(self, value):
        """Test that torch._foreach_addcmul with unbacked float scalar from .item()
        matches tensor scalar value in compiled mode.

        Uses .item() to create an unbacked float symbol that gets passed as a
        kernel argument, exercising the fp64 signature fix in triton_utils.py."""

        def fn_item_scalar(a0, a1, b0, b1, c0, c1, scalar_tensor):
            # .item() creates an unbacked float symbol - passed as kernel arg
            val = scalar_tensor.item()
            return torch._foreach_addcmul([a0, a1], [b0, b1], [c0, c1], value=1 - val)

        def fn_tensor_scalar(a0, a1, b0, b1, c0, c1, val):
            return torch._foreach_addcmul([a0, a1], [b0, b1], [c0, c1], value=1 - val)

        # Use specific values that are sensitive to fp32/fp64 precision differences
        inputs = (
            torch.tensor(
                [[1.0000001192092896, 0.333333333333333333], [1e-7, 3.4028235e30]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
            torch.tensor(
                [[1.0000000000000002, 0.1], [1e-38, 2.718281828459045]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.5, 0.5], [0.5, 0.5]], device=GPU_TYPE, dtype=torch.float32
            ),
            torch.tensor(
                [[1.0, 1.0], [1.0, 1.0]], device=GPU_TYPE, dtype=torch.float32
            ),
            torch.tensor(
                [[2.0, 2.0], [2.0, 2.0]], device=GPU_TYPE, dtype=torch.float32
            ),
            torch.tensor(
                [[0.25, 0.25], [0.25, 0.25]], device=GPU_TYPE, dtype=torch.float32
            ),
        )
        scalar_tensor = torch.tensor(value, device=GPU_TYPE, dtype=torch.float64)

        # Compiled mode comparison - assert bitwise equality
        # The .item() path should preserve fp64 precision just like tensor scalar path
        compiled_item_scalar = torch.compile(fn_item_scalar, fullgraph=True)(
            *inputs, scalar_tensor
        )
        compiled_tensor_scalar = torch.compile(fn_tensor_scalar, fullgraph=True)(
            *inputs, scalar_tensor
        )
        for a, b in zip(compiled_item_scalar, compiled_tensor_scalar):
            self.assertEqual(a, b, atol=0, rtol=0)

    @requires_gpu
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize(
        "op",
        [
            torch._foreach_add,
            torch._foreach_mul,
            torch._foreach_sub,
            # Note: _foreach_div excluded due to pre-existing eager precision
            # mismatch between python scalar and tensor scalar arguments
        ],
        name_fn=lambda op: op.__name__,
    )
    @parametrize(
        "value",
        [
            0.123456789,
            -1.987654321,
            1e-8,
            1e8,
            3.141592653589793,  # pi - more precision than fp32
            -2.718281828459045,  # -e - more precision than fp32
            0.000000001234567890123456,
            # Values that expose fp32/fp64 casting issues:
            1.0000001192092896,  # 1 + 2^-23 (smallest diff from 1.0 in fp32)
            1.00000001,  # Between fp32 representable values
            1e-38,  # Very small but not subnormal
            1.1754944e-38,  # Near fp32 min normal
            0.333333333333333333,  # 1/3 with full fp64 precision
            0.1,  # Cannot be exactly represented in binary
            1.0000000000000002,  # 1 + eps in fp64, rounds to 1.0 in fp32
            0.30000000000000004,  # 0.1 + 0.2 in fp64 (famous precision issue)
        ],
    )
    def test_foreach_scalar_vs_scalar_tensor(self, op, value):
        """Test that foreach binary ops with python scalar second argument
        match tensor scalar second argument in both eager and compiled modes."""

        def fn_python_scalar(a0, a1):
            return op([a0, a1], value)

        def fn_tensor_scalar(a0, a1, val):
            return op([a0, a1], val)

        # Use specific values that are sensitive to fp32/fp64 precision differences
        inputs = (
            torch.tensor(
                [[1.0000001192092896, 0.333333333333333333], [1e-7, 3.4028235e30]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
            torch.tensor(
                [[1.0000000000000002, 0.1], [1e-38, 2.718281828459045]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
        )
        scalar_tensor = torch.tensor(value, device=GPU_TYPE, dtype=torch.float64)

        # Eager mode comparison - assert bitwise equality
        eager_python_scalar = fn_python_scalar(*inputs)
        eager_tensor_scalar = fn_tensor_scalar(*inputs, scalar_tensor)
        for a, b in zip(eager_python_scalar, eager_tensor_scalar):
            self.assertEqual(a, b, atol=0, rtol=0)

        # Compiled mode comparison - assert bitwise equality
        compiled_python_scalar = torch.compile(fn_python_scalar, fullgraph=True)(
            *inputs
        )
        compiled_tensor_scalar = torch.compile(fn_tensor_scalar, fullgraph=True)(
            *inputs, scalar_tensor
        )
        for a, b in zip(compiled_python_scalar, compiled_tensor_scalar):
            self.assertEqual(a, b, atol=0, rtol=0)

        # Cross comparison: eager vs compiled - assert bitwise equality
        for a, b in zip(eager_python_scalar, compiled_python_scalar):
            self.assertEqual(a, b, atol=0, rtol=0)
        for a, b in zip(eager_tensor_scalar, compiled_tensor_scalar):
            self.assertEqual(a, b, atol=0, rtol=0)

    @requires_gpu
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("emulate_precision_casts", True)
    @torch._inductor.config.patch("_use_fp64_for_unbacked_floats", True)
    @parametrize(
        "beta2",
        [
            0.999,  # Typical Adam beta2
            0.9999,  # Higher beta2
            0.99,  # Lower beta2
            0.999999,  # Very high beta2
            0.9990000128746033,  # Between fp32 representable values
            1.0 - 1e-8,  # Near 1.0
            1.0 - 1e-38,  # Very close to 1.0
            0.333333333333333333,  # 1/3 with full fp64 precision
            0.1,  # Cannot be exactly represented in binary
        ],
    )
    def test_adam_ema_update_scalar_precision(self, beta2):
        """Test that Adam-style EMA update compiled matches eager bitwise.

        Tests the pattern:
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1-beta2)
        """

        def fn(exp_avg_sq0, exp_avg_sq1, grad0, grad1):
            exp_avg_sqs = [exp_avg_sq0.clone(), exp_avg_sq1.clone()]
            grads = [grad0, grad1]
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
            return exp_avg_sqs

        # Use values sensitive to fp32/fp64 precision differences
        inputs = (
            torch.tensor(
                [[1.0000001192092896, 0.333333333333333333], [1e-7, 1e-30]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
            torch.tensor(
                [[1.0000000000000002, 0.1], [1e-38, 2.718281828459045]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.01, 0.02], [0.001, 0.0001]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.1, 0.2], [0.01, 0.001]],
                device=GPU_TYPE,
                dtype=torch.float32,
            ),
        )

        # Eager execution
        eager_result = fn(*inputs)

        # Compiled execution
        compiled_result = torch.compile(fn, fullgraph=True)(*inputs)

        # Assert bitwise equality between eager and compiled
        for a, b in zip(eager_result, compiled_result):
            self.assertEqual(a, b, atol=0, rtol=0)

    @requires_gpu
    @foreach_map_un_ops
    def test_foreach_map_backward_unary(self, op):
        from torch._dynamo.polyfills import foreach_map_fn

        def fn(xs):
            outs = op(xs)
            return outs[0].sum() + outs[1].sum() + outs[2].sum()

        def ref_fn(xs):
            outs = foreach_map_fn(op.original_op, xs)
            return outs[0].sum() + outs[1].sum() + outs[2].sum()

        ref_inp = [
            torch.rand(10, 20, device=GPU_TYPE, requires_grad=True),
            torch.rand(10, 30, device=GPU_TYPE, requires_grad=True),
            torch.rand(30, 30, device=GPU_TYPE, requires_grad=True),
        ]

        inp = [x.clone().detach().requires_grad_(True) for x in ref_inp]

        out_ref = ref_fn(ref_inp)
        out_ref.backward()

        # unpacking result, (fw_code, bw_code)
        _, (_, _) = run_fw_bw_and_get_code(lambda: torch.compile(fn)(inp))

        for ref, act in zip(ref_inp, inp):
            torch.allclose(ref.grad, act.grad)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda_and_triton
    @config.patch({"emulate_precision_casts": True})
    def test_foreach_addcmul_fma_bitwise_equal(self):
        """Test that _foreach_addcmul with FMA lowering produces bitwise equal results to eager."""
        self_tensors = [
            torch.randn(64, 64, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]
        tensor1_list = [
            torch.randn(64, 64, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]
        tensor2_list = [
            torch.randn(64, 64, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]

        # ROCm may have small numerical differences
        # For some reason ROCm isn't bitwise equivalent between eager and compiled
        atol = 1e-5 if TEST_WITH_ROCM else 0
        rtol = 1e-5 if TEST_WITH_ROCM else 0

        # Test with default value=1
        eager_result = torch._foreach_addcmul(self_tensors, tensor1_list, tensor2_list)

        @torch.compile
        def fn(s, t1, t2):
            return torch._foreach_addcmul(s, t1, t2)

        compiled_result = fn(self_tensors, tensor1_list, tensor2_list)
        for eager, compiled in zip(eager_result, compiled_result):
            self.assertEqual(eager, compiled, atol=atol, rtol=rtol)

        # Test with value != 1
        eager_result2 = torch._foreach_addcmul(
            self_tensors, tensor1_list, tensor2_list, value=2.5
        )

        @torch.compile
        def fn2(s, t1, t2):
            return torch._foreach_addcmul(s, t1, t2, value=2.5)

        compiled_result2 = fn2(self_tensors, tensor1_list, tensor2_list)
        for eager, compiled in zip(eager_result2, compiled_result2):
            self.assertEqual(eager, compiled, atol=atol, rtol=rtol)

    @requires_cuda_and_triton
    @config.patch({"emulate_precision_casts": True})
    def test_foreach_addcmul_uses_fma_instruction(self):
        """Test that _foreach_addcmul generates code using FMA instruction."""
        from torch._inductor.utils import run_and_get_code

        self_tensors = [
            torch.randn(64, 64, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]
        tensor1_list = [
            torch.randn(64, 64, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]
        tensor2_list = [
            torch.randn(64, 64, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]

        @torch.compile
        def fn(s, t1, t2):
            return torch._foreach_addcmul(s, t1, t2, value=2.0)

        _, code = run_and_get_code(fn, self_tensors, tensor1_list, tensor2_list)
        code = " ".join(code)
        self.assertIn("tl.fma", code, "Expected FMA to be used in generated code")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
