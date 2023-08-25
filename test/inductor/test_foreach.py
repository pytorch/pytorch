# Owner(s): ["module: inductor"]

import sys
import unittest

import torch

import torch._inductor

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    TEST_WITH_ROCM,
    TestCase,
)

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_cuda, requires_cuda
    except ImportError:
        from test_torchinductor import check_model, check_model_cuda, requires_cuda
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


bin_ops_under_test = [
    torch._foreach_add,
    torch._foreach_mul,
    torch._foreach_sub,
    torch._foreach_div,
    torch._foreach_maximum,
    aten._foreach_copy,
]

un_ops_under_test = [torch._foreach_reciprocal, torch._foreach_neg, torch._foreach_sign]
compose_ops = [torch._foreach_addcdiv, torch._foreach_addcmul]
all_ops = parametrize(
    "op", bin_ops_under_test + un_ops_under_test, name_fn=lambda f: f.__name__
)
bin_ops = parametrize("op", bin_ops_under_test, name_fn=lambda f: f.__name__)
scalar_bin_ops = parametrize("op", bin_ops_under_test[:4], name_fn=lambda f: f.__name__)
decomp_ops = parametrize("op", compose_ops, name_fn=lambda f: f.__name__)


def gen_args(op):
    if op in un_ops_under_test:
        return (
            torch.rand(10, 10, device="cuda:0"),
            torch.rand(20, 20, device="cuda:0"),
        )
    else:
        return (
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

        else:

            def fn(a0, a1, b0, b1):
                return op([a0, a1], [b0, b1])

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

    # called in test_cpp_wrapper.py
    @requires_cuda()
    def test_foreach_cpp_wrapper(self):
        self._test_single_list(op=torch._foreach_add)

    @requires_cuda()
    @all_ops
    def test_single_list(self, op):
        self._test_single_list(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
    @scalar_bin_ops
    def test_single_scalar(self, op):
        self._test_single_scalar(op)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
    @all_ops
    def test_scheduler_fusion_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                c = op([a0, a1])
                return torch._foreach_sqrt(c)

        else:

            def fn(a0, a1, b0, b1):
                c = op([a0, a1], [b0, b1])
                return c, torch._foreach_add([a0, a1], c)

        self.check_model_cuda(
            fn,
            gen_args(op),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
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

    @requires_cuda()
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

    @requires_cuda()
    @all_ops
    def test_singleton_lists(self, op):
        if op in un_ops_under_test:

            def fn(a0):
                return op([a0])

            args = (torch.rand(10, 10, device="cuda:0"),)
        else:

            def fn(a0, b0):
                return op([a0], [b0])

            args = (
                torch.rand(10, 10, device="cuda:0"),
                torch.rand(10, 10, device="cuda:0"),
            )

        self.check_model_cuda(
            fn,
            args,
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
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

    @requires_cuda()
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

    @requires_cuda()
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

    @requires_cuda()
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

    @requires_cuda()
    @all_ops
    def test_non_foreach_consumer_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                c = op([a0, a1])
                return torch.mul(c[0], a0)

        else:

            def fn(a0, a1, b0, b1):
                c = op([a0, a1], [b0, b1])
                return torch.mul(c[0], a0)

        self.check_model_cuda(
            fn,
            gen_args(op),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
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

    @requires_cuda()
    @all_ops
    def test_non_foreach_producer_list(self, op):
        if op in un_ops_under_test:

            def fn(a0, a1):
                c0 = torch.add(a0, a0)
                c1 = torch.add(a1, a1)
                return op([c0, c1])

        else:

            def fn(a0, a1, b0, b1):
                c0 = torch.add(a0, b0)
                c1 = torch.add(a1, b1)
                return op([a0, a1], [c0, c1])

        self.check_model_cuda(
            fn, gen_args(op), reference_in_float=False, check_lowp=False
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda()
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

    @requires_cuda()
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

        else:

            def fn(a0, a1, b0, b1):
                c0 = torch.add(a0, b0)
                c1 = torch.add(a1, b1)
                d = op([a0, a1], [c0, c1])
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

    @requires_cuda()
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

    @requires_cuda()
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

    @requires_cuda()
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

    @requires_cuda()
    @bin_ops
    def test_2d_blocking(self, op):
        def fn(a0, a1, b0, b1):
            a0_c = torch.cat((a0, a0), 0)
            b0_c = torch.cat((b0, b0), 0)
            a1_c = torch.cat((a1, a1), 0)
            b1_c = torch.cat((b1, b1), 0)
            return op([a0_c, a1_c], [b0_c, b1_c])

        self.check_model_cuda(
            fn,
            (
                torch.rand(2, 2, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(2, 2, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_cuda()
    @bin_ops
    def test_2d_blocking_partitioning(self, op):
        def fn(a0, a1, b0, b1):
            a0_c = torch.cat((a0, a0), 0)
            b0_c = torch.cat((b0, b0), 0)
            return op([a0_c, a1], [b0_c, b1])

        self.check_model_cuda(
            fn,
            (
                torch.rand(2, 2, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(2, 2, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_cuda()
    @bin_ops
    def test_2d_blocking_partitioning_elems(self, op):
        """2D blocking should be grouped by number of yelems"""

        def fn(a0, a1, a2, b0, b1, b2):
            a0_c = torch.cat((a0, a0), 0)
            b0_c = torch.cat((b0, b0), 0)
            a1_c = torch.cat((a1, a1), 0)
            b1_c = torch.cat((b1, b1), 0)
            a2_c = torch.cat((a2, a2), 0)
            b2_c = torch.cat((b2, b2), 0)
            return op([a0_c, a1_c, a2_c], [b0_c, b1_c, b2_c])

        self.check_model_cuda(
            fn,
            (
                torch.rand(2, 2, 4, 4, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(10, 10, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(2, 2, 4, 4, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(10, 10, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 9)

        torch._inductor.metrics.reset()

        torch._logging.set_logs(output_code=True)
        self.check_model_cuda(
            fn,
            (
                torch.rand(20, 20, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 4, 4, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 3, 3, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 4, 4, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
                torch.rand(20, 20, 2, 2, device="cuda:0").to(
                    memory_format=torch.channels_last
                ),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 7)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
