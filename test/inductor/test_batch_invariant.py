# Owner(s): ["module: inductor"]
"""Batch invariance coverage for Inductor reductions."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._inductor import config, metrics
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils import _pytree as pytree
from torch.utils._triton import has_triton_reduction_ordering


DTYPES = (torch.float16, torch.bfloat16, torch.float32)
BATCH_INVARIANT_CONFIG = {
    "batch_invariant": True,
    "force_disable_caches": True,
    "implicit_fallbacks": False,
}


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON
    and torch.version.hip is None
    and has_triton_reduction_ordering(),
    "requires CUDA and tl.ReductionOrdering",
)
class TestBatchInvariantReductions(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def _assert_bitwise_equal(self, actual, expected, msg=None):
        def tensor_bytes(value):
            data = value.contiguous().view(-1).view(torch.uint8)
            return value.dtype, value.shape, data

        self.assertEqual(
            pytree.tree_map_only(torch.Tensor, tensor_bytes, actual),
            pytree.tree_map_only(torch.Tensor, tensor_bytes, expected),
            msg=msg,
        )

    def _select_rows(self, value, batch_size, reverse):
        value = value[:batch_size]
        return value.flip(0) if reverse else value

    def _apply_layout(self, value, layout):
        if layout == "contiguous":
            return value

        if layout == "outer_strided":
            result = value.new_empty((value.shape[0] * 2, *value.shape[1:]))[::2]
        elif layout == "inner_strided":
            result = value.new_empty((*value.shape[:-1], value.shape[-1] * 2))[..., ::2]
        elif layout in {"transposed", "transposed_fresh"}:
            if value.ndim != 2:
                raise AssertionError("transposed layout requires a 2D input")
            result = value.new_empty((value.shape[1], value.shape[0])).t()
        elif layout == "storage_offset":
            result = value.new_empty((*value.shape[:-1], value.shape[-1] + 1))[..., 1:]
        elif layout == "expanded":
            return value[:1].expand_as(value)
        else:
            raise AssertionError(f"unknown layout: {layout}")

        result.copy_(value)
        return result

    @config.patch(BATCH_INVARIANT_CONFIG)
    def _check_batch_invariance(
        self,
        fn,
        args,
        *,
        layout="contiguous",
        dynamic=False,
        backward=False,
        batch_arg_indices=(0,),
        batch_sizes=None,
    ):
        if backward:
            args = (args[0].detach().requires_grad_(True), *args[1:])
        eager = (fn(*args),)
        if backward:
            grad_output = torch.randn_like(eager[0])
            eager += torch.autograd.grad(eager[0], args[0], grad_output)
        tolerance = {
            torch.float16: 1e-2,
            torch.bfloat16: 2e-2,
            torch.float32: 2e-3,
        }[args[0].dtype]
        compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
        reference = None
        batches = batch_sizes or (args[0].shape[0], 7, 1)
        for batch_size in sorted(set(batches), reverse=True):
            reverse = batch_size < args[0].shape[0] and args[0].is_contiguous()
            values = [
                self._select_rows(arg, batch_size, reverse)
                if i in batch_arg_indices
                else arg
                for i, arg in enumerate(args)
            ]
            if layout == "transposed_fresh":
                values[0] = self._apply_layout(values[0].contiguous(), layout)
            output = (compiled(*values),)
            if backward:
                grad = self._select_rows(grad_output, batch_size, reverse)
                output += torch.autograd.grad(output[0], values[0], grad)
            expected = tuple(self._select_rows(x, batch_size, reverse) for x in eager)
            self.assertEqual(
                output, expected, atol=tolerance, rtol=tolerance, equal_nan=True
            )
            if reference is None:
                reference = output
            else:
                expected = tuple(
                    self._select_rows(x, batch_size, reverse) for x in reference
                )
                self._assert_bitwise_equal(output, expected, msg=f"batch={batch_size}")

    @dtypes(*DTYPES)
    @parametrize(
        "input_shape",
        (
            (4097, 17),
            (4097, 1024),
            (1025, 1025),
            (33, 32769),
            (7, 65537),
        ),
        name_fn=lambda shape: "x".join(map(str, shape)),
    )
    def test_batch_invariant_sum(self, device, dtype, input_shape):
        value = torch.randn(input_shape, device=device, dtype=dtype)
        value[0].zero_()
        value[0, :4] = value.new_tensor([1e4, 1e-4, -1e4, 0])
        self._check_batch_invariance(lambda x: x.sum(-1), (value,))

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_mean(self, device, dtype):
        value = torch.randn(257, 1025, device=device, dtype=dtype)
        value[:, :4] = value.new_tensor([1e4, 1, -1e4, -1])
        self._check_batch_invariance(lambda x: x.mean(-1), (value,))

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_nansum(self, device, dtype):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        value[:, ::17] = float("nan")
        self._check_batch_invariance(lambda x: x.nansum(-1), (value,))

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_nanmean(self, device, dtype):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        value[:, ::17] = float("nan")
        self._check_batch_invariance(lambda x: x.nanmean(-1), (value,))

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_vector_norm(self, device, dtype):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        self._check_batch_invariance(
            lambda x: torch.linalg.vector_norm(x, dim=-1), (value,)
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("width", (1025, 32769))
    def test_batch_invariant_logsumexp(self, device, dtype, width):
        batch = 257 if width == 1025 else 17
        value = torch.randn((batch, width), device=device, dtype=dtype)
        self._check_batch_invariance(
            lambda x: torch.logsumexp(x, -1),
            (value,),
            batch_sizes=None if width == 1025 else (17, 7),
        )

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_masked_sum(self, device, dtype):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        mask = torch.rand_like(value) > 0.2
        self._check_batch_invariance(
            lambda x, m: torch.masked.sum(x, dim=-1, mask=m),
            (value, mask),
            batch_arg_indices=(0, 1),
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("dynamic,backward", ((False, False), (True, True)))
    def test_batch_invariant_masked_mean(self, device, dtype, dynamic, backward):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        mask = torch.rand_like(value) > 0.2
        self._check_batch_invariance(
            lambda x, m: torch.masked.mean(x, dim=-1, mask=m),
            (value, mask),
            batch_arg_indices=(0, 1),
            dynamic=dynamic,
            backward=backward,
        )

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_masked_norm(self, device, dtype):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        mask = torch.rand_like(value) > 0.2
        self._check_batch_invariance(
            lambda x, m: torch.masked.norm(x, ord=2.0, dim=-1, mask=m),
            (value, mask),
            batch_arg_indices=(0, 1),
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("width", (1025, 32769))
    def test_batch_invariant_normalize(self, device, dtype, width):
        batch = 257 if width == 1025 else 17
        value = torch.randn((batch, width), device=device, dtype=dtype)
        self._check_batch_invariance(
            lambda x: F.normalize(x, dim=-1),
            (value,),
            batch_sizes=None if width == 1025 else (17, 7),
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("width", (1025, 32769))
    def test_batch_invariant_rms_norm(self, device, dtype, width):
        batch = 257 if width == 1025 else 17
        value = torch.randn((batch, width), device=device, dtype=dtype)
        weight = torch.randn(width, device=device, dtype=dtype)
        self._check_batch_invariance(
            lambda x, w: F.rms_norm(x, x.shape[1:], w),
            (value, weight),
            batch_sizes=None if width == 1025 else (17, 7),
        )

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_normalize_backward(self, device, dtype):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        self._check_batch_invariance(
            lambda x: F.normalize(x, dim=-1), (value,), dynamic=True, backward=True
        )

    @dtypes(torch.float32, torch.bfloat16)
    def test_batch_invariant_rms_norm_backward(self, device, dtype):
        value = torch.randn((257, 1025), device=device, dtype=dtype)
        weight = torch.randn(1025, device=device, dtype=dtype)
        self._check_batch_invariance(
            lambda x, w: F.rms_norm(x, x.shape[1:], w),
            (value, weight),
            dynamic=True,
            backward=True,
        )

    @dtypes(*DTYPES)
    @parametrize("dynamic,backward", ((False, True), (True, False), (True, True)))
    def test_batch_invariant_sum_modes(self, device, dtype, dynamic, backward):
        value = torch.randn((257, 2049), device=device, dtype=dtype)
        self._check_batch_invariance(
            lambda x: x.sum(-1), (value,), dynamic=dynamic, backward=backward
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize(
        "layout,input_shape",
        (
            ("outer_strided", (1025, 1025)),
            ("inner_strided", (257, 2049)),
            ("transposed", (257, 2049)),
            ("transposed_fresh", (257, 2049)),
            ("storage_offset", (257, 2049)),
            ("expanded", (257, 2049)),
        ),
        name_fn=lambda layout, input_shape: layout,
    )
    def test_batch_invariant_sum_layouts(self, device, dtype, layout, input_shape):
        value = torch.randn(input_shape, device=device, dtype=dtype)
        value = self._apply_layout(value, layout)
        self._check_batch_invariance(
            lambda x: x.sum(-1),
            (value,),
            layout=layout,
            dynamic=layout == "transposed_fresh",
        )

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("pattern", ("signed_zero", "nan", "positive_inf"))
    def test_batch_invariant_sum_special_values(self, device, dtype, pattern):
        value = torch.zeros(257, 1025, device=device, dtype=dtype)
        if pattern == "signed_zero":
            value.fill_(-0.0)
            value[:, ::2] = 0.0
        else:
            value[:, [0, 1023, 1024]] = float("nan" if pattern == "nan" else "inf")
        self._check_batch_invariance(lambda x: x.sum(-1), (value,))

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("input_shape,dim", (((257, 1025, 3), 1), ((257, 17, 33), (1, 2))))
    def test_batch_invariant_sum_multidim(self, device, dtype, input_shape, dim):
        value = torch.randn(input_shape, device=device, dtype=dtype)
        self._check_batch_invariance(lambda x: x.sum(dim), (value,))

    def test_bounded_dynamic_sum(self, device):
        value = torch.randn(7, 17, device=device)
        torch._dynamo.mark_dynamic(value, 1, min=2, max=31)
        with config.patch(BATCH_INVARIANT_CONFIG):
            compiled = torch.compile(lambda x: x.sum(-1), fullgraph=True)
            output, codes = run_and_get_code(compiled, value)
            self.assertEqual(output, value.sum(-1))
            other = torch.randn(7, 31, device=device)
            self.assertEqual(compiled(other), other.sum(-1))
        self.assertIn("'batch_invariant_chunk_size': 32", "\n".join(codes))

    @dtypes(*DTYPES)
    def test_sum_execution_modes_match(self, device, dtype):
        def run(value, overrides):
            torch._dynamo.reset()
            with config.patch({**BATCH_INVARIANT_CONFIG, **overrides}):
                return torch.compile(lambda x: x.sum(-1), fullgraph=True)(value)

        persistent_value = torch.randn(7, 1024, device=device, dtype=dtype)
        persistent = run(persistent_value, {"split_reductions": False})
        loop = run(
            persistent_value,
            {
                "triton.persistent_reductions": False,
                "split_reductions": False,
            },
        )
        self._assert_bitwise_equal(persistent, loop)

        split_value = torch.randn(7, 131073, device=device, dtype=dtype)
        loop = run(
            split_value,
            {
                "triton.persistent_reductions": False,
                "split_reductions": False,
            },
        )
        split = run(
            split_value,
            {"triton.persistent_reductions": False, "split_reductions": True},
        )
        self._assert_bitwise_equal(loop, split)
        cooperative = run(split_value[:1], {"triton.cooperative_reductions": True})
        self._assert_bitwise_equal(cooperative, split[:1])

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("fp32_output", (False, True))
    @parametrize("width", (1024, 65537))
    def test_sum_rounds_before_fused_consumer(self, device, dtype, fp32_output, width):
        delta = torch.finfo(dtype).eps / 2
        row = torch.zeros(width, device=device, dtype=dtype)
        row[0] = 1
        row[1] = delta

        def fn(value):
            total = value.sum(-1)
            return (total.float() if fp32_output else total) + delta

        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        large_batch = 2 * sm_count + 1
        outputs = []
        with config.patch({**BATCH_INVARIANT_CONFIG, "triton.multi_kernel": 0}):
            for batch in (7, large_batch):
                torch._dynamo.reset()
                value = row.expand(batch, -1).clone()
                output = torch.compile(fn, fullgraph=True, dynamic=False)(value)
                self._assert_bitwise_equal(output, fn(value))
                outputs.append(output)
        self._assert_bitwise_equal(outputs[0], outputs[1][:7])

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("dynamic", (False, True))
    def test_sum_rounds_producer_before_splitting(self, device, dtype, dynamic):
        batch = 2 * torch.cuda.get_device_properties(device).multi_processor_count + 1
        value = torch.full(
            (batch, 65537), 1 + torch.finfo(dtype).eps, device=device, dtype=dtype
        )

        def fn(x):
            squared = x * x
            return squared, squared.sum(-1, dtype=torch.float32)

        outputs = []
        with config.patch(BATCH_INVARIANT_CONFIG):
            for size in (batch, 7):
                torch._dynamo.reset()
                output = torch.compile(fn, fullgraph=True, dynamic=dynamic)(
                    value[:size]
                )
                self.assertEqual(output, fn(value[:size]))
                outputs.append(output)
        self._assert_bitwise_equal(tuple(x[:7] for x in outputs[0]), outputs[1])

    @dtypes(*DTYPES)
    def test_sum_tuned_configs_match(self, device, dtype):
        value = torch.randn(33, 12000, device=device, dtype=dtype)

        def run(xblock, rblock, num_warps):
            def forced_configs(*args, **kwargs):
                return [
                    Config(
                        {"XBLOCK": xblock, "R0_BLOCK": rblock},
                        num_warps=num_warps,
                        num_stages=1,
                    )
                ]

            torch._dynamo.reset()
            overrides = {
                **BATCH_INVARIANT_CONFIG,
                "compile_threads": 1,
                "dynamic_scale_rblock": False,
                "triton.persistent_reductions": False,
                "split_reductions": False,
            }
            with (
                config.patch(overrides),
                patch(
                    "torch._inductor.runtime.triton_heuristics._reduction_configs",
                    forced_configs,
                ),
            ):
                return torch.compile(lambda x: x.sum(-1), fullgraph=True)(value)

        reference = run(1, 1024, 4)
        for tuned in ((4, 1024, 8), (1, 8192, 8)):
            self._assert_bitwise_equal(reference, run(*tuned), msg=str(tuned))

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize(
        "operation,dynamic,mode",
        (
            ("sum", False, "batch_invariant"),
            ("logsumexp", False, "batch_invariant"),
            ("logsumexp", True, "batch_invariant"),
            ("sum", False, "default"),
            ("sum", False, "strict"),
        ),
    )
    def test_reduction_tail(self, device, dtype, operation, dynamic, mode):
        def fn(x):
            doubled = x * 2
            return doubled, getattr(torch, operation)(doubled, dim=-1)

        value = torch.randn(7, 4097, device=device, dtype=dtype)
        value[0].fill_(-0.0)
        value[1, :4] = value.new_tensor([10000, 1, -10000, -1])
        value[2, 0] = float("inf")
        value[3, 0] = float("nan")
        value[4].fill_(-float("inf"))
        values = [value]
        if mode == "batch_invariant":
            torch._dynamo.mark_dynamic(value, 0, min=2, max=7)
            values.append(value[:3])
        if dynamic:
            torch._dynamo.mark_dynamic(value, 1, min=2, max=4097)
            values.append(value[:3, :4096].contiguous())

        def run():
            torch._dynamo.reset()
            metrics.reset()
            compiled = torch.compile(fn, fullgraph=True)
            output, codes = run_and_get_code(compiled, values[0])
            kernel_count = metrics.generated_kernel_count
            outputs = [output, *(compiled(x) for x in values[1:])]
            self.assertEqual(metrics.generated_kernel_count, kernel_count)
            return outputs, "\n".join(codes)

        with config.patch(
            {
                **BATCH_INVARIANT_CONFIG,
                "batch_invariant": mode != "default",
                "numerics": "strict" if mode == "strict" else "default",
                "split_reductions": False,
                "triton.persistent_reductions": False,
            }
        ):
            with patch(
                "torch._inductor.codegen.triton.TritonKernel._should_peel_reduction_tail",
                return_value=False,
            ):
                expected, original_code = run()
            actual, code = run()
        self._assert_bitwise_equal(actual, expected)
        if mode == "batch_invariant":
            self._assert_bitwise_equal(actual[1], tuple(x[:3] for x in actual[0]))
        self.assertNotIn("r0_full_numel", original_code)
        self.assertEqual(
            "r0_full_numel" in code, mode == "batch_invariant" and not dynamic
        )

    def test_strict_precedence_and_fma_policy(self, device):
        inputs = tuple(torch.randn(17, 1025, device=device) for _ in range(3))

        def reduction(a, b, c):
            return (a * b + c).sum(-1)

        with config.patch({**BATCH_INVARIANT_CONFIG, "numerics": "strict"}):
            both, both_code = run_and_get_code(torch.compile(reduction), *inputs)
        with config.patch(
            {**BATCH_INVARIANT_CONFIG, "batch_invariant": False, "numerics": "strict"}
        ):
            strict, _ = run_and_get_code(torch.compile(reduction), *inputs)
        self.assertEqual(both.view(torch.uint8), strict.view(torch.uint8))
        self.assertNotIn("'batch_invariant_chunk_size'", "\n".join(both_code))

        with config.patch(BATCH_INVARIANT_CONFIG):
            a = torch.full((17,), 4097.0, device=device)
            bias = torch.full_like(a, -16785408.0)
            output = torch.compile(lambda a, b: a * a + b)(a, bias)
        self.assertEqual(output, torch.ones_like(output))

    def test_fma_policy_is_independent_of_reduction_fusion(self, device):
        def fn(value, bias):
            return value.sum(-1), (value * value + bias).amax(-1)

        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        with config.patch(BATCH_INVARIANT_CONFIG):
            for batch in (2 * sm_count - 1, 2 * sm_count + 1):
                torch._dynamo.reset()
                shape = (batch, 65537)
                value = torch.full(shape, 4097.0, device=device)
                bias = torch.full(shape, -16785408.0, device=device)
                result = torch.compile(fn, fullgraph=True)(value, bias)
                self.assertEqual(result[1], torch.zeros_like(result[1]))

    def test_fma_policy_survives_degenerate_reductions(self, device):
        def fn(value, a, b, c):
            return value.sum(dim=0), a * b + c

        a = torch.full((17,), 4097.0, device=device)
        b = torch.full((17,), 4097.0, device=device)
        c = torch.full((17,), -16785408.0, device=device)
        with config.patch(BATCH_INVARIANT_CONFIG):
            for batch in (2, 1, 0):
                torch._dynamo.reset()
                value = torch.randn(batch, 17, device=device)
                result = torch.compile(fn, fullgraph=True)(value, a, b, c)
                self.assertEqual(result[1], torch.zeros_like(result[1]))

    def test_logsumexp_fuses_default_max_with_planned_sum(self, device):
        value = torch.randn(17, 768, device=device)
        metrics.reset()
        with config.patch(BATCH_INVARIANT_CONFIG):
            compiled = torch.compile(lambda x: torch.logsumexp(x, -1), fullgraph=True)
            output = compiled(value)
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(output, torch.logsumexp(value, -1))

    @dtypes(torch.float32, torch.bfloat16)
    def test_matching_split_final_sums_fuse(self, device, dtype):
        def sum_fn(x):
            return x.sum(-1, dtype=torch.float32)

        def square_sum_fn(x):
            return (x * x).sum(-1, dtype=torch.float32)

        def fn(x):
            total, squares = sum_fn(x), square_sum_fn(x)
            return total, squares, total + squares

        value = torch.randn(7, 65537, device=device, dtype=dtype)
        value[0].fill_(-0.0)
        value[1, ::2], value[1, 1::2] = 10000, -10000
        value[2, 0], value[3, 0] = float("inf"), float("nan")
        value[4, :2] = value.new_tensor([float("inf"), -float("inf")])
        with config.patch(BATCH_INVARIANT_CONFIG):
            metrics.reset()
            output = torch.compile(fn, fullgraph=True)(value)
            self.assertEqual(metrics.generated_kernel_count, 2)
            total = torch.compile(sum_fn, fullgraph=True)(value)
            squares = torch.compile(square_sum_fn, fullgraph=True)(value)
        self._assert_bitwise_equal(output, (total, squares, total + squares))

    def test_split_final_sum_keeps_incompatible_reductions_separate(self, device):
        def fn(x, y):
            return x.sum(-1) + y.sum(-1)

        value = torch.randn(7, 65537, device=device)
        other = torch.randn(7, 65, device=device)
        with config.patch({**BATCH_INVARIANT_CONFIG, "aggressive_fusion": True}):
            metrics.reset()
            output = torch.compile(fn, fullgraph=True)(value, other)
            self.assertEqual(metrics.generated_kernel_count, 3)
            total = torch.compile(lambda x: x.sum(-1), fullgraph=True)(value)
            other_total = torch.compile(lambda y: y.sum(-1), fullgraph=True)(other)
            self._assert_bitwise_equal(output, total + other_total)

    @parametrize("chunks", (2**31 - 1, 2**31))
    def test_dynamic_split_grid_limit(self, device, chunks):
        from torch._inductor.utils import get_code
        from torch._subclasses.fake_tensor import FakeTensorMode

        repeats = chunks * 1024 // 1023
        with FakeTensorMode(), config.patch(BATCH_INVARIANT_CONFIG):
            value = torch.empty(2, 1, 1023, device=device).expand(2, repeats, 1023)
            torch._dynamo.mark_dynamic(value, 1, min=2, max=repeats)
            codes = get_code(
                torch.compile(lambda x: x.sum((1, 2)), fullgraph=True), value
            )
        self.assertEqual("BatchInvariantSplitGrid" in "\n".join(codes), chunks < 2**31)

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("xblock", (1, 4))
    def test_dynamic_split_grid_preserves_chunk_boundaries(self, device, dtype, xblock):
        from torch._inductor.runtime import triton_heuristics

        original_configs = triton_heuristics._persistent_reduction_configs

        def configs(size_hints, reduction_hint, inductor_meta, triton_meta):
            if inductor_meta.get("batch_invariant_chunk_size") == 1024:
                return [Config({"XBLOCK": xblock}, num_warps=4)]
            return original_configs(
                size_hints, reduction_hint, inductor_meta, triton_meta
            )

        values = []
        for batch, width in ((7, 131073), (3, 131072), (2, 1025)):
            value = torch.randn(batch, width, device=device, dtype=dtype)
            value[0].fill_(-0.0)
            value[1, ::2] = 10000
            value[1, 1::2] = -10000
            values.append(value)
        torch._dynamo.mark_dynamic(values[0], 0, min=2, max=7)
        torch._dynamo.mark_dynamic(values[0], 1, min=1025, max=131073)

        with (
            config.patch({**BATCH_INVARIANT_CONFIG, "compile_threads": 1}),
            patch.object(triton_heuristics, "_persistent_reduction_configs", configs),
        ):
            compiled = torch.compile(lambda x: x.sum(-1), fullgraph=True)
            metrics.reset()
            output, codes = run_and_get_code(compiled, values[0])
            self.assertIn("BatchInvariantSplitGrid", "\n".join(codes))
            self.assertEqual(metrics.generated_kernel_count, 2)
            outputs = [output, *(compiled(value) for value in values[1:])]
            self.assertEqual(metrics.generated_kernel_count, 2)

        with config.patch({**BATCH_INVARIANT_CONFIG, "split_reductions": False}):
            torch._dynamo.reset()
            unsplit = torch.compile(lambda x: x.sum(-1), fullgraph=True)
            for value, output in zip(values, outputs):
                self.assertEqual(output, value.sum(-1))
                self._assert_bitwise_equal(output, unsplit(value))


instantiate_device_type_tests(TestBatchInvariantReductions, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
