# Owner(s): ["module: inductor"]

import os
import re
import sys

import torch
import torch._inductor.test_operators
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codegen.triton import TritonCSE, TritonCSEVariable, TritonKernel
from torch._inductor.utils import (
    IndentedBuffer,
    run_and_get_code,
    run_and_get_kernels,
    TRITON_FLOAT8_DTYPES,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_inline_asm_concat import combine_inputs_asm, xor_words


class _SplitCodegen:
    create_cse_var = staticmethod(TritonCSEVariable)
    index_to_str = staticmethod(str)
    _reshape_expr = staticmethod(TritonKernel._reshape_expr)
    _bitcast_reshape_expr = TritonKernel._bitcast_reshape_expr
    _emit_recursive_split = TritonKernel._emit_recursive_split
    emit_split_via_reshape = TritonKernel.emit_split_via_reshape

    def __init__(self, persistent):
        self.compute = IndentedBuffer()
        self.cse = TritonCSE()
        self.persistent_reduction = persistent


@instantiate_parametrized_tests
class TestLaneSplitValidation(TestCase):
    @parametrize("factor", [0, 1, 3, 5, 6, 9])
    @parametrize("dtype", [torch.int16, torch.float16, torch.bfloat16, torch.float32])
    def test_invalid_factor(self, factor, dtype):
        emitter = _SplitCodegen(persistent=False)
        shape = (4, factor)
        names = [f"lane{i}" for i in range(factor)]
        with V.set_kernel_handler(emitter):
            value = emitter.cse.newvar(dtype=dtype, shape=shape)
            with self.assertRaisesRegex(
                AssertionError, f"split factor must be a power of two: {factor}$"
            ):
                emitter.emit_split_via_reshape(value, shape, names)


class TestRecursiveLaneSplit(TestCase):
    @parametrize("persistent", [False, True])
    @parametrize("rank", [2, 3])
    @parametrize("factor", [2, 4, 8, 32, 64])
    @parametrize(
        "dtype",
        [torch.int16, torch.float16, torch.bfloat16, torch.float32, *TRITON_FLOAT8_DTYPES],
    )
    def test_storage_bits(self, device, rank, factor, dtype, persistent):
        prefix = (4,) if rank == 2 else (2, 2)
        shape = (*prefix, factor)
        count = 4 * factor
        emitter = _SplitCodegen(persistent)
        code = emitter.compute
        code.writelines(
            [
                "import triton",
                "import triton.language as tl",
                "@triton.jit",
                "def split_kernel(x, y):",
            ]
        )
        with code.indent(), V.set_kernel_handler(emitter):
            code.writeline(f"offset = tl.program_id(0) * {count}")
            code.writeline(f"indices = offset + tl.arange(0, {count})")
            value = emitter.cse.newvar(dtype=dtype, shape=(count,))
            code.writeline(f"{value} = tl.load(x + indices)")
            lanes = [f"lane{i}" for i in range(factor)]
            emitter.emit_split_via_reshape(value, shape, lanes)
            code.writeline(f"rows = tl.reshape(tl.arange(0, 4), {prefix})")
            for lane, name in enumerate(lanes):
                code.writeline(
                    f"tl.store(y + offset + rows * {factor} + {lane}, {name})"
                )
        kernel = PyCodeCache.load(code.getvalue()).split_kernel
        raw = torch.randint(
            256, (32 * count * dtype.itemsize,), dtype=torch.uint8, device=device
        )
        raw[:256] = torch.arange(256, device=device).to(torch.uint8)
        x = raw.view(dtype)
        y = torch.empty_like(x)
        kernel[(32,)](x, y, num_warps=4)
        self.assertEqual(y.view(torch.uint8), raw)


def realized_rms_groups(x, group=32, dtype=torch.bfloat16):
    value = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)).to(dtype)
    return torch.ops._inductor_test.realize(value).view(x.shape[0], -1, group)


def lane_input(device, width=6144):
    columns = (torch.arange(width, device=device) % 32 - 16).float() / 8
    return torch.arange(17, device=device).float()[:, None] / 8 + columns


class NestedLaneTestCase(TestCase):
    def compare(self, fn, x, persistent, exact=True):
        class Choices(InductorChoices):
            @staticmethod
            def should_use_cooperative_reduction(*args, **kwargs):
                return False

            @staticmethod
            def should_use_persistent_reduction(features, cooperative_reduction):
                if features.reduction_numel == x.shape[-1]:
                    return persistent
                return InductorChoices.should_use_persistent_reduction(
                    features, cooperative_reduction
                )

        settings = {
            "triton.multi_kernel": 0,
            "loop_ordering_after_fusion": True,
            "split_reductions": False,
            "emulate_precision_casts": True,
            "force_disable_caches": True,
        }
        with config.patch({**settings, "triton.nested_reduction": False}):
            expected = torch.compile(fn, fullgraph=True)(x)
        torch._dynamo.reset()
        metrics.reset()
        with (
            V.set_choices_handler(Choices()),
            config.patch({**settings, "triton.nested_reduction": True}),
        ):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        if exact:
            self.assertEqual(actual, expected, atol=0, rtol=0)
        else:
            self.assertEqual(actual, expected)
        return "\n".join(code)


class TestPitchedLaneSources(TestCase):
    @parametrize(
        "mode", ["escape", "dynamic", "misaligned", "cross_row", "input_mutation"]
    )
    @config.patch(
        {
            "triton.nested_reduction": True,
            "loop_ordering_after_fusion": True,
            "comprehensive_padding": True,
            "emulate_precision_casts": True,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_pitched_group_lanes_and_aliases(self, device, mode):
        pitch = 14 if mode == "misaligned" else 16

        def fn(x):
            value = (x - x.mean(-1, keepdim=True)).to(torch.bfloat16)
            pitched = torch.empty_strided(
                x.shape, (pitch, 1), dtype=value.dtype, device=x.device
            )
            pitched.copy_(value)
            groups = pitched.view(x.shape[0], 3, 4)
            lane = groups[..., 1]
            if mode == "cross_row":
                lane = lane.roll(1, dims=0)
            result = groups.abs().amax(-1) + lane
            if mode == "input_mutation":
                x.add_(1)
                return result, x
            return (result,) if mode == "cross_row" else (result, pitched)

        def make_input(batch):
            row = (torch.arange(12, device=device) % 4 - 2).float() / 8
            signs = (torch.arange(batch, device=device) % 2 * 2 - 1).float()
            return signs[:, None] * row[None, :]

        x = make_input(37)
        if mode == "dynamic":
            torch._dynamo.mark_dynamic(x, 0)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        expected = fn(x.clone())
        metrics.reset()
        actual, kernels = run_and_get_kernels(compiled, x)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(
            tuple(t.stride() for t in actual), tuple(t.stride() for t in expected)
        )
        if mode == "dynamic":
            for batch in (2, 17, 129):
                value = make_input(batch)
                result, ref = compiled(value), fn(value)
                self.assertEqual(result, ref, atol=0, rtol=0)
                self.assertEqual(
                    tuple(t.stride() for t in result), tuple(t.stride() for t in ref)
                )
        self.assertEqual(counter.frame_count, 1)
        if mode == "input_mutation":
            self.assertEqual(actual[1].data_ptr(), x.data_ptr())
        if len(actual) == 2:
            actual[0].fill_(17)
            expected[0].fill_(17)
            self.assertEqual(actual, expected, atol=0, rtol=0)
            actual[1].add_(5)
            expected[1].add_(5)
            self.assertEqual(actual, expected, atol=0, rtol=0)
        fused = mode not in ("misaligned", "cross_row")
        self.assertEqual(metrics.codegen_nested_reduction, int(fused))
        self.assertEqual(len(kernels), 1 if mode in ("escape", "dynamic") else 2)
        if mode in ("escape", "dynamic"):
            self.assertIn("tl.sum(", kernels[0])
            self.assertTrue(
                "triton_helpers.max2(" in kernels[0]
                or kernels[0].count("tl.maximum(") >= 3
            )
            self.assertNotIn("in_ptr1", kernels[0])
            self.assertEqual(kernels[0].count("tl.store("), 2)


class TestNestedAsmLaneInputs(NestedLaneTestCase):
    @parametrize("dynamic_batch", [False, True])
    @config.patch(
        {
            "triton.nested_reduction": True,
            "loop_ordering_after_fusion": True,
            "comprehensive_padding": True,
            "emulate_precision_casts": True,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_pitched_layer_norm_lanes(self, device, dynamic_batch):
        asm, constraints = combine_inputs_asm(3)
        width = 1056

        def fn(x, weight, bias, native=True):
            value = torch.nn.functional.layer_norm(
                x.float(), (width,), weight.float(), bias.float()
            ).to(torch.bfloat16)
            groups = value.view(x.shape[0], -1, 32)
            maximum = groups.abs().amax(-1)
            bits = groups.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            if native:
                words = inline_asm_elementwise(
                    *pairs.unbind(-1),
                    maximum.to(torch.int32),
                    asm_str=asm,
                    constraints=constraints,
                    dtype=(torch.int32,) * 3,
                )
            else:
                words = []
                for i in range(3):
                    word = pairs[..., i] ^ maximum.to(torch.int32)
                    for j in range(i + 3, 16, 3):
                        word = word ^ pairs[..., j]
                    words.append(word)
            return torch.stack(words, -1).reshape(x.shape[0], -1), maximum

        def make_input(batch):
            # Repeated signed rows keep BF16 rounding away from midpoints.
            row = (torch.arange(width, device=device) % 32 - 16).float() / 8
            signs = (torch.arange(batch, device=device) % 2 * 2 - 1).float()
            return (signs[:, None] * row[None, :]).half()

        weight = ((torch.arange(width, device=device) % 5 + 1).float() / 8).half()
        bias = ((torch.arange(width, device=device) % 3 - 1).float() / 16).half()
        x = make_input(128)
        if dynamic_batch:
            torch._dynamo.mark_dynamic(x, 0)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        metrics.reset()
        actual, kernels = run_and_get_kernels(compiled, x, weight, bias)
        self.assertEqual(actual, fn(x, weight, bias, False), atol=0, rtol=0)
        for batch in (2, 17, 129) if dynamic_batch else (128,):
            value = make_input(batch)
            actual = compiled(value, weight, bias)
            expected = fn(value, weight, bias, False)
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertEqual(
                tuple(t.stride() for t in actual), tuple(t.stride() for t in expected)
            )
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        compute = [kernel for kernel in kernels if "tl.load(" in kernel]
        self.assertEqual(len(compute), 1)
        self.assertTrue("welford" in compute[0] or "tl.sum(" in compute[0])
        self.assertIn("triton_helpers.max2(", compute[0])
        self.assertIn("tl.inline_asm_elementwise(", compute[0])
        self.assertNotIn("in_ptr3", compute[0])  # No BF16 intermediate input.
        # Three FP6 words and the scale.
        self.assertEqual(compute[0].count("tl.store("), 4)

    @parametrize("persistent", [False, True])
    @parametrize("width,shift_groups", [(6144, False), (1024, True)])
    def test_parent_lane_inputs(self, device, persistent, width, shift_groups):
        asm, constraints = combine_inputs_asm(6)

        def fn(x):
            groups = realized_rms_groups(x)
            maximum = groups.abs().amax(-1)
            if shift_groups:
                groups = groups.roll(1, dims=1)
            bits = groups.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            words = inline_asm_elementwise(
                *pairs.unbind(-1),
                maximum.to(torch.int32),
                asm_str=asm,
                constraints=constraints,
                dtype=(torch.int32,) * 6,
            )
            return torch.stack(words, -1).view(torch.uint8), maximum

        self.compare(fn, lane_input(device, width), persistent)
        if shift_groups:
            self.assertGreater(metrics.generated_kernel_count, 1)
        else:
            self.assertEqual(metrics.codegen_nested_reduction, 1)
            self.assertEqual(metrics.generated_kernel_count, 1)

    @parametrize("persistent", [False, True])
    @parametrize("quantized_output", [False, True])
    def test_parent_full_lane_source(self, device, persistent, quantized_output):
        # The lanes read a value computed from the grouped reduction's output.
        asm, constraints = combine_inputs_asm(3)

        def fn(x):
            groups = realized_rms_groups(x)
            maximum = groups.abs().amax(-1)
            scale = maximum[..., None].float() + 1
            quantized = (groups.float() / scale).to(torch.bfloat16)
            if not quantized_output:
                quantized = torch.ops._inductor_test.realize(quantized)
            bits = quantized.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            words = inline_asm_elementwise(
                *pairs.unbind(-1),
                maximum.to(torch.int32),
                asm_str=asm,
                constraints=constraints,
                dtype=(torch.int32,) * 3,
            )
            packed = torch.stack(words, -1).view(torch.uint8)
            return (packed, maximum, quantized) if quantized_output else (packed,)

        self.compare(fn, lane_input(device), persistent)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(metrics.generated_kernel_count, 1)

    @parametrize("rows", [128, 256])
    @config.patch(
        {
            "triton.nested_reduction": True,
            "loop_ordering_after_fusion": True,
            "emulate_precision_casts": True,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_tiled_group_axes(self, device, rows):
        # With 128 rows the first axis of the GEMM scale tile has size 1.
        asm, constraints = combine_inputs_asm(3)
        width, padded = 1056, 1152

        def fn(x, y, native=True):
            value = (x + (x.float() * y.float()).to(torch.float16)).to(torch.bfloat16)
            value = torch.nn.functional.pad(value, (0, padded - width))
            groups = (x.shape[0] // 128, 4, 32, padded // 192, 3, 2)
            maximum = value.float().view(*groups, 32).abs().amax(-1)
            bits = (value.view(torch.int16).int() & 65535).view(*groups, 16, 2)
            pairs = bits[..., 0] | (bits[..., 1] << 16)
            scale = maximum.view(torch.int32)
            if native:
                words = inline_asm_elementwise(
                    *pairs.unbind(-1),
                    scale,
                    asm_str=asm,
                    constraints=constraints,
                    dtype=(torch.int32,) * 3,
                )
            else:
                words = xor_words(pairs, scale, 3)
            return torch.stack(words, -1).view(x.shape[0], -1), maximum

        x = torch.randn(rows, width, device=device, dtype=torch.float16)
        y = torch.randn_like(x)
        metrics.reset()
        compiled = torch.compile(fn, fullgraph=True)
        actual, kernels = run_and_get_kernels(compiled, x, y)
        self.assertEqual(actual, fn(x, y, False), atol=0, rtol=0)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(len([k for k in kernels if "tl.load(" in k]), 1)

    @parametrize("rows", [128, 1024])
    @config.patch(
        {
            "triton.nested_reduction": True,
            "loop_ordering_after_fusion": True,
            "emulate_precision_casts": True,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_padded_two_pass_layer_norm(self, device, rows):
        # Padding the input (instead of padding the packed outputs with
        # constants) lets the padded groups flow through the same kernel. The
        # payload padding is only zero because the conversion maps zero inputs
        # to zero, which Inductor cannot see through inline asm, so the graph
        # has to be written this way rather than rewritten automatically.
        asm, constraints = combine_inputs_asm(3)
        width, padded = 1056, 1152

        def fn(x, weight, bias, native=True):
            value = torch.nn.functional.pad(x, (0, padded - width)).float()
            columns = torch.arange(padded, device=x.device)
            mean = value.sum(-1, keepdim=True) / width
            centered = torch.where(columns < width, value - mean, 0)
            variance = centered.square().sum(-1, keepdim=True) / width
            normalized = centered * torch.rsqrt(variance + 1e-5)
            weight = torch.nn.functional.pad(weight, (0, padded - width))
            bias = torch.nn.functional.pad(bias, (0, padded - width))
            normalized = normalized * weight.float() + bias.float()
            value = torch.where(columns < width, normalized.to(torch.bfloat16), 0)
            groups = value.view(x.shape[0], -1, 32)
            maximum = groups.float().abs().amax(-1)
            bits = groups.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            scale = maximum.to(torch.int32)
            if native:
                words = inline_asm_elementwise(
                    *pairs.unbind(-1),
                    scale,
                    asm_str=asm,
                    constraints=constraints,
                    dtype=(torch.int32,) * 3,
                )
            else:
                words = xor_words(pairs, scale, 3)
            return torch.stack(words, -1).view(x.shape[0], -1), maximum

        # Repeated signed rows keep the sums exact and BF16 rounding away from
        # midpoints, so the reference can use a different reduction order.
        row = (torch.arange(width, device=device) % 32 - 16).float() / 8
        signs = (torch.arange(rows, device=device) % 2 * 2 - 1).float()
        x = (signs[:, None] * row[None, :]).half()
        weight = ((torch.arange(width, device=device) % 5 + 1).float() / 8).half()
        bias = ((torch.arange(width, device=device) % 3 - 1).float() / 16).half()
        with config.patch({"triton.nested_reduction": False}):
            expected = torch.compile(lambda *a: fn(*a, native=False))(x, weight, bias)
        torch._dynamo.reset()
        actual, kernels = run_and_get_kernels(
            torch.compile(fn, fullgraph=True), x, weight, bias
        )
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(len(kernels), 1)
        # The 32 bf16 lanes are split once, as 16 packed uint32 pairs.
        self.assertEqual(kernels[0].count("tl.split("), 16)
        self.assertIn("tl.uint32", kernels[0])


class TestNestedLaneForwarding(NestedLaneTestCase):
    @parametrize("persistent", [False, True])
    @parametrize("source_before_reduction", [False, True])
    def test_source_lifetime(self, device, persistent, source_before_reduction):
        width = 6144

        def fn(x):
            source = torch.ops._inductor_test.realize(x + 1)
            mean_square = source.square().mean(-1, keepdim=True)
            value = source * torch.rsqrt(mean_square + 1e-5)
            value = torch.ops._inductor_test.realize(value)
            maximum = value.view(x.shape[0], -1, 32).abs().amax(-1)
            selected = source if source_before_reduction else value
            lanes = selected.view(x.shape[0], -1, 32)
            return maximum + lanes[..., 0], maximum + lanes[..., 31]

        torch.manual_seed(1234)
        x = torch.randn(16, width, device=device)
        self.compare(fn, x, persistent, exact=False)
        if source_before_reduction:
            self.assertGreater(metrics.generated_kernel_count, 1)
        else:
            self.assertEqual(metrics.generated_kernel_count, 1)
            self.assertEqual(metrics.codegen_nested_reduction, 1)


class TestNestedLanePairs(NestedLaneTestCase):
    @parametrize("persistent", [False, True])
    @parametrize("operation", ["sub", "pack"])
    @parametrize(
        "group_size,pattern",
        [(2, "aligned"), (2, "reversed")]
        + [
            (32, pattern)
            for pattern in ("aligned", "reversed", "unaligned", "different_source")
        ],
    )
    def test_pair_lane_inputs(
        self, device, persistent, group_size, operation, pattern
    ):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, group_size)
            maximum = groups.abs().amax(-1)
            other = groups
            if pattern == "different_source":
                other = torch.ops._inductor_test.realize(groups.view_as(x) + 1).view_as(
                    groups
                )
            outputs = []
            for base in (0, group_size - 2):
                left, right = base, base + 1
                if pattern == "reversed":
                    left, right = right, left
                elif pattern == "unaligned":
                    left, right = (left + 1) % group_size, (right + 1) % group_size
                a, b = groups[..., left], other[..., right]
                if operation == "pack":
                    a = a.view(torch.int16).to(torch.int32) & 65535
                    b = b.view(torch.int16).to(torch.int32) & 65535
                    outputs.append((a | (b << 16)) ^ maximum.to(torch.int32))
                else:
                    outputs.append(a.float() - b.float() + maximum)
            return tuple(outputs)

        x = lane_input(device, width)
        source = self.compare(fn, x, persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        # Exact counts catch a second split tree of the same source.
        splits = {2: {"sub": 1, "pack": 2}, 32: {"sub": 31, "pack": 31}}
        expected = splits[group_size][operation]
        if pattern == "different_source":
            expected *= 2
        self.assertEqual(source.count("tl.split("), expected)

    @parametrize("persistent", [False, True])
    @parametrize("operation", ["recursive", "mixed_factor", "expensive"])
    def test_pair_composition(self, device, persistent, operation):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, 8)
            maximum = groups.abs().amax(-1)
            a, b, c, d = (groups[..., lane].float() for lane in range(4))
            if operation == "recursive":
                return ((a - b) - (c - d)) + maximum
            if operation == "mixed_factor":
                pair = a - b
                return pair - c + maximum, pair + maximum
            return torch.sin(a) - torch.sin(b) + maximum

        x = lane_input(device, width)
        source = self.compare(fn, x, persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(source.count("tl.split("), 7)
        if operation == "expensive":
            self.assertLess(source.index("tl.split("), source.index(".sin("))


class TestNestedLaneCasts(NestedLaneTestCase):
    @parametrize("persistent", [False, True])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("lanes", [(0, 31), tuple(range(32))])
    @parametrize("pattern", ["lane_constants", "pair_constants", "mixed"])
    def test_shared_bitcast_before_lane_split(
        self, device, persistent, dtype, lanes, pattern
    ):
        def fn(x):
            groups = realized_rms_groups(x, dtype=dtype)
            maximum = groups.abs().amax(-1).to(torch.int32)
            result = maximum
            for lane in lanes:
                if pattern == "mixed" and lane != 0:
                    result = result ^ groups[..., lane].to(torch.int32)
                    continue
                bits = groups[..., lane].view(torch.int16).to(torch.int32)
                if pattern == "pair_constants":
                    other = groups[..., lane ^ 1].view(torch.int16).to(torch.int32)
                    result = result ^ (bits - other * (lane + 1))
                else:
                    result = result ^ (bits * (lane + 1))
            return result

        source = self.compare(fn, lane_input(device), persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        bitcast = ".to(tl.int16, bitcast=True)"
        # Integer carrier unpacking does not repeat the source's floating-point cast.
        source_bitcasts = source.count(bitcast) - source.count(".to(tl.uint16)" + bitcast)
        if len(lanes) == 32:
            self.assertEqual(source_bitcasts, 1)
            self.assertLess(
                source.index(".to(tl.int16, bitcast=True)"), source.index("tl.split(")
            )
            # The int16 tile is split as packed uint32 pairs.
            self.assertIn("tl.uint32", source)
            splits = 16
        else:
            count = {"lane_constants": 2, "pair_constants": 4, "mixed": 1}[pattern]
            self.assertEqual(source_bitcasts, count)
            splits = 31
        # Mixed also splits the floating-point source for its plain lane reads.
        self.assertEqual(source.count("tl.split("), splits + 31 * (pattern == "mixed"))

    @parametrize("persistent", [False, True])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("bitcast", [False, True])
    def test_cast_before_lane_split(self, device, persistent, dtype, bitcast):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, dtype=dtype)
            maximum = groups.abs().amax(-1)
            lanes = tuple(groups[..., lane] for lane in (0, 1, 15, 31))
            if bitcast:
                lanes = tuple(lane.view(torch.int16) for lane in lanes)
            return tuple(lane.to(torch.int32) + maximum for lane in lanes)

        x = lane_input(device, width)
        source = self.compare(fn, x, persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        # Each lane casts at lane width on one shared split of the source
        # rather than casting the whole tile and splitting the result.
        self.assertEqual(source.count("tl.split("), 31)
        self.assertEqual(source.count(".to(tl.int32)"), 4)
        self.assertLess(source.index("tl.split("), source.index(".to(tl.int32)"))


class TestNestedLaneSplits(NestedLaneTestCase):
    @parametrize("persistent", [False, True])
    @parametrize("group_size", [8, 64])
    def test_lane_specific_constants(self, device, persistent, group_size):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, group_size)
            total = groups.float().abs().amax(-1)
            for lane in range(group_size):
                total = total + groups[..., lane].float() * (lane + 1)
            return total

        source = self.compare(fn, lane_input(device, width), persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        # Each lane scales by its own constant, so one lane split must serve
        # every lane instead of a split tree per scaled parent.
        self.assertEqual(source.count("tl.split("), group_size - 1)

    @parametrize("persistent", [False, True])
    @parametrize("lanes", [(1,), (0, 1, 2, 3)])
    def test_no_unused_splits(self, device, persistent, lanes):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, 4)
            total = groups.float().abs().amax(-1)
            for lane in lanes:
                total = total + groups[..., lane].float() * 2
            return total

        source = self.compare(fn, lane_input(device, width), persistent)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        unused = []
        for match in re.finditer(r"((?:tmp\d+, )+tmp\d+) = tl\.split\(", source):
            later = source[match.end() :]
            unused += [
                name
                for name in match.group(1).split(", ")
                if not re.search(rf"\b{name}\b", later)
            ]
        self.assertEqual(unused, [])


# The inline asm strings are PTX or AMDGCN.
instantiate_device_type_tests(TestRecursiveLaneSplit, globals(), only_for="cuda")
instantiate_device_type_tests(TestNestedAsmLaneInputs, globals(), only_for="cuda")

instantiate_device_type_tests(TestPitchedLaneSources, globals(), only_for=GPU_TYPE)
instantiate_device_type_tests(TestNestedLaneForwarding, globals(), only_for=GPU_TYPE)
instantiate_device_type_tests(TestNestedLaneCasts, globals(), only_for=GPU_TYPE)
instantiate_device_type_tests(TestNestedLanePairs, globals(), only_for=GPU_TYPE)
instantiate_device_type_tests(TestNestedLaneSplits, globals(), only_for=GPU_TYPE)


if __name__ == "__main__":
    run_tests()
