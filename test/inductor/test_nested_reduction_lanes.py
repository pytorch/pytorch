# Owner(s): ["module: inductor"]

import re

import torch
import torch._inductor.test_operators
from torch._dynamo.testing import CompileCounterWithBackend
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




class TestNestedLaneSplits(NestedLaneTestCase):
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


instantiate_device_type_tests(TestRecursiveLaneSplit, globals(), only_for="cuda")
instantiate_device_type_tests(TestPitchedLaneSources, globals(), only_for=GPU_TYPE)
instantiate_device_type_tests(TestNestedLaneSplits, globals(), only_for=GPU_TYPE)


if __name__ == "__main__":
    run_tests()
