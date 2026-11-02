# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._inductor import config, metrics
from torch._inductor.utils import run_and_get_kernels
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE


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


instantiate_device_type_tests(TestPitchedLaneSources, globals(), only_for=GPU_TYPE)


if __name__ == "__main__":
    run_tests()
