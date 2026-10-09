# Owner(s): ["module: inductor"]

import torch
import torch.nn.functional as F
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def rounded_groups(x, y, group):
    x = F.pad(x, (0, group))
    y = F.pad(y, (0, group))
    product = (x.float() * y.float()).to(torch.float16)
    value = (x + product).to(torch.bfloat16)
    return value.reshape(-1, group)


def stacked_lanes(x, y):
    value = rounded_groups(x, y, 32)
    maximum = value.float().abs().amax(-1)
    return torch.stack([maximum + value[:, lane].float() for lane in (0, 7, 31)], -1)


class TestPersistentReductionLanes(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            config.patch(
                {
                    "emulate_precision_casts": True,
                    "loop_ordering_after_fusion": True,
                    "triton.nested_reduction": True,
                    "triton.multi_kernel": 1,
                    "force_disable_caches": True,
                }
            )
        )

    def check(self, fn, device, group=32, reference=None, extra=(), kernels=1):
        x = torch.randn(37, 3 * group, device=device, dtype=torch.float16)
        inputs = (x, torch.randn_like(x), *extra)
        expected = (reference or fn)(*inputs)
        torch._dynamo.reset()
        metrics.reset()
        actual, sources = run_and_get_code(torch.compile(fn, fullgraph=True), *inputs)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        source = "\n".join(sources)
        if kernels is not None:
            self.assertEqual(metrics.generated_kernel_count, kernels)
        if kernels == 1:
            self.assertEqual(source.count("tl.load("), 2)
        return source

    @parametrize("group,all_lanes", [(8, False), (64, False), (32, True)])
    def test_internal_rounded_source(self, device, group, all_lanes):
        def fn(x, y):
            value = rounded_groups(x, y, group)
            maximum = value.float().abs().amax(-1)
            if all_lanes:
                return tuple(maximum + lane.float() for lane in value.unbind(-1))
            return maximum + value[:, 0].float() + value[:, group - 1].float()

        source = self.check(fn, device, group)
        self.assertNotIn("for r0_offset", source)

    @parametrize("pointwise_cat", [False, True])
    def test_disjoint_stack_outputs(self, device, pointwise_cat):
        def fn(x, y):
            product = (x.float() * y.float()).to(torch.float16)
            value = (x + product).to(torch.bfloat16).reshape(-1, 32)
            maximum = value.float().abs().amax(-1)
            return torch.stack(
                [maximum + value[:, lane].float() for lane in range(0, 32, 2)], -1
            )

        limit = 16 if pointwise_cat else 0
        patches = {
            "max_complex_pointwise_cat_inputs": limit,
            "max_pointwise_cat_inputs": limit,
        }
        with config.patch(patches):
            self.check(fn, device)

    def test_three_output_asm_stack(self, device):
        op, suffix, reg = (
            ("v_add_f32", "", "v") if torch.version.hip else ("add.f32", ";", "f")
        )
        asm = "\n".join(
            f"{op} ${output}, ${3 + lane}, $35{suffix}"
            for output, lane in enumerate((0, 7, 31))
        )
        constraints = ",".join([f"=&{reg}"] * 3 + [reg] * 33)

        def fn(x, y):
            value = rounded_groups(x, y, 32).float()
            maximum = value.abs().amax(-1)
            words = inline_asm_elementwise(
                *value.unbind(-1),
                maximum,
                asm_str=asm,
                constraints=constraints,
                dtype=(torch.float32,) * 3,
            )
            return torch.stack(words, -1)

        self.check(fn, device, reference=stacked_lanes)

    def test_concat_destination_reader_is_separate(self, device):
        def fn(x, y):
            packed = stacked_lanes(x, y)
            return packed, packed[:, 0] + packed[:, 1]

        self.check(fn, device, kernels=2)
        self.assertEqual(metrics.codegen_nested_reduction, 1)

    def test_concat_with_foreign_column(self, device):
        # Only one column of the stack is a lane epilogue; the other column is
        # written by an unrelated reduction.
        def fn(x, y, z):
            value = rounded_groups(x, y, 32)
            maximum = value.float().abs().amax(-1)
            return torch.stack((maximum + value[:, 0].float(), z.amax(-1)), -1)

        z = torch.randn(148, 8, device=device)
        self.check(fn, device, extra=(z,), kernels=None)
        self.assertEqual(metrics.codegen_nested_reduction, 1)

    def test_source_mutation_preserves_versions(self, device):
        def fn(x, y):
            maximum = x.float().abs().amax(-1)
            x.add_(y)
            return maximum + x[:, 0].float() + x[:, 31].float()

        x = torch.randn(37, 32, device=device, dtype=torch.float16)
        y = torch.randn_like(x)
        expected_x = x.clone()
        expected = fn(expected_x, y)
        actual_x = x.clone()
        torch._dynamo.reset()
        actual = torch.compile(fn, fullgraph=True)(actual_x, y)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual_x, expected_x, atol=0, rtol=0)

    @parametrize("disable_config", [False, True])
    def test_forced_looped_falls_back(self, device, disable_config):
        class LoopedChoices(InductorChoices):
            @staticmethod
            def should_use_persistent_reduction(features, cooperative_reduction):
                return False

        def fn(x, y):
            value = rounded_groups(x, y, 32)
            return value.float().abs().amax(-1) + value[:, 7].float()

        choices = InductorChoices() if disable_config else LoopedChoices()
        with (
            config.patch({"triton.persistent_reductions": not disable_config}),
            V.set_choices_handler(choices),
        ):
            self.check(fn, device, kernels=None)
        self.assertEqual(metrics.codegen_nested_reduction, 0)

    def test_single_group_is_not_forwarded(self, device):
        def fn(x):
            value = (x.float() * 1.5).to(torch.bfloat16).view(-1, 32)
            return value.float().abs().amax(-1) + value[:, 0].float()

        x = torch.randn(1, 32, device=device)
        expected = fn(x)
        torch._dynamo.reset()
        metrics.reset()
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(metrics.codegen_nested_reduction, 0)

    # Multi-kernel with a dynamic reduction size fails upstream independently.
    @config.patch({"triton.multi_kernel": 0})
    def test_dynamic_group_count(self, device):
        # Recompiling at one group per row specializes the group count to 1
        # while the frame keeps it symbolic.
        def fn(x, y):
            value = (x + (x.float() * y.float()).half()).float()
            groups = value.view(x.shape[0], -1, 32)
            return value, groups[..., 17] - groups.amin(-1)

        compiled = torch.compile(fn, fullgraph=True)
        for groups in (8, 1):
            x = torch.randn(6, groups * 32, device=device).half()
            y = torch.randn_like(x)
            torch._dynamo.mark_dynamic(x, 1)
            torch._dynamo.maybe_mark_dynamic(y, 1)
            self.assertEqual(compiled(x, y), fn(x, y), atol=0, rtol=0)

    @parametrize("group", [31, 32])
    def test_cross_row_lane_is_not_forwarded(self, device, group):
        def fn(x, y):
            value = rounded_groups(x, y, group)
            shifted = torch.roll(value, shifts=1, dims=0)
            return value.float().abs().amax(-1) + shifted[:, 0].float()

        self.check(fn, device, group, kernels=None)
        self.assertEqual(metrics.codegen_nested_reduction, 0)


instantiate_device_type_tests(
    TestPersistentReductionLanes, globals(), only_for=("cuda",)
)

if __name__ == "__main__":
    run_tests()
