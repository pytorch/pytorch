# Owner(s): ["module: inductor"]

"""End-to-end tests for padded scatter lowering and Triton fusion."""

import torch
import torch._inductor.config as inductor_config
from torch._inductor import inductor_prims, metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def _layernorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, correction=0)
    return (x - mean) / torch.sqrt(variance + 1.0e-6)


class _PersistentReductionChoices(InductorChoices):
    @staticmethod
    def should_use_cooperative_reduction(*args, **kwargs):
        return False

    @staticmethod
    def should_use_persistent_reduction(*args, **kwargs):
        return True


@inductor_config.patch("force_disable_caches", True)
class PaddedScatterTest(TestCase):
    __unittest_skip__ = not HAS_GPU

    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.reset()

    def test_invalid_dynamic_layout_rejected(self):
        cases = (
            ((97, 4), 128, 4),
            ((1, 5), 128, 4),
            ((1, 4), 192, 4),
        )
        for enabled in (False, True):
            for shape, padded_rows, padded_cols in cases:
                with self.subTest(
                    enabled=enabled,
                    shape=shape,
                    padded_rows=padded_rows,
                    padded_cols=padded_cols,
                ):
                    torch._dynamo.reset()
                    values = torch.empty(shape, dtype=torch.uint8, device=GPU_TYPE)
                    torch._dynamo.mark_dynamic(values, 0, min=1, max=max(shape[0], 192))

                    def f(values):
                        return inductor_prims.padded_xdl_scale_scatter(
                            values,
                            padded_rows,
                            padded_cols,
                            96,
                            128,
                            32,
                            4,
                            2,
                            127,
                        )

                    with (
                        inductor_config.patch(
                            "triton.enable_fuse_auxiliary_writes", enabled
                        ),
                        self.assertRaisesRegex(
                            torch._dynamo.exc.BackendCompilerFailed,
                            "expect_true failed",
                        ),
                    ):
                        torch.compile(f, fullgraph=True)(values)

    @inductor_config.patch(
        {
            "triton.enable_fuse_auxiliary_writes": True,
            "triton.nested_reduction": False,
        }
    )
    def test_dynamic_quantization_fusion(self):
        cols, group = 3, 32

        def f(x):
            rows = x.shape[0]
            grouped = x.reshape(rows, cols, group)
            block_amax = grouped.abs().amax(dim=-1)
            scale = block_amax.clamp_min(1.0e-12)
            quantized = (grouped / scale.unsqueeze(-1)).to(torch.float16)
            scales = (block_amax > 0).to(torch.uint8) * 17
            padded_rows = (rows + 95) // 96 * 128
            padded = inductor_prims.padded_xdl_scale_scatter(
                scales,
                padded_rows,
                4,
                96,
                128,
                32,
                4,
                2,
                127,
            )
            return quantized, padded

        example = torch.randn(101, cols * group, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(example, 0, min=1, max=192)
        compiled = torch.compile(f, fullgraph=True)
        with (
            fresh_inductor_cache(),
            V.set_choices_handler(_PersistentReductionChoices()),
        ):
            actual, code = run_and_get_code(compiled, example)

        expected = f(example)
        self.assertEqual(expected[0], actual[0], atol=1.0e-3, rtol=1.0e-3)
        self.assertEqual(expected[1], actual[1], atol=0, rtol=0)
        self.assertEqual(metrics.codegen_nested_reduction, 0)
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(len(code), 1)
        FileCheck().check("for auxiliary_offset_").run(code[0])
        FileCheck().check("tl.arange(0, 256)").run(code[0])

        for rows in (37, 95, 96, 97, 192):
            x = torch.randn(rows, cols * group, device=GPU_TYPE)
            expected = f(x)
            actual = compiled(x)
            self.assertEqual(expected[0], actual[0], atol=1.0e-3, rtol=1.0e-3)
            self.assertEqual(expected[1], actual[1], atol=0, rtol=0)

        self.assertEqual(metrics.generated_kernel_count, 1)

    @inductor_config.patch(
        {
            "loop_ordering_after_fusion": True,
            "triton.enable_fuse_auxiliary_writes": True,
            "triton.nested_reduction": True,
        }
    )
    def test_dynamic_nested_reduction_fusion(self):
        cols, group = 128, 32

        def f(x):
            rows = x.shape[0]
            normalized = _layernorm(x)
            grouped = normalized.reshape(rows, cols, group)
            block_amax = grouped.abs().amax(dim=-1)
            scale = block_amax.clamp_min(1.0e-12)
            quantized = (grouped / scale.unsqueeze(-1)).to(torch.float16)
            scales = (block_amax > 0).to(torch.uint8) * 17
            padded_rows = (rows + 95) // 96 * 128
            padded = inductor_prims.padded_xdl_scale_scatter(
                scales,
                padded_rows,
                cols,
                96,
                128,
                32,
                4,
                2,
                127,
            )
            return quantized, padded

        example = torch.randn(101, cols * group, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(example, 0, min=1, max=192)
        compiled = torch.compile(f, fullgraph=True)
        with (
            fresh_inductor_cache(),
            V.set_choices_handler(_PersistentReductionChoices()),
        ):
            actual, code = run_and_get_code(compiled, example)

        expected = f(example)
        self.assertEqual(expected[0], actual[0], atol=1.0e-3, rtol=1.0e-3)
        self.assertEqual(expected[1], actual[1], atol=0, rtol=0)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(len(code), 1)
        FileCheck().check("for auxiliary_offset_").run(code[0])
        FileCheck().check("tl.arange(0, 256)").run(code[0])

        for rows in (37, 95, 96, 97, 192):
            x = torch.randn(rows, cols * group, device=GPU_TYPE)
            expected = f(x)
            actual = compiled(x)
            self.assertEqual(expected[0], actual[0], atol=1.0e-3, rtol=1.0e-3)
            self.assertEqual(expected[1], actual[1], atol=0, rtol=0)

        self.assertEqual(metrics.generated_kernel_count, 1)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
