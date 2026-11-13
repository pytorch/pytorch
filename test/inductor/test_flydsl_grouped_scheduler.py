# Owner(s): ["module: inductor"]
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize


try:
    from . import test_flydsl_template as input_factory
except ImportError:
    import test_flydsl_template as input_factory


class TestFlyDSLGroupedSchedulerPolicy(TestCase):
    def setUp(self):
        super().setUp()
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

    def test_grid_cap_and_cache_modes(self):
        import dataclasses

        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp8_grouped_gemm_gfx950 as kernel,
        )

        param = kernel.make_mxfp8_grouped_gemm_param(512, 2048, 8, 128, 128)
        props = SimpleNamespace(multi_processor_count=256)
        for rows in (0, 1, 128, 4096, 32768):
            bound, _ = kernel.get_mxfp8_grouped_gemm_grid_size(param, rows)
            grid = kernel.get_mxfp8_grouped_gemm_persistent_grid_size(
                param, rows, props
            )
            self.assertEqual(grid, max(1, min(bound, 256)))
        single = dataclasses.replace(param, persistent=False)
        self.assertNotEqual(param.__cache_signature__(), single.__cache_signature__())
        self.assertNotEqual(
            kernel.make_mxfp8_grouped_gemm_kernel_name(param),
            kernel.make_mxfp8_grouped_gemm_kernel_name(single),
        )
        self.assertIs(
            kernel.cached_launch(*param.key()), kernel.cached_launch(*param.key())
        )


@unittest.skipIf(torch.version.hip is None, "requires ROCm")
class TestFlyDSLGroupedScheduler(TestCase):
    def setUp(self):
        super().setUp()
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        if arch.split(":")[0] != "gfx950":
            self.skipTest("requires gfx950")

    @parametrize(
        "tile", ((64, 128), (64, 256), (128, 128), (128, 256), (256, 128), (256, 256))
    )
    @parametrize("k", (512, 640, 2048))
    def test_persistent_tiles(self, device, tile, k):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp8_grouped_gemm_gfx950 as kernel,
        )
        from torch._inductor.runtime.flydsl_cache import flydsl_tensor_arg

        n = 384  # Partial column tile for BLOCK_C=256.
        param = kernel.make_mxfp8_grouped_gemm_param(k, n, 6, *tile)
        # Same compiled specialization, different routing and token counts.
        for sizes in ([0, 1, 257, 0, 63, 513], [65, 0, 0, 129, 17, 0]):
            a, b, sa, sb, offs, reference = (
                input_factory.TestFlyDSLTemplate._make_mxfp8_grouped_inputs(
                    sizes, k, n, device=device
                )
            )
            output = torch.empty_like(reference)
            # Non-power-of-two grid, repeated tiles, and empty groups.
            with mock.patch.object(
                kernel,
                "get_mxfp8_grouped_gemm_persistent_grid_size",
                return_value=3,
            ):
                for _ in range(8):
                    output.fill_(float("nan"))
                    kernel.launch_mxfp8_grouped_gemm_gfx950(
                        output,
                        a,
                        b.permute(0, 2, 1),
                        sa,
                        sb,
                        offs,
                        param,
                        torch.cuda.current_stream(),
                        tensor_arg=flydsl_tensor_arg,
                    )
                    self.assertEqual(
                        output.float(), reference.float(), atol=6e-2, rtol=6e-2
                    )

    @parametrize("grid", (1, 3, 256))
    def test_empty_routing_leaves_output_untouched(self, device, grid):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp8_grouped_gemm_gfx950 as kernel,
        )
        from torch._inductor.runtime.flydsl_cache import flydsl_tensor_arg

        a, b, sa, sb, offs, reference = (
            input_factory.TestFlyDSLTemplate._make_mxfp8_grouped_inputs(
                [0, 17, 0, 0], 512, 384, device=device
            )
        )
        param = kernel.make_mxfp8_grouped_gemm_param(512, 384, 4, 64, 256)
        output = torch.full_like(reference, 42)
        offs.zero_()
        with mock.patch.object(
            kernel, "get_mxfp8_grouped_gemm_persistent_grid_size", return_value=grid
        ):
            kernel.launch_mxfp8_grouped_gemm_gfx950(
                output,
                a,
                b.permute(0, 2, 1),
                sa,
                sb,
                offs,
                param,
                torch.cuda.current_stream(),
                tensor_arg=flydsl_tensor_arg,
            )
        self.assertEqual(output, torch.full_like(output, 42))

    def test_persistent_routing_changes_in_place(self, device):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp8_grouped_gemm_gfx950 as kernel,
        )
        from torch._inductor.runtime.flydsl_cache import flydsl_tensor_arg

        a, b, sa, sb, offs, _ = (
            input_factory.TestFlyDSLTemplate._make_mxfp8_grouped_inputs(
                [256] * 4, 2048, 384, device=device
            )
        )
        a_scales = torch.pow(2.0, sa.view(torch.uint8).float() - 127)
        b_scales = torch.pow(2.0, sb.view(torch.uint8).float() - 127).reshape(
            4, 384, -1
        )
        param = kernel.make_mxfp8_grouped_gemm_param(2048, 384, 4, 128, 256)
        output = torch.empty((1024, 384), device=device, dtype=torch.bfloat16)
        with mock.patch.object(
            kernel, "get_mxfp8_grouped_gemm_persistent_grid_size", return_value=3
        ):
            for boundaries in (
                [0, 1, 513, 1024],
                [257, 257, 1024, 1024],
                [0, 0, 17, 17],
            ):
                offs.copy_(torch.tensor(boundaries, device=device, dtype=offs.dtype))
                reference = input_factory.TestFlyDSLTemplate._mxfp8_grouped_reference(
                    a, a_scales, b.transpose(-2, -1), b_scales, offs
                )
                output.fill_(42)
                kernel.launch_mxfp8_grouped_gemm_gfx950(
                    output,
                    a,
                    b.permute(0, 2, 1),
                    sa,
                    sb,
                    offs,
                    param,
                    torch.cuda.current_stream(),
                    tensor_arg=flydsl_tensor_arg,
                )
                end = boundaries[-1]
                self.assertEqual(
                    output[:end].float(), reference[:end].float(), atol=0.06, rtol=0.06
                )
                self.assertEqual(output[end:], torch.full_like(output[end:], 42))


instantiate_device_type_tests(TestFlyDSLGroupedScheduler, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
