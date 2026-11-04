# Owner(s): ["module: dsl-native-ops"]
# Smoke tests for few-row, large-N cross-CTA reduction; OpInfo covers numerics.

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass

from torch._native.cutedsl import launch as _L
from torch._native.ops.reductions import kernel_xcta, tile, traits as T


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestKernelXcta(TestCase):
    def test_reduce_row_xcta(self):
        # A misaligned input needs an element-aligned descriptor.
        raw = torch.randn((1 << 20) + 1, device="cuda")
        x = raw[1:].view(1, -1)
        self.assertEqual(
            _L.supported_alignment(x, tile.TRANSFER_ALIGNMENT),
            x.element_size(),
        )
        out = kernel_xcta.reduce_row_xcta(
            T.SumOps(acc=cutlass.Float32),
            "smoke",
            x,
            torch.float32,
        )
        self.assertEqual(out, x.double().sum(dim=1).float(), atol=2e-2, rtol=1e-4)

    def test_plans_key_and_require_the_current_device(self):
        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.randn(2, 1 << 20, device="cuda")
        before = (set(kernel_xcta._GEOM), set(kernel_xcta._PLAN))
        # One GPU has no other device, so pretend another one is current.
        other = x.get_device() + 1
        with mock.patch.object(torch.cuda, "current_device", return_value=other):
            with self.assertRaisesRegex(AssertionError, "is current"):
                kernel_xcta.reduce_row_xcta(trait, "bydevice", x, torch.float32)
        self.assertEqual((set(kernel_xcta._GEOM), set(kernel_xcta._PLAN)), before)
        kernel_xcta.reduce_row_xcta(trait, "bydevice", x, torch.float32)
        # Geometry and plan keys both end with the device index.
        geom = [k for k in kernel_xcta._GEOM if k[0] == "bydevice"]
        plans = [k for k in kernel_xcta._PLAN if k[1] == "bydevice"]
        self.assertEqual(len(geom), 1)
        self.assertEqual(len(plans), 1)
        self.assertEqual({k[-1] for k in geom + plans}, {x.get_device()})

    def test_two_output_split(self):
        # Stage 2 projects both outputs from one accumulator; this path rejects index traits.
        x = torch.randn(2, 1 << 20, device="cuda")
        res = kernel_xcta.reduce_row_xcta_2out(
            T.AMinMaxOps(acc=cutlass.Float32),
            "smoke_2out",
            x,
            [torch.float32, torch.float32],
        )
        self.assertIsNotNone(res, "the split declined a shape it is meant to serve")
        lo, hi = res
        want = torch.aminmax(x, dim=1)
        self.assertEqual(lo, want.min)
        self.assertEqual(hi, want.max)

    def test_one_kernel_per_vec_class(self):
        # Runtime sub-row geometry lets one kernel serve every M and N in a vector class.
        key = "vecclass"
        trait = T.SumOps(acc=cutlass.Float32)

        def compiled():
            return len([k for k in kernel_xcta._PLAN if key in repr(k)])

        # Multiples of 4096 guarantee a legal exact split; assert it to keep the count meaningful.
        def run(m, n):
            x = torch.randn(m, n, device="cuda")
            out = kernel_xcta.reduce_row_xcta(trait, key, x, torch.float32)
            self.assertIsNotNone(out, f"declined ({m}, {n})")
            torch.testing.assert_close(
                out, x.double().sum(dim=1).float(), atol=2e-2, rtol=1e-4
            )

        kernel_xcta._PLAN.clear()
        for m, n in ((1, 1 << 20), (3, 1 << 20), (2, 1 << 21)):
            run(m, n)
        few = compiled()
        self.assertGreater(few, 0, "nothing compiled -- the count is measuring nothing")
        for n in ((1 << 20) + 4096, (1 << 20) + 8192, (1 << 20) + 12288):
            run(1, n)
        self.assertEqual(
            compiled(), few, "a new N in the same vec class compiled a new kernel"
        )

    def test_declines_are_deliberate(self):
        # Return None for unsplittable rows so the dispatcher can choose a faster path.
        trait = T.SumOps(acc=cutlass.Float32)
        # Below the sub-row floor, no C > 1 is legal.
        self.assertIsNone(
            kernel_xcta.reduce_row_xcta(
                trait, "decline_short", torch.randn(1, 64, device="cuda"), torch.float32
            )
        )
        # Prime N lacks an in-window divisor; decline instead of folding 65537 scalars.
        self.assertIsNone(
            kernel_xcta.reduce_row_xcta(
                trait,
                "decline_prime",
                torch.randn(1, 65537, device="cuda"),
                torch.float32,
            )
        )

    def test_b200_welford_subrow_target(self):
        # Only single-row Welford at the 256 MiB and 2 GiB anchors shortens sub-rows.
        default = kernel_xcta._XctaConfig().subrow_target
        welford, plain = T.WelfordOps(correction=0), T.SumOps(acc=cutlass.Float32)
        n = 128 << 20
        cases = (
            ((1, n), welford, "var0", 16384),
            ((2, n), welford, "var0", default),
            ((1, n - 2048), welford, "var0", default),
            ((1, n), plain, "sum", default),
        )
        caps = SimpleNamespace(cc=(10, 0))
        for shape, trait, key, want in cases:
            x = torch.empty(shape, device="cuda", dtype=torch.float16)
            with (
                self.subTest(shape=shape, key=key),
                mock.patch.object(kernel_xcta._hw, "caps", return_value=caps),
                mock.patch.dict(kernel_xcta._GEOM, clear=True),
            ):
                kernel_xcta.reduce_row_xcta(trait, key, x, torch.float32)
                # Geometry keys carry the sub-row target sixth.
                self.assertEqual({k[5] for k in kernel_xcta._GEOM}, {want})


if __name__ == "__main__":
    run_tests()
