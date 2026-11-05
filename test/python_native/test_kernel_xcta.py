# Owner(s): ["module: dsl-native-ops"]
# Smoke tests for few-row, large-N cross-CTA reduction; OpInfo covers numerics.

import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


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

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_welford_long_subrows(self, dtype):
        """Split one Welford row into 16384-element sub-rows, the geometry B200's Welford
        rule launches; TestXctaConfig pins the selector."""
        n = 1 << 22
        x = torch.randn(1, n, device="cuda", dtype=dtype)
        key = f"long_subrows_{dtype}"
        f32 = [torch.float32] * 2
        trait = T.VarMeanOps(correction=0)
        res = kernel_xcta.reduce_row_xcta_2out(trait, key, x, f32, subrow_target=16384)
        self.assertIsNotNone(res, "the split declined a shape it is meant to serve")
        # Geometry entries lead with the sub-row count C.
        splits = {geom[0] for k, geom in kernel_xcta._GEOM.items() if k[0] == key}
        self.assertEqual(splits, {n // 16384})
        want = torch.var_mean(x.double(), dim=1, correction=0)
        self.assertEqual(res, tuple(w.float() for w in want), atol=1e-4, rtol=1e-4)


class TestXctaConfig(TestCase):
    @parametrize(
        "case,updates,subrow_target",
        [
            ("var_256mib", {}, 16384),
            ("fp32_2gib", {"dtype": torch.float32, "N": 512 << 20}, 16384),
            ("var_mean", {"nouts": 2, "welford_outs": 2}, 16384),
            (
                "var_mean_trait",
                {"nouts": 2, "welford_outs": T.welford_outputs(T.VarMeanOps())},
                16384,
            ),
            (
                "return_mean",
                {"welford_outs": T.welford_outputs(T.WelfordOps(return_mean=True))},
                8192,
            ),
            ("two_rows", {"M": 2}, 8192),
            ("off_anchor", {"N": (128 << 20) - 2048}, 8192),
            ("sum", {"welford_outs": 0}, 8192),
            ("acc64", {"acc_bits": 64}, 8192),
            ("sm90", {"cc": (9, 0)}, 8192),
        ],
        name_fn=lambda case, updates, subrow_target: case,
    )
    def test_b200_welford_subrow_target(self, case, updates, subrow_target):
        """Lengthen sub-rows only for single-row Welford at the 256 MiB and 2 GiB anchors."""
        args = dict(
            cc=(10, 0),
            dtype=torch.float16,
            N=128 << 20,
            M=1,
            nouts=1,
            welford_outs=1,
            acc_bits=32,
        )
        cfg = kernel_xcta.select_xcta_config(**(args | updates))
        self.assertEqual(cfg, kernel_xcta._XctaConfig(subrow_target=subrow_target))


instantiate_parametrized_tests(TestKernelXcta)
instantiate_parametrized_tests(TestXctaConfig)


if __name__ == "__main__":
    run_tests()
