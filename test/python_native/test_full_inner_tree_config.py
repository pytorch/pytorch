# Owner(s): ["module: dsl-native-ops"]

import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


@unittest.skipUnless(TEST_CUTEDSL, "requires CuTeDSL")
class TestFullInnerTreeConfig(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "dtype,mib,key,fields,nouts,unroll",
        [
            (torch.float32, 256, "sum", 1, 1, 128),
            (torch.bfloat16, 2048, "argmaxi32", 2, 1, 128),
            (torch.float32, 256, "var1", 3, 1, 64),
            (torch.bfloat16, 2048, "stdmean0", 3, 2, 64),
        ],
    )
    def test_sm107_full_profiles(self, dtype, mib, key, fields, nouts, unroll):
        """Pin each distinct SM107 full-reduction policy branch."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        output = (
            torch.float32
            if key == "sum"
            else torch.int64
            if key == "argmaxi32"
            else dtype
        )
        arch = rt.select_full_itree_arch(
            (10, 7),
            dtype,
            key,
            (mib << 20) // dtype.itemsize,
            1,
            field_bits=(32,) * fields,
            out_dtypes=(output,) * nouts,
            alignment=16,
            contiguous=True,
        )
        welford = fields == 3
        self.assertEqual(
            arch,
            rt._ItreeArch(
                4,
                32,
                16,
                False,
                32,
                2048,
                unroll,
                uniform_count=welford,
                combine_weights=welford,
            ),
        )

    @parametrize(
        "updates",
        [
            {"cc": (10, 0)},
            {"N": (256 << 20) // 4 - 1},
            {"trait_key": "mean"},
        ],
    )
    def test_sm107_full_guards(self, updates):
        """Keep calls outside the SM107 policy contract on the generic path."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_key="sum",
            N=(256 << 20) // 4,
            M=1,
            field_bits=(32,),
            out_dtypes=(torch.float32,),
            alignment=16,
            contiguous=True,
        )
        args.update(updates)
        self.assertIsNone(rt.select_full_itree_arch(**args))


@unittest.skipUnless(TEST_CUTEDSL, "requires CuTeDSL")
@unittest.skipUnless(TEST_CUDA and SM90OrLater, "requires Hopper or later")
class TestFullInnerTreeDevice(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_sm107_profile_reaches_every_plan(self):
        """Thread the SM107 profile through the split, re-plan, and combine builders."""
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        real_arch, real_plan, real_combine = (
            rt._full_itree_arch,
            rt.itree_plan,
            rt.itree_combine_plan,
        )
        sm107 = mock.Mock(cc=(10, 7))
        profiles, archs = [], []

        def full_arch(*args):
            with mock.patch.object(rt._hw, "caps", return_value=sm107):
                profiles.append(real_arch(*args))
            return profiles[-1]

        def recorded(real):
            def plan(*args, **kwargs):
                archs.append(kwargs.get("arch"))
                return real(*args, **kwargs)

            return plan

        x = torch.randn(1, (256 << 20) // 4, device="cuda")
        with (
            mock.patch.object(rt, "_full_itree_arch", full_arch),
            mock.patch.object(rt, "itree_plan", recorded(real_plan)),
            mock.patch.object(rt, "itree_combine_plan", recorded(real_combine)),
        ):
            (got,) = rt.reduce_row_tile(
                T.SumOps(acc=cutlass.Float32),
                "sum",
                x,
                [torch.float32],
                order="inner_tree",
            )
        self.assertIsNotNone(profiles[0])
        self.assertEqual(profiles[0].split_stage_rows, 4)
        self.assertGreaterEqual(len(archs), 2)
        self.assertTrue(all(arch is profiles[0] for arch in archs))
        self.assertEqual(got, x.double().sum(dim=1).float(), atol=1e-1, rtol=1e-5)


instantiate_parametrized_tests(TestFullInnerTreeConfig)


if __name__ == "__main__":
    run_tests()
