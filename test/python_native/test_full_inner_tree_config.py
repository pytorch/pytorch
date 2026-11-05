# Owner(s): ["module: dsl-native-ops"]

import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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
            welford_outs=nouts if fields == 3 else 0,
            alignment=16,
            contiguous=True,
        )
        welford = fields == 3
        want = rt._ItreeArch(
            split_stage_rows=4,
            combine_rpb=32,
            combine_group_bytes=16,
            stage_rows=False,
            combine_async=32,
            combine_max=2048,
            combine_unroll=unroll,
            lowp_thread_elems=32,
            uniform_count=welford,
            combine_weights=welford,
        )
        self.assertEqual(arch, want)

    @parametrize(
        "updates",
        [
            {"cc": (10, 0)},
            {"N": (256 << 20) // 4 - 1},
            {"trait_key": "mean"},
            {"dtype": torch.float16, "N": (256 << 20) // 2},
            {"M": 2},
            {"contiguous": False},
            {"alignment": 8},
            {"out_dtypes": (torch.bfloat16,)},
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
            welford_outs=0,
            alignment=16,
            contiguous=True,
        )
        self.assertIsNotNone(rt.select_full_itree_arch(**args))
        args.update(updates)
        self.assertIsNone(rt.select_full_itree_arch(**args))

    @parametrize(
        "updates",
        [
            {"field_bits": (64,) * 3},
            {"welford_outs": 0},
            {"out_dtypes": (torch.bfloat16,)},
            {"out_dtypes": (torch.float32,) * 2},
        ],
    )
    def test_sm107_full_welford_guards(self, updates):
        """Keep Welford calls outside the SM107 Welford contract on the generic path."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_key="var1",
            N=(256 << 20) // 4,
            M=1,
            field_bits=(32,) * 3,
            out_dtypes=(torch.float32,),
            welford_outs=1,
            alignment=16,
            contiguous=True,
        )
        self.assertIsNotNone(rt.select_full_itree_arch(**args))
        args.update(updates)
        self.assertIsNone(rt.select_full_itree_arch(**args))


@unittest.skipUnless(TEST_CUTEDSL, "requires CuTeDSL")
@unittest.skipUnless(TEST_CUDA and SM90OrLater, "requires Hopper or later")
class TestFullInnerTreeDevice(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("key", ["sum", "var1"])
    def test_sm107_profile_reaches_every_plan(self, device, key):
        """Thread the SM107 profile through the split and combine builders."""
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        real_arch, real_plan, real_combine = (
            rt._full_itree_arch,
            rt.itree_plan,
            rt.itree_combine_plan,
        )
        sm107 = mock.Mock(cc=(10, 7))
        profiles, archs, plans = [], [], []

        def full_arch(*args):
            with mock.patch.object(rt._hw, "caps", return_value=sm107):
                profiles.append(real_arch(*args))
            return profiles[-1]

        def recorded(real):
            def plan(*args, **kwargs):
                archs.append(kwargs.get("arch"))
                plans.append(real(*args, **kwargs))
                return plans[-1]

            return plan

        welford = key == "var1"
        trait = (T.WelfordOps if welford else T.SumOps)(acc=cutlass.Float32)
        x = torch.randn(1, (256 << 20) // 4, device=device)
        with (
            mock.patch.object(rt, "_full_itree_arch", full_arch),
            mock.patch.object(rt, "itree_plan", recorded(real_plan)),
            mock.patch.object(rt, "itree_combine_plan", recorded(real_combine)),
        ):
            (got,) = rt.reduce_row_tile(
                trait, key, x, [torch.float32], order="inner_tree"
            )
        self.assertIsNotNone(profiles[0])
        self.assertTrue(all(arch is profiles[0] for arch in archs))
        split, c = plans
        staged = (split.shape, split.rows_per_block, split.stage_e)
        self.assertEqual(staged, ("split", 4, 32))
        folded = (c.combine_tile, c.combine_unroll, c.combine_count, c.combine_weights)
        # Welford partials come from uniform 8192-element batches, so they are weighted.
        want = (2048, 64, 8192, True) if welford else (2048, 128, 0, False)
        self.assertEqual(folded, want)
        # Each stage's cache key ends with its plan's sig.
        for tag, plan in (("rowitree1", split), ("rowitree2", c)):
            self.assertIn(plan.sig, {k[-1] for k in rt._CACHE if k[:2] == (tag, key)})
        ref = x.double().var(dim=1) if welford else x.double().sum(dim=1)
        tol = 1e-4 if welford else 1e-1
        self.assertEqual(got, ref.float(), atol=tol, rtol=1e-5)


instantiate_parametrized_tests(TestFullInnerTreeConfig)
instantiate_device_type_tests(TestFullInnerTreeDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
