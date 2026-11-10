# Owner(s): ["module: dsl-native-ops"]

import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


if not TEST_CUTEDSL:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass

from torch._native.ops.reductions import kernel_rowtile as rt, traits as T


class TestFullInnerTreeConfig(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "dtype,mib,trait_type,fields,nouts,unroll",
        [
            (torch.float32, 256, T.SumOps, 1, 1, 128),
            (torch.bfloat16, 2048, T.ArgMaxOps, 2, 1, 128),
            (torch.float32, 256, T.WelfordOps, 3, 1, 64),
            (torch.bfloat16, 2048, T.VarMeanOps, 3, 2, 64),
        ],
        name_fn=lambda dtype, mib, trait_type, *_: f"{trait_type.__name__}_{mib}mib",
    )
    def test_sm107_full_profiles(self, dtype, mib, trait_type, fields, nouts, unroll):
        """Pin each distinct SM107 full-reduction policy branch."""
        natural = {T.SumOps: torch.float32, T.ArgMaxOps: torch.int64}
        output = natural.get(trait_type, dtype)
        arch = rt.select_full_itree_arch(
            (10, 7),
            dtype,
            (mib << 20) // dtype.itemsize,
            1,
            trait_type=trait_type,
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
            {"trait_type": T.MeanOps},
            {"trait_type": T.NanSumOps},
            {"dtype": torch.float16, "N": (256 << 20) // 2},
            # Only fp32 sums were measured, not bf16 sums into fp32.
            {"dtype": torch.bfloat16, "N": (256 << 20) // 2},
            {"M": 2},
            {"contiguous": False},
            {"alignment": 8},
            {"out_dtypes": (torch.bfloat16,)},
        ],
    )
    def test_sm107_full_guards(self, updates):
        """Keep calls outside the SM107 policy contract on the generic path."""
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_type=T.SumOps,
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
            {"dtype": torch.float16, "N": 128 << 20, "out_dtypes": (torch.float16,)},
            {"welford_outs": 0, "out_dtypes": ()},
            {"out_dtypes": (torch.bfloat16,)},
            {"out_dtypes": (torch.float32,) * 2},
        ],
    )
    def test_sm107_full_welford_guards(self, updates):
        """Keep Welford calls outside the SM107 Welford contract on the generic path."""
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_type=T.WelfordOps,
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


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "requires Hopper or later")
class TestFullInnerTreeDevice(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("key", ["sum", "var1"])
    @largeTensorTest("1GB", device="cuda")
    def test_sm107_profile_reaches_every_plan(self, device, key):
        """Thread the SM107 profile through both row entry points' split and combine
        builders, bitwise equal to the default profile."""
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
            # The ordered entry point takes the same profile and plans, compiling its own
            # kernels under a distinct key.
            ordered = torch.empty(1, device=device)
            self.assertTrue(rt.reduce_row_itree(trait, key + "_ordered", x, ordered))
        self.assertEqual(len(profiles), 2)
        self.assertIsNotNone(profiles[0])
        self.assertTrue(all(arch == profiles[0] for arch in archs))
        split, c, ordered_split, ordered_c = plans
        self.assertEqual((ordered_split.sig, ordered_c.sig), (split.sig, c.sig))
        self.assertEqual(ordered.view(torch.int32), got.view(torch.int32))
        staged = (split.shape, split.rows_per_block, split.stage_e)
        self.assertEqual(staged, ("split", 4, 32))
        folded = (c.combine_tile, c.combine_unroll, c.combine_count, c.combine_weights)
        # Welford partials come from uniform 8192-element batches, so they are weighted.
        want = (2048, 64, 8192, True) if welford else (2048, 128, 0, False)
        self.assertEqual(folded, want)
        # Each stage's cache key ends with its plan's sig.
        for tag, plan in (("rowitree1", split), ("rowitree2", c)):
            self.assertIn(plan.sig, {k[-1] for k in rt._CACHE if k[:2] == (tag, key)})
        # Every _ItreeArch knob is bit-neutral, so the default profile gives the same bits.
        default = rt._ITREE_ARCH["default"]
        ref_plan = rt.trait_itree_plan(trait, x.shape[1], 1, 4, arch=default)
        self.assertNotEqual(ref_plan.sig, split.sig)
        (want,) = rt._run_itree(
            trait, key + "_ref", x, [torch.float32], ref_plan, arch=default
        )
        self.assertEqual(got.view(torch.int32), want.view(torch.int32))


instantiate_parametrized_tests(TestFullInnerTreeConfig)
instantiate_device_type_tests(TestFullInnerTreeDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
