# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


@unittest.skipUnless(TEST_CUTEDSL, "requires CuTeDSL")
class TestFullInnerTreeConfig(TestCase):
    @parametrize(
        "dtype,mib,key,fields,nouts,unroll",
        [
            (torch.float32, 256, "sum", 1, 1, 0),
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


instantiate_parametrized_tests(TestFullInnerTreeConfig)


if __name__ == "__main__":
    run_tests()
