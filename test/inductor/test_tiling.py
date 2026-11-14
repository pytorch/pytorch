# Owner(s): ["module: inductor"]

from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestTiling(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_prefix_ranks(self):
        from torch._inductor.runtime.tiling import (
            pointwise_prefixes,
            reduction_prefixes,
        )

        for rank, expected in enumerate([(), ("x",), ("y", "x"), ("z0_", "y", "x")]):
            self.assertEqual(pointwise_prefixes(rank), expected)
            self.assertEqual(
                reduction_prefixes(rank), tuple(f"r{i}_" for i in range(rank))
            )
        self.assertEqual(pointwise_prefixes(5), ("z2_", "z1_", "z0_", "y", "x"))
        for helper in (pointwise_prefixes, reduction_prefixes):
            with self.assertRaises(ValueError):
                helper(-1)

    def test_numeric_axis_order(self):
        from torch._inductor.runtime.tiling import (
            get_pointwise_prefixes,
            get_reduction_prefixes,
            get_z_prefixes,
        )

        prefixes = ("r10_", "z10_", "r2_", "z2_", "y", "r0_", "x", "z0_", "invalid")
        self.assertEqual(get_reduction_prefixes(prefixes), ("r0_", "r2_", "r10_"))
        self.assertEqual(get_z_prefixes(prefixes), ("z0_", "z2_", "z10_"))
        self.assertEqual(
            get_pointwise_prefixes(iter(prefixes)), ("z10_", "z2_", "z0_", "y", "x")
        )
        for prefixes, expected in (
            ((), ()),
            (("x",), ("x",)),
            (("y",), ("y",)),
            (("x", "y", "x"), ("y", "x")),
            (("xoffset", "yindex", "zfoo_", "r0_"), ()),
        ):
            self.assertEqual(get_pointwise_prefixes(prefixes), expected)

    def test_block_limits_and_names(self):
        from torch._inductor.runtime.tiling import (
            block_to_prefix,
            get_axis_prefix,
            get_max_block,
            is_reduction_prefix,
            is_z_prefix,
        )

        limits = {"X": 16, "Y": 32, "Z0_": 64, "R0_": 128, "R1_": 256, "R2_": 512}
        for prefix, expected in (
            ("x", 16),
            ("y", 32),
            ("z10_", 64),
            ("r0_", 128),
            ("r1_", 256),
            ("r2_", 512),
            ("r10_", 256),
        ):
            self.assertEqual(get_max_block(prefix, limits), expected)
            self.assertEqual(block_to_prefix(f"{prefix.upper()}BLOCK"), prefix)
            self.assertEqual(get_axis_prefix(f"{prefix.upper()}BLOCK"), prefix)
        for prefix in ("r0_", "r2_", "r10_"):
            self.assertTrue(is_reduction_prefix(prefix))
            self.assertFalse(is_z_prefix(prefix))
        for prefix in ("z0_", "z10_"):
            self.assertTrue(is_z_prefix(prefix))
            self.assertFalse(is_reduction_prefix(prefix))
        for prefix in ("", "r", "r0", "rfoo_", "z", "z0", "zfoo_", "x"):
            self.assertFalse(is_reduction_prefix(prefix))
            self.assertFalse(is_z_prefix(prefix))
        self.assertIsNone(get_axis_prefix("nested_R0_REDUCED_BLOCK"))
        with self.assertRaises(KeyError):
            get_max_block("invalid", limits)


if __name__ == "__main__":
    run_tests()
