# Owner(s): ["module: inductor"]

from torch._inductor.runtime.tiling import (
    block_to_prefix,
    get_axis_prefix,
    get_max_block,
    get_max_block_prefix,
    get_pointwise_prefixes,
    get_reduction_prefixes,
    get_yz_grid_extents,
    get_z_prefixes,
    is_reduction_prefix,
    is_z_prefix,
    pointwise_prefixes,
    prefix_to_family,
    reduction_prefixes,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


_UNORDERED_PREFIXES = ("r10_", "z10_", "r2_", "z2_", "y", "r0_", "x", "z0_", "invalid")


@instantiate_parametrized_tests
class TestTiling(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "prefix,expected",
        [
            ("x", "x"),
            ("y", "y"),
            ("z0_", "z"),
            ("z10_", "z"),
            ("r0_", "r"),
            ("r10_", "r"),
        ],
    )
    def test_prefix_to_family(self, prefix, expected):
        self.assertEqual(prefix_to_family(prefix), expected)

    @parametrize(
        "rank,expected",
        [
            (0, ()),
            (1, ("x",)),
            (2, ("y", "x")),
            (3, ("z0_", "y", "x")),
            (5, ("z2_", "z1_", "z0_", "y", "x")),
        ],
    )
    def test_prefix_ranks(self, rank, expected):
        self.assertEqual(pointwise_prefixes(rank), expected)
        self.assertEqual(reduction_prefixes(rank), tuple(f"r{i}_" for i in range(rank)))

    @parametrize(
        "kind,helper",
        [("pointwise", pointwise_prefixes), ("reduction", reduction_prefixes)],
    )
    def test_negative_rank(self, kind, helper):
        with self.assertRaisesRegex(ValueError, f"Invalid {kind} rank: -1"):
            helper(-1)

    @parametrize(
        "prefixes",
        [
            subtest(_UNORDERED_PREFIXES, name="tuple"),
            subtest(list(_UNORDERED_PREFIXES), name="list"),
            subtest(iter(_UNORDERED_PREFIXES), name="iterator"),
        ],
    )
    def test_reduction_numeric_axis_order(self, prefixes):
        self.assertEqual(get_reduction_prefixes(prefixes), ("r0_", "r2_", "r10_"))

    @parametrize(
        "prefixes",
        [
            subtest(_UNORDERED_PREFIXES, name="tuple"),
            subtest(list(_UNORDERED_PREFIXES), name="list"),
            subtest(iter(_UNORDERED_PREFIXES), name="iterator"),
        ],
    )
    def test_z_numeric_axis_order(self, prefixes):
        self.assertEqual(get_z_prefixes(prefixes), ("z0_", "z2_", "z10_"))

    @parametrize(
        "prefixes",
        [
            subtest(_UNORDERED_PREFIXES, name="tuple"),
            subtest(list(_UNORDERED_PREFIXES), name="list"),
            subtest(iter(_UNORDERED_PREFIXES), name="iterator"),
        ],
    )
    def test_pointwise_numeric_axis_order(self, prefixes):
        self.assertEqual(
            get_pointwise_prefixes(prefixes), ("z10_", "z2_", "z0_", "y", "x")
        )

    @parametrize(
        "prefixes,expected",
        [
            ((), ()),
            (("x",), ("x",)),
            (("y",), ("y",)),
            (("x", "y", "x"), ("y", "x")),
            (("z0_", "z0_", "x", "x"), ("z0_", "x")),
            (("xoffset", "yindex", "zfoo_", "r0_"), ()),
        ],
    )
    def test_pointwise_filter(self, prefixes, expected):
        self.assertEqual(get_pointwise_prefixes(iter(prefixes)), expected)

    @parametrize(
        "helper,prefixes,expected",
        [
            (get_reduction_prefixes, ("r0_", "r0_"), ("r0_",)),
            (
                get_reduction_prefixes,
                ("r10_", "r2_", "r10_", "r0_", "r2_"),
                ("r0_", "r2_", "r10_"),
            ),
            (get_z_prefixes, ("z0_", "z0_", "x", "x"), ("z0_",)),
        ],
    )
    def test_numbered_prefix_duplicates(self, helper, prefixes, expected):
        self.assertEqual(helper(iter(prefixes)), expected)

    @parametrize(
        "prefix,expected,key",
        [
            ("x", 16, "X"),
            ("y", 32, "Y"),
            ("z10_", 64, "Z0_"),
            ("r0_", 128, "R0_"),
            ("r1_", 256, "R1_"),
            ("r2_", 512, "R2_"),
            ("r10_", 256, "R1_"),
        ],
    )
    def test_block_limits_and_names(self, prefix, expected, key):
        limits = {"X": 16, "Y": 32, "Z0_": 64, "R0_": 128, "R1_": 256, "R2_": 512}
        self.assertEqual(get_max_block_prefix(prefix, limits), key)
        self.assertEqual(get_max_block(prefix, limits), expected)
        self.assertEqual(block_to_prefix(f"{prefix.upper()}BLOCK"), prefix)
        self.assertEqual(get_axis_prefix(f"{prefix.upper()}BLOCK"), prefix)

    @parametrize(
        "prefix,reduction,z",
        [
            ("r0_", True, False),
            ("r2_", True, False),
            ("r10_", True, False),
            ("z0_", False, True),
            ("z10_", False, True),
            ("", False, False),
            ("r", False, False),
            ("r0", False, False),
            ("rfoo_", False, False),
            ("z", False, False),
            ("z0", False, False),
            ("zfoo_", False, False),
            ("x", False, False),
        ],
    )
    def test_prefix_predicates(self, prefix, reduction, z):
        self.assertEqual(is_reduction_prefix(prefix), reduction)
        self.assertEqual(is_z_prefix(prefix), z)

    def test_named_constant_has_no_axis(self):
        self.assertIsNone(get_axis_prefix("nested_R0_REDUCED_BLOCK"))

    def test_invalid_block_limit(self):
        with self.assertRaisesRegex(KeyError, "INVALID"):
            get_max_block("invalid", {})

    @parametrize(
        "size_hints,blocks,expected",
        [
            ({"x": 8, "y": 65536}, {"x": 1, "y": 1}, {}),
            (
                {"x": 8, "y": 7, "z0_": 9, "z1_": 5},
                {"x": 1, "y": 2, "z0_": 4, "z1_": 2},
                {("y",): 4, ("z0_", "z1_"): 9},
            ),
            (
                {"x": 8, "y": 7, "z0_": 9, "z1_": 5},
                {"x": 1, "y": 2, "z0_": 4},
                {("y",): 4},
            ),
            (
                {"x": 8, "z0_": 9},
                {"x": 1, "z0_": 4},
                {("z0_",): 3},
            ),
        ],
    )
    def test_yz_grid_extents(self, size_hints, blocks, expected):
        self.assertEqual(get_yz_grid_extents(size_hints, blocks), expected)


if __name__ == "__main__":
    run_tests()
