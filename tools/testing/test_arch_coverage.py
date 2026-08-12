# Owner(s): ["module: ci"]

# Plain unittest, not torch's TestCase: arch_coverage must work in the lint image,
# which has no torch build, so its test must not require one either.
import ast
import unittest

import arch_coverage


def classify(decorator_src):
    """-> the required predicate name, or None if the decorator is not a requirement."""
    node = ast.parse(f"{decorator_src}\ndef test_x(): pass").body[0].decorator_list[0]
    pred = arch_coverage.requires(node)
    return None if pred is None else arch_coverage.predicate_name(pred)


class TestRequires(unittest.TestCase):
    def test_negated_skip_if_is_a_requirement(self):
        self.assertEqual(
            classify("@skipIf(not SM90OrLater, 'needs sm90')"), "SM90OrLater"
        )
        self.assertEqual(
            classify("@unittest.skipIf(not has_triton_tma_device(), 'tma')"),
            "has_triton_tma_device",
        )
        self.assertEqual(
            classify("@skipCUDAIf(not SM80OrLater, 'bf16')"), "SM80OrLater"
        )

    def test_skip_unless_is_a_requirement(self):
        self.assertEqual(
            classify("@unittest.skipUnless(SM90OrLater, 'sm90')"), "SM90OrLater"
        )

    def test_bare_skip_if_excludes_rather_than_requires(self):
        # The case an earlier text-matching version got backwards: this skips *on*
        # sm90, so the test does not require an H100.
        self.assertIsNone(classify("@unittest.skipIf(SM90OrLater, 'fails on sm90')"))

    def test_compound_conditions_are_declined(self):
        self.assertIsNone(classify("@skipIf(not SM90OrLater or SM120OrLater, 'x')"))
        self.assertIsNone(classify("@skipIf(IS_WINDOWS and SM89OrLater, 'x')"))
        self.assertIsNone(classify("@skipUnless(SM90OrLater and TEST_CUDA, 'x')"))

    def test_non_skip_decorators_are_ignored(self):
        self.assertIsNone(classify("@parametrize('x', [SM90OrLater])"))


class TestRanges(unittest.TestCase):
    def test_tables_load_without_torch(self):
        ranges, exempt = arch_coverage.load_tables()
        self.assertIn("SM90OrLater", ranges)
        self.assertEqual(ranges["SM90OrLater"], ((9, 0), None))
        # Bounded, not a floor: the predicate is SM90OrLater and not SM100OrLater.
        self.assertEqual(
            ranges["PLATFORM_SUPPORTS_FP8_GROUPED_GEMM"], ((9, 0), (10, 0))
        )
        self.assertIn("PLATFORM_SUPPORTS_SYMM_MEM", exempt)

    def test_targets_follow_the_declared_ranges(self):
        ranges, _ = arch_coverage.load_tables()
        gates = [
            arch_coverage.Gate("f", "t_floor", 1, "SM90OrLater"),
            arch_coverage.Gate(
                "f", "t_blackwell", 2, "has_datacenter_blackwell_tma_device"
            ),
            arch_coverage.Gate("f", "t_exact90", 3, "IS_SM90"),
            arch_coverage.Gate("f", "t_ampere", 4, "SM80OrLater"),
        ]
        plan = arch_coverage.needed(gates, ranges)
        h100 = plan["test_python_smoke"]["f"]
        b200 = plan["test_python_smoke_b200"]["f"]
        # sm90+ runs on both; blackwell only on b200; exactly-sm90 only on h100.
        self.assertEqual(h100, {"t_floor", "t_exact90"})
        self.assertEqual(b200, {"t_floor", "t_blackwell"})
        # sm80 is reached by auto-discovery, so it needs no curated entry.
        self.assertNotIn("t_ampere", h100 | b200)


if __name__ == "__main__":
    unittest.main()
