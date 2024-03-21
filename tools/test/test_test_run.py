import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))
    from tools.testing.test_run import ShardedTest, TestRun
except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


class TestTestRun(unittest.TestCase):
    def test_union_with_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    def test_union_with_inclusions(self) -> None:
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo::baz")

        expected = TestRun("foo", included=["bar", "baz"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_non_overlapping_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar"])
        run2 = TestRun("foo", excluded=["baz"])

        expected = TestRun("foo")

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_overlapping_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "car"])
        run2 = TestRun("foo", excluded=["bar", "caz"])

        expected = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_mixed_inclusion_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["baz", "car"])
        run2 = TestRun("foo", included=["baz"])

        expected = TestRun("foo", excluded=["car"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_mixed_files_fails(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("bar")

        with self.assertRaises(AssertionError):
            run1 | run2

    def test_union_with_empty_file_yields_orig_file(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    def test_subtracting_full_run_fails(self) -> None:
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo")

        self.assertEqual(run1 - run2, TestRun.empty())

    def test_subtracting_empty_file_yields_orig_file(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        self.assertEqual(run1 - run2, run1)
        self.assertEqual(run2 - run1, TestRun.empty())

    def test_empty_is_falsey(self) -> None:
        self.assertFalse(TestRun.empty())

    def test_subtracting_inclusion_from_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        expected = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 - run2, expected)

    def test_subtracting_inclusion_from_overlapping_inclusion(self) -> None:
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo::baz")

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))

    def test_subtracting_inclusion_from_nonoverlapping_inclusion(self) -> None:
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo", included=["car"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar", "baz"]))

    def test_subtracting_exclusion_from_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))

    def test_subtracting_exclusion_from_superset_exclusion(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["baz"])

        self.assertEqual(run1 - run2, TestRun.empty())
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))

    def test_subtracting_exclusion_from_nonoverlapping_exclusion(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["car"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["car"]))
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar", "baz"]))

    def test_subtracting_inclusion_from_exclusion_without_overlaps(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", included=["bar"])

        self.assertEqual(run1 - run2, run1)
        self.assertEqual(run2 - run1, run2)

    def test_subtracting_inclusion_from_exclusion_with_overlaps(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", included=["bar", "car"])

        self.assertEqual(run1 - run2, TestRun("foo", excluded=["bar", "baz", "car"]))
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))

    def test_and(self) -> None:
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo", included=["bar", "car"])

        self.assertEqual(run1 & run2, TestRun("foo", included=["bar"]))

    def test_and_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["bar", "car"])

        self.assertEqual(run1 & run2, TestRun("foo", excluded=["bar", "baz", "car"]))


class TestShardedTest(unittest.TestCase):
    def test_get_pytest_args(self) -> None:
        test = TestRun("foo", included=["bar", "baz"])
        sharded_test = ShardedTest(test, 1, 1)

        expected_args = ["-k", "bar or baz"]

        self.assertListEqual(sharded_test.get_pytest_args(), expected_args)


if __name__ == "__main__":
    unittest.main()
