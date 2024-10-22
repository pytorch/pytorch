from __future__ import annotations

import functools
import random
import sys
import unittest
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))
    from tools.testing.test_run import ShardedTest, TestRun
    from tools.testing.test_selections import calculate_shards, THRESHOLD
except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


def gen_class_times(test_times: dict[str, float]) -> dict[str, dict[str, float]]:
    return {k: {"class1": v} for k, v in test_times.items()}


class TestCalculateShards(unittest.TestCase):
    tests: list[TestRun] = [
        TestRun("super_long_test"),
        TestRun("long_test1"),
        TestRun("long_test2"),
        TestRun("normal_test1"),
        TestRun("normal_test2"),
        TestRun("normal_test3"),
        TestRun("short_test1"),
        TestRun("short_test2"),
        TestRun("short_test3"),
        TestRun("short_test4"),
        TestRun("short_test5"),
    ]

    test_times: dict[str, float] = {
        "super_long_test": 55,
        "long_test1": 22,
        "long_test2": 18,
        "normal_test1": 9,
        "normal_test2": 7,
        "normal_test3": 5,
        "short_test1": 1,
        "short_test2": 0.6,
        "short_test3": 0.4,
        "short_test4": 0.3,
        "short_test5": 0.01,
    }

    test_class_times: dict[str, dict[str, float]] = {
        "super_long_test": {"class1": 55},
        "long_test1": {"class1": 1, "class2": 21},
        "long_test2": {"class1": 10, "class2": 8},
        "normal_test1": {"class1": 9},
        "normal_test2": {"class1": 7},
        "normal_test3": {"class1": 5},
        "short_test1": {"class1": 1},
        "short_test2": {"class1": 0.6},
        "short_test3": {"class1": 0.4},
        "short_test4": {"class1": 0.3},
        "short_test5": {"class1": 0.01},
    }

    def assert_shards_equal(
        self,
        expected_shards: list[tuple[float, list[ShardedTest]]],
        actual_shards: list[tuple[float, list[ShardedTest]]],
    ) -> None:
        for expected, actual in zip(expected_shards, actual_shards):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertListEqual(expected[1], actual[1])

    def test_no_times(self) -> None:
        # Check that round robin sharding is used when no times are provided
        expected_shards = [
            (
                0.0,
                [
                    ShardedTest(
                        test="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(2, self.tests, {}, {}, sort_by_time=False),
        )

    def test_some_times_with_not_sort_by_time(self) -> None:
        expected_shards = [
            (
                400.0,
                [
                    ShardedTest(test="test_1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="test_2", shard=1, num_shards=1, time=400),
                    ShardedTest(test="test_5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                300.0,
                [
                    ShardedTest(test="test_3", shard=1, num_shards=1, time=300),
                    ShardedTest(test="test_4", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [
                    TestRun("test_1"),
                    TestRun("test_2"),
                    TestRun("test_3"),
                    TestRun("test_4"),
                    TestRun("test_5"),
                ],
                {"test_2": 400, "test_3": 300},
                {},
                sort_by_time=False,
            ),
        )

    def test_serial_parallel_interleaving(self) -> None:
        expected_shards = [
            (
                300.0,
                [
                    ShardedTest(test="test_1", shard=1, num_shards=1, time=None),
                    ShardedTest(test="test_3", shard=1, num_shards=1, time=300),
                    ShardedTest(test="test_4", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                400.0,
                [
                    ShardedTest(test="test_2", shard=1, num_shards=1, time=400),
                    ShardedTest(test="test_5", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [
                    TestRun("test_1"),
                    TestRun("test_2"),
                    TestRun("test_3"),
                    TestRun("test_4"),
                    TestRun("test_5"),
                ],
                {"test_2": 400, "test_3": 300},
                {},
                must_serial=lambda x: x in ["test_1", "test_3"],
                sort_by_time=False,
            ),
        )

    def test_calculate_2_shards_with_complete_test_times(self) -> None:
        expected_shards = [
            (
                60.0,
                [
                    ShardedTest(test="super_long_test", shard=1, num_shards=1, time=55),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=5),
                ],
            ),
            (
                58.31,
                [
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=18),
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(2, self.tests, self.test_times, self.test_class_times),
        )

    def test_calculate_1_shard_with_complete_test_times(self) -> None:
        tests = self.tests.copy()
        class_test1 = TestRun("long_test1", excluded=["class2"])
        class_test2 = TestRun("long_test1", included=["class2"])
        tests.append(class_test1)
        tests.append(class_test2)

        expected_shards = [
            (
                140.31,
                [
                    ShardedTest(test="super_long_test", shard=1, num_shards=1, time=55),
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(class_test2, shard=1, num_shards=1, time=21),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=18),
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=5),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(class_test1, shard=1, num_shards=1, time=1),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            )
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(1, tests, self.test_times, self.test_class_times),
        )

    def test_calculate_5_shards_with_complete_test_times(self) -> None:
        expected_shards = [
            (
                55.0,
                [ShardedTest(test="super_long_test", shard=1, num_shards=1, time=55)],
            ),
            (22.0, [ShardedTest(test="long_test1", shard=1, num_shards=1, time=22)]),
            (18.0, [ShardedTest(test="long_test2", shard=1, num_shards=1, time=18)]),
            (
                11.31,
                [
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            ),
            (
                12.0,
                [
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=5),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(5, self.tests, self.test_times, self.test_class_times),
        )

    def test_calculate_2_shards_with_incomplete_test_times(self) -> None:
        incomplete_test_times = {
            k: v for k, v in self.test_times.items() if "test1" in k
        }
        expected_shards = [
            (
                22.0,
                [
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(
                        test="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                10.0,
                [
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                self.tests,
                incomplete_test_times,
                gen_class_times(incomplete_test_times),
            ),
        )

    def test_calculate_5_shards_with_incomplete_test_times(self) -> None:
        incomplete_test_times = {
            k: v for k, v in self.test_times.items() if "test1" in k
        }
        expected_shards = [
            (
                22.0,
                [
                    ShardedTest(test="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(
                        test="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    ShardedTest(test="short_test3", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                9.0,
                [
                    ShardedTest(test="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(test="long_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                1.0,
                [
                    ShardedTest(test="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(test="normal_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(test="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    ShardedTest(test="normal_test3", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    ShardedTest(test="short_test2", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                5,
                self.tests,
                incomplete_test_times,
                gen_class_times(incomplete_test_times),
            ),
        )

    def test_split_shards(self) -> None:
        test_times: dict[str, float] = {"test1": THRESHOLD, "test2": THRESHOLD}
        expected_shards = [
            (600.0, [ShardedTest(test="test1", shard=1, num_shards=1, time=THRESHOLD)]),
            (600.0, [ShardedTest(test="test2", shard=1, num_shards=1, time=THRESHOLD)]),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [TestRun(t) for t in test_times.keys()],
                test_times,
                gen_class_times(test_times),
            ),
        )

        test_times = {"test1": THRESHOLD * 4, "test2": THRESHOLD * 2.5}
        expected_shards = [
            (
                2200.0,
                [
                    ShardedTest(test="test1", shard=1, num_shards=4, time=600.0),
                    ShardedTest(test="test1", shard=3, num_shards=4, time=600.0),
                    ShardedTest(test="test2", shard=1, num_shards=3, time=500.0),
                    ShardedTest(test="test2", shard=3, num_shards=3, time=500.0),
                ],
            ),
            (
                1700.0,
                [
                    ShardedTest(test="test1", shard=2, num_shards=4, time=600.0),
                    ShardedTest(test="test1", shard=4, num_shards=4, time=600.0),
                    ShardedTest(test="test2", shard=2, num_shards=3, time=500.0),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [TestRun(t) for t in test_times.keys()],
                test_times,
                gen_class_times(test_times),
            ),
        )

        test_times = {"test1": THRESHOLD / 2, "test2": THRESHOLD}
        expected_shards = [
            (600.0, [ShardedTest(test="test2", shard=1, num_shards=1, time=THRESHOLD)]),
            (
                300.0,
                [ShardedTest(test="test1", shard=1, num_shards=1, time=THRESHOLD / 2)],
            ),
        ]
        self.assert_shards_equal(
            expected_shards,
            calculate_shards(
                2,
                [TestRun(t) for t in test_times.keys()],
                test_times,
                gen_class_times(test_times),
            ),
        )

    def test_zero_tests(self) -> None:
        self.assertListEqual([(0.0, []), (0.0, [])], calculate_shards(2, [], {}, None))

    def test_split_shards_random(self) -> None:
        random.seed(120)
        for _ in range(100):
            num_shards = random.randint(1, 10)
            num_tests = random.randint(1, 100)
            test_names = [str(i) for i in range(num_tests)]
            tests = [TestRun(x) for x in test_names]
            serial = [x for x in test_names if random.randint(0, 1) == 0]
            has_times = [x for x in test_names if random.randint(0, 1) == 0]
            random_times: dict[str, float] = {
                i: random.randint(0, THRESHOLD * 10) for i in has_times
            }
            sort_by_time = random.randint(0, 1) == 0

            shards = calculate_shards(
                num_shards,
                tests,
                random_times,
                None,
                must_serial=lambda x: x in serial,
                sort_by_time=sort_by_time,
            )

            times = [x[0] for x in shards]
            max_diff = max(times) - min(times)
            self.assertTrue(max_diff <= THRESHOLD + (num_tests - len(has_times)) * 60)

            all_sharded_tests: dict[str, list[ShardedTest]] = defaultdict(list)
            for _, sharded_tests in shards:
                for sharded_test in sharded_tests:
                    all_sharded_tests[sharded_test.name].append(sharded_test)

            # Check that all test files are represented in the shards
            self.assertListEqual(sorted(test_names), sorted(all_sharded_tests.keys()))
            # Check that for each test file, the pytest shards' times adds up to
            # original and all shards are present
            for test, sharded_tests in all_sharded_tests.items():
                if random_times.get(test) is None:
                    self.assertTrue(len(sharded_tests) == 1)
                    self.assertTrue(sharded_tests[0].time is None)
                else:
                    # x.time is not None because of the above check
                    self.assertAlmostEqual(
                        random_times[test],
                        sum(x.time for x in sharded_tests),  # type: ignore[misc]
                    )
                self.assertListEqual(
                    list(range(sharded_tests[0].num_shards)),
                    sorted(x.shard - 1 for x in sharded_tests),
                )
            # Check that sort_by_time is respected
            if sort_by_time:

                def comparator(a: ShardedTest, b: ShardedTest) -> int:
                    # serial comes first
                    if a.name in serial and b.name not in serial:
                        return -1
                    if a.name not in serial and b.name in serial:
                        return 1
                    # known test times come first
                    if a.time is not None and b.time is None:
                        return -1
                    if a.time is None and b.time is not None:
                        return 1
                    if a.time == b.time:
                        return 0
                    # not None due to the above checks
                    return -1 if a.time > b.time else 1  # type: ignore[operator]

            else:

                def comparator(a: ShardedTest, b: ShardedTest) -> int:
                    # serial comes first
                    if a.name in serial and b.name not in serial:
                        return -1
                    if a.name not in serial and b.name in serial:
                        return 1
                    return test_names.index(a.name) - test_names.index(b.name)

            for _, sharded_tests in shards:
                self.assertListEqual(
                    sorted(sharded_tests, key=functools.cmp_to_key(comparator)),
                    sharded_tests,
                )

    def test_calculate_2_shards_against_optimal_shards(self) -> None:
        random.seed(120)
        for _ in range(100):
            random_times = {k.test_file: random.random() * 10 for k in self.tests}
            # all test times except first two
            rest_of_tests = [
                i
                for k, i in random_times.items()
                if k != "super_long_test" and k != "long_test1"
            ]
            sum_of_rest = sum(rest_of_tests)
            random_times["super_long_test"] = max(sum_of_rest / 2, *rest_of_tests)
            random_times["long_test1"] = sum_of_rest - random_times["super_long_test"]
            # An optimal sharding would look like the below, but we don't need to compute this for the test:
            # optimal_shards = [
            #     (sum_of_rest, ['super_long_test', 'long_test1']),
            #     (sum_of_rest, [i for i in self.tests if i != 'super_long_test' and i != 'long_test1']),
            # ]
            calculated_shards = calculate_shards(
                2, self.tests, random_times, gen_class_times(random_times)
            )
            max_shard_time = max(calculated_shards[0][0], calculated_shards[1][0])
            if sum_of_rest != 0:
                # The calculated shard should not have a ratio worse than 7/6 for num_shards = 2
                self.assertGreaterEqual(7.0 / 6.0, max_shard_time / sum_of_rest)
                sorted_tests = sorted([t.test_file for t in self.tests])
                sorted_shard_tests = sorted(
                    calculated_shards[0][1] + calculated_shards[1][1]
                )
                # All the tests should be represented by some shard
                self.assertEqual(sorted_tests, [x.name for x in sorted_shard_tests])


if __name__ == "__main__":
    unittest.main()
