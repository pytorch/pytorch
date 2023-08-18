import io
import json
import pathlib
import random
import sys
import unittest
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))
    from tools.testing.test_selections import (
        _get_previously_failing_tests,
        calculate_shards,
        get_reordered_tests,
        log_time_savings,
        ShardedTest,
        THRESHOLD,
    )
except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    exit(1)


class TestCalculateShards(unittest.TestCase):
    tests: List[str] = [
        "super_long_test",
        "long_test1",
        "long_test2",
        "normal_test1",
        "normal_test2",
        "normal_test3",
        "short_test1",
        "short_test2",
        "short_test3",
        "short_test4",
        "short_test5",
    ]

    test_times: Dict[str, float] = {
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

    def assert_shards_equal(
        self,
        expected_shards: List[Tuple[float, List[ShardedTest]]],
        actual_shards: List[Tuple[float, List[ShardedTest]]],
    ) -> None:
        for expected, actual in zip(expected_shards, actual_shards):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertListEqual(expected[1], actual[1])

    def test_calculate_2_shards_with_complete_test_times(self) -> None:
        expected_shards = [
            (
                60.0,
                [
                    ShardedTest(name="super_long_test", shard=1, num_shards=1, time=55),
                    ShardedTest(name="normal_test3", shard=1, num_shards=1, time=5),
                ],
            ),
            (
                58.31,
                [
                    ShardedTest(name="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(name="long_test2", shard=1, num_shards=1, time=18),
                    ShardedTest(name="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(name="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(name="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(name="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(name="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(name="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(name="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(2, self.tests, self.test_times)
        )

    def test_calculate_1_shard_with_complete_test_times(self) -> None:
        expected_shards = [
            (
                118.31,
                [
                    ShardedTest(name="super_long_test", shard=1, num_shards=1, time=55),
                    ShardedTest(name="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(name="long_test2", shard=1, num_shards=1, time=18),
                    ShardedTest(name="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(name="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(name="normal_test3", shard=1, num_shards=1, time=5),
                    ShardedTest(name="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(name="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(name="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(name="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(name="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            )
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(1, self.tests, self.test_times)
        )

    def test_calculate_5_shards_with_complete_test_times(self) -> None:
        expected_shards = [
            (
                55.0,
                [ShardedTest(name="super_long_test", shard=1, num_shards=1, time=55)],
            ),
            (22.0, [ShardedTest(name="long_test1", shard=1, num_shards=1, time=22)]),
            (18.0, [ShardedTest(name="long_test2", shard=1, num_shards=1, time=18)]),
            (
                11.31,
                [
                    ShardedTest(name="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(name="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(name="short_test2", shard=1, num_shards=1, time=0.6),
                    ShardedTest(name="short_test3", shard=1, num_shards=1, time=0.4),
                    ShardedTest(name="short_test4", shard=1, num_shards=1, time=0.3),
                    ShardedTest(name="short_test5", shard=1, num_shards=1, time=0.01),
                ],
            ),
            (
                12.0,
                [
                    ShardedTest(name="normal_test2", shard=1, num_shards=1, time=7),
                    ShardedTest(name="normal_test3", shard=1, num_shards=1, time=5),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(5, self.tests, self.test_times)
        )

    def test_calculate_2_shards_with_incomplete_test_times(self) -> None:
        incomplete_test_times = {
            k: v for k, v in self.test_times.items() if "test1" in k
        }
        expected_shards = [
            (
                22.0,
                [
                    ShardedTest(name="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(name="long_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(name="normal_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(name="short_test3", shard=1, num_shards=1, time=None),
                    ShardedTest(name="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                10.0,
                [
                    ShardedTest(name="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(name="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(
                        name="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    ShardedTest(name="normal_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(name="short_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(name="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(2, self.tests, incomplete_test_times)
        )

    def test_calculate_5_shards_with_incomplete_test_times(self) -> None:
        incomplete_test_times = {
            k: v for k, v in self.test_times.items() if "test1" in k
        }
        expected_shards = [
            (
                22.0,
                [
                    ShardedTest(name="long_test1", shard=1, num_shards=1, time=22),
                    ShardedTest(name="normal_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(name="short_test5", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                9.0,
                [
                    ShardedTest(name="normal_test1", shard=1, num_shards=1, time=9),
                    ShardedTest(name="normal_test3", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                1.0,
                [
                    ShardedTest(name="short_test1", shard=1, num_shards=1, time=1),
                    ShardedTest(name="short_test2", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    ShardedTest(
                        name="super_long_test", shard=1, num_shards=1, time=None
                    ),
                    ShardedTest(name="short_test3", shard=1, num_shards=1, time=None),
                ],
            ),
            (
                0.0,
                [
                    ShardedTest(name="long_test2", shard=1, num_shards=1, time=None),
                    ShardedTest(name="short_test4", shard=1, num_shards=1, time=None),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(5, self.tests, incomplete_test_times)
        )

    def test_split_shards(self) -> None:
        test_times: Dict[str, float] = {"test1": THRESHOLD, "test2": THRESHOLD}
        expected_shards = [
            (600.0, [ShardedTest(name="test1", shard=1, num_shards=1, time=THRESHOLD)]),
            (600.0, [ShardedTest(name="test2", shard=1, num_shards=1, time=THRESHOLD)]),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(2, list(test_times.keys()), test_times)
        )

        test_times = {"test1": THRESHOLD * 4, "test2": THRESHOLD * 2.5}
        expected_shards = [
            (
                2200.0,
                [
                    ShardedTest(name="test1", shard=1, num_shards=4, time=600.0),
                    ShardedTest(name="test1", shard=3, num_shards=4, time=600.0),
                    ShardedTest(name="test2", shard=1, num_shards=3, time=500.0),
                    ShardedTest(name="test2", shard=3, num_shards=3, time=500.0),
                ],
            ),
            (
                1700.0,
                [
                    ShardedTest(name="test1", shard=2, num_shards=4, time=600.0),
                    ShardedTest(name="test1", shard=4, num_shards=4, time=600.0),
                    ShardedTest(name="test2", shard=2, num_shards=3, time=500.0),
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(2, list(test_times.keys()), test_times)
        )

        test_times = {"test1": THRESHOLD / 2, "test2": THRESHOLD}
        expected_shards = [
            (600.0, [ShardedTest(name="test2", shard=1, num_shards=1, time=THRESHOLD)]),
            (
                300.0,
                [ShardedTest(name="test1", shard=1, num_shards=1, time=THRESHOLD / 2)],
            ),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(2, list(test_times.keys()), test_times)
        )

    def test_split_shards_random(self) -> None:
        random.seed(120)
        for _ in range(100):
            num_shards = random.randint(1, 10)
            num_tests = random.randint(1, 100)
            random_times: Dict[str, float] = {
                str(i): random.randint(0, THRESHOLD * 10) for i in range(num_tests)
            }

            shards = calculate_shards(
                num_shards, list(random_times.keys()), random_times
            )

            times = [x[0] for x in shards]
            max_diff = max(times) - min(times)
            self.assertTrue(max_diff <= THRESHOLD)

            all_sharded_tests = defaultdict(list)
            for time, sharded_tests in shards:
                self.assertEqual(time, sum(x.time for x in sharded_tests))
                for sharded_test in sharded_tests:
                    all_sharded_tests[sharded_test.name].append(sharded_test)

            self.assertListEqual(
                sorted(random_times.keys()), sorted(all_sharded_tests.keys())
            )
            for test, sharded_tests in all_sharded_tests.items():
                self.assertAlmostEqual(
                    random_times[test], sum(x.time or 0 for x in sharded_tests)
                )
                self.assertListEqual(
                    list(range(sharded_tests[0].num_shards)),
                    sorted(x.shard - 1 for x in sharded_tests),
                )

    def test_calculate_2_shards_against_optimal_shards(self) -> None:
        random.seed(120)
        for _ in range(100):
            random_times = {k: random.random() * 10 for k in self.tests}
            # all test times except first two
            rest_of_tests = [
                i
                for k, i in random_times.items()
                if k != "super_long_test" and k != "long_test1"
            ]
            sum_of_rest = sum(rest_of_tests)
            random_times["super_long_test"] = max(sum_of_rest / 2, max(rest_of_tests))
            random_times["long_test1"] = sum_of_rest - random_times["super_long_test"]
            # An optimal sharding would look like the below, but we don't need to compute this for the test:
            # optimal_shards = [
            #     (sum_of_rest, ['super_long_test', 'long_test1']),
            #     (sum_of_rest, [i for i in self.tests if i != 'super_long_test' and i != 'long_test1']),
            # ]
            calculated_shards = calculate_shards(2, self.tests, random_times)
            max_shard_time = max(calculated_shards[0][0], calculated_shards[1][0])
            if sum_of_rest != 0:
                # The calculated shard should not have a ratio worse than 7/6 for num_shards = 2
                self.assertGreaterEqual(7.0 / 6.0, max_shard_time / sum_of_rest)
                sorted_tests = sorted(self.tests)
                sorted_shard_tests = sorted(
                    calculated_shards[0][1] + calculated_shards[1][1]
                )
                # All the tests should be represented by some shard
                self.assertEqual(sorted_tests, [x.name for x in sorted_shard_tests])


def mocked_file(contents: Dict[Any, Any]) -> io.IOBase:
    file_object = io.StringIO()
    json.dump(contents, file_object)
    file_object.seek(0)
    return file_object


def never_serial(test_name: str) -> bool:
    return False


class TestParsePrevTests(unittest.TestCase):
    @mock.patch("pathlib.Path.exists", return_value=False)
    def test_cache_does_not_exist(self, mock_exists: Any) -> None:
        expected_failing_test_files: Set[str] = set()

        found_tests = _get_previously_failing_tests()

        self.assertSetEqual(expected_failing_test_files, found_tests)

    @mock.patch("pathlib.Path.exists", return_value=True)
    @mock.patch("builtins.open", return_value=mocked_file({"": True}))
    def test_empty_cache(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files: Set[str] = set()

        found_tests = _get_previously_failing_tests()

        self.assertSetEqual(expected_failing_test_files, found_tests)
        mock_open.assert_called()

    lastfailed_with_multiple_tests_per_file = {
        "test/test_car.py::TestCar::test_num[17]": True,
        "test/test_car.py::TestBar::test_num[25]": True,
        "test/test_far.py::TestFar::test_fun_copy[17]": True,
        "test/test_bar.py::TestBar::test_fun_copy[25]": True,
    }

    @mock.patch("pathlib.Path.exists", return_value=True)
    @mock.patch(
        "builtins.open",
        return_value=mocked_file(lastfailed_with_multiple_tests_per_file),
    )
    def test_dedupes_failing_test_files(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files = {"test_car", "test_bar", "test_far"}
        found_tests = _get_previously_failing_tests()

        self.assertSetEqual(expected_failing_test_files, found_tests)

    @mock.patch(
        "tools.testing.test_selections._get_previously_failing_tests",
        return_value={"test4"},
    )
    @mock.patch(
        "tools.testing.test_selections._get_modified_tests",
        return_value={"test2", "test4"},
    )
    @mock.patch(
        "tools.testing.test_selections._get_file_rating_tests", return_value=["test1"]
    )
    def test_get_reordered_tests(
        self,
        mock_get_prev_failing_tests: Any,
        mock_get_modified_tests: Any,
        mock_get_file_rating_tests: Any,
    ) -> None:
        tests = ["test1", "test2", "test3", "test4", "test5"]

        expected_prioritized_tests = ["test4", "test2", "test1"]
        expected_remaining_tests = {"test3", "test5"}

        prioritized_tests, remaining_tests = get_reordered_tests(tests)

        self.assertListEqual(expected_prioritized_tests, prioritized_tests)
        self.assertSetEqual(expected_remaining_tests, set(remaining_tests))

    def test_compute_prioritization_time_savings_with_multiple_threads(self) -> None:
        tests = [
            ShardedTest(name="test1", shard=1, num_shards=2, time=7.0),
            ShardedTest(name="test2", shard=1, num_shards=2, time=5.0),
            ShardedTest(name="test3", shard=1, num_shards=2, time=4.0),
            ShardedTest(name="test4", shard=1, num_shards=2, time=3.0),
            ShardedTest(name="test5", shard=1, num_shards=2, time=2.0),
            ShardedTest(name="test6", shard=1, num_shards=2, time=1.0),
        ]
        prioritized_tests = [
            test for test in tests if test.name in ["test4", "test5", "test8"]
        ]

        expected_time_savings = 9.0

        time_savings = log_time_savings(
            tests, prioritized_tests, is_serial_test_fn=never_serial, num_procs=2
        )
        self.assertEqual(
            time_savings, expected_time_savings, "Received an unexpected time savings"
        )

    def test_compute_prioritization_time_savings_with_multiple_threads_and_many_prioritized_tests(
        self,
    ) -> None:
        tests = [
            ShardedTest(name="test1", shard=1, num_shards=2, time=4.0),
            ShardedTest(name="test2", shard=1, num_shards=2, time=3.0),
            ShardedTest(name="test3", shard=1, num_shards=2, time=2.0),
            ShardedTest(name="test4", shard=1, num_shards=2, time=3.0),
            ShardedTest(name="test5", shard=1, num_shards=2, time=4.0),
            ShardedTest(name="test6", shard=1, num_shards=2, time=3.0),
            ShardedTest(name="test7", shard=1, num_shards=2, time=5.0),
        ]
        prioritized_tests = [
            test for test in tests if test.name in ["test2", "test3", "test7"]
        ]

        # Drawing out the math here since this is a complicated example

        # Logic for original execution assuming 2 procs
        # Test  | Proc 1 | Proc 2
        # test1 |   4    |
        # test2 |        |   3
        # test3 |        |   2
        # test4 |   3    |
        # test5 |        |   4
        # test6 |   3    |
        # test7 |        |   5   <- starts at time 9 ( 3 + 2 + 4)

        # Logic for new execution's prioritized pool:
        # Test  | Proc 1 | Proc 2
        # test3 |   2    |
        # test4 |        |   3
        # test7 |   5    |       <- now starts at time 2

        # Time savings = 9 - 2 = 7

        expected_time_savings = 7.0

        time_savings = log_time_savings(
            tests, prioritized_tests, is_serial_test_fn=never_serial, num_procs=2
        )
        self.assertEqual(
            time_savings, expected_time_savings, "Received an unexpected time savings"
        )
        pass

    def test_compute_prioritization_time_savings_with_serialized_test(self) -> None:
        tests = [
            ShardedTest(name="test1", shard=1, num_shards=2, time=7.0),
            ShardedTest(name="test2", shard=1, num_shards=2, time=5.0),
            ShardedTest(name="test3", shard=1, num_shards=2, time=4.0),
            ShardedTest(name="test4", shard=1, num_shards=2, time=3.0),
            ShardedTest(name="test5", shard=1, num_shards=2, time=2.0),
            ShardedTest(name="test6", shard=1, num_shards=2, time=1.0),
        ]
        prioritized_tests = [test for test in tests if test.name in ["test3", "test6"]]

        def serialized(test: str) -> bool:
            return test in ["test4", "test6"]

        expected_time_savings = 8.0

        time_savings = log_time_savings(
            tests, prioritized_tests, is_serial_test_fn=serialized, num_procs=2
        )
        self.assertEqual(
            time_savings, expected_time_savings, "Received an unexpected time savings"
        )
        pass


if __name__ == "__main__":
    unittest.main()
