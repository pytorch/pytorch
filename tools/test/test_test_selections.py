import pathlib
import random
import sys
import unittest
from collections import defaultdict
from typing import Dict, List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))
    from tools.testing.test_selections import calculate_shards, ShardedTest, THRESHOLD
except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


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
            random_times["super_long_test"] = max(sum_of_rest / 2, *rest_of_tests)
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


if __name__ == "__main__":
    unittest.main()
