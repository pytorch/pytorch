import random
import unittest

from tools.testing.test_selections import calculate_shards
from typing import Dict, List, Tuple


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
        expected_shards: List[Tuple[float, List[str]]],
        actual_shards: List[Tuple[float, List[str]]],
    ) -> None:
        for expected, actual in zip(expected_shards, actual_shards):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertListEqual(expected[1], actual[1])

    def test_calculate_2_shards_with_complete_test_times(self) -> None:
        expected_shards = [
            (60, ["super_long_test", "normal_test3"]),
            (
                58.31,
                [
                    "long_test1",
                    "long_test2",
                    "normal_test1",
                    "normal_test2",
                    "short_test1",
                    "short_test2",
                    "short_test3",
                    "short_test4",
                    "short_test5",
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
                ],
            ),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(1, self.tests, self.test_times)
        )

    def test_calculate_5_shards_with_complete_test_times(self) -> None:
        expected_shards = [
            (55.0, ["super_long_test"]),
            (
                22.0,
                [
                    "long_test1",
                ],
            ),
            (
                18.0,
                [
                    "long_test2",
                ],
            ),
            (
                11.31,
                [
                    "normal_test1",
                    "short_test1",
                    "short_test2",
                    "short_test3",
                    "short_test4",
                    "short_test5",
                ],
            ),
            (12.0, ["normal_test2", "normal_test3"]),
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
                    "long_test1",
                    "long_test2",
                    "normal_test3",
                    "short_test3",
                    "short_test5",
                ],
            ),
            (
                10.0,
                [
                    "normal_test1",
                    "short_test1",
                    "super_long_test",
                    "normal_test2",
                    "short_test2",
                    "short_test4",
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
            (22.0, ["long_test1", "normal_test2", "short_test5"]),
            (9.0, ["normal_test1", "normal_test3"]),
            (1.0, ["short_test1", "short_test2"]),
            (0.0, ["super_long_test", "short_test3"]),
            (0.0, ["long_test2", "short_test4"]),
        ]
        self.assert_shards_equal(
            expected_shards, calculate_shards(5, self.tests, incomplete_test_times)
        )

    def test_calculate_2_shards_against_optimal_shards(self) -> None:
        for _ in range(100):
            random.seed(120)
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
                self.assertEqual(sorted_tests, sorted_shard_tests)


if __name__ == "__main__":
    unittest.main()
