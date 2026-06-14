import unittest

from tools.testing.update_slow_tests import merge_slow_tests


class TestUpdateSlowTests(unittest.TestCase):
    def test_merge_slow_tests_preserves_existing_unmeasured_entries(self) -> None:
        existing = {
            "test_still_slow (__main__.FooTests)": 95.0,
            "test_recently_measured (__main__.FooTests)": 61.0,
        }
        measured = {
            "test_recently_measured (__main__.FooTests)": 70.0,
            "test_new_slow (__main__.BarTests)": 80.0,
        }

        self.assertEqual(
            merge_slow_tests(existing, measured),
            {
                "test_new_slow (__main__.BarTests)": 80.0,
                "test_recently_measured (__main__.FooTests)": 70.0,
                "test_still_slow (__main__.FooTests)": 95.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
