from unittest import main, TestCase

from pytest_caching_utils import _merged_lastfailed_content


class TestPytestCachingUtils(TestCase):
    def test_merged_lastfailed_content_with_overlap(self) -> None:
        last_failed_source = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
        }
        last_failed_dest = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_without_overlap(self) -> None:
        last_failed_source = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
        }
        last_failed_dest = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_with_empty_source(self) -> None:
        last_failed_source = {
            "": True,
        }
        last_failed_dest = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_with_empty_dest(self) -> None:
        last_failed_source = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_dest = {
            "": True,
        }
        last_failed_merged = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)


if __name__ == "__main__":
    main()
