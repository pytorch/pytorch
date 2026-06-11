# Owner(s): ["module: dynamo"]
"""Tests for tp_str / generic_str foundation behavior in Dynamo."""

from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class TpStrTests(TestCase):
    @make_dynamo_test
    def test_str_int(self):
        assert str(42) == "42"  # noqa: S101
        assert str(-1) == "-1"  # noqa: S101
        assert str(0) == "0"  # noqa: S101

    @make_dynamo_test
    def test_str_float(self):
        assert str(3.14) == "3.14"  # noqa: S101
        assert str(0.0) == "0.0"  # noqa: S101
        assert str(-2.5) == "-2.5"  # noqa: S101

    @make_dynamo_test
    def test_str_bool(self):
        assert str(True) == "True"  # noqa: S101
        assert str(False) == "False"  # noqa: S101

    @make_dynamo_test
    def test_str_none(self):
        assert str(None) == "None"  # noqa: S101

    @make_dynamo_test
    def test_str_string_identity(self):
        s = "hello"
        empty = ""
        assert str(s) == "hello"  # noqa: S101
        assert str(empty) == ""  # noqa: S101

    @make_dynamo_test
    def test_str_dunder_constant(self):
        assert (42).__str__() == "42"  # noqa: S101
        assert (3.14).__str__() == "3.14"  # noqa: S101
        assert True.__str__() == "True"  # noqa: S101

    @make_dynamo_test
    def test_str_unbound_dunder_constant(self):
        assert int.__str__(42) == "42"  # noqa: S101
        assert float.__str__(3.14) == "3.14"  # noqa: S101
        assert bool.__str__(True) == "True"  # noqa: S101

    @make_dynamo_test
    def test_str_unbound_dunder_string(self):
        assert str.__str__("hello") == "hello"  # noqa: S101
        assert str.__str__("") == ""  # noqa: S101

    @make_dynamo_test
    def test_str_list_falls_back_to_repr(self):
        assert str([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101


if __name__ == "__main__":
    run_tests()
