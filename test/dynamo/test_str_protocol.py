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


class FStringMutationTests(TestCase):
    """Tests for f-string mutation ordering (issue #177582).

    Dynamo must evaluate f-string formatting at the correct bytecode point
    so that mutations between two f-strings are reflected in the output.
    """

    def _check(self, fn, *args_factory):
        import copy

        import torch
        import torch._dynamo.testing

        eager_result = fn(*copy.deepcopy(args_factory))
        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt)
        compiled_result = compiled_fn(*copy.deepcopy(args_factory))
        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(cnt.frame_count, 1)

    def test_fstring_tracks_user_defined_object_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"obj = {obj}"
            obj.val.append(0)
            s2 = f"obj = {obj}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_fstring_tracks_frozen_dataclass_field_mutations(self):
        from dataclasses import dataclass

        import torch

        @dataclass(frozen=True)
        class FrozenObj:
            val: list

            def __repr__(self):
                return f"FrozenObj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"obj = {obj}"
            obj.val.append(0)
            s2 = f"obj = {obj}"
            return x, s1, s2

        self._check(fn, torch.randn(3), FrozenObj([1, 2]))

    def test_fstring_str_conversion_tracks_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"{obj!s}"
            obj.val.append(0)
            s2 = f"{obj!s}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_fstring_repr_conversion_tracks_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"{obj!r}"
            obj.val.append(0)
            s2 = f"{obj!r}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_explicit_str_tracks_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = str(obj)
            obj.val.append(0)
            s2 = str(obj)
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))


if __name__ == "__main__":
    run_tests()
