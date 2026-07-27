# Owner(s): ["module: dynamo"]
"""Tests for tp_str / generic_str behavior in Dynamo."""

import collections
import typing

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class _OpaqueStrDescriptorObject:
    __str__ = str.upper


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

    @make_dynamo_test
    def test_object_dunder_str_on_string_uses_repr(self):
        assert object.__str__("hello") == "'hello'"  # noqa: S101
        assert object.__str__("") == "''"  # noqa: S101

    @make_dynamo_test
    def test_object_dunder_str_on_list_uses_repr(self):
        assert object.__str__([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101


class TpStrUserDefinedTests(TestCase):
    def test_counter_str(self):
        def fn(x):
            return str(collections.Counter("aba"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_defined_str(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return str(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_dunder_str(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return obj.__str__()

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_default_object_str(self):
        class Plain:
            pass

        def fn(x, obj):
            return str(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_repr_fallback_for_str(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return str(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_str_on_plain_instance(self):
        class Plain:
            pass

        def fn(x, obj):
            return object.__str__(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_str_ignores_user_defined_str(self):
        class MyObj:
            def __str__(self):
                return "MyObj"

        def fn(x, obj):
            return object.__str__(obj)

        obj = MyObj()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, repr(obj))
        self.assertNotEqual(out, str(obj))

    def test_object_dunder_str_uses_user_defined_repr(self):
        class MyObj:
            def __repr__(self):
                return "MyObjRepr"

            def __str__(self):
                return "MyObjStr"

        def fn(x, obj):
            return object.__str__(obj)

        obj = MyObj()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, repr(obj))
        self.assertNotEqual(out, str(obj))

    def test_str_returning_non_string_raises(self):
        class BadStr:
            def __str__(self):
                return 3  # noqa: PLE0307

        def fn(x, obj):
            try:
                return str(obj)
            except TypeError as e:
                return str(e)

        obj = BadStr()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertIn("__str__ returned non-string", out)

    def test_user_defined_opaque_str_descriptor_raises_type_error(self):
        def fn(x, obj):
            try:
                return str(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        eager_result = fn(x, _OpaqueStrDescriptorObject())
        self.assertIn(
            "descriptor 'upper' for 'str' objects doesn't apply",
            eager_result,
        )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(eager_result, compiled(x, _OpaqueStrDescriptorObject()))

    def test_metaclass_str(self):
        class Meta(type):
            def __repr__(cls):
                return f"<MetaRepr {cls.__name__}>"

            def __str__(cls):
                return f"<MetaStr {cls.__name__}>"

        class MyClass(metaclass=Meta):
            pass

        def fn(x):
            return str(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_type_dunder_str_on_class(self):
        class Meta(type):
            def __repr__(cls):
                return f"<MetaRepr {cls.__name__}>"

            def __str__(cls):
                return f"<MetaStr {cls.__name__}>"

        class MyClass(metaclass=Meta):
            pass

        def fn(x):
            return type.__str__(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))
        self.assertEqual(compiled(x), type.__str__(MyClass))

    def test_user_function_str(self):
        def helper(y):
            return y + 1

        def fn(x):
            return str(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_lambda_str(self):
        helper = lambda: None  # noqa: E731

        def fn(x):
            return str(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_str(self):
        def fn(x):
            return str(collections.defaultdict(int, {"a": 1}))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_str_with_nested_function_factory_unsupported(self):
        def fn(x):
            def factory():
                return x

            return str(collections.defaultdict(factory, {"a": 1}))

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"repr\(\) on nested function with non-constructible closure",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    def test_ordereddict_and_namedtuple_str_track_nested_repr(self):
        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        class Named(typing.NamedTuple):
            obj: object

        def fn(x, obj):
            ordered = collections.OrderedDict([("obj", obj)])
            named = Named(obj)
            y = x + 1
            s1 = (str(ordered), str(named))
            obj.val.append(0)
            s2 = (str(ordered), str(named))
            return y, s1, s2

        x = torch.randn(4)
        eager_result = fn(x, Obj([1, 2]))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            x, Obj([1, 2])
        )
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1:], compiled_result[1:])

    def test_structseq_str_with_tensor_graph_breaks(self):
        def fn(x):
            return str(torch.max(x, dim=0))

        x = torch.randn(3, 2)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), str(torch.max(x, dim=0)))


class TpStrExceptionTests(TestCase):
    @make_dynamo_test
    def test_exception_no_args(self):
        assert str(ValueError()) == ""  # noqa: S101

    @make_dynamo_test
    def test_exception_one_arg(self):
        assert str(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_exception_one_int_arg(self):
        assert str(ValueError(42)) == "42"  # noqa: S101

    @make_dynamo_test
    def test_exception_multiple_args(self):
        assert str(ValueError("error", 42)) == "('error', 42)"  # noqa: S101

    @make_dynamo_test
    def test_exception_dunder(self):
        assert TypeError("bad type").__str__() == "bad type"  # noqa: S101

    @make_dynamo_test
    def test_exception_unbound_dunder(self):
        assert ValueError.__str__(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_runtime_error(self):
        assert str(RuntimeError("runtime failure")) == "runtime failure"  # noqa: S101

    def test_user_defined_exception_subclass_str(self):
        class MyError(ValueError):
            pass

        def fn(x):
            return str(MyError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_defined_exception_subclass_custom_str(self):
        class MyError(ValueError):
            def __str__(self):
                return f"MyError({self.args[0]!r})"

        def fn(x):
            return str(MyError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))


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
