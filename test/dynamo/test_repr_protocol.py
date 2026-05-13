# Owner(s): ["module: dynamo"]
"""Tests for tp_repr / generic_repr: repr() via PyObject_Repr in Dynamo."""

import collections
import enum
import typing

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class _Color(enum.Enum):
    RED = 1
    BLUE = 2


class _OpaqueReprDescriptorObject:
    __repr__ = str.upper


class TpReprTests(TestCase):
    @make_dynamo_test
    def test_int_repr(self):
        assert repr(3) == "3"  # noqa: S101

    @make_dynamo_test
    def test_int_dunder_repr(self):
        assert (3).__repr__() == "3"  # noqa: S101

    @make_dynamo_test
    def test_list_repr(self):
        assert repr([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101

    @make_dynamo_test
    def test_list_dunder_repr(self):
        assert list.__repr__([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101

    @make_dynamo_test
    def test_set_repr(self):
        assert repr({1, 2, 3}) == "{1, 2, 3}"  # noqa: S101

    def test_counter_repr(self):
        def fn(x):
            return repr(collections.Counter("aba"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    @make_dynamo_test
    def test_enum_member_repr(self):
        assert repr(_Color.RED) == "<_Color.RED: 1>"  # noqa: S101

    def test_user_defined_repr(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return repr(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_dunder_repr(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return obj.__repr__()

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_default_object_repr(self):
        class Plain:
            pass

        def fn(x, obj):
            return repr(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_repr_on_plain_instance(self):
        class Plain:
            pass

        def fn(x, obj):
            return object.__repr__(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_repr_returning_non_string_raises(self):
        class BadRepr:
            def __repr__(self):
                return 3

        def fn(x, obj):
            try:
                return repr(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        obj = BadRepr()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertIn("__repr__ returned non-string", out)

    def test_dunder_repr_returning_non_string_raises(self):
        class BadRepr:
            def __repr__(self):
                return 3

        def fn(x, obj):
            return obj.__repr__()

        x = torch.randn(4)
        obj = BadRepr()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, 3)

    def test_user_defined_opaque_repr_descriptor_raises_type_error(self):
        def fn(x, obj):
            try:
                return repr(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        eager_result = fn(x, _OpaqueReprDescriptorObject())
        self.assertIn(
            "descriptor 'upper' for 'str' objects doesn't apply",
            eager_result,
        )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(eager_result, compiled(x, _OpaqueReprDescriptorObject()))

    def test_metaclass_repr(self):
        class Meta(type):
            def __repr__(cls):
                return f"<Meta {cls.__name__}>"

        class MyClass(metaclass=Meta):
            pass

        def fn(x):
            return repr(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_type_dunder_repr_on_class(self):
        class Meta(type):
            def __repr__(cls):
                return f"<Meta {cls.__name__}>"

        class MyClass(metaclass=Meta):
            pass

        def fn(x):
            return type.__repr__(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))
        self.assertEqual(compiled(x), type.__repr__(MyClass))

    def test_exception_repr(self):
        def fn(x):
            return repr(ValueError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_function_repr(self):
        def helper(y):
            return y + 1

        def fn(x):
            return repr(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_tensor_repr(self):
        def fn(x):
            return repr(x)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(x),
            "Tensor(shape=(4,), dtype=torch.float32)",
        )

    def test_tensor_dunder_repr(self):
        def fn(x):
            return x.__repr__()

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(x),
            "Tensor(shape=(4,), dtype=torch.float32)",
        )

    def test_list_repr_with_tensor(self):
        def fn(x):
            return repr([x])

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(x),
            "[Tensor(shape=(4,), dtype=torch.float32)]",
        )

    def test_list_repr_tracks_user_defined_object_mutations(self):
        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            y = x + 1
            s1 = repr([obj])
            obj.val.append(0)
            s2 = repr([obj])
            return y, s1, s2

        x = torch.randn(4)
        eager_result = fn(x, Obj([1, 2]))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            x, Obj([1, 2])
        )
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1:], compiled_result[1:])

    def test_dict_repr_with_tensor(self):
        def fn(x):
            return repr({"x": x})

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(x),
            "{'x': Tensor(shape=(4,), dtype=torch.float32)}",
        )

    def test_nn_module_repr(self):
        mod = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, 4))

        def fn(x):
            return repr(mod)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_dict_view_repr(self):
        def fn(x):
            return repr({"a": 1, "b": 2}.keys())

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_dict_view_repr_with_tensor(self):
        def fn(x):
            return repr({"x": x}.keys())

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), "dict_keys(['x'])")

    def test_defaultdict_repr(self):
        def fn(x):
            return repr(collections.defaultdict(int, {"a": 1}))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_repr_with_nested_function_factory_unsupported(self):
        def fn(x):
            def factory():
                return x

            return repr(collections.defaultdict(factory, {"a": 1}))

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"repr\(\) on nested function with non-constructible closure",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    def test_defaultdict_repr_uses_factory_repr(self):
        class Factory:
            def __call__(self):
                return 0

            def __repr__(self):
                return "Factory()"

        def fn(x, factory):
            return repr(collections.defaultdict(factory, {"a": 1}))

        x = torch.randn(4)
        factory = Factory()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, factory), compiled(x, factory))

    def test_user_defined_dict_subclass_repr(self):
        class MyDict(dict):
            pass

        def fn(x, obj):
            return repr(obj)

        x = torch.randn(4)
        obj = MyDict({"a": 1})
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_ordereddict_and_namedtuple_repr_track_nested_repr(self):
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
            s1 = (repr(ordered), repr(named))
            obj.val.append(0)
            s2 = (repr(ordered), repr(named))
            return y, s1, s2

        x = torch.randn(4)
        eager_result = fn(x, Obj([1, 2]))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            x, Obj([1, 2])
        )
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1:], compiled_result[1:])

    def test_structseq_repr(self):
        def fn(x):
            return repr(torch.max(x, dim=0))

        x = torch.randn(3, 2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(x),
            "torch.return_types.max(values=Tensor(shape=(2,), dtype=torch.float32), indices=Tensor(shape=(2,), dtype=torch.int64))",
        )

    def test_nested_function_repr_with_tensor_closure_unsupported(self):
        def fn(x):
            def inner():
                return x + 1

            return repr(inner)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"repr\(\) on nested function with non-constructible closure",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    def test_self_ref_list_repr(self):
        def fn():
            l = [1, 2, 3]
            l[0] = l
            return repr(l)

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        self.assertEqual(compiled(), fn())

    def test_self_ref_dict_repr(self):
        def fn():
            d = {}
            d["self"] = d
            return repr(d)

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        self.assertEqual(compiled(), fn())

    def test_mutual_ref_repr(self):
        def fn():
            a = [1]
            b = [2]
            a.append(b)
            b.append(a)
            return repr(a)

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        self.assertEqual(compiled(), fn())


if __name__ == "__main__":
    run_tests()
