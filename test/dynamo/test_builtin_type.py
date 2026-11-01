# Owner(s): ["module: dynamo"]

import enum

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


def method_name(name, *args):
    return name.strip("_")


@instantiate_parametrized_tests
class BuiltinTypeVariableTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def assertCompiledMatchesEager(self, fn):
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(), fn())

    def assertUnboundCall(self, ty, name, args):
        def fn():
            return getattr(ty, name)(*args)

        self.assertCompiledMatchesEager(fn)

    def assertUnboundMutatingCall(self, ty, name, make_args):
        # The arguments are built inside the traced function so mutations are tracked
        def fn():
            args = make_args()
            return getattr(ty, name)(*args), args

        self.assertCompiledMatchesEager(fn)

    @staticmethod
    def make_subclass(base, override=None):
        # The override returns a marker, so running it instead of the base
        # type's method is visible in the result.
        ns = {} if override is None else {override: lambda self, *a, **kw: "override"}
        return type(f"Sub{base.__name__}", (base,), ns)

    @parametrize("ty", [dict, list, bytearray])
    def test_reconstruct(self, ty):
        def fn(x, ty):
            return type(ty()), x + 1

        x = torch.randn(3)
        res_ty, res_x = torch.compile(fn, backend="eager", fullgraph=True)(x, ty)
        self.assertIs(res_ty, ty)
        self.assertEqual(res_x, x + 1)

    @parametrize("ty", [dict, list, bytearray])
    def test_reconstruct_across_graph_break(self, ty):
        def fn(x, ty):
            res_ty = type(ty())
            torch._dynamo.graph_break()
            return res_ty, x + 1

        x = torch.randn(3)
        res_ty, res_x = torch.compile(fn, backend="eager")(x, ty)
        self.assertIs(res_ty, ty)
        self.assertEqual(res_x, x + 1)

    @parametrize("ty", [dict, list, bytearray])
    def test_repr_str(self, ty):
        def fn(ty):
            return repr(ty), str(ty)

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(ty), fn(ty))

    @parametrize("ty", [dict, list, bytearray])
    def test_type_members(self, ty):
        def fn(ty):
            return ty.__mro__, ty.__bases__, ty.__base__, ty.__flags__, ty.__basicsize__

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(ty), fn(ty))

    @parametrize(
        "name,args",
        [
            ("__radd__", (1, 2)),
            ("__mul__", (3, 4)),
            ("__truediv__", (7, 2)),
            ("__pow__", (2, 5)),
            ("__neg__", (4,)),
            ("__abs__", (-4,)),
            ("__invert__", (4,)),
            ("__eq__", (1, 1)),
            ("__ne__", (1, "a")),
            ("__lt__", (1, 2)),
            ("__hash__", (5,)),
            ("__bool__", (5,)),
            ("__index__", (5,)),
            ("__float__", (5,)),
            ("__repr__", (5,)),
            ("__lshift__", (1, 3)),
            ("__and__", (6, 3)),
            ("bit_length", (255,)),
            ("to_bytes", (258, 2, "big")),
            ("conjugate", (3,)),
            ("from_bytes", (b"\x01\x02", "big")),
            ("__new__", (int, "12")),
        ],
        name_fn=method_name,
    )
    def test_int(self, name, args):
        self.assertUnboundCall(int, name, args)

    @parametrize(
        "name,args",
        [
            ("__add__", (1.5, 2)),
            ("__neg__", (1.5,)),
            ("__eq__", (1.0, 1)),
            ("__hash__", (1.5,)),
            ("__int__", (1.5,)),
            ("__pow__", (2.0, 3)),
            ("hex", (1.5,)),
            ("is_integer", (2.0,)),
            ("fromhex", ("0x1.8p0",)),
            ("__new__", (float, "1.5")),
        ],
        name_fn=method_name,
    )
    def test_float(self, name, args):
        self.assertUnboundCall(float, name, args)

    @parametrize(
        "name,args",
        [
            ("__add__", (1 + 1j, 2)),
            ("__eq__", (1 + 1j, "a")),
            ("__neg__", (1 + 1j,)),
            ("__abs__", (3 + 4j,)),
            ("conjugate", (1 + 1j,)),
            ("__hash__", (1 + 1j,)),
        ],
        name_fn=method_name,
    )
    def test_complex(self, name, args):
        self.assertUnboundCall(complex, name, args)

    @parametrize(
        "name,args",
        [
            ("__and__", (True, False)),
            ("__invert__", (True,)),
            ("__repr__", (True,)),
            ("__new__", (bool, 3)),
        ],
        name_fn=method_name,
    )
    def test_bool(self, name, args):
        self.assertUnboundCall(bool, name, args)

    @parametrize(
        "name,args",
        [
            ("join", (",", ["a", "b"])),
            ("format", ("{}-{}", 1, 2)),
            ("startswith", ("abc", "a")),
            ("maketrans", ("a", "b")),
            ("__len__", ("abc",)),
            ("__mul__", ("a", 3)),
            ("__mod__", ("%d", 3)),
            ("__contains__", ("abc", "b")),
            ("__eq__", ("a", "a")),
            ("__lt__", ("a", "b")),
            ("__hash__", ("a",)),
            ("__repr__", ("a",)),
            ("__new__", (str, "x")),
        ],
        name_fn=method_name,
    )
    def test_str(self, name, args):
        self.assertUnboundCall(str, name, args)

    @parametrize(
        "name,args",
        [
            ("decode", (b"abc",)),
            ("hex", (b"ab",)),
            ("fromhex", ("6162",)),
            ("__len__", (b"abc",)),
            ("__add__", (b"a", b"b")),
            ("__getitem__", (b"abc", 1)),
        ],
        name_fn=method_name,
    )
    def test_bytes(self, name, args):
        self.assertUnboundCall(bytes, name, args)

    @parametrize(
        "name,make_args",
        [
            ("__len__", lambda: (bytearray(b"ab"),)),
            ("hex", lambda: (bytearray(b"ab"),)),
            ("fromhex", lambda: ("6162",)),
            ("append", lambda: (bytearray(b"a"), 98)),
        ],
        name_fn=method_name,
    )
    def test_bytearray(self, name, make_args):
        self.assertUnboundMutatingCall(bytearray, name, make_args)

    @parametrize(
        "name,args",
        [
            ("__len__", ((1, 2),)),
            ("__add__", ((1,), (2,))),
            ("__mul__", ((1,), 3)),
            ("__contains__", ((1, 2), 2)),
            ("__eq__", ((1,), (1,))),
            ("__lt__", ((1,), (2,))),
            ("__hash__", ((1, 2),)),
            ("index", ((1, 2, 3), 2)),
            ("count", ((1, 1, 2), 1)),
            ("__new__", (tuple, [1, 2])),
        ],
        name_fn=method_name,
    )
    def test_tuple(self, name, args):
        self.assertUnboundCall(tuple, name, args)

    @parametrize(
        "name,args",
        [
            ("__len__", (frozenset({1, 2}),)),
            ("__or__", (frozenset({1}), frozenset({2}))),
            ("__hash__", (frozenset({1}),)),
            ("__contains__", (frozenset({1}), 1)),
            ("union", (frozenset({1}), {2})),
        ],
        name_fn=method_name,
    )
    def test_frozenset(self, name, args):
        self.assertUnboundCall(frozenset, name, args)

    @parametrize(
        "name,make_args",
        [
            ("__len__", lambda: ([1, 2, 3],)),
            ("__add__", lambda: ([1], [2])),
            ("__mul__", lambda: ([1], 3)),
            ("__contains__", lambda: ([1, 2], 2)),
            ("__eq__", lambda: ([1], [1])),
            ("__lt__", lambda: ([1], [2])),
            ("__repr__", lambda: ([1],)),
            ("index", lambda: ([1, 2, 3], 2)),
            ("count", lambda: ([1, 1, 2], 1)),
            ("copy", lambda: ([1, 2],)),
            ("append", lambda: ([1], 2)),
            ("extend", lambda: ([1], [2, 3])),
            ("insert", lambda: ([1, 3], 1, 2)),
            ("pop", lambda: ([1, 2, 3],)),
            ("reverse", lambda: ([1, 2, 3],)),
            ("__setitem__", lambda: ([1, 2], 0, 9)),
            ("__delitem__", lambda: ([1, 2], 0)),
            ("__iadd__", lambda: ([1], [2])),
            ("__init__", lambda: ([], [1, 2])),
        ],
        name_fn=method_name,
    )
    def test_list(self, name, make_args):
        self.assertUnboundMutatingCall(list, name, make_args)

    @parametrize(
        "name,make_args",
        [
            ("__len__", lambda: ({1, 2},)),
            ("__contains__", lambda: ({1, 2}, 2)),
            ("__or__", lambda: ({1}, {2})),
            ("__and__", lambda: ({1, 2}, {2})),
            ("__sub__", lambda: ({1, 2}, {2})),
            ("__xor__", lambda: ({1, 2}, {2, 3})),
            ("__eq__", lambda: ({1}, {1})),
            ("__le__", lambda: ({1}, {1, 2})),
            ("update", lambda: ({1}, [2, 3])),
            ("discard", lambda: ({1, 2}, 2)),
            ("remove", lambda: ({1, 2}, 2)),
            ("pop", lambda: ({1},)),
            ("copy", lambda: ({1, 2},)),
            ("union", lambda: ({1}, {2})),
            ("intersection", lambda: ({1, 2}, {2})),
            ("difference", lambda: ({1, 2}, {2})),
            ("issubset", lambda: ({1}, {1, 2})),
            ("__ior__", lambda: ({1}, {2})),
            ("__init__", lambda: (set(), [1, 2])),
            ("__new__", lambda: (set,)),
        ],
        name_fn=method_name,
    )
    def test_set(self, name, make_args):
        self.assertUnboundMutatingCall(set, name, make_args)

    @parametrize(
        "name,make_args",
        [
            ("__len__", lambda: ({1: 2},)),
            ("__getitem__", lambda: ({1: 2}, 1)),
            ("__contains__", lambda: ({1: 2}, 1)),
            ("__eq__", lambda: ({1: 2}, {1: 2})),
            ("__or__", lambda: ({1: 2}, {3: 4})),
            ("__repr__", lambda: ({1: 2},)),
            ("get", lambda: ({1: 2}, 1)),
            ("copy", lambda: ({1: 2},)),
            ("pop", lambda: ({1: 2}, 1)),
            ("setdefault", lambda: ({}, 1, 2)),
            ("update", lambda: ({1: 2}, {3: 4})),
            ("__delitem__", lambda: ({1: 2}, 1)),
            ("__new__", lambda: (dict,)),
        ],
        name_fn=method_name,
    )
    def test_dict(self, name, make_args):
        self.assertUnboundMutatingCall(dict, name, make_args)

    @parametrize(
        "ty,name,value",
        [
            (str, "__iter__", "abc"),
            (list, "__iter__", [1, 2]),
            (tuple, "__iter__", (1, 2)),
            (set, "__iter__", {1}),
            (dict, "values", {1: 2}),
        ],
        name_fn=lambda ty, name, value: f"{ty.__name__}_{name.strip('_')}",
    )
    def test_iteration(self, ty, name, value):
        self.assertCompiledMatchesEager(lambda: list(getattr(ty, name)(value)))

    @parametrize(
        "name,args",
        [
            ("__eq__", (1, 1)),
            ("__ne__", (1, 2)),
        ],
        name_fn=method_name,
    )
    def test_object(self, name, args):
        self.assertUnboundCall(object, name, args)

    def test_object_hash(self):
        class A:
            pass

        self.assertCompiledMatchesEager(lambda: isinstance(object.__hash__(A()), int))

    @parametrize(
        "name,args",
        [
            ("__call__", (int, 3)),
            ("__getattribute__", (int, "__name__")),
            ("__instancecheck__", (int, 3)),
            ("__subclasscheck__", (int, bool)),
        ],
        name_fn=method_name,
    )
    def test_type(self, name, args):
        self.assertUnboundCall(type, name, args)

    def test_type_or(self):
        self.assertCompiledMatchesEager(lambda: type.__or__(int, str).__args__)

    @parametrize(
        "override", [False, True], name_fn=lambda o: "override" if o else "plain"
    )
    @parametrize(
        "base,name,args,value",
        [
            (int, "__add__", (2,), 1),
            (int, "__radd__", (2,), 1),
            (int, "__neg__", (), 4),
            (int, "__int__", (), 4),
            (int, "__index__", (), 4),
            (int, "__eq__", (2,), 1),
            (int, "__lt__", (2,), 1),
            (int, "__hash__", (), 3),
            (int, "__bool__", (), 3),
            (int, "__repr__", (), 3),
            (int, "bit_length", (), 8),
            (float, "__add__", (2,), 1.5),
            (float, "__neg__", (), 1.5),
            (float, "__eq__", (1.5,), 1.5),
            (str, "__add__", ("b",), "a"),
            (str, "__len__", (), "abc"),
            (str, "upper", (), "abc"),
            (str, "__eq__", ("a",), "a"),
            (list, "__len__", (), [1, 2]),
            (list, "__getitem__", (0,), [1, 2]),
            (list, "__add__", ([2],), [1]),
            (list, "__contains__", (1,), [1]),
            (list, "__eq__", ([1],), [1]),
            (list, "__iadd__", ([2],), [1]),
            (list, "append", (1,), []),
            (tuple, "__len__", (), (1, 2)),
            (tuple, "__add__", ((2,),), (1,)),
            (tuple, "__getitem__", (1,), (1, 2)),
            (dict, "__len__", (), {1: 2}),
            (dict, "__getitem__", (1,), {1: 2}),
            (dict, "__contains__", (1,), {1: 2}),
            (dict, "get", (1,), {1: 2}),
            (set, "__len__", (), {1, 2}),
            (set, "__contains__", (1,), {1}),
            (set, "__or__", ({2},), {1}),
            (set, "add", (1,), set()),
        ],
        name_fn=lambda base, name, args, value: f"{base.__name__}_{name.strip('_')}",
    )
    def test_subclass_instance(self, base, name, args, value, override):
        # CPython runs the base type's method and ignores subclass overrides
        sub = self.make_subclass(base, name if override else None)

        def fn():
            obj = sub(value)
            return getattr(base, name)(obj, *args), base.__repr__(obj)

        self.assertCompiledMatchesEager(fn)

    @parametrize(
        "ty,name,wrong_self",
        [(int, "__add__", []), (list, "__len__", 1), (dict, "__len__", [])],
        name_fn=lambda ty, name, wrong_self: f"{ty.__name__}_{name.strip('_')}",
    )
    def test_wrong_self_raises_type_error(self, ty, name, wrong_self):
        def fn():
            try:
                getattr(ty, name)(wrong_self, 2)
            except TypeError:
                return "TypeError"
            return "no error"

        self.assertEqual(fn(), "TypeError")
        self.assertCompiledMatchesEager(fn)

    @parametrize("ty", [list, set, dict])
    def test_unhashable(self, ty):
        self.assertCompiledMatchesEager(lambda: ty.__hash__ is None)

    @parametrize("base", [tuple, list, set, dict])
    def test_new_of_subclass(self, base):
        sub = self.make_subclass(base)
        self.assertCompiledMatchesEager(lambda: type(base.__new__(sub)) is sub)

    def test_int_hash_in_dunder_hash(self):
        class A:
            def __hash__(self):
                return int.__hash__(3)

        self.assertCompiledMatchesEager(lambda: hash(A()))

    def test_int_enum(self):
        class E(enum.IntEnum):
            A = 1

        self.assertCompiledMatchesEager(
            lambda: (int.__eq__(E.A, 1), int.__add__(E.A, 1))
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
