# Owner(s): ["module: dynamo"]
"""Unbound calls on builtin types, e.g. int.__add__(1, 2), eager vs compiled."""

import enum
import re

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class IntEnum_(enum.IntEnum):
    A = 1
    B = 2


class MyInt(int):
    def __add__(self, o):
        return 100


class MyList(list):
    def __len__(self):
        return 42

    def append(self, x):
        super().append(x * 2)


class MyDict(dict):
    def __getitem__(self, k):
        return 7


class MyStr(str):
    def upper(self):
        return "custom"


class MyTuple(tuple):
    pass


class MySet(set):
    def add(self, x):
        super().add(x + 1)


class PlainInt(int):
    pass


class PlainFloat(float):
    pass


class PlainStr(str):
    pass


class PlainList(list):
    pass


class PlainDict(dict):
    pass


class PlainSet(set):
    pass


class PlainTuple(tuple):
    pass


class MyFloat(float):
    def __add__(self, o):
        return 100.0

    def __neg__(self):
        return -100.0


class MyIntAll(int):
    def __radd__(self, o):
        return 100

    def __neg__(self):
        return -100

    def __int__(self):
        return 100

    def __eq__(self, o):
        return True

    def __hash__(self):
        return 100

    def __repr__(self):
        return "custom"

    def __bool__(self):
        return False


class MyListAll(list):
    def __getitem__(self, i):
        return 7

    def __iter__(self):
        return iter([9])

    def __contains__(self, x):
        return True

    def __iadd__(self, o):
        return "custom"

    def __eq__(self, o):
        return True


def _cases():
    c = {}

    def add(name, fn):
        c[name] = fn

    # int
    add("int.__add__", lambda: int.__add__(1, 2))
    add("int.__radd__", lambda: int.__radd__(1, 2))
    add("int.__sub__", lambda: int.__sub__(5, 2))
    add("int.__mul__", lambda: int.__mul__(3, 4))
    add("int.__floordiv__", lambda: int.__floordiv__(7, 2))
    add("int.__truediv__", lambda: int.__truediv__(7, 2))
    add("int.__mod__", lambda: int.__mod__(7, 4))
    add("int.__pow__", lambda: int.__pow__(2, 5))
    add("int.__neg__", lambda: int.__neg__(4))
    add("int.__pos__", lambda: int.__pos__(4))
    add("int.__abs__", lambda: int.__abs__(-4))
    add("int.__invert__", lambda: int.__invert__(4))
    add("int.__eq__", lambda: int.__eq__(1, 1))
    add("int.__ne__", lambda: int.__ne__(1, 2))
    add("int.__lt__", lambda: int.__lt__(1, 2))
    add("int.__ge__", lambda: int.__ge__(1, 2))
    add("int.__eq__ notimpl", lambda: int.__eq__(1, "a"))
    add("int.__hash__", lambda: int.__hash__(5))
    add("int.__bool__", lambda: int.__bool__(5))
    add("int.__int__", lambda: int.__int__(5))
    add("int.__index__", lambda: int.__index__(5))
    add("int.__float__", lambda: int.__float__(5))
    add("int.__repr__", lambda: int.__repr__(5))
    add("int.__str__", lambda: int.__str__(5))
    add("int.__lshift__", lambda: int.__lshift__(1, 3))
    add("int.__and__", lambda: int.__and__(6, 3))
    add("int.__or__", lambda: int.__or__(6, 3))
    add("int.__xor__", lambda: int.__xor__(6, 3))
    add("int.bit_length", lambda: int.bit_length(255))
    add("int.to_bytes", lambda: int.to_bytes(258, 2, "big"))
    add("int.from_bytes", lambda: int.from_bytes(b"\x01\x02", "big"))
    add("int.conjugate", lambda: int.conjugate(3))
    add("int.__new__", lambda: int.__new__(int, "12"))
    add("int.__add__ subclass", lambda: int.__add__(MyInt(1), 2))
    add("int.__eq__ intenum", lambda: int.__eq__(IntEnum_.A, 1))
    add("int.__add__ enum", lambda: int.__add__(IntEnum_.A, 1))
    add("int.__add__ bad arg", lambda: int.__add__(1, "a"))
    add("int.__hash__ in __hash__", lambda: hash(_HashesViaInt()))

    # float
    add("float.__add__", lambda: float.__add__(1.5, 2))
    add("float.__rsub__", lambda: float.__rsub__(3.0, 1))
    add("float.__neg__", lambda: float.__neg__(1.5))
    add("float.__abs__", lambda: float.__abs__(-1.5))
    add("float.__eq__", lambda: float.__eq__(1.0, 1))
    add("float.__lt__", lambda: float.__lt__(1.0, 2.0))
    add("float.__hash__", lambda: float.__hash__(1.5))
    add("float.__float__", lambda: float.__float__(1.5))
    add("float.__int__", lambda: float.__int__(1.5))
    add("float.hex", lambda: float.hex(1.5))
    add("float.fromhex", lambda: float.fromhex("0x1.8p0"))
    add("float.is_integer", lambda: float.is_integer(2.0))
    add("float.__new__", lambda: float.__new__(float, "1.5"))
    add("float.__pow__", lambda: float.__pow__(2.0, 3))

    # complex
    add("complex.__add__", lambda: complex.__add__(1 + 1j, 2))
    add("complex.__eq__", lambda: complex.__eq__(1 + 1j, 1 + 1j))
    add("complex.__eq__ notimpl", lambda: complex.__eq__(1 + 1j, "a"))
    add("complex.__neg__", lambda: complex.__neg__(1 + 1j))
    add("complex.__abs__", lambda: complex.__abs__(3 + 4j))
    add("complex.conjugate", lambda: complex.conjugate(1 + 1j))
    add("complex.__hash__", lambda: complex.__hash__(1 + 1j))

    # bool
    add("bool.__and__", lambda: bool.__and__(True, False))
    add("bool.__or__", lambda: bool.__or__(True, False))
    add("bool.__invert__", lambda: bool.__invert__(True))
    add("bool.__repr__", lambda: bool.__repr__(True))
    add("bool.__new__", lambda: bool.__new__(bool, 3))

    # str
    add("str.upper", lambda: str.upper("abc"))
    add("str.join", lambda: str.join(",", ["a", "b"]))
    add("str.split", lambda: str.split("a b c"))
    add("str.format", lambda: str.format("{}-{}", 1, 2))
    add("str.startswith", lambda: str.startswith("abc", "a"))
    add("str.__len__", lambda: str.__len__("abc"))
    add("str.__add__", lambda: str.__add__("a", "b"))
    add("str.__mul__", lambda: str.__mul__("a", 3))
    add("str.__mod__", lambda: str.__mod__("%d", 3))
    add("str.__getitem__", lambda: str.__getitem__("abc", 1))
    add("str.__contains__", lambda: str.__contains__("abc", "b"))
    add("str.__iter__", lambda: list(str.__iter__("abc")))
    add("str.__eq__", lambda: str.__eq__("a", "a"))
    add("str.__lt__", lambda: str.__lt__("a", "b"))
    add("str.__hash__", lambda: str.__hash__("a"))
    add("str.__repr__", lambda: str.__repr__("a"))
    add("str.__str__", lambda: str.__str__("a"))
    add("str.__new__", lambda: str.__new__(str, "x"))
    add("str.upper subclass", lambda: str.upper(MyStr("abc")))
    add("str.maketrans", lambda: str.maketrans("a", "b"))

    # bytes / bytearray
    add("bytes.decode", lambda: bytes.decode(b"abc"))
    add("bytes.__len__", lambda: bytes.__len__(b"abc"))
    add("bytes.__add__", lambda: bytes.__add__(b"a", b"b"))
    add("bytes.__getitem__", lambda: bytes.__getitem__(b"abc", 1))
    add("bytes.hex", lambda: bytes.hex(b"ab"))
    add("bytes.fromhex", lambda: bytes.fromhex("6162"))
    add("bytearray.__len__", lambda: bytearray.__len__(bytearray(b"ab")))
    add("bytearray.hex", lambda: bytearray.hex(bytearray(b"ab")))
    add("bytearray.fromhex", lambda: bytearray.fromhex("6162"))
    add("bytearray.append", lambda: _ba_append())

    # list
    add("list.__len__", lambda: list.__len__([1, 2, 3]))
    add("list.__len__ subclass", lambda: list.__len__(MyList([1, 2])))
    add("list.__getitem__", lambda: list.__getitem__([1, 2, 3], 1))
    add("list.__getitem__ slice", lambda: list.__getitem__([1, 2, 3], slice(0, 2)))
    add("list.__add__", lambda: list.__add__([1], [2]))
    add("list.__mul__", lambda: list.__mul__([1], 3))
    add("list.__contains__", lambda: list.__contains__([1, 2], 2))
    add("list.__iter__", lambda: list(list.__iter__([1, 2])))
    add("list.__eq__", lambda: list.__eq__([1], [1]))
    add("list.__lt__", lambda: list.__lt__([1], [2]))
    add("list.__repr__", lambda: list.__repr__([1]))
    add("list.index", lambda: list.index([1, 2, 3], 2))
    add("list.count", lambda: list.count([1, 1, 2], 1))
    add("list.copy", lambda: list.copy([1, 2]))
    add("list.append", lambda: _list_append())
    add("list.append subclass", lambda: _mylist_append())
    add("list.extend", lambda: _list_extend())
    add("list.__setitem__", lambda: _list_setitem())
    add("list.__delitem__", lambda: _list_delitem())
    add("list.__iadd__", lambda: _list_iadd())
    add("list.__init__", lambda: _list_init())
    add("list.__new__", lambda: list.__new__(list))
    add("list.sort", lambda: _list_sort())
    add("list.reverse", lambda: _list_reverse())
    add("list.pop", lambda: _list_pop())
    add("list.insert", lambda: _list_insert())
    add("list.__hash__ is None", lambda: list.__hash__ is None)

    # tuple
    add("tuple.__len__", lambda: tuple.__len__((1, 2)))
    add("tuple.__getitem__", lambda: tuple.__getitem__((1, 2, 3), 1))
    add("tuple.__add__", lambda: tuple.__add__((1,), (2,)))
    add("tuple.__mul__", lambda: tuple.__mul__((1,), 3))
    add("tuple.__contains__", lambda: tuple.__contains__((1, 2), 2))
    add("tuple.__iter__", lambda: list(tuple.__iter__((1, 2))))
    add("tuple.__eq__", lambda: tuple.__eq__((1,), (1,)))
    add("tuple.__lt__", lambda: tuple.__lt__((1,), (2,)))
    add("tuple.__hash__", lambda: tuple.__hash__((1, 2)))
    add("tuple.index", lambda: tuple.index((1, 2, 3), 2))
    add("tuple.count", lambda: tuple.count((1, 1, 2), 1))
    add("tuple.__new__", lambda: tuple.__new__(tuple, [1, 2]))
    add("tuple.__new__ subclass", lambda: tuple.__new__(MyTuple, [1, 2]))
    add("tuple.__len__ subclass", lambda: tuple.__len__(MyTuple((1, 2))))

    # set / frozenset
    add("set.__len__", lambda: set.__len__({1, 2}))
    add("set.__contains__", lambda: set.__contains__({1, 2}, 2))
    add("set.__or__", lambda: set.__or__({1}, {2}))
    add("set.__and__", lambda: set.__and__({1, 2}, {2}))
    add("set.__sub__", lambda: set.__sub__({1, 2}, {2}))
    add("set.__xor__", lambda: set.__xor__({1, 2}, {2, 3}))
    add("set.__eq__", lambda: set.__eq__({1}, {1}))
    add("set.__le__", lambda: set.__le__({1}, {1, 2}))
    add("set.__iter__", lambda: list(set.__iter__({1})))
    add("set.add", lambda: _set_add())
    add("set.add subclass", lambda: _myset_add())
    add("set.update", lambda: _set_update())
    add("set.discard", lambda: _set_discard())
    add("set.remove", lambda: _set_remove())
    add("set.pop", lambda: _set_pop())
    add("set.copy", lambda: set.copy({1, 2}))
    add("set.union", lambda: set.union({1}, {2}))
    add("set.intersection", lambda: set.intersection({1, 2}, {2}))
    add("set.difference", lambda: set.difference({1, 2}, {2}))
    add("set.issubset", lambda: set.issubset({1}, {1, 2}))
    add("set.__ior__", lambda: _set_ior())
    add("set.__init__", lambda: _set_init())
    add("set.__new__", lambda: set.__new__(set))
    add("set.__hash__ is None", lambda: set.__hash__ is None)
    add("frozenset.__len__", lambda: frozenset.__len__(frozenset({1, 2})))
    add("frozenset.__or__", lambda: frozenset.__or__(frozenset({1}), frozenset({2})))
    add("frozenset.__hash__", lambda: frozenset.__hash__(frozenset({1})))
    add("frozenset.union", lambda: frozenset.union(frozenset({1}), {2}))
    add("frozenset.__contains__", lambda: frozenset.__contains__(frozenset({1}), 1))

    # dict
    add("dict.__len__", lambda: dict.__len__({1: 2}))
    add("dict.__getitem__", lambda: dict.__getitem__({1: 2}, 1))
    add("dict.__getitem__ subclass", lambda: dict.__getitem__(MyDict({1: 2}), 1))
    add("dict.__contains__", lambda: dict.__contains__({1: 2}, 1))
    add("dict.__iter__", lambda: list(dict.__iter__({1: 2})))
    add("dict.__eq__", lambda: dict.__eq__({1: 2}, {1: 2}))
    add("dict.__or__", lambda: dict.__or__({1: 2}, {3: 4}))
    add("dict.__repr__", lambda: dict.__repr__({1: 2}))
    add("dict.get", lambda: dict.get({1: 2}, 1))
    add("dict.keys", lambda: list(dict.keys({1: 2})))
    add("dict.values", lambda: list(dict.values({1: 2})))
    add("dict.items", lambda: list(dict.items({1: 2})))
    add("dict.copy", lambda: dict.copy({1: 2}))
    add("dict.fromkeys", lambda: dict.fromkeys([1, 2], 0))
    add("dict.__setitem__", lambda: _dict_setitem())
    add("dict.__delitem__", lambda: _dict_delitem())
    add("dict.update", lambda: _dict_update())
    add("dict.pop", lambda: _dict_pop())
    add("dict.setdefault", lambda: _dict_setdefault())
    add("dict.__init__", lambda: _dict_init())
    add("dict.__new__", lambda: dict.__new__(dict))
    add("dict.__hash__ is None", lambda: dict.__hash__ is None)

    # object
    add("object.__init__", lambda: _object_init())
    add("object.__new__", lambda: type(object.__new__(object)))
    add("object.__eq__", lambda: object.__eq__(1, 1))
    add("object.__ne__", lambda: object.__ne__(1, 2))
    add("object.__hash__", lambda: isinstance(object.__hash__(_Plain()), int))
    add("object.__repr__ type", lambda: type(object.__repr__(_Plain())))
    add("object.__str__ type", lambda: type(object.__str__(_Plain())))
    add("object.__getattribute__", lambda: object.__getattribute__(_Plain(), "x"))
    add("object.__setattr__", lambda: _object_setattr())
    add("object.__class__", lambda: object.__class__)

    # type
    add("type.__call__", lambda: type.__call__(int, 3))
    add("type.__getattribute__", lambda: type.__getattribute__(int, "__name__"))
    add("type.__instancecheck__", lambda: type.__instancecheck__(int, 3))
    add("type.__subclasscheck__", lambda: type.__subclasscheck__(int, bool))
    add("type.__or__", lambda: type.__or__(int, str))

    # UDOV: user subclass instances. CPython runs the base type's slot and
    # ignores subclass overrides.
    # no overrides
    add("udov int.__add__ plain", lambda: int.__add__(PlainInt(1), 2))
    add("udov int.__radd__ plain", lambda: int.__radd__(PlainInt(1), 2))
    add("udov int.__sub__ plain", lambda: int.__sub__(PlainInt(5), 2))
    add("udov int.__neg__ plain", lambda: int.__neg__(PlainInt(4)))
    add("udov int.__int__ plain", lambda: int.__int__(PlainInt(4)))
    add("udov int.__index__ plain", lambda: int.__index__(PlainInt(4)))
    add("udov int.__eq__ plain", lambda: int.__eq__(PlainInt(1), 1))
    add("udov int.__lt__ plain", lambda: int.__lt__(PlainInt(1), 2))
    add("udov int.__hash__ plain", lambda: int.__hash__(PlainInt(3)))
    add("udov int.__bool__ plain", lambda: int.__bool__(PlainInt(3)))
    add("udov int.__repr__ plain", lambda: int.__repr__(PlainInt(3)))
    add("udov int.bit_length plain", lambda: int.bit_length(PlainInt(8)))
    add("udov float.__add__ plain", lambda: float.__add__(PlainFloat(1.5), 2))
    add("udov float.__neg__ plain", lambda: float.__neg__(PlainFloat(1.5)))
    add("udov float.__eq__ plain", lambda: float.__eq__(PlainFloat(1.5), 1.5))
    add("udov str.__add__ plain", lambda: str.__add__(PlainStr("a"), "b"))
    add("udov str.__len__ plain", lambda: str.__len__(PlainStr("abc")))
    add("udov str.upper plain", lambda: str.upper(PlainStr("abc")))
    add("udov str.__eq__ plain", lambda: str.__eq__(PlainStr("a"), "a"))
    add("udov list.__len__ plain", lambda: list.__len__(PlainList([1, 2])))
    add("udov list.__getitem__ plain", lambda: list.__getitem__(PlainList([1, 2]), 0))
    add("udov list.__add__ plain", lambda: list.__add__(PlainList([1]), [2]))
    add("udov list.__contains__ plain", lambda: list.__contains__(PlainList([1]), 1))
    add("udov list.__eq__ plain", lambda: list.__eq__(PlainList([1]), [1]))
    add("udov list.__iter__ plain", lambda: list(list.__iter__(PlainList([1, 2]))))
    add("udov list.append plain", lambda: _udov_append(PlainList()))
    add("udov list.__iadd__ plain", lambda: list.__iadd__(PlainList([1]), [2]))
    add("udov tuple.__len__ plain", lambda: tuple.__len__(PlainTuple((1, 2))))
    add("udov tuple.__add__ plain", lambda: tuple.__add__(PlainTuple((1,)), (2,)))
    add(
        "udov tuple.__getitem__ plain", lambda: tuple.__getitem__(PlainTuple((1, 2)), 1)
    )
    add("udov dict.__getitem__ plain", lambda: dict.__getitem__(PlainDict({1: 2}), 1))
    add("udov dict.__len__ plain", lambda: dict.__len__(PlainDict({1: 2})))
    add("udov dict.__contains__ plain", lambda: dict.__contains__(PlainDict({1: 2}), 1))
    add("udov dict.get plain", lambda: dict.get(PlainDict({1: 2}), 1))
    add("udov dict.keys plain", lambda: list(dict.keys(PlainDict({1: 2}))))
    add("udov set.__len__ plain", lambda: set.__len__(PlainSet({1, 2})))
    add("udov set.__contains__ plain", lambda: set.__contains__(PlainSet({1}), 1))
    add("udov set.__or__ plain", lambda: set.__or__(PlainSet({1}), {2}))
    add("udov set.add plain", lambda: _udov_set_add(PlainSet()))
    # subclass overrides: the base slot must win
    add("udov int.__add__ override", lambda: int.__add__(MyInt(1), 2))
    add("udov int.__radd__ override", lambda: int.__radd__(MyIntAll(1), 2))
    add("udov int.__neg__ override", lambda: int.__neg__(MyIntAll(4)))
    add("udov int.__int__ override", lambda: int.__int__(MyIntAll(4)))
    add("udov int.__eq__ override", lambda: int.__eq__(MyIntAll(1), 2))
    add("udov int.__hash__ override", lambda: int.__hash__(MyIntAll(3)))
    add("udov int.__repr__ override", lambda: int.__repr__(MyIntAll(3)))
    add("udov int.__bool__ override", lambda: int.__bool__(MyIntAll(3)))
    add("udov float.__add__ override", lambda: float.__add__(MyFloat(1.5), 2))
    add("udov float.__neg__ override", lambda: float.__neg__(MyFloat(1.5)))
    add("udov str.upper override", lambda: str.upper(MyStr("abc")))
    add("udov list.__len__ override", lambda: list.__len__(MyList([1, 2])))
    add(
        "udov list.__getitem__ override", lambda: list.__getitem__(MyListAll([1, 2]), 0)
    )
    add("udov list.__iter__ override", lambda: list(list.__iter__(MyListAll([1, 2]))))
    add("udov list.__contains__ override", lambda: list.__contains__(MyListAll([1]), 5))
    add("udov list.__iadd__ override", lambda: list.__iadd__(MyListAll([1]), [2]))
    add("udov list.__eq__ override", lambda: list.__eq__(MyListAll([1]), [2]))
    add("udov list.append override", lambda: _udov_append(MyList()))
    add("udov dict.__getitem__ override", lambda: dict.__getitem__(MyDict({1: 2}), 1))
    add("udov set.add override", lambda: _udov_set_add(MySet()))
    # wrong receiver type: must raise TypeError like CPython
    add("udov int.__add__ wrong self", lambda: int.__add__(PlainList(), 2))
    add("udov list.__len__ wrong self", lambda: list.__len__(PlainInt(1)))
    add("udov dict.__len__ wrong self", lambda: dict.__len__(PlainList()))

    return c


def _udov_append(x):
    list.append(x, 1)
    return list(list.__iter__(x))


def _udov_set_add(x):
    set.add(x, 1)
    return set(x)


class _Plain:
    x = 3


class _HashesViaInt:
    def __hash__(self):
        return int.__hash__(3)


def _ba_append():
    b = bytearray(b"a")
    bytearray.append(b, 98)
    return bytes(b)


def _list_append():
    a = [1]
    list.append(a, 2)
    return a


def _mylist_append():
    a = MyList([1])
    list.append(a, 2)
    return list(a)


def _list_extend():
    a = [1]
    list.extend(a, [2, 3])
    return a


def _list_setitem():
    a = [1, 2]
    list.__setitem__(a, 0, 9)
    return a


def _list_delitem():
    a = [1, 2]
    list.__delitem__(a, 0)
    return a


def _list_iadd():
    a = [1]
    list.__iadd__(a, [2])
    return a


def _list_init():
    a = []
    list.__init__(a, [1, 2])
    return a


def _list_sort():
    a = [3, 1, 2]
    list.sort(a)
    return a


def _list_reverse():
    a = [1, 2, 3]
    list.reverse(a)
    return a


def _list_pop():
    a = [1, 2, 3]
    return list.pop(a), a


def _list_insert():
    a = [1, 3]
    list.insert(a, 1, 2)
    return a


def _set_add():
    s = {1}
    set.add(s, 2)
    return s


def _myset_add():
    s = MySet({1})
    set.add(s, 2)
    return set(s)


def _set_update():
    s = {1}
    set.update(s, [2, 3])
    return s


def _set_discard():
    s = {1, 2}
    set.discard(s, 2)
    return s


def _set_remove():
    s = {1, 2}
    set.remove(s, 2)
    return s


def _set_pop():
    s = {1}
    return set.pop(s), s


def _set_ior():
    s = {1}
    set.__ior__(s, {2})
    return s


def _set_init():
    s = set()
    set.__init__(s, [1, 2])
    return s


def _dict_setitem():
    d = {}
    dict.__setitem__(d, 1, 2)
    return d


def _dict_delitem():
    d = {1: 2}
    dict.__delitem__(d, 1)
    return d


def _dict_update():
    d = {1: 2}
    dict.update(d, {3: 4})
    return d


def _dict_pop():
    d = {1: 2}
    return dict.pop(d, 1), d


def _dict_setdefault():
    d = {}
    dict.setdefault(d, 1, 2)
    return d


def _dict_init():
    d = {}
    dict.__init__(d, a=1)
    return d


def _object_init():
    o = _Plain()
    object.__init__(o)
    return o.x


def _object_setattr():
    o = _Plain()
    object.__setattr__(o, "y", 5)
    return o.y


_CASES = _cases()


def _case_name(name):
    return re.sub(r"\W+", "_", name).strip("_")


@instantiate_parametrized_tests
class UnboundCallTests(torch._dynamo.test_case.TestCase):
    @parametrize("name", list(_CASES), name_fn=_case_name)
    def test_unbound_call(self, name):
        fn = _CASES[name]
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        try:
            expected = fn()
        except Exception as e:
            with self.assertRaises(type(e)):
                compiled()
            return
        self.assertEqual(compiled(), expected)


if __name__ == "__main__":
    run_tests()
