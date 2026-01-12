# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import collections
import collections.abc
import copy
import gc
import operator
import pickle
import unittest
import warnings
import weakref

from test import support
from torch.testing._internal.common_utils import TestCase
from torch.utils._ordered_set import OrderedSet


class PassThru(Exception):
    pass


def check_pass_thru():
    raise PassThru
    yield 1


class BadCmp:
    def __hash__(self):
        return 1

    def __eq__(self, other):
        raise RuntimeError


class ReprWrapper:
    "Used to test self-referential repr() calls"

    def __repr__(self):
        return repr(self.value)


class HashCountingInt(int):
    "int-like object that counts the number of times __hash__ is called"

    def __init__(self, *args):
        self.hash_count = 0

    def __hash__(self):
        self.hash_count += 1
        return int.__hash__(self)


class TestJointOps(TestCase):
    # Tests common to both OrderedSet and frozenset
    thetype = OrderedSet
    basetype = OrderedSet

    def setUp(self):
        super().setUp()
        self.word = word = "simsalabim"
        self.otherword = "madagascar"
        self.letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.s = self.thetype(word)
        self.d = dict.fromkeys(word)

    def test_new_or_init(self):
        self.assertRaises(TypeError, self.thetype, [], 2)
        self.assertRaises(TypeError, OrderedSet().__init__, a=1)

    def test_uniquification(self):
        actual = sorted(self.s)
        expected = sorted(self.d)
        self.assertEqual(actual, expected)
        self.assertRaises(PassThru, self.thetype, check_pass_thru())
        self.assertRaises(TypeError, self.thetype, [[]])

    def test_len(self):
        self.assertEqual(len(self.s), len(self.d))

    def test_contains(self):
        for c in self.letters:
            self.assertEqual(c in self.s, c in self.d)
        self.assertRaises(TypeError, self.s.__contains__, [[]])
        #
        # s = self.thetype([frozenset(self.letters)])
        # self.assertIn(self.thetype(self.letters), s)

    def test_union(self):
        u = self.s.union(self.otherword)
        for c in self.letters:
            self.assertEqual(c in u, c in self.d or c in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(u), self.basetype)
        self.assertRaises(PassThru, self.s.union, check_pass_thru())
        self.assertRaises(TypeError, self.s.union, [[]])
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(self.thetype("abcba").union(C("cdc")), OrderedSet("abcd"))
            self.assertEqual(
                self.thetype("abcba").union(C("efgfe")), OrderedSet("abcefg")
            )
            self.assertEqual(self.thetype("abcba").union(C("ccb")), OrderedSet("abc"))
            self.assertEqual(self.thetype("abcba").union(C("ef")), OrderedSet("abcef"))
            self.assertEqual(
                self.thetype("abcba").union(C("ef"), C("fg")), OrderedSet("abcefg")
            )

        # Issue #6573
        x = self.thetype()
        self.assertEqual(
            x.union(OrderedSet([1]), x, OrderedSet([2])), self.thetype([1, 2])
        )

    def test_or(self):
        i = self.s.union(self.otherword)
        self.assertEqual(self.s | OrderedSet(self.otherword), i)
        self.assertEqual(self.s | frozenset(self.otherword), i)
        try:
            self.s | self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s|t did not screen-out general iterables")

    def test_intersection(self):
        i = self.s.intersection(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, c in self.d and c in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.intersection, check_pass_thru())
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(
                self.thetype("abcba").intersection(C("cdc")), OrderedSet("cc")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("efgfe")), OrderedSet("")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("ccb")), OrderedSet("bc")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("ef")), OrderedSet("")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("cbcf"), C("bag")), OrderedSet("b")
            )
        s = self.thetype("abcba")
        z = s.intersection()
        if self.thetype == frozenset():
            self.assertEqual(id(s), id(z))
        else:
            self.assertNotEqual(id(s), id(z))

    def test_isdisjoint(self):
        def f(s1, s2):
            "Pure python equivalent of isdisjoint()"
            return not OrderedSet(s1).intersection(s2)

        for large in "", "a", "ab", "abc", "ababac", "cdc", "cc", "efgfe", "ccb", "ef":
            s1 = self.thetype(large)
            for rarg in (
                "",
                "a",
                "ab",
                "abc",
                "ababac",
                "cdc",
                "cc",
                "efgfe",
                "ccb",
                "ef",
            ):
                for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                    s2 = C(rarg)
                    actual = s1.isdisjoint(s2)
                    expected = f(s1, s2)
                    self.assertEqual(actual, expected)
                    self.assertTrue(actual is True or actual is False)

    def test_and(self):
        i = self.s.intersection(self.otherword)
        self.assertEqual(self.s & OrderedSet(self.otherword), i)
        self.assertEqual(self.s & frozenset(self.otherword), i)
        try:
            self.s & self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s&t did not screen-out general iterables")

    def test_difference(self):
        i = self.s.difference(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, c in self.d and c not in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.difference, check_pass_thru())
        self.assertRaises(TypeError, self.s.difference, [[]])
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(
                self.thetype("abcba").difference(C("cdc")), OrderedSet("ab")
            )
            self.assertEqual(
                self.thetype("abcba").difference(C("efgfe")), OrderedSet("abc")
            )
            self.assertEqual(
                self.thetype("abcba").difference(C("ccb")), OrderedSet("a")
            )
            self.assertEqual(
                self.thetype("abcba").difference(C("ef")), OrderedSet("abc")
            )
            self.assertEqual(self.thetype("abcba").difference(), OrderedSet("abc"))
            self.assertEqual(
                self.thetype("abcba").difference(C("a"), C("b")), OrderedSet("c")
            )

    def test_sub(self):
        i = self.s.difference(self.otherword)
        self.assertEqual(self.s - OrderedSet(self.otherword), i)
        self.assertEqual(self.s - frozenset(self.otherword), i)
        try:
            self.s - self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s-t did not screen-out general iterables")

    def test_symmetric_difference(self):
        i = self.s.symmetric_difference(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, (c in self.d) ^ (c in self.otherword))
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.symmetric_difference, check_pass_thru())
        self.assertRaises(TypeError, self.s.symmetric_difference, [[]])
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("cdc")),
                OrderedSet("abd"),  # codespell:ignore
            )
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("efgfe")),
                OrderedSet("abcefg"),
            )
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("ccb")), OrderedSet("a")
            )
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("ef")), OrderedSet("abcef")
            )

    def test_xor(self):
        i = self.s.symmetric_difference(self.otherword)
        self.assertEqual(self.s ^ OrderedSet(self.otherword), i)
        self.assertEqual(self.s ^ frozenset(self.otherword), i)
        try:
            self.s ^ self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s^t did not screen-out general iterables")

    def test_equality(self):
        self.assertEqual(self.s, OrderedSet(self.word))
        self.assertEqual(self.s, frozenset(self.word))
        self.assertEqual(self.s == self.word, False)
        self.assertNotEqual(self.s, OrderedSet(self.otherword))
        self.assertNotEqual(self.s, frozenset(self.otherword))
        self.assertEqual(self.s != self.word, True)

    def test_setOfFrozensets(self):
        t = map(frozenset, ["abcdef", "bcd", "bdcb", "fed", "fedccba"])
        s = self.thetype(t)
        self.assertEqual(len(s), 3)

    def test_sub_and_super(self):
        p, q, r = map(self.thetype, ["ab", "abcde", "def"])
        self.assertTrue(p < q)
        self.assertTrue(p <= q)
        self.assertTrue(q <= q)
        self.assertTrue(q > p)
        self.assertTrue(q >= p)
        self.assertFalse(q < r)
        self.assertFalse(q <= r)
        self.assertFalse(q > r)
        self.assertFalse(q >= r)
        self.assertTrue(OrderedSet("a").issubset("abc"))
        self.assertTrue(OrderedSet("abc").issuperset("a"))
        self.assertFalse(OrderedSet("a").issubset("cbs"))
        self.assertFalse(OrderedSet("cbs").issuperset("a"))

    def test_pickling(self):
        for i in range(pickle.HIGHEST_PROTOCOL + 1):
            if type(self.s) not in (OrderedSet, frozenset):
                self.s.x = ["x"]
                self.s.z = ["z"]
            p = pickle.dumps(self.s, i)
            dup = pickle.loads(p)
            self.assertEqual(self.s, dup, "%s != %s" % (self.s, dup))  # noqa: UP031
            if type(self.s) not in (OrderedSet, frozenset):
                self.assertEqual(self.s.x, dup.x)
                self.assertEqual(self.s.z, dup.z)
                self.assertFalse(hasattr(self.s, "y"))
                del self.s.x, self.s.z

    @unittest.skip("Pickling nyi")
    def test_iterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            itorg = iter(self.s)
            data = self.thetype(self.s)
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            # Set iterators unpickle as list iterators due to the
            # undefined order of OrderedSet items.
            # self.assertEqual(type(itorg), type(it))
            self.assertIsInstance(it, collections.abc.Iterator)
            self.assertEqual(self.thetype(it), data)

            it = pickle.loads(d)
            try:
                drop = next(it)
            except StopIteration:
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(self.thetype(it), data - self.thetype((drop,)))

    def test_deepcopy(self):
        class Tracer:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return self.value

            def __deepcopy__(self, memo=None):
                return Tracer(self.value + 1)

        t = Tracer(10)
        s = self.thetype([t])
        dup = copy.deepcopy(s)
        self.assertNotEqual(id(s), id(dup))
        for elem in dup:
            newt = elem
        self.assertNotEqual(id(t), id(newt))
        self.assertEqual(t.value + 1, newt.value)

    def test_gc(self):
        # Create a nest of cycles to exercise overall ref count check
        class A:
            pass

        s = OrderedSet(A() for i in range(1000))
        for elem in s:
            elem.cycle = s
            elem.sub = elem
            elem.OrderedSet = OrderedSet([elem])

    def test_subclass_with_custom_hash(self):
        # Bug #1257731
        class H(self.thetype):
            def __hash__(self):
                return int(id(self) & 0x7FFFFFFF)

        s = H()
        f = OrderedSet()
        f.add(s)
        self.assertIn(s, f)
        f.remove(s)
        f.add(s)
        f.discard(s)

    def test_badcmp(self):
        s = self.thetype([BadCmp()])
        # Detect comparison errors during insertion and lookup
        self.assertRaises(RuntimeError, self.thetype, [BadCmp(), BadCmp()])
        self.assertRaises(RuntimeError, s.__contains__, BadCmp())
        # Detect errors during mutating operations
        if hasattr(s, "add"):
            self.assertRaises(RuntimeError, s.add, BadCmp())
            self.assertRaises(RuntimeError, s.discard, BadCmp())
            self.assertRaises(RuntimeError, s.remove, BadCmp())

    @unittest.skip("Different repr")
    def test_cyclical_repr(self):
        w = ReprWrapper()
        s = self.thetype([w])
        w.value = s
        if self.thetype == OrderedSet:
            self.assertEqual(repr(s), "{OrderedSet(...)}")
        else:
            name = repr(s).partition("(")[0]  # strip class name
            self.assertEqual(repr(s), "%s({%s(...)})" % (name, name))  # noqa: UP031

    @unittest.skip("Different hashing")
    def test_do_not_rehash_dict_keys(self):
        n = 10
        d = dict.fromkeys(map(HashCountingInt, range(n)))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        s = self.thetype(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        s.difference(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        if hasattr(s, "symmetric_difference_update"):
            s.symmetric_difference_update(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d2 = dict.fromkeys(OrderedSet(d))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d3 = dict.fromkeys(frozenset(d))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d3 = dict.fromkeys(frozenset(d), 123)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        self.assertEqual(d3, dict.fromkeys(d, 123))

    def test_container_iterator(self):
        # Bug #3680: tp_traverse was not implemented for OrderedSet iterator object
        class C:
            pass

        obj = C()
        ref = weakref.ref(obj)
        container = OrderedSet([obj, 1])
        obj.x = iter(container)
        del obj, container
        gc.collect()
        self.assertTrue(ref() is None, "Cycle was not collected")

    def test_free_after_iterating(self):
        support.check_free_after_iterating(self, iter, self.thetype)


class TestSet(TestJointOps, TestCase):
    thetype = OrderedSet
    basetype = OrderedSet

    def test_init(self):
        s = self.thetype()
        s.__init__(self.word)
        self.assertEqual(s, OrderedSet(self.word))
        s.__init__(self.otherword)
        self.assertEqual(s, OrderedSet(self.otherword))
        self.assertRaises(TypeError, s.__init__, s, 2)
        self.assertRaises(TypeError, s.__init__, 1)

    def test_constructor_identity(self):
        s = self.thetype(range(3))
        t = self.thetype(s)
        self.assertNotEqual(id(s), id(t))

    def test_set_literal(self):
        s = OrderedSet([1, 2, 3])
        t = {1, 2, 3}
        self.assertEqual(s, t)

    def test_set_literal_insertion_order(self):
        # SF Issue #26020 -- Expect left to right insertion
        s = {1, 1.0, True}  # noqa: B033
        self.assertEqual(len(s), 1)
        stored_value = s.pop()
        self.assertEqual(type(stored_value), int)

    def test_set_literal_evaluation_order(self):
        # Expect left to right expression evaluation
        events = []

        def record(obj):
            events.append(obj)

        s = {record(1), record(2), record(3)}
        self.assertEqual(events, [1, 2, 3])

    def test_hash(self):
        self.assertRaises(TypeError, hash, self.s)

    def test_clear(self):
        self.s.clear()
        self.assertEqual(self.s, OrderedSet())
        self.assertEqual(len(self.s), 0)

    def test_copy(self):
        dup = self.s.copy()
        self.assertEqual(self.s, dup)
        self.assertNotEqual(id(self.s), id(dup))
        self.assertEqual(type(dup), self.basetype)

    def test_add(self):
        self.s.add("Q")
        self.assertIn("Q", self.s)
        dup = self.s.copy()
        self.s.add("Q")
        self.assertEqual(self.s, dup)
        self.assertRaises(TypeError, self.s.add, [])

    def test_remove(self):
        self.s.remove("a")
        self.assertNotIn("a", self.s)
        self.assertRaises(KeyError, self.s.remove, "Q")
        self.assertRaises(TypeError, self.s.remove, [])
        # NYI: __as_immutable__
        # s = self.thetype([frozenset(self.word)])
        # self.assertIn(self.thetype(self.word), s)
        # s.remove(self.thetype(self.word))
        # self.assertNotIn(self.thetype(self.word), s)
        # self.assertRaises(KeyError, self.s.remove, self.thetype(self.word))

    def test_remove_keyerror_unpacking(self):
        # https://bugs.python.org/issue1576657
        for v1 in ["Q", (1,)]:
            try:
                self.s.remove(v1)
            except KeyError as e:
                v2 = e.args[0]
                self.assertEqual(v1, v2)
            else:
                self.fail()

    def test_remove_keyerror_set(self):
        key = self.thetype([3, 4])
        try:
            self.s.remove(key)
        except Exception:
            pass
            # self.assertTrue(e.args[0] is key,
            #              "KeyError should be {0}, not {1}".format(key,
            #                                                       e.args[0]))
        else:
            self.fail()

    def test_discard(self):
        self.s.discard("a")
        self.assertNotIn("a", self.s)
        self.s.discard("Q")
        self.assertRaises(TypeError, self.s.discard, [])
        # NYI: __as_immutable__
        # s = self.thetype([frozenset(self.word)])
        # self.assertIn(self.thetype(self.word), s)
        # s.discard(self.thetype(self.word))
        # self.assertNotIn(self.thetype(self.word), s)
        # s.discard(self.thetype(self.word))

    def test_pop(self):
        for _ in range(len(self.s)):
            elem = self.s.pop()
            self.assertNotIn(elem, self.s)
        self.assertRaises(KeyError, self.s.pop)

    def test_update(self):
        retval = self.s.update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            self.assertIn(c, self.s)
        self.assertRaises(PassThru, self.s.update, check_pass_thru())
        self.assertRaises(TypeError, self.s.update, [[]])
        for p, q in (
            ("cdc", "abcd"),
            ("efgfe", "abcefg"),
            ("ccb", "abc"),
            ("ef", "abcef"),
        ):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))
        for p in ("cdc", "efgfe", "ccb", "ef", "abcda"):
            q = "ahi"
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.update(C(p), C(q)), None)
                self.assertEqual(s, OrderedSet(s) | OrderedSet(p) | OrderedSet(q))

    def test_ior(self):
        self.s |= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            self.assertIn(c, self.s)

    def test_intersection_update(self):
        retval = self.s.intersection_update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            if c in self.otherword and c in self.word:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(PassThru, self.s.intersection_update, check_pass_thru())
        self.assertRaises(TypeError, self.s.intersection_update, [[]])
        for p, q in (("cdc", "c"), ("efgfe", ""), ("ccb", "bc"), ("ef", "")):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.intersection_update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))
                ss = "abcba"
                s = self.thetype(ss)
                t = "cbc"
                self.assertEqual(s.intersection_update(C(p), C(t)), None)
                self.assertEqual(s, OrderedSet("abcba") & OrderedSet(p) & OrderedSet(t))

    def test_iand(self):
        self.s &= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            if c in self.otherword and c in self.word:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_difference_update(self):
        retval = self.s.difference_update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            if c in self.word and c not in self.otherword:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(PassThru, self.s.difference_update, check_pass_thru())
        self.assertRaises(TypeError, self.s.difference_update, [[]])
        self.assertRaises(TypeError, self.s.symmetric_difference_update, [[]])
        for p, q in (("cdc", "ab"), ("efgfe", "abc"), ("ccb", "a"), ("ef", "abc")):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.difference_update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))

                s = self.thetype("abcdefghih")
                s.difference_update()
                self.assertEqual(s, self.thetype("abcdefghih"))

                s = self.thetype("abcdefghih")
                s.difference_update(C("aba"))
                self.assertEqual(s, self.thetype("cdefghih"))

                s = self.thetype("abcdefghih")
                s.difference_update(C("cdc"), C("aba"))
                self.assertEqual(s, self.thetype("efghih"))

    def test_isub(self):
        self.s -= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            if c in self.word and c not in self.otherword:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_symmetric_difference_update(self):
        retval = self.s.symmetric_difference_update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            if (c in self.word) ^ (c in self.otherword):
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(
            PassThru, self.s.symmetric_difference_update, check_pass_thru()
        )
        self.assertRaises(TypeError, self.s.symmetric_difference_update, [[]])
        for p, q in (
            ("cdc", "abd"),  # codespell:ignore
            ("efgfe", "abcefg"),
            ("ccb", "a"),
            ("ef", "abcef"),
        ):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.symmetric_difference_update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))

    def test_ixor(self):
        self.s ^= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            if (c in self.word) ^ (c in self.otherword):
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_inplace_on_self(self):
        t = self.s.copy()
        t |= t
        self.assertEqual(t, self.s)
        t &= t
        self.assertEqual(t, self.s)
        t -= t
        self.assertEqual(t, self.thetype())
        t = self.s.copy()
        t ^= t
        self.assertEqual(t, self.thetype())

    @unittest.skip("Slots interferes with weakrefs")
    def test_weakref(self):
        s = self.thetype("gallahad")
        p = weakref.proxy(s)
        self.assertEqual(str(p), str(s))
        s = None
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, str, p)

    def test_rich_compare(self):
        class TestRichSetCompare:
            def __gt__(self, some_set):
                self.gt_called = True
                return False

            def __lt__(self, some_set):
                self.lt_called = True
                return False

            def __ge__(self, some_set):
                self.ge_called = True
                return False

            def __le__(self, some_set):
                self.le_called = True
                return False

        # This first tries the builtin rich OrderedSet comparison, which doesn't know
        # how to handle the custom object. Upon returning NotImplemented, the
        # corresponding comparison on the right object is invoked.
        myset = {1, 2, 3}

        myobj = TestRichSetCompare()
        myset < myobj  # noqa: B015
        self.assertTrue(myobj.gt_called)

        myobj = TestRichSetCompare()
        myset > myobj  # noqa: B015
        self.assertTrue(myobj.lt_called)

        myobj = TestRichSetCompare()
        myset <= myobj  # noqa: B015
        self.assertTrue(myobj.ge_called)

        myobj = TestRichSetCompare()
        myset >= myobj  # noqa: B015
        self.assertTrue(myobj.le_called)


# Tests taken from test_sets.py =============================================

empty_set = OrderedSet()

# ==============================================================================


class TestBasicOps(TestCase):
    @unittest.skip("Different repr")
    def test_repr(self):
        if self.repr is not None:
            self.assertEqual(repr(self.OrderedSet), self.repr)

    def check_repr_against_values(self):
        text = repr(self.OrderedSet)
        self.assertTrue(text.startswith("{"))
        self.assertTrue(text.endswith("}"))

        result = text[1:-1].split(", ")
        result.sort()
        sorted_repr_values = [repr(value) for value in self.values]
        sorted_repr_values.sort()
        self.assertEqual(result, sorted_repr_values)

    def test_length(self):
        self.assertEqual(len(self.OrderedSet), self.length)

    def test_self_equality(self):
        self.assertEqual(self.OrderedSet, self.OrderedSet)

    def test_equivalent_equality(self):
        self.assertEqual(self.OrderedSet, self.dup)

    def test_copy(self):
        self.assertEqual(self.OrderedSet.copy(), self.dup)

    def test_self_union(self):
        result = self.OrderedSet | self.OrderedSet
        self.assertEqual(result, self.dup)

    def test_empty_union(self):
        result = self.OrderedSet | empty_set
        self.assertEqual(result, self.dup)

    def test_union_empty(self):
        result = empty_set | self.OrderedSet
        self.assertEqual(result, self.dup)

    def test_self_intersection(self):
        result = self.OrderedSet & self.OrderedSet
        self.assertEqual(result, self.dup)

    def test_empty_intersection(self):
        result = self.OrderedSet & empty_set
        self.assertEqual(result, empty_set)

    def test_intersection_empty(self):
        result = empty_set & self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_self_isdisjoint(self):
        result = self.OrderedSet.isdisjoint(self.OrderedSet)
        self.assertEqual(result, not self.OrderedSet)

    def test_empty_isdisjoint(self):
        result = self.OrderedSet.isdisjoint(empty_set)
        self.assertEqual(result, True)

    def test_isdisjoint_empty(self):
        result = empty_set.isdisjoint(self.OrderedSet)
        self.assertEqual(result, True)

    def test_self_symmetric_difference(self):
        result = self.OrderedSet ^ self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_empty_symmetric_difference(self):
        result = self.OrderedSet ^ empty_set
        self.assertEqual(result, self.OrderedSet)

    def test_self_difference(self):
        result = self.OrderedSet - self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_empty_difference(self):
        result = self.OrderedSet - empty_set
        self.assertEqual(result, self.dup)

    def test_empty_difference_rev(self):
        result = empty_set - self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_iteration(self):
        for v in self.OrderedSet:
            self.assertIn(v, self.values)
        # setiter = iter(self.OrderedSet)
        # self.assertEqual(setiter.__length_hint__(), len(self.OrderedSet))

    def test_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            p = pickle.dumps(self.OrderedSet, proto)
            copy = pickle.loads(p)
            self.assertEqual(
                self.OrderedSet,
                copy,
                "%s != %s" % (self.OrderedSet, copy),  # noqa: UP031
            )

    def test_issue_37219(self):
        with self.assertRaises(TypeError):
            OrderedSet().difference(123)
        with self.assertRaises(TypeError):
            OrderedSet().difference_update(123)


# ------------------------------------------------------------------------------


class TestBasicOpsEmpty(TestBasicOps, TestCase):
    def setUp(self):
        super().setUp()
        self.case = "empty OrderedSet"
        self.values = []
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 0
        self.repr = "OrderedSet()"


# ------------------------------------------------------------------------------


class TestBasicOpsSingleton(TestBasicOps, TestCase):
    def setUp(self):
        super().setUp()
        self.case = "unit OrderedSet (number)"
        self.values = [3]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 1
        self.repr = "{3}"

    def test_in(self):
        self.assertIn(3, self.OrderedSet)

    def test_not_in(self):
        self.assertNotIn(2, self.OrderedSet)


# ------------------------------------------------------------------------------


class TestBasicOpsTuple(TestBasicOps, TestCase):
    def setUp(self):
        super().setUp()
        self.case = "unit OrderedSet (tuple)"
        self.values = [(0, "zero")]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 1
        self.repr = "{(0, 'zero')}"

    def test_in(self):
        self.assertIn((0, "zero"), self.OrderedSet)

    def test_not_in(self):
        self.assertNotIn(9, self.OrderedSet)


# ------------------------------------------------------------------------------


class TestBasicOpsTriple(TestBasicOps, TestCase):
    def setUp(self):
        super().setUp()
        self.case = "triple OrderedSet"
        self.values = [0, "zero", operator.add]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 3
        self.repr = None


# ------------------------------------------------------------------------------


class TestBasicOpsString(TestBasicOps, TestCase):
    def setUp(self):
        super().setUp()
        self.case = "string OrderedSet"
        self.values = ["a", "b", "c"]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 3

    @unittest.skip("Different repr")
    def test_repr(self):
        self.check_repr_against_values()


# ------------------------------------------------------------------------------


class TestBasicOpsBytes(TestBasicOps, TestCase):
    def setUp(self):
        super().setUp()
        self.case = "bytes OrderedSet"
        self.values = [b"a", b"b", b"c"]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 3

    @unittest.skip("Different repr")
    def test_repr(self):
        self.check_repr_against_values()


# ------------------------------------------------------------------------------


class TestBasicOpsMixedStringBytes(TestBasicOps, TestCase):
    def setUp(self):
        super().setUp()
        warnings.simplefilter("ignore", BytesWarning)
        self.case = "string and bytes OrderedSet"
        self.values = ["a", "b", b"a", b"b"]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 4

    @unittest.skip("Different repr")
    def test_repr(self):
        self.check_repr_against_values()


del TestBasicOps
# ==============================================================================


def baditer():
    raise TypeError
    yield True


def gooditer():
    yield True


class TestExceptionPropagation(TestCase):
    """SF 628246:  Set constructor should not trap iterator TypeErrors"""

    def test_instanceWithException(self):
        self.assertRaises(TypeError, OrderedSet, baditer())

    def test_instancesWithoutException(self):
        # All of these iterables should load without exception.
        OrderedSet([1, 2, 3])
        OrderedSet((1, 2, 3))
        OrderedSet({"one": 1, "two": 2, "three": 3})
        OrderedSet(range(3))
        OrderedSet("abc")
        OrderedSet(gooditer())

    def test_changingSizeWhileIterating(self):
        s = OrderedSet([1, 2, 3])
        try:
            for _ in s:
                s.update([4])  # noqa: B909
        except RuntimeError:
            pass
        else:
            self.fail("no exception when changing size during iteration")


# ==============================================================================


class TestSetOfSets(TestCase):
    def test_constructor(self):
        inner = frozenset([1])
        outer = OrderedSet([inner])
        element = outer.pop()
        self.assertEqual(type(element), frozenset)
        outer.add(inner)  # Rebuild OrderedSet of sets with .add method
        outer.remove(inner)
        self.assertEqual(outer, OrderedSet())  # Verify that remove worked
        outer.discard(inner)  # Absence of KeyError indicates working fine


# ==============================================================================


class TestBinaryOps(TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet((2, 4, 6))

    def test_eq(self):  # SF bug 643115
        self.assertEqual(self.OrderedSet, OrderedSet({2: 1, 4: 3, 6: 5}))

    def test_union_subset(self):
        result = self.OrderedSet | OrderedSet([2])
        self.assertEqual(result, OrderedSet((2, 4, 6)))

    def test_union_superset(self):
        result = self.OrderedSet | OrderedSet([2, 4, 6, 8])
        self.assertEqual(result, OrderedSet([2, 4, 6, 8]))

    def test_union_overlap(self):
        result = self.OrderedSet | OrderedSet([3, 4, 5])
        self.assertEqual(result, OrderedSet([2, 3, 4, 5, 6]))

    def test_union_non_overlap(self):
        result = self.OrderedSet | OrderedSet([8])
        self.assertEqual(result, OrderedSet([2, 4, 6, 8]))

    def test_intersection_subset(self):
        result = self.OrderedSet & OrderedSet((2, 4))
        self.assertEqual(result, OrderedSet((2, 4)))

    def test_intersection_superset(self):
        result = self.OrderedSet & OrderedSet([2, 4, 6, 8])
        self.assertEqual(result, OrderedSet([2, 4, 6]))

    def test_intersection_overlap(self):
        result = self.OrderedSet & OrderedSet([3, 4, 5])
        self.assertEqual(result, OrderedSet([4]))

    def test_intersection_non_overlap(self):
        result = self.OrderedSet & OrderedSet([8])
        self.assertEqual(result, empty_set)

    def test_isdisjoint_subset(self):
        result = self.OrderedSet.isdisjoint(OrderedSet((2, 4)))
        self.assertEqual(result, False)

    def test_isdisjoint_superset(self):
        result = self.OrderedSet.isdisjoint(OrderedSet([2, 4, 6, 8]))
        self.assertEqual(result, False)

    def test_isdisjoint_overlap(self):
        result = self.OrderedSet.isdisjoint(OrderedSet([3, 4, 5]))
        self.assertEqual(result, False)

    def test_isdisjoint_non_overlap(self):
        result = self.OrderedSet.isdisjoint(OrderedSet([8]))
        self.assertEqual(result, True)

    def test_sym_difference_subset(self):
        result = self.OrderedSet ^ OrderedSet((2, 4))
        self.assertEqual(result, OrderedSet([6]))

    def test_sym_difference_superset(self):
        result = self.OrderedSet ^ OrderedSet((2, 4, 6, 8))
        self.assertEqual(result, OrderedSet([8]))

    def test_sym_difference_overlap(self):
        result = self.OrderedSet ^ OrderedSet((3, 4, 5))
        self.assertEqual(result, OrderedSet([2, 3, 5, 6]))

    def test_sym_difference_non_overlap(self):
        result = self.OrderedSet ^ OrderedSet([8])
        self.assertEqual(result, OrderedSet([2, 4, 6, 8]))


# ==============================================================================


class TestUpdateOps(TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet((2, 4, 6))

    def test_union_subset(self):
        self.OrderedSet |= OrderedSet([2])
        self.assertEqual(self.OrderedSet, OrderedSet((2, 4, 6)))

    def test_union_superset(self):
        self.OrderedSet |= OrderedSet([2, 4, 6, 8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6, 8]))

    def test_union_overlap(self):
        self.OrderedSet |= OrderedSet([3, 4, 5])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 4, 5, 6]))

    def test_union_non_overlap(self):
        self.OrderedSet |= OrderedSet([8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6, 8]))

    def test_union_method_call(self):
        self.OrderedSet.update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 4, 5, 6]))

    def test_intersection_subset(self):
        self.OrderedSet &= OrderedSet((2, 4))
        self.assertEqual(self.OrderedSet, OrderedSet((2, 4)))

    def test_intersection_superset(self):
        self.OrderedSet &= OrderedSet([2, 4, 6, 8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6]))

    def test_intersection_overlap(self):
        self.OrderedSet &= OrderedSet([3, 4, 5])
        self.assertEqual(self.OrderedSet, OrderedSet([4]))

    def test_intersection_non_overlap(self):
        self.OrderedSet &= OrderedSet([8])
        self.assertEqual(self.OrderedSet, empty_set)

    def test_intersection_method_call(self):
        self.OrderedSet.intersection_update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([4]))

    def test_sym_difference_subset(self):
        self.OrderedSet ^= OrderedSet((2, 4))
        self.assertEqual(self.OrderedSet, OrderedSet([6]))

    def test_sym_difference_superset(self):
        self.OrderedSet ^= OrderedSet((2, 4, 6, 8))
        self.assertEqual(self.OrderedSet, OrderedSet([8]))

    def test_sym_difference_overlap(self):
        self.OrderedSet ^= OrderedSet((3, 4, 5))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 5, 6]))

    def test_sym_difference_non_overlap(self):
        self.OrderedSet ^= OrderedSet([8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6, 8]))

    def test_sym_difference_method_call(self):
        self.OrderedSet.symmetric_difference_update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 5, 6]))

    def test_difference_subset(self):
        self.OrderedSet -= OrderedSet((2, 4))
        self.assertEqual(self.OrderedSet, OrderedSet([6]))

    def test_difference_superset(self):
        self.OrderedSet -= OrderedSet((2, 4, 6, 8))
        self.assertEqual(self.OrderedSet, OrderedSet([]))

    def test_difference_overlap(self):
        self.OrderedSet -= OrderedSet((3, 4, 5))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 6]))

    def test_difference_non_overlap(self):
        self.OrderedSet -= OrderedSet([8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6]))

    def test_difference_method_call(self):
        self.OrderedSet.difference_update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 6]))


# ==============================================================================


class TestMutate(TestCase):
    def setUp(self):
        super().setUp()
        self.values = ["a", "b", "c"]
        self.OrderedSet = OrderedSet(self.values)

    def test_add_present(self):
        self.OrderedSet.add("c")
        self.assertEqual(self.OrderedSet, OrderedSet("abc"))

    def test_add_absent(self):
        self.OrderedSet.add("d")
        self.assertEqual(self.OrderedSet, OrderedSet("abcd"))

    def test_add_until_full(self):
        tmp = OrderedSet()
        expected_len = 0
        for v in self.values:
            tmp.add(v)
            expected_len += 1  # noqa: SIM113
            self.assertEqual(len(tmp), expected_len)
        self.assertEqual(tmp, self.OrderedSet)

    def test_remove_present(self):
        self.OrderedSet.remove("b")
        self.assertEqual(self.OrderedSet, OrderedSet("ac"))

    def test_remove_absent(self):
        try:
            self.OrderedSet.remove("d")
            self.fail("Removing missing element should have raised LookupError")
        except LookupError:
            pass

    def test_remove_until_empty(self):
        expected_len = len(self.OrderedSet)
        for v in self.values:
            self.OrderedSet.remove(v)
            expected_len -= 1
            self.assertEqual(len(self.OrderedSet), expected_len)

    def test_discard_present(self):
        self.OrderedSet.discard("c")
        self.assertEqual(self.OrderedSet, OrderedSet("ab"))

    def test_discard_absent(self):
        self.OrderedSet.discard("d")
        self.assertEqual(self.OrderedSet, OrderedSet("abc"))

    def test_clear(self):
        self.OrderedSet.clear()
        self.assertEqual(len(self.OrderedSet), 0)

    def test_pop(self):
        popped = {}
        while self.OrderedSet:
            popped[self.OrderedSet.pop()] = None
        self.assertEqual(len(popped), len(self.values))
        for v in self.values:
            self.assertIn(v, popped)

    def test_update_empty_tuple(self):
        self.OrderedSet.update(())
        self.assertEqual(self.OrderedSet, OrderedSet(self.values))

    def test_update_unit_tuple_overlap(self):
        self.OrderedSet.update(("a",))
        self.assertEqual(self.OrderedSet, OrderedSet(self.values))

    def test_update_unit_tuple_non_overlap(self):
        self.OrderedSet.update(("a", "z"))
        self.assertEqual(self.OrderedSet, OrderedSet(self.values + ["z"]))


# ==============================================================================


class TestSubsets(TestCase):
    case2method = {
        "<=": "issubset",
        ">=": "issuperset",
    }

    reverse = {
        "==": "==",
        "!=": "!=",
        "<": ">",
        ">": "<",
        "<=": ">=",
        ">=": "<=",
    }

    def test_issubset(self):
        if type(self) is TestSubsets:
            raise unittest.SkipTest("Only meant to be run as subclass")
        x = self.left
        y = self.right
        for case in "!=", "==", "<", "<=", ">", ">=":
            expected = case in self.cases
            # Test the binary infix spelling.
            result = eval("x" + case + "y", locals())
            self.assertEqual(result, expected)
            # Test the "friendly" method-name spelling, if one exists.
            if case in TestSubsets.case2method:
                method = getattr(x, TestSubsets.case2method[case])
                result = method(y)
                self.assertEqual(result, expected)

            # Now do the same for the operands reversed.
            rcase = TestSubsets.reverse[case]
            result = eval("y" + rcase + "x", locals())
            self.assertEqual(result, expected)
            if rcase in TestSubsets.case2method:
                method = getattr(y, TestSubsets.case2method[rcase])
                result = method(x)
                self.assertEqual(result, expected)


# ------------------------------------------------------------------------------


class TestSubsetEqualEmpty(TestSubsets, TestCase):
    left = OrderedSet()
    right = OrderedSet()
    name = "both empty"
    cases = "==", "<=", ">="


# ------------------------------------------------------------------------------


class TestSubsetEqualNonEmpty(TestSubsets, TestCase):
    left = OrderedSet([1, 2])
    right = OrderedSet([1, 2])
    name = "equal pair"
    cases = "==", "<=", ">="


# ------------------------------------------------------------------------------


class TestSubsetEmptyNonEmpty(TestSubsets, TestCase):
    left = OrderedSet()
    right = OrderedSet([1, 2])
    name = "one empty, one non-empty"
    cases = "!=", "<", "<="


# ------------------------------------------------------------------------------


class TestSubsetPartial(TestSubsets, TestCase):
    left = OrderedSet([1])
    right = OrderedSet([1, 2])
    name = "one a non-empty proper subset of other"
    cases = "!=", "<", "<="


# ------------------------------------------------------------------------------


class TestSubsetNonOverlap(TestSubsets, TestCase):
    left = OrderedSet([1])
    right = OrderedSet([2])
    name = "neither empty, neither contains"
    cases = "!="


# ==============================================================================


class TestOnlySetsInBinaryOps(TestCase):
    def test_eq_ne(self):
        # Unlike the others, this is testing that == and != *are* allowed.
        self.assertEqual(self.other == self.OrderedSet, False)
        self.assertEqual(self.OrderedSet == self.other, False)
        self.assertEqual(self.other != self.OrderedSet, True)
        self.assertEqual(self.OrderedSet != self.other, True)

    def test_ge_gt_le_lt(self):
        pass
        # self.assertRaises(TypeError, lambda: self.OrderedSet < self.other)
        # self.assertRaises(TypeError, lambda: self.OrderedSet <= self.other)
        # self.assertRaises(TypeError, lambda: self.OrderedSet > self.other)
        # self.assertRaises(TypeError, lambda: self.OrderedSet >= self.other)

        # self.assertRaises(TypeError, lambda: self.other < self.OrderedSet)
        # self.assertRaises(TypeError, lambda: self.other <= self.OrderedSet)
        # self.assertRaises(TypeError, lambda: self.other > self.OrderedSet)
        # self.assertRaises(TypeError, lambda: self.other >= self.OrderedSet)

    def test_update_operator(self):
        try:
            self.OrderedSet |= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_update(self):
        if self.otherIsIterable:
            self.OrderedSet.update(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.update, self.other)

    def test_union(self):
        # self.assertRaises(TypeError, lambda: self.OrderedSet | self.other)
        # self.assertRaises(TypeError, lambda: self.other | self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.union(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.union, self.other)

    def test_intersection_update_operator(self):
        try:
            self.OrderedSet &= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_intersection_update(self):
        if self.otherIsIterable:
            self.OrderedSet.intersection_update(self.other)
        else:
            self.assertRaises(
                TypeError, self.OrderedSet.intersection_update, self.other
            )

    def test_intersection(self):
        # self.assertRaises(TypeError, lambda: self.OrderedSet & self.other)
        # self.assertRaises(TypeError, lambda: self.other & self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.intersection(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.intersection, self.other)

    def test_sym_difference_update_operator(self):
        try:
            self.OrderedSet ^= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_sym_difference_update(self):
        if self.otherIsIterable:
            self.OrderedSet.symmetric_difference_update(self.other)
        else:
            self.assertRaises(
                TypeError, self.OrderedSet.symmetric_difference_update, self.other
            )

    def test_sym_difference(self):
        # self.assertRaises(TypeError, lambda: self.OrderedSet ^ self.other)
        # self.assertRaises(TypeError, lambda: self.other ^ self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.symmetric_difference(self.other)
        else:
            self.assertRaises(
                TypeError, self.OrderedSet.symmetric_difference, self.other
            )

    def test_difference_update_operator(self):
        try:
            self.OrderedSet -= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_difference_update(self):
        if self.otherIsIterable:
            self.OrderedSet.difference_update(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.difference_update, self.other)

    def test_difference(self):
        # self.assertRaises(TypeError, lambda: self.OrderedSet - self.other)
        # self.assertRaises(TypeError, lambda: self.other - self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.difference(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.difference, self.other)


# ------------------------------------------------------------------------------


class TestOnlySetsNumeric(TestOnlySetsInBinaryOps, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = 19
        self.otherIsIterable = False


# ------------------------------------------------------------------------------


class TestOnlySetsDict(TestOnlySetsInBinaryOps, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = {1: 2, 3: 4}
        self.otherIsIterable = True


# ------------------------------------------------------------------------------


class TestOnlySetsOperator(TestOnlySetsInBinaryOps, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = operator.add
        self.otherIsIterable = False


# ------------------------------------------------------------------------------


class TestOnlySetsTuple(TestOnlySetsInBinaryOps, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = (2, 4, 6)
        self.otherIsIterable = True


# ------------------------------------------------------------------------------


class TestOnlySetsString(TestOnlySetsInBinaryOps, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = "abc"
        self.otherIsIterable = True


# ------------------------------------------------------------------------------


class TestOnlySetsGenerator(TestOnlySetsInBinaryOps, TestCase):
    def setUp(self):
        super().setUp()

        def gen():
            for i in range(0, 10, 2):  # noqa: UP028
                yield i

        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = gen()
        self.otherIsIterable = True


del TestOnlySetsInBinaryOps
# ==============================================================================


class TestCopying:
    def test_copy(self):
        dup = self.OrderedSet.copy()
        dup_list = sorted(dup, key=repr)
        set_list = sorted(self.OrderedSet, key=repr)
        self.assertEqual(len(dup_list), len(set_list))
        for i in range(len(dup_list)):
            self.assertTrue(dup_list[i] is set_list[i])

    def test_deep_copy(self):
        dup = copy.deepcopy(self.OrderedSet)
        # print type(dup), repr(dup)
        dup_list = sorted(dup, key=repr)
        set_list = sorted(self.OrderedSet, key=repr)
        self.assertEqual(len(dup_list), len(set_list))
        for i in range(len(dup_list)):
            self.assertEqual(dup_list[i], set_list[i])


# ------------------------------------------------------------------------------


class TestCopyingEmpty(TestCopying, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet()


# ------------------------------------------------------------------------------


class TestCopyingSingleton(TestCopying, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet(["hello"])


# ------------------------------------------------------------------------------


class TestCopyingTriple(TestCopying, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet(["zero", 0, None])


# ------------------------------------------------------------------------------


class TestCopyingTuple(TestCopying, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet([(1, 2)])


# ------------------------------------------------------------------------------


class TestCopyingNested(TestCopying, TestCase):
    def setUp(self):
        super().setUp()
        self.OrderedSet = OrderedSet([((1, 2), (3, 4))])


del TestCopying

# ==============================================================================


class TestIdentities(TestCase):
    def setUp(self):
        super().setUp()
        self.a = OrderedSet("abracadabra")
        self.b = OrderedSet("alacazam")

    def test_binopsVsSubsets(self):
        a, b = self.a, self.b
        self.assertTrue(a - b < a)
        self.assertTrue(b - a < b)
        self.assertTrue(a & b < a)
        self.assertTrue(a & b < b)
        self.assertTrue(a | b > a)
        self.assertTrue(a | b > b)
        self.assertTrue(a ^ b < a | b)

    def test_commutativity(self):
        a, b = self.a, self.b
        self.assertEqual(a & b, b & a)
        self.assertEqual(a | b, b | a)
        self.assertEqual(a ^ b, b ^ a)
        if a != b:
            self.assertNotEqual(a - b, b - a)

    def test_summations(self):
        # check that sums of parts equal the whole
        a, b = self.a, self.b
        self.assertEqual((a - b) | (a & b) | (b - a), a | b)
        self.assertEqual((a & b) | (a ^ b), a | b)
        self.assertEqual(a | (b - a), a | b)
        self.assertEqual((a - b) | b, a | b)
        self.assertEqual((a - b) | (a & b), a)
        self.assertEqual((b - a) | (a & b), b)
        self.assertEqual((a - b) | (b - a), a ^ b)

    def test_exclusion(self):
        # check that inverse operations show non-overlap
        a, b, zero = self.a, self.b, OrderedSet()
        self.assertEqual((a - b) & b, zero)
        self.assertEqual((b - a) & a, zero)
        self.assertEqual((a & b) & (a ^ b), zero)


# Tests derived from test_itertools.py =======================================


def R(seqn):
    "Regular generator"
    for i in seqn:  # noqa: UP028
        yield i


class G:
    "Sequence using __getitem__"

    def __init__(self, seqn):
        self.seqn = seqn

    def __getitem__(self, i):
        return self.seqn[i]


class I:  # noqa: E742
    "Sequence using iterator protocol"

    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class Ig:
    "Sequence using iterator protocol defined with a generator"

    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        for val in self.seqn:  # noqa: UP028
            yield val


class X:
    "Missing __getitem__ and __iter__"

    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0

    def __next__(self):
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class N:
    "Iterator missing __next__()"

    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        return self


class E:
    "Test propagation of exceptions"

    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        3 // 0


class S:
    "Test immediate stop"

    def __init__(self, seqn):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


from itertools import chain


def L(seqn):
    "Test multiple tiers of iterators"
    return chain(map(lambda x: x, R(Ig(G(seqn)))))  # noqa: C417


class TestVariousIteratorArgs(TestCase):
    def test_constructor(self):
        for cons in (OrderedSet, frozenset):
            for s in ("123", "", range(1000), ("do", 1.2), range(2000, 2200, 5)):
                for g in (G, I, Ig, S, L, R):
                    self.assertEqual(
                        sorted(cons(g(s)), key=repr), sorted(g(s), key=repr)
                    )
                self.assertRaises(TypeError, cons, X(s))
                self.assertRaises(TypeError, cons, N(s))
                self.assertRaises(ZeroDivisionError, cons, E(s))

    def test_inline_methods(self):
        s = OrderedSet("november")
        for data in (
            "123",
            "",
            range(1000),
            ("do", 1.2),
            range(2000, 2200, 5),
            "december",
        ):
            for meth in (
                s.union,
                s.intersection,
                s.difference,
                s.symmetric_difference,
                s.isdisjoint,
            ):
                for g in (G, I, Ig, L, R):
                    # Only iterables supported, not sequences
                    if g is G:
                        continue
                    expected = meth(data)
                    actual = meth(g(data))
                    if isinstance(expected, bool):
                        self.assertEqual(actual, expected)
                    else:
                        self.assertEqual(
                            sorted(actual, key=repr), sorted(expected, key=repr)
                        )
                self.assertRaises(TypeError, meth, X(s))
                self.assertRaises(TypeError, meth, N(s))
                self.assertRaises(ZeroDivisionError, meth, E(s))

    def test_inplace_methods(self):
        for data in (
            "123",
            "",
            range(1000),
            ("do", 1.2),
            range(2000, 2200, 5),
            "december",
        ):
            for methname in (
                "update",
                "intersection_update",
                "difference_update",
                "symmetric_difference_update",
            ):
                for g in (G, I, Ig, S, L, R):
                    # Only Iterables supported, not Sequence
                    if g is G:
                        continue

                    s = OrderedSet("january")
                    t = s.copy()
                    getattr(s, methname)(list(g(data)))
                    getattr(t, methname)(g(data))
                    self.assertEqual(sorted(s, key=repr), sorted(t, key=repr))

                self.assertRaises(
                    TypeError, getattr(OrderedSet("january"), methname), X(data)
                )
                self.assertRaises(
                    TypeError, getattr(OrderedSet("january"), methname), N(data)
                )
                self.assertRaises(
                    ZeroDivisionError, getattr(OrderedSet("january"), methname), E(data)
                )


class bad_eq:
    def __eq__(self, other):
        if be_bad:
            set2.clear()
            raise ZeroDivisionError
        return self is other

    def __hash__(self):
        return 0


class bad_dict_clear:
    def __eq__(self, other):
        if be_bad:
            dict2.clear()
        return self is other

    def __hash__(self):
        return 0


class TestWeirdBugs(TestCase):
    def test_8420_set_merge(self):
        # This used to segfault
        global be_bad, set2, dict2
        be_bad = False
        set1 = {bad_eq()}
        set2 = {bad_eq() for i in range(75)}
        be_bad = True
        self.assertRaises(ZeroDivisionError, set1.update, set2)

        be_bad = False
        set1 = {bad_dict_clear()}
        dict2 = {bad_dict_clear(): None}
        be_bad = True
        set1.symmetric_difference_update(dict2)

    def test_iter_and_mutate(self):
        # Issue #24581
        s = OrderedSet(range(100))
        s.clear()
        s.update(range(100))
        si = iter(s)
        s.clear()
        a = list(range(100))
        s.update(range(100))
        list(si)

    def test_merge_and_mutate(self):
        class X:
            def __hash__(self):
                return hash(0)

            def __eq__(self, o):
                other.clear()
                return False

        other = OrderedSet()
        other = {X() for i in range(10)}
        s = {0}
        s.update(other)


# Application tests (based on David Eppstein's graph recipes ====================================


def powerset(U):
    """Generates all subsets of a OrderedSet or sequence U."""
    U = iter(U)
    try:
        x = frozenset([next(U)])
        for S in powerset(U):
            yield S
            yield S | x
    except StopIteration:
        yield frozenset()


def cube(n):
    """Graph of n-dimensional hypercube."""
    singletons = [frozenset([x]) for x in range(n)]
    return dict(  # noqa: C404
        [(x, frozenset([x ^ s for s in singletons])) for x in powerset(range(n))]
    )


def linegraph(G):
    """Graph, the vertices of which are edges of G,
    with two vertices being adjacent iff the corresponding
    edges share a vertex."""
    L = {}
    for x in G:
        for y in G[x]:
            nx = [frozenset([x, z]) for z in G[x] if z != y]
            ny = [frozenset([y, z]) for z in G[y] if z != x]
            L[frozenset([x, y])] = frozenset(nx + ny)
    return L


def faces(G):
    "Return a OrderedSet of faces in G.  Where a face is a OrderedSet of vertices on that face"
    # currently limited to triangles,squares, and pentagons
    f = OrderedSet()
    for v1, edges in G.items():
        for v2 in edges:
            for v3 in G[v2]:
                if v1 == v3:
                    continue
                if v1 in G[v3]:
                    f.add(frozenset([v1, v2, v3]))
                else:
                    for v4 in G[v3]:
                        if v4 == v2:
                            continue
                        if v1 in G[v4]:
                            f.add(frozenset([v1, v2, v3, v4]))
                        else:
                            for v5 in G[v4]:
                                if v5 == v3 or v5 == v2:  # noqa: SIM109
                                    continue
                                if v1 in G[v5]:
                                    f.add(frozenset([v1, v2, v3, v4, v5]))
    return f


class TestGraphs(TestCase):
    def test_cube(self):
        g = cube(3)  # vert --> {v1, v2, v3}
        vertices1 = OrderedSet(g)
        self.assertEqual(len(vertices1), 8)  # eight vertices
        for edge in g.values():
            self.assertEqual(len(edge), 3)  # each vertex connects to three edges
        vertices2 = OrderedSet(v for edges in g.values() for v in edges)
        self.assertEqual(vertices1, vertices2)  # edge vertices in original OrderedSet

        cubefaces = faces(g)
        self.assertEqual(len(cubefaces), 6)  # six faces
        for face in cubefaces:
            self.assertEqual(len(face), 4)  # each face is a square

    def test_cuboctahedron(self):
        # http://en.wikipedia.org/wiki/Cuboctahedron
        # 8 triangular faces and 6 square faces
        # 12 identical vertices each connecting a triangle and square

        g = cube(3)
        cuboctahedron = linegraph(g)  # V( --> {V1, V2, V3, V4}
        self.assertEqual(len(cuboctahedron), 12)  # twelve vertices

        vertices = OrderedSet(cuboctahedron)
        for edges in cuboctahedron.values():
            self.assertEqual(
                len(edges), 4
            )  # each vertex connects to four other vertices
        othervertices = OrderedSet(
            edge for edges in cuboctahedron.values() for edge in edges
        )
        self.assertEqual(
            vertices, othervertices
        )  # edge vertices in original OrderedSet

        cubofaces = faces(cuboctahedron)
        facesizes = collections.defaultdict(int)
        for face in cubofaces:
            facesizes[len(face)] += 1
        self.assertEqual(facesizes[3], 8)  # eight triangular faces
        self.assertEqual(facesizes[4], 6)  # six square faces

        for vertex in cuboctahedron:
            edge = vertex  # Cuboctahedron vertices are edges in Cube
            self.assertEqual(len(edge), 2)  # Two cube vertices define an edge
            for cubevert in edge:
                self.assertIn(cubevert, g)


# ==============================================================================

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
