# Owner(s): ["module: meta tensors"]

import copy
import gc
import random
import threading

import unittest

import torch
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    TestCase,
)
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary


def C():
    return torch.randn(1)


# These tests are ported from cpython/Lib/test/test_weakref.py,
# but adapted to use tensor rather than object
class WeakTest(TestCase):
    COUNT = 10

    def test_make_weak_keyed_dict_from_dict(self):
        o = torch.randn(2)
        dict = WeakIdKeyDictionary({o: 364})
        self.assertEqual(dict[o], 364)

    def test_make_weak_keyed_dict_from_weak_keyed_dict(self):
        o = torch.randn(3)
        dict = WeakIdKeyDictionary({o: 364})
        dict2 = WeakIdKeyDictionary(dict)
        self.assertEqual(dict[o], 364)

    def check_popitem(self, klass, key1, value1, key2, value2):
        weakdict = klass()
        weakdict[key1] = value1
        weakdict[key2] = value2
        self.assertEqual(len(weakdict), 2)
        k, v = weakdict.popitem()
        self.assertEqual(len(weakdict), 1)
        if k is key1:
            self.assertIs(v, value1)
        else:
            self.assertIs(v, value2)
        k, v = weakdict.popitem()
        self.assertEqual(len(weakdict), 0)
        if k is key1:
            self.assertIs(v, value1)
        else:
            self.assertIs(v, value2)

    def test_weak_keyed_dict_popitem(self):
        self.check_popitem(WeakIdKeyDictionary, C(), "value 1", C(), "value 2")

    def check_setdefault(self, klass, key, value1, value2):
        self.assertIsNot(
            value1,
            value2,
            "invalid test -- value parameters must be distinct objects",
        )
        weakdict = klass()
        o = weakdict.setdefault(key, value1)
        self.assertIs(o, value1)
        self.assertIn(key, weakdict)
        self.assertIs(weakdict.get(key), value1)
        self.assertIs(weakdict[key], value1)

        o = weakdict.setdefault(key, value2)
        self.assertIs(o, value1)
        self.assertIn(key, weakdict)
        self.assertIs(weakdict.get(key), value1)
        self.assertIs(weakdict[key], value1)

    def test_weak_keyed_dict_setdefault(self):
        self.check_setdefault(WeakIdKeyDictionary, C(), "value 1", "value 2")

    def check_update(self, klass, dict):
        #
        #  This exercises d.update(), len(d), d.keys(), k in d,
        #  d.get(), d[].
        #
        weakdict = klass()
        weakdict.update(dict)
        self.assertEqual(len(weakdict), len(dict))
        for k in weakdict.keys():
            self.assertIn(k, dict, "mysterious new key appeared in weak dict")
            v = dict.get(k)
            self.assertIs(v, weakdict[k])
            self.assertIs(v, weakdict.get(k))
        for k in dict.keys():
            self.assertIn(k, weakdict, "original key disappeared in weak dict")
            v = dict[k]
            self.assertIs(v, weakdict[k])
            self.assertIs(v, weakdict.get(k))

    def test_weak_keyed_dict_update(self):
        self.check_update(WeakIdKeyDictionary, {C(): 1, C(): 2, C(): 3})

    def test_weak_keyed_delitem(self):
        d = WeakIdKeyDictionary()
        o1 = torch.randn(1)
        o2 = torch.randn(2)
        d[o1] = "something"
        d[o2] = "something"
        self.assertEqual(len(d), 2)
        del d[o1]
        self.assertEqual(len(d), 1)
        self.assertEqual(list(d.keys()), [o2])

    def test_weak_keyed_union_operators(self):
        try:
            {} | {}
        except TypeError:
            self.skipTest("dict union not supported in this Python")

        o1 = C()
        o2 = C()
        o3 = C()
        wkd1 = WeakIdKeyDictionary({o1: 1, o2: 2})
        wkd2 = WeakIdKeyDictionary({o3: 3, o1: 4})
        wkd3 = wkd1.copy()
        d1 = {o2: "5", o3: "6"}
        pairs = [(o2, 7), (o3, 8)]

        tmp1 = wkd1 | wkd2  # Between two WeakKeyDictionaries
        self.assertEqual(dict(tmp1), dict(wkd1) | dict(wkd2))
        self.assertIs(type(tmp1), WeakIdKeyDictionary)
        wkd1 |= wkd2
        self.assertEqual(wkd1, tmp1)

        tmp2 = wkd2 | d1  # Between WeakKeyDictionary and mapping
        self.assertEqual(dict(tmp2), dict(wkd2) | d1)
        self.assertIs(type(tmp2), WeakIdKeyDictionary)
        wkd2 |= d1
        self.assertEqual(wkd2, tmp2)

        tmp3 = wkd3.copy()  # Between WeakKeyDictionary and iterable key, value
        tmp3 |= pairs
        self.assertEqual(dict(tmp3), dict(wkd3) | dict(pairs))
        self.assertIs(type(tmp3), WeakIdKeyDictionary)

        tmp4 = d1 | wkd3  # Testing .__ror__
        self.assertEqual(dict(tmp4), d1 | dict(wkd3))
        self.assertIs(type(tmp4), WeakIdKeyDictionary)

        del o1
        self.assertNotIn(4, tmp1.values())
        self.assertNotIn(4, tmp2.values())
        self.assertNotIn(1, tmp3.values())
        self.assertNotIn(1, tmp4.values())

    def test_weak_keyed_bad_delitem(self):
        d = WeakIdKeyDictionary()
        o = torch.randn(1)
        # An attempt to delete an object that isn't there should raise
        # KeyError.  It didn't before 2.3.
        self.assertRaises(KeyError, d.__delitem__, o)
        self.assertRaises(KeyError, d.__getitem__, o)

        # If a key isn't of a weakly referencable type, __getitem__ and
        # __setitem__ raise TypeError.  __delitem__ should too.
        self.assertRaises(TypeError, d.__delitem__, 13)
        self.assertRaises(TypeError, d.__getitem__, 13)
        self.assertRaises(TypeError, d.__setitem__, 13, 13)

    def test_make_weak_keyed_dict_repr(self):
        dict = WeakIdKeyDictionary()
        self.assertRegex(repr(dict), "<WeakIdKeyDictionary at 0x.*>")

    def check_threaded_weak_dict_copy(self, type_, deepcopy):
        # `deepcopy` should be either True or False.
        exc = []

        # Cannot give these slots as weakrefs weren't supported
        # on these objects until later versions of Python
        class DummyKey:  # noqa: B903
            def __init__(self, ctr):
                self.ctr = ctr

        class DummyValue:  # noqa: B903
            def __init__(self, ctr):
                self.ctr = ctr

        def dict_copy(d, exc):
            try:
                if deepcopy is True:
                    _ = copy.deepcopy(d)
                else:
                    _ = d.copy()
            except Exception as ex:
                exc.append(ex)

        def pop_and_collect(lst):
            gc_ctr = 0
            while lst:
                i = random.randint(0, len(lst) - 1)
                gc_ctr += 1
                lst.pop(i)
                if gc_ctr % 10000 == 0:
                    gc.collect()  # just in case

        d = type_()
        keys = []
        values = []
        # Initialize d with many entries
        for i in range(70000):
            k, v = DummyKey(i), DummyValue(i)
            keys.append(k)
            values.append(v)
            d[k] = v
            del k
            del v

        t_copy = threading.Thread(
            target=dict_copy,
            args=(
                d,
                exc,
            ),
        )
        t_collect = threading.Thread(target=pop_and_collect, args=(keys,))

        t_copy.start()
        t_collect.start()

        t_copy.join()
        t_collect.join()

        # Test exceptions
        if exc:
            raise exc[0]

    def test_threaded_weak_key_dict_copy(self):
        # Issue #35615: Weakref keys or values getting GC'ed during dict
        # copying should not result in a crash.
        self.check_threaded_weak_dict_copy(WeakIdKeyDictionary, False)

    def test_threaded_weak_key_dict_deepcopy(self):
        # Issue #35615: Weakref keys or values getting GC'ed during dict
        # copying should not result in a crash.
        self.check_threaded_weak_dict_copy(WeakIdKeyDictionary, True)


# Adapted from cpython/Lib/test/mapping_tests.py
class WeakKeyDictionaryTestCase(TestCase):
    __ref = {torch.randn(1): 1, torch.randn(2): 2, torch.randn(3): 3}
    type2test = WeakIdKeyDictionary

    def _reference(self):
        return self.__ref.copy()

    def _empty_mapping(self):
        """Return an empty mapping object"""
        return self.type2test()

    def _full_mapping(self, data):
        """Return a mapping object with the value contained in data
        dictionary"""
        x = self._empty_mapping()
        for key, value in data.items():
            x[key] = value
        return x

    def __init__(self, *args, **kw):
        unittest.TestCase.__init__(self, *args, **kw)
        self.reference = self._reference().copy()

        # A (key, value) pair not in the mapping
        key, value = self.reference.popitem()
        self.other = {key: value}

        # A (key, value) pair in the mapping
        key, value = self.reference.popitem()
        self.inmapping = {key: value}
        self.reference[key] = value

    def test_read(self):
        # Test for read only operations on mapping
        p = self._empty_mapping()
        p1 = dict(p)  # workaround for singleton objects
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        # Indexing
        for key, value in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = next(iter(self.other.keys()))
        self.assertRaises(KeyError, lambda: d[knownkey])
        # len
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        # __contains__
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        # cmp
        self.assertTrue(
            p == p
        )  # NB: don't use assertEqual, that doesn't actually use ==
        self.assertTrue(d == d)
        self.assertTrue(p != d)
        self.assertTrue(d != p)
        # bool
        if p:
            self.fail("Empty mapping must compare to False")
        if not d:
            self.fail("Full mapping must compare to True")

        # keys(), items(), iterkeys() ...
        def check_iterandlist(iter, lst, ref):
            self.assertTrue(hasattr(iter, "__next__"))
            self.assertTrue(hasattr(iter, "__iter__"))
            x = list(iter)
            self.assertTrue(set(x) == set(lst) == set(ref))

        check_iterandlist(iter(d.keys()), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()), self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()), self.reference.items())
        # get
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)

    def test_write(self):
        # Test for write operations on mapping
        p = self._empty_mapping()
        # Indexing
        for key, value in self.reference.items():
            p[key] = value
            self.assertEqual(p[key], value)
        for key in self.reference.keys():
            del p[key]
            self.assertRaises(KeyError, lambda: p[key])
        p = self._empty_mapping()
        # update
        p.update(self.reference)
        self.assertEqual(dict(p), self.reference)
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        self.assertEqual(dict(p), self.reference)
        d = self._full_mapping(self.reference)
        # setdefault
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)
        # pop
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)
        # popitem
        key, value = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])
        p = self._empty_mapping()
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self.reference)
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self.reference
        self.assertIn(next(iter(self.inmapping.keys())), d.keys())
        self.assertNotIn(next(iter(self.other.keys())), d.keys())
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        d = self._empty_mapping()
        self.assertEqual(list(d.values()), [])

        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        d = self._empty_mapping()
        self.assertEqual(list(d.items()), [])

        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        d = self._empty_mapping()
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        d = self.reference
        self.assertEqual(
            d[next(iter(self.inmapping.keys()))], next(iter(self.inmapping.values()))
        )

        self.assertRaises(TypeError, d.__getitem__)

    def test_update(self):
        # mapping argument
        d = self._empty_mapping()
        d.update(self.other)
        self.assertEqual(list(d.items()), list(self.other.items()))

        # No argument
        d = self._empty_mapping()
        d.update()
        self.assertEqual(d, self._empty_mapping())

        # item sequence
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))

        # Iterator
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))

        # FIXME: Doesn't work with UserDict
        # self.assertRaises((TypeError, AttributeError), d.update, None)
        self.assertRaises((TypeError, AttributeError), d.update, 42)

        outerself = self

        class SimpleUserDict:
            def __init__(self):
                self.d = outerself.reference

            def keys(self):
                return self.d.keys()

            def __getitem__(self, i):
                return self.d[i]

        d.clear()
        d.update(SimpleUserDict())
        i1 = sorted((id(k), v) for k, v in d.items())
        i2 = sorted((id(k), v) for k, v in self.reference.items())
        self.assertEqual(i1, i2)

        class Exc(Exception):
            pass

        d = self._empty_mapping()

        class FailingUserDict:
            def keys(self):
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d.clear()

        class FailingUserDict:
            def keys(self):
                class BogonIter:
                    def __init__(self):
                        self.i = 1

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.i:
                            self.i = 0
                            return "a"
                        raise Exc

                return BogonIter()

            def __getitem__(self, key):
                return key

        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            def keys(self):
                class BogonIter:
                    def __init__(self):
                        self.i = ord("a")

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.i <= ord("z"):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration

                return BogonIter()

            def __getitem__(self, key):
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d = self._empty_mapping()

        class badseq:
            def __iter__(self):
                return self

            def __next__(self):
                raise Exc()

        self.assertRaises(Exc, d.update, badseq())

        self.assertRaises(ValueError, d.update, [(1, 2, 3)])

    # no test_fromkeys or test_copy as both os.environ and selves don't support it

    def test_get(self):
        d = self._empty_mapping()
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        d = self.reference
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys()))),
            next(iter(self.inmapping.values())),
        )
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys())), 3),
            next(iter(self.inmapping.values())),
        )
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        d = self._empty_mapping()
        self.assertRaises(TypeError, d.setdefault)

    def test_popitem(self):
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)
        self.assertRaises(TypeError, d.popitem, 42)

    def test_pop(self):
        d = self._empty_mapping()
        k, v = next(iter(self.inmapping.items()))
        d[k] = v
        self.assertRaises(KeyError, d.pop, next(iter(self.other.keys())))

        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)

        self.assertRaises(KeyError, d.pop, k)


# Adapted from cpython/Lib/test/mapping_tests.py
class WeakKeyDictionaryScriptObjectTestCase(TestCase):
    def _reference(self):
        self.__ref = {
            torch.classes._TorchScriptTesting._Foo(1, 2): 1,
            torch.classes._TorchScriptTesting._Foo(2, 3): 2,
            torch.classes._TorchScriptTesting._Foo(3, 4): 3,
        }
        return self.__ref.copy()

    def _empty_mapping(self):
        """Return an empty mapping object"""
        return WeakIdKeyDictionary(ref_type=_WeakHashRef)

    def _full_mapping(self, data):
        """Return a mapping object with the value contained in data
        dictionary"""
        x = self._empty_mapping()
        for key, value in data.items():
            x[key] = value
        return x

    def setUp(self):
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")

    def __init__(self, *args, **kw):
        unittest.TestCase.__init__(self, *args, **kw)
        if IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        elif IS_MACOS:
            # don't load the library, just skip the tests in setUp
            return
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
            if IS_WINDOWS:
                lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))

        self.reference = self._reference().copy()

        # A (key, value) pair not in the mapping
        key, value = self.reference.popitem()
        self.other = {key: value}

        # A (key, value) pair in the mapping
        key, value = self.reference.popitem()
        self.inmapping = {key: value}
        self.reference[key] = value

    def test_read(self):
        # Test for read only operations on mapping
        p = self._empty_mapping()
        p1 = dict(p)  # workaround for singleton objects
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        # Indexing
        for key, value in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = next(iter(self.other.keys()))
        self.assertRaises(KeyError, lambda: d[knownkey])
        # len
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        # __contains__
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        # cmp
        self.assertTrue(
            p == p
        )  # NB: don't use assertEqual, that doesn't actually use ==
        self.assertTrue(d == d)
        self.assertTrue(p != d)
        self.assertTrue(d != p)
        # bool
        if p:
            self.fail("Empty mapping must compare to False")
        if not d:
            self.fail("Full mapping must compare to True")

        # keys(), items(), iterkeys() ...
        def check_iterandlist(iter, lst, ref):
            self.assertTrue(hasattr(iter, "__next__"))
            self.assertTrue(hasattr(iter, "__iter__"))
            x = list(iter)
            self.assertTrue(set(x) == set(lst) == set(ref))

        check_iterandlist(iter(d.keys()), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()), self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()), self.reference.items())
        # get
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)

    def test_write(self):
        # Test for write operations on mapping
        p = self._empty_mapping()
        # Indexing
        for key, value in self.reference.items():
            p[key] = value
            self.assertEqual(p[key], value)
        for key in self.reference.keys():
            del p[key]
            self.assertRaises(KeyError, lambda: p[key])
        p = self._empty_mapping()
        # update
        p.update(self.reference)
        self.assertEqual(dict(p), self.reference)
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        self.assertEqual(dict(p), self.reference)
        d = self._full_mapping(self.reference)
        # setdefault
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)
        # pop
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)
        # popitem
        key, value = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])
        p = self._empty_mapping()
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self.reference)
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self.reference
        self.assertIn(next(iter(self.inmapping.keys())), d.keys())
        self.assertNotIn(next(iter(self.other.keys())), d.keys())
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        d = self._empty_mapping()
        self.assertEqual(list(d.values()), [])

        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        d = self._empty_mapping()
        self.assertEqual(list(d.items()), [])

        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        d = self._empty_mapping()
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        d = self.reference
        self.assertEqual(
            d[next(iter(self.inmapping.keys()))], next(iter(self.inmapping.values()))
        )

        self.assertRaises(TypeError, d.__getitem__)

    def test_update(self):
        # mapping argument
        d = self._empty_mapping()
        d.update(self.other)
        self.assertEqual(list(d.items()), list(self.other.items()))

        # No argument
        d = self._empty_mapping()
        d.update()
        self.assertEqual(d, self._empty_mapping())

        # item sequence
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))

        # Iterator
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))

        # FIXME: Doesn't work with UserDict
        # self.assertRaises((TypeError, AttributeError), d.update, None)
        self.assertRaises((TypeError, AttributeError), d.update, 42)

        outerself = self

        class SimpleUserDict:
            def __init__(self):
                self.d = outerself.reference

            def keys(self):
                return self.d.keys()

            def __getitem__(self, i):
                return self.d[i]

        d.clear()
        d.update(SimpleUserDict())
        i1 = sorted((id(k), v) for k, v in d.items())
        i2 = sorted((id(k), v) for k, v in self.reference.items())
        self.assertEqual(i1, i2)

        class Exc(Exception):
            pass

        d = self._empty_mapping()

        class FailingUserDict:
            def keys(self):
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d.clear()

        class FailingUserDict:
            def keys(self):
                class BogonIter:
                    def __init__(self):
                        self.i = 1

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.i:
                            self.i = 0
                            return "a"
                        raise Exc

                return BogonIter()

            def __getitem__(self, key):
                return key

        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            def keys(self):
                class BogonIter:
                    def __init__(self):
                        self.i = ord("a")

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.i <= ord("z"):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration

                return BogonIter()

            def __getitem__(self, key):
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d = self._empty_mapping()

        class badseq:
            def __iter__(self):
                return self

            def __next__(self):
                raise Exc()

        self.assertRaises(Exc, d.update, badseq())

        self.assertRaises(ValueError, d.update, [(1, 2, 3)])

    # no test_fromkeys or test_copy as both os.environ and selves don't support it

    def test_get(self):
        d = self._empty_mapping()
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        d = self.reference
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys()))),
            next(iter(self.inmapping.values())),
        )
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys())), 3),
            next(iter(self.inmapping.values())),
        )
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        d = self._empty_mapping()
        self.assertRaises(TypeError, d.setdefault)

    def test_popitem(self):
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)
        self.assertRaises(TypeError, d.popitem, 42)

    def test_pop(self):
        d = self._empty_mapping()
        k, v = next(iter(self.inmapping.items()))
        d[k] = v
        self.assertRaises(KeyError, d.pop, next(iter(self.other.keys())))

        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)

        self.assertRaises(KeyError, d.pop, k)


if __name__ == "__main__":
    run_tests()
