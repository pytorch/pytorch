# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_class.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo

# ======= END DYNAMO PATCH =======

"Test the functionality of Python classes implementing operators."

import unittest
from test import support
from test.support import cpython_only, import_helper, script_helper

testmeths = [

# Binary operations
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "matmul",
    "rmatmul",
    "truediv",
    "rtruediv",
    "floordiv",
    "rfloordiv",
    "mod",
    "rmod",
    "divmod",
    "rdivmod",
    "pow",
    "rpow",
    "rshift",
    "rrshift",
    "lshift",
    "rlshift",
    "and",
    "rand",
    "or",
    "ror",
    "xor",
    "rxor",

# List/dict operations
    "contains",
    "getitem",
    "setitem",
    "delitem",

# Unary operations
    "neg",
    "pos",
    "abs",

# generic operations
    "init",
    ]

# These need to return something other than None
#    "hash",
#    "str",
#    "repr",
#    "int",
#    "float",

# These are separate because they can influence the test of other methods.
#    "getattr",
#    "setattr",
#    "delattr",

callLst = []
def trackCall(f):
    def track(*args, **kwargs):
        callLst.append((f.__name__, args))
        return f(*args, **kwargs)
    return track

statictests = """
@trackCall
def __hash__(self, *args):
    return hash(id(self))

@trackCall
def __str__(self, *args):
    return "AllTests"

@trackCall
def __repr__(self, *args):
    return "AllTests"

@trackCall
def __int__(self, *args):
    return 1

@trackCall
def __index__(self, *args):
    return 1

@trackCall
def __float__(self, *args):
    return 1.0

@trackCall
def __eq__(self, *args):
    return True

@trackCall
def __ne__(self, *args):
    return False

@trackCall
def __lt__(self, *args):
    return False

@trackCall
def __le__(self, *args):
    return True

@trackCall
def __gt__(self, *args):
    return False

@trackCall
def __ge__(self, *args):
    return True
"""

# Synthesize all the other AllTests methods from the names in testmeths.

method_template = """\
@trackCall
def __%s__(self, *args):
    pass
"""

d = {}
exec(statictests, globals(), d)
for method in testmeths:
    exec(method_template % method, globals(), d)
AllTests = type("AllTests", (object,), d)
del d, statictests, method, method_template

class ClassTests(CPythonTestCase):
    def setUp(self):
        super().setUp()
        callLst[:] = []

    def assertCallStack(self, expected_calls):
        actualCallList = callLst[:]  # need to copy because the comparison below will add
                                     # additional calls to callLst
        if expected_calls != actualCallList:
            self.fail("Expected call list:\n  %s\ndoes not match actual call list\n  %s" %
                      (expected_calls, actualCallList))

    def testInit(self):
        foo = AllTests()
        self.assertCallStack([("__init__", (foo,))])

    def testBinaryOps(self):
        testme = AllTests()
        # Binary operations

        callLst[:] = []
        testme + 1
        self.assertCallStack([("__add__", (testme, 1))])

        callLst[:] = []
        1 + testme
        self.assertCallStack([("__radd__", (testme, 1))])

        callLst[:] = []
        testme - 1
        self.assertCallStack([("__sub__", (testme, 1))])

        callLst[:] = []
        1 - testme
        self.assertCallStack([("__rsub__", (testme, 1))])

        callLst[:] = []
        testme * 1
        self.assertCallStack([("__mul__", (testme, 1))])

        callLst[:] = []
        1 * testme
        self.assertCallStack([("__rmul__", (testme, 1))])

        callLst[:] = []
        testme @ 1
        self.assertCallStack([("__matmul__", (testme, 1))])

        callLst[:] = []
        1 @ testme
        self.assertCallStack([("__rmatmul__", (testme, 1))])

        callLst[:] = []
        testme / 1
        self.assertCallStack([("__truediv__", (testme, 1))])


        callLst[:] = []
        1 / testme
        self.assertCallStack([("__rtruediv__", (testme, 1))])

        callLst[:] = []
        testme // 1
        self.assertCallStack([("__floordiv__", (testme, 1))])


        callLst[:] = []
        1 // testme
        self.assertCallStack([("__rfloordiv__", (testme, 1))])

        callLst[:] = []
        testme % 1
        self.assertCallStack([("__mod__", (testme, 1))])

        callLst[:] = []
        1 % testme
        self.assertCallStack([("__rmod__", (testme, 1))])


        callLst[:] = []
        divmod(testme,1)
        self.assertCallStack([("__divmod__", (testme, 1))])

        callLst[:] = []
        divmod(1, testme)
        self.assertCallStack([("__rdivmod__", (testme, 1))])

        callLst[:] = []
        testme ** 1
        self.assertCallStack([("__pow__", (testme, 1))])

        callLst[:] = []
        1 ** testme
        self.assertCallStack([("__rpow__", (testme, 1))])

        callLst[:] = []
        testme >> 1
        self.assertCallStack([("__rshift__", (testme, 1))])

        callLst[:] = []
        1 >> testme
        self.assertCallStack([("__rrshift__", (testme, 1))])

        callLst[:] = []
        testme << 1
        self.assertCallStack([("__lshift__", (testme, 1))])

        callLst[:] = []
        1 << testme
        self.assertCallStack([("__rlshift__", (testme, 1))])

        callLst[:] = []
        testme & 1
        self.assertCallStack([("__and__", (testme, 1))])

        callLst[:] = []
        1 & testme
        self.assertCallStack([("__rand__", (testme, 1))])

        callLst[:] = []
        testme | 1
        self.assertCallStack([("__or__", (testme, 1))])

        callLst[:] = []
        1 | testme
        self.assertCallStack([("__ror__", (testme, 1))])

        callLst[:] = []
        testme ^ 1
        self.assertCallStack([("__xor__", (testme, 1))])

        callLst[:] = []
        1 ^ testme
        self.assertCallStack([("__rxor__", (testme, 1))])

    def testListAndDictOps(self):
        testme = AllTests()

        # List/dict operations

        with torch._dynamo.error_on_graph_break(False):
            class Empty: pass

        try:
            1 in Empty()
            self.fail('failed, should have raised TypeError')
        except TypeError:
            pass

        callLst[:] = []
        1 in testme
        self.assertCallStack([('__contains__', (testme, 1))])

        callLst[:] = []
        testme[1]
        self.assertCallStack([('__getitem__', (testme, 1))])

        callLst[:] = []
        testme[1] = 1
        self.assertCallStack([('__setitem__', (testme, 1, 1))])

        callLst[:] = []
        del testme[1]
        self.assertCallStack([('__delitem__', (testme, 1))])

        callLst[:] = []
        testme[:42]
        self.assertCallStack([('__getitem__', (testme, slice(None, 42)))])

        callLst[:] = []
        testme[:42] = "The Answer"
        self.assertCallStack([('__setitem__', (testme, slice(None, 42),
                                               "The Answer"))])

        callLst[:] = []
        del testme[:42]
        self.assertCallStack([('__delitem__', (testme, slice(None, 42)))])

        callLst[:] = []
        testme[2:1024:10]
        self.assertCallStack([('__getitem__', (testme, slice(2, 1024, 10)))])

        callLst[:] = []
        testme[2:1024:10] = "A lot"
        self.assertCallStack([('__setitem__', (testme, slice(2, 1024, 10),
                                                                    "A lot"))])
        callLst[:] = []
        del testme[2:1024:10]
        self.assertCallStack([('__delitem__', (testme, slice(2, 1024, 10)))])

        callLst[:] = []
        testme[:42, ..., :24:, 24, 100]
        self.assertCallStack([('__getitem__', (testme, (slice(None, 42, None),
                                                        Ellipsis,
                                                        slice(None, 24, None),
                                                        24, 100)))])
        callLst[:] = []
        testme[:42, ..., :24:, 24, 100] = "Strange"
        self.assertCallStack([('__setitem__', (testme, (slice(None, 42, None),
                                                        Ellipsis,
                                                        slice(None, 24, None),
                                                        24, 100), "Strange"))])
        callLst[:] = []
        del testme[:42, ..., :24:, 24, 100]
        self.assertCallStack([('__delitem__', (testme, (slice(None, 42, None),
                                                        Ellipsis,
                                                        slice(None, 24, None),
                                                        24, 100)))])

    def testUnaryOps(self):
        testme = AllTests()

        callLst[:] = []
        -testme
        self.assertCallStack([('__neg__', (testme,))])
        callLst[:] = []
        +testme
        self.assertCallStack([('__pos__', (testme,))])
        callLst[:] = []
        abs(testme)
        self.assertCallStack([('__abs__', (testme,))])
        callLst[:] = []
        int(testme)
        self.assertCallStack([('__int__', (testme,))])
        callLst[:] = []
        float(testme)
        self.assertCallStack([('__float__', (testme,))])
        callLst[:] = []
        oct(testme)
        self.assertCallStack([('__index__', (testme,))])
        callLst[:] = []
        hex(testme)
        self.assertCallStack([('__index__', (testme,))])


    def testMisc(self):
        testme = AllTests()

        callLst[:] = []
        hash(testme)
        self.assertCallStack([('__hash__', (testme,))])

        callLst[:] = []
        repr(testme)
        self.assertCallStack([('__repr__', (testme,))])

        callLst[:] = []
        str(testme)
        self.assertCallStack([('__str__', (testme,))])

        callLst[:] = []
        testme == 1
        self.assertCallStack([('__eq__', (testme, 1))])

        callLst[:] = []
        testme < 1
        self.assertCallStack([('__lt__', (testme, 1))])

        callLst[:] = []
        testme > 1
        self.assertCallStack([('__gt__', (testme, 1))])

        callLst[:] = []
        testme != 1
        self.assertCallStack([('__ne__', (testme, 1))])

        callLst[:] = []
        1 == testme
        self.assertCallStack([('__eq__', (1, testme))])

        callLst[:] = []
        1 < testme
        self.assertCallStack([('__gt__', (1, testme))])

        callLst[:] = []
        1 > testme
        self.assertCallStack([('__lt__', (1, testme))])

        callLst[:] = []
        1 != testme
        self.assertCallStack([('__ne__', (1, testme))])


    def testGetSetAndDel(self):
        # Interfering tests
        with torch._dynamo.error_on_graph_break(False):
            class ExtraTests(AllTests):
                @trackCall
                def __getattr__(self, *args):
                    return "SomeVal"

                @trackCall
                def __setattr__(self, *args):
                    pass

                @trackCall
                def __delattr__(self, *args):
                    pass

        testme = ExtraTests()

        callLst[:] = []
        testme.spam
        self.assertCallStack([('__getattr__', (testme, "spam"))])

        callLst[:] = []
        testme.eggs = "spam, spam, spam and ham"
        self.assertCallStack([('__setattr__', (testme, "eggs",
                                               "spam, spam, spam and ham"))])

        callLst[:] = []
        del testme.cardinal
        self.assertCallStack([('__delattr__', (testme, "cardinal"))])

    def testHasAttrString(self):
        import sys
        from test.support import import_helper
        _testlimitedcapi = import_helper.import_module('_testlimitedcapi')

        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __init__(self):
                    self.attr = 1

        a = A()
        self.assertEqual(_testlimitedcapi.object_hasattrstring(a, b"attr"), 1)
        self.assertEqual(_testlimitedcapi.object_hasattrstring(a, b"noattr"), 0)
        self.assertIsNone(sys.exception())

    def testDel(self):
        x = []

        class DelTest:
            def __del__(self):
                x.append("crab people, crab people")
        testme = DelTest()
        del testme
        import gc
        gc.collect()
        self.assertEqual(["crab people, crab people"], x)

    def testBadTypeReturned(self):
        # return values of some method are type-checked
        with torch._dynamo.error_on_graph_break(False):
            class BadTypeClass:
                def __int__(self):
                    return None
                __float__ = __int__
                __complex__ = __int__
                __str__ = __int__
                __repr__ = __int__
                __bytes__ = __int__
                __bool__ = __int__
                __index__ = __int__
        def index(x):
            return [][x]

        for f in [float, complex, str, repr, bytes, bin, oct, hex, bool, index]:
            self.assertRaises(TypeError, f, BadTypeClass())

    def testHashStuff(self):
        # Test correct errors from hash() on objects with comparisons but
        #  no __hash__

        with torch._dynamo.error_on_graph_break(False):
            class C0:
                pass

        hash(C0()) # This should work; the next two should raise TypeError

        with torch._dynamo.error_on_graph_break(False):
            class C2:
                def __eq__(self, other): return 1

        self.assertRaises(TypeError, hash, C2())

    def testPredefinedAttrs(self):
        o = object()

        with torch._dynamo.error_on_graph_break(False):
            class Custom:
                pass

        c = Custom()

        methods = (
            '__class__', '__delattr__', '__dir__', '__eq__', '__format__',
            '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__',
            '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__',
            '__new__', '__reduce__', '__reduce_ex__', '__repr__',
            '__setattr__', '__sizeof__', '__str__', '__subclasshook__'
        )
        for name in methods:
            with self.subTest(name):
                self.assertTrue(callable(getattr(object, name, None)))
                self.assertTrue(callable(getattr(o, name, None)))
                self.assertTrue(callable(getattr(Custom, name, None)))
                self.assertTrue(callable(getattr(c, name, None)))

        not_defined = [
            '__abs__', '__aenter__', '__aexit__', '__aiter__', '__anext__',
            '__await__', '__bool__', '__bytes__', '__ceil__',
            '__complex__', '__contains__', '__del__', '__delete__',
            '__delitem__', '__divmod__', '__enter__', '__exit__',
            '__float__', '__floor__', '__get__', '__getattr__', '__getitem__',
            '__index__', '__int__', '__invert__', '__iter__', '__len__',
            '__length_hint__', '__missing__', '__neg__', '__next__',
            '__objclass__', '__pos__', '__rdivmod__', '__reversed__',
            '__round__', '__set__', '__setitem__', '__trunc__'
        ]
        augment = (
            'add', 'and', 'floordiv', 'lshift', 'matmul', 'mod', 'mul', 'pow',
            'rshift', 'sub', 'truediv', 'xor'
        )
        not_defined.extend(map("__{}__".format, augment))
        not_defined.extend(map("__r{}__".format, augment))
        not_defined.extend(map("__i{}__".format, augment))
        for name in not_defined:
            with self.subTest(name):
                self.assertFalse(hasattr(object, name))
                self.assertFalse(hasattr(o, name))
                self.assertFalse(hasattr(Custom, name))
                self.assertFalse(hasattr(c, name))

        # __call__() is defined on the metaclass but not the class
        self.assertFalse(hasattr(o, "__call__"))
        self.assertFalse(hasattr(c, "__call__"))

    def testSFBug532646(self):
        # Test for SF bug 532646

        with torch._dynamo.error_on_graph_break(False):
            class A:
                pass
        A.__call__ = A()
        a = A()

        try:
            a() # This should not segfault
        except RecursionError:
            pass
        else:
            self.fail("Failed to raise RecursionError")

    def testForExceptionsRaisedInInstanceGetattr2(self):
        # Tests for exceptions raised in instance_getattr2().

        def booh(self):
            raise AttributeError("booh")

        with torch._dynamo.error_on_graph_break(False):
            class A:
                a = property(booh)
        try:
            A().a # Raised AttributeError: A instance has no attribute 'a'
        except AttributeError as x:
            if str(x) != "booh":
                self.fail("attribute error for A().a got masked: %s" % x)

        with torch._dynamo.error_on_graph_break(False):
            class E:
                __eq__ = property(booh)
        E() == E() # In debug mode, caused a C-level assert() to fail

        with torch._dynamo.error_on_graph_break(False):
            class I:
                __init__ = property(booh)
        try:
            # In debug mode, printed XXX undetected error and
            #  raises AttributeError
            I()
        except AttributeError:
            pass
        else:
            self.fail("attribute error for I.__init__ got masked")

    def assertNotOrderable(self, a, b):
        with self.assertRaises(TypeError):
            a < b
        with self.assertRaises(TypeError):
            a > b
        with self.assertRaises(TypeError):
            a <= b
        with self.assertRaises(TypeError):
            a >= b

    def testHashComparisonOfMethods(self):
        # Test comparison and hash of methods
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __init__(self, x):
                    self.x = x
                def f(self):
                    pass
                def g(self):
                    pass
                def __eq__(self, other):
                    return True
                def __hash__(self):
                    raise TypeError
        with torch._dynamo.error_on_graph_break(False):
            class B(A):
                pass

        a1 = A(1)
        a2 = A(1)
        self.assertTrue(a1.f == a1.f)
        self.assertFalse(a1.f != a1.f)
        self.assertFalse(a1.f == a2.f)
        self.assertTrue(a1.f != a2.f)
        self.assertFalse(a1.f == a1.g)
        self.assertTrue(a1.f != a1.g)
        self.assertNotOrderable(a1.f, a1.f)
        self.assertEqual(hash(a1.f), hash(a1.f))

        self.assertFalse(A.f == a1.f)
        self.assertTrue(A.f != a1.f)
        self.assertFalse(A.f == A.g)
        self.assertTrue(A.f != A.g)
        self.assertTrue(B.f == A.f)
        self.assertFalse(B.f != A.f)
        self.assertNotOrderable(A.f, A.f)
        self.assertEqual(hash(B.f), hash(A.f))

        # the following triggers a SystemError in 2.4
        a = A(hash(A.f)^(-1))
        hash(a.f)

    def testSetattrWrapperNameIntern(self):
        # Issue #25794: __setattr__ should intern the attribute name
        with torch._dynamo.error_on_graph_break(False):
            class A:
                pass

        def add(self, other):
            return 'summa'

        name = str(b'__add__', 'ascii')  # shouldn't be optimized
        self.assertIsNot(name, '__add__')  # not interned
        type.__setattr__(A, name, add)
        self.assertEqual(A() + 1, 'summa')

        name2 = str(b'__add__', 'ascii')
        self.assertIsNot(name2, '__add__')
        self.assertIsNot(name2, name)
        type.__delattr__(A, name2)
        with self.assertRaises(TypeError):
            A() + 1

    def testSetattrNonStringName(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                pass

        with self.assertRaises(TypeError):
            type.__setattr__(A, b'x', None)

    def testTypeAttributeAccessErrorMessages(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                pass

        error_msg = "type object 'A' has no attribute 'x'"
        with self.assertRaisesRegex(AttributeError, error_msg):
            A.x
        with self.assertRaisesRegex(AttributeError, error_msg):
            del A.x

    def testObjectAttributeAccessErrorMessages(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                pass
            class B:
                y = 0
                __slots__ = ('z',)
            class C:
                __slots__ = ("y",)

                def __setattr__(self, name, value) -> None:
                    if name == "z":
                        super().__setattr__("y", 1)
                    else:
                        super().__setattr__(name, value)

        error_msg = "'A' object has no attribute 'x'"
        with self.assertRaisesRegex(AttributeError, error_msg):
            A().x
        with self.assertRaisesRegex(AttributeError, error_msg):
            del A().x

        error_msg = "'B' object has no attribute 'x'"
        with self.assertRaisesRegex(AttributeError, error_msg):
            B().x
        with self.assertRaisesRegex(AttributeError, error_msg):
            del B().x
        with self.assertRaisesRegex(
            AttributeError,
            "'B' object has no attribute 'x' and no __dict__ for setting new attributes"
        ):
            B().x = 0
        with self.assertRaisesRegex(
            AttributeError,
            "'C' object has no attribute 'x'"
        ):
            C().x = 0

        error_msg = "'B' object attribute 'y' is read-only"
        with self.assertRaisesRegex(AttributeError, error_msg):
            del B().y
        with self.assertRaisesRegex(AttributeError, error_msg):
            B().y = 0

        error_msg = 'z'
        with self.assertRaisesRegex(AttributeError, error_msg):
            B().z
        with self.assertRaisesRegex(AttributeError, error_msg):
            del B().z

    def testConstructorErrorMessages(self):
        # bpo-31506: Improves the error message logic for object_new & object_init

        # Class without any method overrides
        with torch._dynamo.error_on_graph_break(False):
            class C:
                pass

        error_msg = r'C.__init__\(\) takes exactly one argument \(the instance to initialize\)'

        with self.assertRaisesRegex(TypeError, r'C\(\) takes no arguments'):
            C(42)

        with self.assertRaisesRegex(TypeError, r'C\(\) takes no arguments'):
            C.__new__(C, 42)

        with self.assertRaisesRegex(TypeError, error_msg):
            C().__init__(42)

        with self.assertRaisesRegex(TypeError, r'C\(\) takes no arguments'):
            object.__new__(C, 42)

        with self.assertRaisesRegex(TypeError, error_msg):
            object.__init__(C(), 42)

        # Class with both `__init__` & `__new__` method overridden
        with torch._dynamo.error_on_graph_break(False):
            class D:
                def __new__(cls, *args, **kwargs):
                    super().__new__(cls, *args, **kwargs)
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

        error_msg =  r'object.__new__\(\) takes exactly one argument \(the type to instantiate\)'

        with self.assertRaisesRegex(TypeError, error_msg):
            D(42)

        with self.assertRaisesRegex(TypeError, error_msg):
            D.__new__(D, 42)

        with self.assertRaisesRegex(TypeError, error_msg):
            object.__new__(D, 42)

        # Class that only overrides __init__
        with torch._dynamo.error_on_graph_break(False):
            class E:
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

        error_msg = r'object.__init__\(\) takes exactly one argument \(the instance to initialize\)'

        with self.assertRaisesRegex(TypeError, error_msg):
            E().__init__(42)

        with self.assertRaisesRegex(TypeError, error_msg):
            object.__init__(E(), 42)

    def testClassWithExtCall(self):
        with torch._dynamo.error_on_graph_break(False):
            class Meta(int):
                def __init__(*args, **kwargs):
                    pass

                def __new__(cls, name, bases, attrs, **kwargs):
                    return bases, kwargs

        d = {'metaclass': Meta}

        with torch._dynamo.error_on_graph_break(False):
            class A(**d): pass
        self.assertEqual(A, ((), {}))
        with torch._dynamo.error_on_graph_break(False):
            class A(0, 1, 2, 3, 4, 5, 6, 7, **d): pass
        self.assertEqual(A, (tuple(range(8)), {}))
        with torch._dynamo.error_on_graph_break(False):
            class A(0, *range(1, 8), **d, foo='bar'): pass
        self.assertEqual(A, (tuple(range(8)), {'foo': 'bar'}))

    def testClassCallRecursionLimit(self):
        class C:
            def __init__(self):
                self.c = C()

        with self.assertRaises(RecursionError):
            C()

        def add_one_level():
            #Each call to C() consumes 2 levels, so offset by 1.
            C()

        with self.assertRaises(RecursionError):
            add_one_level()

    def testMetaclassCallOptimization(self):
        calls = 0

        with torch._dynamo.error_on_graph_break(False):
            class TypeMetaclass(type):
                def __call__(cls, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    return type.__call__(cls, *args, **kwargs)

            class Type(metaclass=TypeMetaclass):
                def __init__(self, obj):
                    self._obj = obj

        for i in range(100):
            Type(i)
        self.assertEqual(calls, 100)

    def test_specialization_class_call_doesnt_crash(self):
        # gh-123185

        with torch._dynamo.error_on_graph_break(False):
            class Foo:
                def __init__(self, arg):
                    pass

        for _ in range(8):
            try:
                Foo()
            except:
                pass


from _testinternalcapi import has_inline_values

Py_TPFLAGS_MANAGED_DICT = (1 << 2)

class Plain:
    pass


class WithAttrs:

    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        self.d = 4


@skipIfTorchDynamo("CPython only")
class TestInlineValues(CPythonTestCase):

    def test_flags(self):
        self.assertEqual(Plain.__flags__ & Py_TPFLAGS_MANAGED_DICT, Py_TPFLAGS_MANAGED_DICT)
        self.assertEqual(WithAttrs.__flags__ & Py_TPFLAGS_MANAGED_DICT, Py_TPFLAGS_MANAGED_DICT)

    def test_has_inline_values(self):
        c = Plain()
        self.assertTrue(has_inline_values(c))
        del c.__dict__
        self.assertFalse(has_inline_values(c))

    def test_instances(self):
        self.assertTrue(has_inline_values(Plain()))
        self.assertTrue(has_inline_values(WithAttrs()))

    def test_inspect_dict(self):
        for cls in (Plain, WithAttrs):
            c = cls()
            c.__dict__
            self.assertTrue(has_inline_values(c))

    def test_update_dict(self):
        d = { "e": 5, "f": 6 }
        for cls in (Plain, WithAttrs):
            c = cls()
            c.__dict__.update(d)
            self.assertTrue(has_inline_values(c))

    @staticmethod
    def set_100(obj):
        for i in range(100):
            setattr(obj, f"a{i}", i)

    def check_100(self, obj):
        for i in range(100):
            self.assertEqual(getattr(obj, f"a{i}"), i)

    def test_many_attributes(self):
        with torch._dynamo.error_on_graph_break(False):
            class C: pass
        c = C()
        self.assertTrue(has_inline_values(c))
        self.set_100(c)
        self.assertFalse(has_inline_values(c))
        self.check_100(c)
        c = C()
        self.assertTrue(has_inline_values(c))

    def test_many_attributes_with_dict(self):
        class C: pass
        c = C()
        d = c.__dict__
        self.assertTrue(has_inline_values(c))
        self.set_100(c)
        self.assertFalse(has_inline_values(c))
        self.check_100(c)

    def test_bug_117750(self):
        "Aborted on 3.13a6"
        class C:
            def __init__(self):
                self.__dict__.clear()

        obj = C()
        self.assertEqual(obj.__dict__, {})
        obj.foo = None # Aborted here
        self.assertEqual(obj.__dict__, {"foo":None})

    def test_store_attr_deleted_dict(self):
        class Foo:
            pass

        f = Foo()
        del f.__dict__
        f.a = 3
        self.assertEqual(f.a, 3)

    def test_rematerialize_object_dict(self):
        # gh-121860: rematerializing an object's managed dictionary after it
        # had been deleted caused a crash.
        class Foo: pass
        f = Foo()
        f.__dict__["attr"] = 1
        del f.__dict__

        # Using a str subclass is a way to trigger the re-materialization
        class StrSubclass(str): pass
        self.assertFalse(hasattr(f, StrSubclass("attr")))

        # Changing the __class__ also triggers the re-materialization
        class Bar: pass
        f.__class__ = Bar
        self.assertIsInstance(f, Bar)
        self.assertEqual(f.__dict__, {})

    def test_store_attr_type_cache(self):
        """Verifies that the type cache doesn't provide a value which  is
        inconsistent from the dict."""
        class X:
            def __del__(inner_self):
                v = C.a
                self.assertEqual(v, C.__dict__['a'])

        class C:
            a = X()

        # prime the cache
        C.a
        C.a

        # destructor shouldn't be able to see inconsistent state
        C.a = X()
        C.a = X()

    @cpython_only
    def test_detach_materialized_dict_no_memory(self):
        # Skip test if _testcapi is not available:
        import_helper.import_module('_testcapi')

        code = """if 1:
            import test.support
            import _testcapi

            class A:
                def __init__(self):
                    self.a = 1
                    self.b = 2
            a = A()
            d = a.__dict__
            with test.support.catch_unraisable_exception() as ex:
                _testcapi.set_nomemory(0, 1)
                del a
                assert ex.unraisable.exc_type is MemoryError
            try:
                d["a"]
            except KeyError:
                pass
            else:
                assert False, "KeyError not raised"
        """
        rc, out, err = script_helper.assert_python_ok("-c", code)
        self.assertEqual(rc, 0)
        self.assertFalse(out, msg=out.decode('utf-8'))
        self.assertFalse(err, msg=err.decode('utf-8'))

if __name__ == "__main__":
    run_tests()
