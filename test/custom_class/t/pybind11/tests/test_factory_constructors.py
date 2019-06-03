import pytest
import re

from pybind11_tests import factory_constructors as m
from pybind11_tests.factory_constructors import tag
from pybind11_tests import ConstructorStats


def test_init_factory_basic():
    """Tests py::init_factory() wrapper around various ways of returning the object"""

    cstats = [ConstructorStats.get(c) for c in [m.TestFactory1, m.TestFactory2, m.TestFactory3]]
    cstats[0].alive()  # force gc
    n_inst = ConstructorStats.detail_reg_inst()

    x1 = m.TestFactory1(tag.unique_ptr, 3)
    assert x1.value == "3"
    y1 = m.TestFactory1(tag.pointer)
    assert y1.value == "(empty)"
    z1 = m.TestFactory1("hi!")
    assert z1.value == "hi!"

    assert ConstructorStats.detail_reg_inst() == n_inst + 3

    x2 = m.TestFactory2(tag.move)
    assert x2.value == "(empty2)"
    y2 = m.TestFactory2(tag.pointer, 7)
    assert y2.value == "7"
    z2 = m.TestFactory2(tag.unique_ptr, "hi again")
    assert z2.value == "hi again"

    assert ConstructorStats.detail_reg_inst() == n_inst + 6

    x3 = m.TestFactory3(tag.shared_ptr)
    assert x3.value == "(empty3)"
    y3 = m.TestFactory3(tag.pointer, 42)
    assert y3.value == "42"
    z3 = m.TestFactory3("bye")
    assert z3.value == "bye"

    with pytest.raises(TypeError) as excinfo:
        m.TestFactory3(tag.null_ptr)
    assert str(excinfo.value) == "pybind11::init(): factory function returned nullptr"

    assert [i.alive() for i in cstats] == [3, 3, 3]
    assert ConstructorStats.detail_reg_inst() == n_inst + 9

    del x1, y2, y3, z3
    assert [i.alive() for i in cstats] == [2, 2, 1]
    assert ConstructorStats.detail_reg_inst() == n_inst + 5
    del x2, x3, y1, z1, z2
    assert [i.alive() for i in cstats] == [0, 0, 0]
    assert ConstructorStats.detail_reg_inst() == n_inst

    assert [i.values() for i in cstats] == [
        ["3", "hi!"],
        ["7", "hi again"],
        ["42", "bye"]
    ]
    assert [i.default_constructions for i in cstats] == [1, 1, 1]


def test_init_factory_signature(msg):
    with pytest.raises(TypeError) as excinfo:
        m.TestFactory1("invalid", "constructor", "arguments")
    assert msg(excinfo.value) == """
        __init__(): incompatible constructor arguments. The following argument types are supported:
            1. m.factory_constructors.TestFactory1(arg0: m.factory_constructors.tag.unique_ptr_tag, arg1: int)
            2. m.factory_constructors.TestFactory1(arg0: str)
            3. m.factory_constructors.TestFactory1(arg0: m.factory_constructors.tag.pointer_tag)
            4. m.factory_constructors.TestFactory1(arg0: handle, arg1: int, arg2: handle)

        Invoked with: 'invalid', 'constructor', 'arguments'
    """  # noqa: E501 line too long

    assert msg(m.TestFactory1.__init__.__doc__) == """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: m.factory_constructors.TestFactory1, arg0: m.factory_constructors.tag.unique_ptr_tag, arg1: int) -> None

        2. __init__(self: m.factory_constructors.TestFactory1, arg0: str) -> None

        3. __init__(self: m.factory_constructors.TestFactory1, arg0: m.factory_constructors.tag.pointer_tag) -> None

        4. __init__(self: m.factory_constructors.TestFactory1, arg0: handle, arg1: int, arg2: handle) -> None
    """  # noqa: E501 line too long


def test_init_factory_casting():
    """Tests py::init_factory() wrapper with various upcasting and downcasting returns"""

    cstats = [ConstructorStats.get(c) for c in [m.TestFactory3, m.TestFactory4, m.TestFactory5]]
    cstats[0].alive()  # force gc
    n_inst = ConstructorStats.detail_reg_inst()

    # Construction from derived references:
    a = m.TestFactory3(tag.pointer, tag.TF4, 4)
    assert a.value == "4"
    b = m.TestFactory3(tag.shared_ptr, tag.TF4, 5)
    assert b.value == "5"
    c = m.TestFactory3(tag.pointer, tag.TF5, 6)
    assert c.value == "6"
    d = m.TestFactory3(tag.shared_ptr, tag.TF5, 7)
    assert d.value == "7"

    assert ConstructorStats.detail_reg_inst() == n_inst + 4

    # Shared a lambda with TF3:
    e = m.TestFactory4(tag.pointer, tag.TF4, 8)
    assert e.value == "8"

    assert ConstructorStats.detail_reg_inst() == n_inst + 5
    assert [i.alive() for i in cstats] == [5, 3, 2]

    del a
    assert [i.alive() for i in cstats] == [4, 2, 2]
    assert ConstructorStats.detail_reg_inst() == n_inst + 4

    del b, c, e
    assert [i.alive() for i in cstats] == [1, 0, 1]
    assert ConstructorStats.detail_reg_inst() == n_inst + 1

    del d
    assert [i.alive() for i in cstats] == [0, 0, 0]
    assert ConstructorStats.detail_reg_inst() == n_inst

    assert [i.values() for i in cstats] == [
        ["4", "5", "6", "7", "8"],
        ["4", "5", "8"],
        ["6", "7"]
    ]


def test_init_factory_alias():
    """Tests py::init_factory() wrapper with value conversions and alias types"""

    cstats = [m.TestFactory6.get_cstats(), m.TestFactory6.get_alias_cstats()]
    cstats[0].alive()  # force gc
    n_inst = ConstructorStats.detail_reg_inst()

    a = m.TestFactory6(tag.base, 1)
    assert a.get() == 1
    assert not a.has_alias()
    b = m.TestFactory6(tag.alias, "hi there")
    assert b.get() == 8
    assert b.has_alias()
    c = m.TestFactory6(tag.alias, 3)
    assert c.get() == 3
    assert c.has_alias()
    d = m.TestFactory6(tag.alias, tag.pointer, 4)
    assert d.get() == 4
    assert d.has_alias()
    e = m.TestFactory6(tag.base, tag.pointer, 5)
    assert e.get() == 5
    assert not e.has_alias()
    f = m.TestFactory6(tag.base, tag.alias, tag.pointer, 6)
    assert f.get() == 6
    assert f.has_alias()

    assert ConstructorStats.detail_reg_inst() == n_inst + 6
    assert [i.alive() for i in cstats] == [6, 4]

    del a, b, e
    assert [i.alive() for i in cstats] == [3, 3]
    assert ConstructorStats.detail_reg_inst() == n_inst + 3
    del f, c, d
    assert [i.alive() for i in cstats] == [0, 0]
    assert ConstructorStats.detail_reg_inst() == n_inst

    class MyTest(m.TestFactory6):
        def __init__(self, *args):
            m.TestFactory6.__init__(self, *args)

        def get(self):
            return -5 + m.TestFactory6.get(self)

    # Return Class by value, moved into new alias:
    z = MyTest(tag.base, 123)
    assert z.get() == 118
    assert z.has_alias()

    # Return alias by value, moved into new alias:
    y = MyTest(tag.alias, "why hello!")
    assert y.get() == 5
    assert y.has_alias()

    # Return Class by pointer, moved into new alias then original destroyed:
    x = MyTest(tag.base, tag.pointer, 47)
    assert x.get() == 42
    assert x.has_alias()

    assert ConstructorStats.detail_reg_inst() == n_inst + 3
    assert [i.alive() for i in cstats] == [3, 3]
    del x, y, z
    assert [i.alive() for i in cstats] == [0, 0]
    assert ConstructorStats.detail_reg_inst() == n_inst

    assert [i.values() for i in cstats] == [
        ["1", "8", "3", "4", "5", "6", "123", "10", "47"],
        ["hi there", "3", "4", "6", "move", "123", "why hello!", "move", "47"]
    ]


def test_init_factory_dual():
    """Tests init factory functions with dual main/alias factory functions"""
    from pybind11_tests.factory_constructors import TestFactory7

    cstats = [TestFactory7.get_cstats(), TestFactory7.get_alias_cstats()]
    cstats[0].alive()  # force gc
    n_inst = ConstructorStats.detail_reg_inst()

    class PythFactory7(TestFactory7):
        def get(self):
            return 100 + TestFactory7.get(self)

    a1 = TestFactory7(1)
    a2 = PythFactory7(2)
    assert a1.get() == 1
    assert a2.get() == 102
    assert not a1.has_alias()
    assert a2.has_alias()

    b1 = TestFactory7(tag.pointer, 3)
    b2 = PythFactory7(tag.pointer, 4)
    assert b1.get() == 3
    assert b2.get() == 104
    assert not b1.has_alias()
    assert b2.has_alias()

    c1 = TestFactory7(tag.mixed, 5)
    c2 = PythFactory7(tag.mixed, 6)
    assert c1.get() == 5
    assert c2.get() == 106
    assert not c1.has_alias()
    assert c2.has_alias()

    d1 = TestFactory7(tag.base, tag.pointer, 7)
    d2 = PythFactory7(tag.base, tag.pointer, 8)
    assert d1.get() == 7
    assert d2.get() == 108
    assert not d1.has_alias()
    assert d2.has_alias()

    # Both return an alias; the second multiplies the value by 10:
    e1 = TestFactory7(tag.alias, tag.pointer, 9)
    e2 = PythFactory7(tag.alias, tag.pointer, 10)
    assert e1.get() == 9
    assert e2.get() == 200
    assert e1.has_alias()
    assert e2.has_alias()

    f1 = TestFactory7(tag.shared_ptr, tag.base, 11)
    f2 = PythFactory7(tag.shared_ptr, tag.base, 12)
    assert f1.get() == 11
    assert f2.get() == 112
    assert not f1.has_alias()
    assert f2.has_alias()

    g1 = TestFactory7(tag.shared_ptr, tag.invalid_base, 13)
    assert g1.get() == 13
    assert not g1.has_alias()
    with pytest.raises(TypeError) as excinfo:
        PythFactory7(tag.shared_ptr, tag.invalid_base, 14)
    assert (str(excinfo.value) ==
            "pybind11::init(): construction failed: returned holder-wrapped instance is not an "
            "alias instance")

    assert [i.alive() for i in cstats] == [13, 7]
    assert ConstructorStats.detail_reg_inst() == n_inst + 13

    del a1, a2, b1, d1, e1, e2
    assert [i.alive() for i in cstats] == [7, 4]
    assert ConstructorStats.detail_reg_inst() == n_inst + 7
    del b2, c1, c2, d2, f1, f2, g1
    assert [i.alive() for i in cstats] == [0, 0]
    assert ConstructorStats.detail_reg_inst() == n_inst

    assert [i.values() for i in cstats] == [
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "100", "11", "12", "13", "14"],
        ["2", "4", "6", "8", "9", "100", "12"]
    ]


def test_no_placement_new(capture):
    """Prior to 2.2, `py::init<...>` relied on the type supporting placement
    new; this tests a class without placement new support."""
    with capture:
        a = m.NoPlacementNew(123)

    found = re.search(r'^operator new called, returning (\d+)\n$', str(capture))
    assert found
    assert a.i == 123
    with capture:
        del a
        pytest.gc_collect()
    assert capture == "operator delete called on " + found.group(1)

    with capture:
        b = m.NoPlacementNew()

    found = re.search(r'^operator new called, returning (\d+)\n$', str(capture))
    assert found
    assert b.i == 100
    with capture:
        del b
        pytest.gc_collect()
    assert capture == "operator delete called on " + found.group(1)


def test_multiple_inheritance():
    class MITest(m.TestFactory1, m.TestFactory2):
        def __init__(self):
            m.TestFactory1.__init__(self, tag.unique_ptr, 33)
            m.TestFactory2.__init__(self, tag.move)

    a = MITest()
    assert m.TestFactory1.value.fget(a) == "33"
    assert m.TestFactory2.value.fget(a) == "(empty2)"


def create_and_destroy(*args):
    a = m.NoisyAlloc(*args)
    print("---")
    del a
    pytest.gc_collect()


def strip_comments(s):
    return re.sub(r'\s+#.*', '', s)


def test_reallocations(capture, msg):
    """When the constructor is overloaded, previous overloads can require a preallocated value.
    This test makes sure that such preallocated values only happen when they might be necessary,
    and that they are deallocated properly"""

    pytest.gc_collect()

    with capture:
        create_and_destroy(1)
    assert msg(capture) == """
        noisy new
        noisy placement new
        NoisyAlloc(int 1)
        ---
        ~NoisyAlloc()
        noisy delete
    """
    with capture:
        create_and_destroy(1.5)
    assert msg(capture) == strip_comments("""
        noisy new               # allocation required to attempt first overload
        noisy delete            # have to dealloc before considering factory init overload
        noisy new               # pointer factory calling "new", part 1: allocation
        NoisyAlloc(double 1.5)  # ... part two, invoking constructor
        ---
        ~NoisyAlloc()  # Destructor
        noisy delete   # operator delete
    """)

    with capture:
        create_and_destroy(2, 3)
    assert msg(capture) == strip_comments("""
        noisy new          # pointer factory calling "new", allocation
        NoisyAlloc(int 2)  # constructor
        ---
        ~NoisyAlloc()  # Destructor
        noisy delete   # operator delete
    """)

    with capture:
        create_and_destroy(2.5, 3)
    assert msg(capture) == strip_comments("""
        NoisyAlloc(double 2.5)  # construction (local func variable: operator_new not called)
        noisy new               # return-by-value "new" part 1: allocation
        ~NoisyAlloc()           # moved-away local func variable destruction
        ---
        ~NoisyAlloc()  # Destructor
        noisy delete   # operator delete
    """)

    with capture:
        create_and_destroy(3.5, 4.5)
    assert msg(capture) == strip_comments("""
        noisy new               # preallocation needed before invoking placement-new overload
        noisy placement new     # Placement new
        NoisyAlloc(double 3.5)  # construction
        ---
        ~NoisyAlloc()  # Destructor
        noisy delete   # operator delete
    """)

    with capture:
        create_and_destroy(4, 0.5)
    assert msg(capture) == strip_comments("""
        noisy new          # preallocation needed before invoking placement-new overload
        noisy delete       # deallocation of preallocated storage
        noisy new          # Factory pointer allocation
        NoisyAlloc(int 4)  # factory pointer construction
        ---
        ~NoisyAlloc()  # Destructor
        noisy delete   # operator delete
    """)

    with capture:
        create_and_destroy(5, "hi")
    assert msg(capture) == strip_comments("""
        noisy new            # preallocation needed before invoking first placement new
        noisy delete         # delete before considering new-style constructor
        noisy new            # preallocation for second placement new
        noisy placement new  # Placement new in the second placement new overload
        NoisyAlloc(int 5)    # construction
        ---
        ~NoisyAlloc()  # Destructor
        noisy delete   # operator delete
    """)


@pytest.unsupported_on_py2
def test_invalid_self():
    """Tests invocation of the pybind-registered base class with an invalid `self` argument.  You
    can only actually do this on Python 3: Python 2 raises an exception itself if you try."""
    class NotPybindDerived(object):
        pass

    # Attempts to initialize with an invalid type passed as `self`:
    class BrokenTF1(m.TestFactory1):
        def __init__(self, bad):
            if bad == 1:
                a = m.TestFactory2(tag.pointer, 1)
                m.TestFactory1.__init__(a, tag.pointer)
            elif bad == 2:
                a = NotPybindDerived()
                m.TestFactory1.__init__(a, tag.pointer)

    # Same as above, but for a class with an alias:
    class BrokenTF6(m.TestFactory6):
        def __init__(self, bad):
            if bad == 1:
                a = m.TestFactory2(tag.pointer, 1)
                m.TestFactory6.__init__(a, tag.base, 1)
            elif bad == 2:
                a = m.TestFactory2(tag.pointer, 1)
                m.TestFactory6.__init__(a, tag.alias, 1)
            elif bad == 3:
                m.TestFactory6.__init__(NotPybindDerived.__new__(NotPybindDerived), tag.base, 1)
            elif bad == 4:
                m.TestFactory6.__init__(NotPybindDerived.__new__(NotPybindDerived), tag.alias, 1)

    for arg in (1, 2):
        with pytest.raises(TypeError) as excinfo:
            BrokenTF1(arg)
        assert str(excinfo.value) == "__init__(self, ...) called with invalid `self` argument"

    for arg in (1, 2, 3, 4):
        with pytest.raises(TypeError) as excinfo:
            BrokenTF6(arg)
        assert str(excinfo.value) == "__init__(self, ...) called with invalid `self` argument"
