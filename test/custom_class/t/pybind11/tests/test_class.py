import pytest

from pybind11_tests import class_ as m
from pybind11_tests import UserType, ConstructorStats


def test_repr():
    # In Python 3.3+, repr() accesses __qualname__
    assert "pybind11_type" in repr(type(UserType))
    assert "UserType" in repr(UserType)


def test_instance(msg):
    with pytest.raises(TypeError) as excinfo:
        m.NoConstructor()
    assert msg(excinfo.value) == "m.class_.NoConstructor: No constructor defined!"

    instance = m.NoConstructor.new_instance()

    cstats = ConstructorStats.get(m.NoConstructor)
    assert cstats.alive() == 1
    del instance
    assert cstats.alive() == 0


def test_docstrings(doc):
    assert doc(UserType) == "A `py::class_` type for testing"
    assert UserType.__name__ == "UserType"
    assert UserType.__module__ == "pybind11_tests"
    assert UserType.get_value.__name__ == "get_value"
    assert UserType.get_value.__module__ == "pybind11_tests"

    assert doc(UserType.get_value) == """
        get_value(self: m.UserType) -> int

        Get value using a method
    """
    assert doc(UserType.value) == "Get/set value using a property"

    assert doc(m.NoConstructor.new_instance) == """
        new_instance() -> m.class_.NoConstructor

        Return an instance
    """


def test_qualname(doc):
    """Tests that a properly qualified name is set in __qualname__ (even in pre-3.3, where we
    backport the attribute) and that generated docstrings properly use it and the module name"""
    assert m.NestBase.__qualname__ == "NestBase"
    assert m.NestBase.Nested.__qualname__ == "NestBase.Nested"

    assert doc(m.NestBase.__init__) == """
        __init__(self: m.class_.NestBase) -> None
    """
    assert doc(m.NestBase.g) == """
        g(self: m.class_.NestBase, arg0: m.class_.NestBase.Nested) -> None
    """
    assert doc(m.NestBase.Nested.__init__) == """
        __init__(self: m.class_.NestBase.Nested) -> None
    """
    assert doc(m.NestBase.Nested.fn) == """
        fn(self: m.class_.NestBase.Nested, arg0: int, arg1: m.class_.NestBase, arg2: m.class_.NestBase.Nested) -> None
    """  # noqa: E501 line too long
    assert doc(m.NestBase.Nested.fa) == """
        fa(self: m.class_.NestBase.Nested, a: int, b: m.class_.NestBase, c: m.class_.NestBase.Nested) -> None
    """  # noqa: E501 line too long
    assert m.NestBase.__module__ == "pybind11_tests.class_"
    assert m.NestBase.Nested.__module__ == "pybind11_tests.class_"


def test_inheritance(msg):
    roger = m.Rabbit('Rabbit')
    assert roger.name() + " is a " + roger.species() == "Rabbit is a parrot"
    assert m.pet_name_species(roger) == "Rabbit is a parrot"

    polly = m.Pet('Polly', 'parrot')
    assert polly.name() + " is a " + polly.species() == "Polly is a parrot"
    assert m.pet_name_species(polly) == "Polly is a parrot"

    molly = m.Dog('Molly')
    assert molly.name() + " is a " + molly.species() == "Molly is a dog"
    assert m.pet_name_species(molly) == "Molly is a dog"

    fred = m.Hamster('Fred')
    assert fred.name() + " is a " + fred.species() == "Fred is a rodent"

    assert m.dog_bark(molly) == "Woof!"

    with pytest.raises(TypeError) as excinfo:
        m.dog_bark(polly)
    assert msg(excinfo.value) == """
        dog_bark(): incompatible function arguments. The following argument types are supported:
            1. (arg0: m.class_.Dog) -> str

        Invoked with: <m.class_.Pet object at 0>
    """

    with pytest.raises(TypeError) as excinfo:
        m.Chimera("lion", "goat")
    assert "No constructor defined!" in str(excinfo.value)


def test_automatic_upcasting():
    assert type(m.return_class_1()).__name__ == "DerivedClass1"
    assert type(m.return_class_2()).__name__ == "DerivedClass2"
    assert type(m.return_none()).__name__ == "NoneType"
    # Repeat these a few times in a random order to ensure no invalid caching is applied
    assert type(m.return_class_n(1)).__name__ == "DerivedClass1"
    assert type(m.return_class_n(2)).__name__ == "DerivedClass2"
    assert type(m.return_class_n(0)).__name__ == "BaseClass"
    assert type(m.return_class_n(2)).__name__ == "DerivedClass2"
    assert type(m.return_class_n(2)).__name__ == "DerivedClass2"
    assert type(m.return_class_n(0)).__name__ == "BaseClass"
    assert type(m.return_class_n(1)).__name__ == "DerivedClass1"


def test_isinstance():
    objects = [tuple(), dict(), m.Pet("Polly", "parrot")] + [m.Dog("Molly")] * 4
    expected = (True, True, True, True, True, False, False)
    assert m.check_instances(objects) == expected


def test_mismatched_holder():
    import re

    with pytest.raises(RuntimeError) as excinfo:
        m.mismatched_holder_1()
    assert re.match('generic_type: type ".*MismatchDerived1" does not have a non-default '
                    'holder type while its base ".*MismatchBase1" does', str(excinfo.value))

    with pytest.raises(RuntimeError) as excinfo:
        m.mismatched_holder_2()
    assert re.match('generic_type: type ".*MismatchDerived2" has a non-default holder type '
                    'while its base ".*MismatchBase2" does not', str(excinfo.value))


def test_override_static():
    """#511: problem with inheritance + overwritten def_static"""
    b = m.MyBase.make()
    d1 = m.MyDerived.make2()
    d2 = m.MyDerived.make()

    assert isinstance(b, m.MyBase)
    assert isinstance(d1, m.MyDerived)
    assert isinstance(d2, m.MyDerived)


def test_implicit_conversion_life_support():
    """Ensure the lifetime of temporary objects created for implicit conversions"""
    assert m.implicitly_convert_argument(UserType(5)) == 5
    assert m.implicitly_convert_variable(UserType(5)) == 5

    assert "outside a bound function" in m.implicitly_convert_variable_fail(UserType(5))


def test_operator_new_delete(capture):
    """Tests that class-specific operator new/delete functions are invoked"""

    class SubAliased(m.AliasedHasOpNewDelSize):
        pass

    with capture:
        a = m.HasOpNewDel()
        b = m.HasOpNewDelSize()
        d = m.HasOpNewDelBoth()
    assert capture == """
        A new 8
        B new 4
        D new 32
    """
    sz_alias = str(m.AliasedHasOpNewDelSize.size_alias)
    sz_noalias = str(m.AliasedHasOpNewDelSize.size_noalias)
    with capture:
        c = m.AliasedHasOpNewDelSize()
        c2 = SubAliased()
    assert capture == (
        "C new " + sz_noalias + "\n" +
        "C new " + sz_alias + "\n"
    )

    with capture:
        del a
        pytest.gc_collect()
        del b
        pytest.gc_collect()
        del d
        pytest.gc_collect()
    assert capture == """
        A delete
        B delete 4
        D delete
    """

    with capture:
        del c
        pytest.gc_collect()
        del c2
        pytest.gc_collect()
    assert capture == (
        "C delete " + sz_noalias + "\n" +
        "C delete " + sz_alias + "\n"
    )


def test_bind_protected_functions():
    """Expose protected member functions to Python using a helper class"""
    a = m.ProtectedA()
    assert a.foo() == 42

    b = m.ProtectedB()
    assert b.foo() == 42

    class C(m.ProtectedB):
        def __init__(self):
            m.ProtectedB.__init__(self)

        def foo(self):
            return 0

    c = C()
    assert c.foo() == 0


def test_brace_initialization():
    """ Tests that simple POD classes can be constructed using C++11 brace initialization """
    a = m.BraceInitialization(123, "test")
    assert a.field1 == 123
    assert a.field2 == "test"

    # Tests that a non-simple class doesn't get brace initialization (if the
    # class defines an initializer_list constructor, in particular, it would
    # win over the expected constructor).
    b = m.NoBraceInitialization([123, 456])
    assert b.vec == [123, 456]


@pytest.unsupported_on_pypy
def test_class_refcount():
    """Instances must correctly increase/decrease the reference count of their types (#1029)"""
    from sys import getrefcount

    class PyDog(m.Dog):
        pass

    for cls in m.Dog, PyDog:
        refcount_1 = getrefcount(cls)
        molly = [cls("Molly") for _ in range(10)]
        refcount_2 = getrefcount(cls)

        del molly
        pytest.gc_collect()
        refcount_3 = getrefcount(cls)

        assert refcount_1 == refcount_3
        assert refcount_2 > refcount_1


def test_reentrant_implicit_conversion_failure(msg):
    # ensure that there is no runaway reentrant implicit conversion (#1035)
    with pytest.raises(TypeError) as excinfo:
        m.BogusImplicitConversion(0)
    assert msg(excinfo.value) == '''
        __init__(): incompatible constructor arguments. The following argument types are supported:
            1. m.class_.BogusImplicitConversion(arg0: m.class_.BogusImplicitConversion)

        Invoked with: 0
    '''


def test_error_after_conversions():
    with pytest.raises(TypeError) as exc_info:
        m.test_error_after_conversions("hello")
    assert str(exc_info.value).startswith(
        "Unable to convert function return value to a Python type!")


def test_aligned():
    if hasattr(m, "Aligned"):
        p = m.Aligned().ptr()
        assert p % 1024 == 0
