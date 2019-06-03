import pytest
from pybind11_tests import opaque_types as m
from pybind11_tests import ConstructorStats, UserType


def test_string_list():
    lst = m.StringList()
    lst.push_back("Element 1")
    lst.push_back("Element 2")
    assert m.print_opaque_list(lst) == "Opaque list: [Element 1, Element 2]"
    assert lst.back() == "Element 2"

    for i, k in enumerate(lst, start=1):
        assert k == "Element {}".format(i)
    lst.pop_back()
    assert m.print_opaque_list(lst) == "Opaque list: [Element 1]"

    cvp = m.ClassWithSTLVecProperty()
    assert m.print_opaque_list(cvp.stringList) == "Opaque list: []"

    cvp.stringList = lst
    cvp.stringList.push_back("Element 3")
    assert m.print_opaque_list(cvp.stringList) == "Opaque list: [Element 1, Element 3]"


def test_pointers(msg):
    living_before = ConstructorStats.get(UserType).alive()
    assert m.get_void_ptr_value(m.return_void_ptr()) == 0x1234
    assert m.get_void_ptr_value(UserType())  # Should also work for other C++ types
    assert ConstructorStats.get(UserType).alive() == living_before

    with pytest.raises(TypeError) as excinfo:
        m.get_void_ptr_value([1, 2, 3])  # This should not work
    assert msg(excinfo.value) == """
        get_void_ptr_value(): incompatible function arguments. The following argument types are supported:
            1. (arg0: capsule) -> int

        Invoked with: [1, 2, 3]
    """  # noqa: E501 line too long

    assert m.return_null_str() is None
    assert m.get_null_str_value(m.return_null_str()) is not None

    ptr = m.return_unique_ptr()
    assert "StringList" in repr(ptr)
    assert m.print_opaque_list(ptr) == "Opaque list: [some value]"
