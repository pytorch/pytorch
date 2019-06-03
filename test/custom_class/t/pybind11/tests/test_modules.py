from pybind11_tests import modules as m
from pybind11_tests.modules import subsubmodule as ms
from pybind11_tests import ConstructorStats


def test_nested_modules():
    import pybind11_tests
    assert pybind11_tests.__name__ == "pybind11_tests"
    assert pybind11_tests.modules.__name__ == "pybind11_tests.modules"
    assert pybind11_tests.modules.subsubmodule.__name__ == "pybind11_tests.modules.subsubmodule"
    assert m.__name__ == "pybind11_tests.modules"
    assert ms.__name__ == "pybind11_tests.modules.subsubmodule"

    assert ms.submodule_func() == "submodule_func()"


def test_reference_internal():
    b = ms.B()
    assert str(b.get_a1()) == "A[1]"
    assert str(b.a1) == "A[1]"
    assert str(b.get_a2()) == "A[2]"
    assert str(b.a2) == "A[2]"

    b.a1 = ms.A(42)
    b.a2 = ms.A(43)
    assert str(b.get_a1()) == "A[42]"
    assert str(b.a1) == "A[42]"
    assert str(b.get_a2()) == "A[43]"
    assert str(b.a2) == "A[43]"

    astats, bstats = ConstructorStats.get(ms.A), ConstructorStats.get(ms.B)
    assert astats.alive() == 2
    assert bstats.alive() == 1
    del b
    assert astats.alive() == 0
    assert bstats.alive() == 0
    assert astats.values() == ['1', '2', '42', '43']
    assert bstats.values() == []
    assert astats.default_constructions == 0
    assert bstats.default_constructions == 1
    assert astats.copy_constructions == 0
    assert bstats.copy_constructions == 0
    # assert astats.move_constructions >= 0  # Don't invoke any
    # assert bstats.move_constructions >= 0  # Don't invoke any
    assert astats.copy_assignments == 2
    assert bstats.copy_assignments == 0
    assert astats.move_assignments == 0
    assert bstats.move_assignments == 0


def test_importing():
    from pybind11_tests.modules import OD
    from collections import OrderedDict

    assert OD is OrderedDict
    assert str(OD([(1, 'a'), (2, 'b')])) == "OrderedDict([(1, 'a'), (2, 'b')])"


def test_pydoc():
    """Pydoc needs to be able to provide help() for everything inside a pybind11 module"""
    import pybind11_tests
    import pydoc

    assert pybind11_tests.__name__ == "pybind11_tests"
    assert pybind11_tests.__doc__ == "pybind11 test module"
    assert pydoc.text.docmodule(pybind11_tests)


def test_duplicate_registration():
    """Registering two things with the same name"""

    assert m.duplicate_registration() == []
