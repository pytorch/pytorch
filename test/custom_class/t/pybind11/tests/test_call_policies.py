import pytest
from pybind11_tests import call_policies as m
from pybind11_tests import ConstructorStats


def test_keep_alive_argument(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChild(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChildKeepAlive(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == "Allocating child."
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_keep_alive_return_value(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChild()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChildKeepAlive()
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == "Allocating child."
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


# https://bitbucket.org/pypy/pypy/issues/2447
@pytest.unsupported_on_pypy
def test_alive_gc(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    p = m.ParentGC()
    p.addChildKeepAlive(m.Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)   # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_alive_gc_derived(capture):
    class Derived(m.Parent):
        pass

    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(m.Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)   # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_alive_gc_multi_derived(capture):
    class Derived(m.Parent, m.Child):
        def __init__(self):
            m.Parent.__init__(self)
            m.Child.__init__(self)

    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(m.Child())
    # +3 rather than +2 because Derived corresponds to two registered instances
    assert ConstructorStats.detail_reg_inst() == n_inst + 3
    lst = [p]
    lst.append(lst)   # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
        Releasing child.
    """


def test_return_none(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveChild()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == ""
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveParent()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == ""
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."


def test_keep_alive_constructor(capture):
    n_inst = ConstructorStats.detail_reg_inst()

    with capture:
        p = m.Parent(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == """
        Allocating child.
        Allocating parent.
    """
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_call_guard():
    assert m.unguarded_call() == "unguarded"
    assert m.guarded_call() == "guarded"

    assert m.multiple_guards_correct_order() == "guarded & guarded"
    assert m.multiple_guards_wrong_order() == "unguarded & guarded"

    if hasattr(m, "with_gil"):
        assert m.with_gil() == "GIL held"
        assert m.without_gil() == "GIL released"
