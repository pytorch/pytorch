import pytest
from pybind11_tests import kwargs_and_defaults as m


def test_function_signatures(doc):
    assert doc(m.kw_func0) == "kw_func0(arg0: int, arg1: int) -> str"
    assert doc(m.kw_func1) == "kw_func1(x: int, y: int) -> str"
    assert doc(m.kw_func2) == "kw_func2(x: int = 100, y: int = 200) -> str"
    assert doc(m.kw_func3) == "kw_func3(data: str = 'Hello world!') -> None"
    assert doc(m.kw_func4) == "kw_func4(myList: List[int] = [13, 17]) -> str"
    assert doc(m.kw_func_udl) == "kw_func_udl(x: int, y: int = 300) -> str"
    assert doc(m.kw_func_udl_z) == "kw_func_udl_z(x: int, y: int = 0) -> str"
    assert doc(m.args_function) == "args_function(*args) -> tuple"
    assert doc(m.args_kwargs_function) == "args_kwargs_function(*args, **kwargs) -> tuple"
    assert doc(m.KWClass.foo0) == \
        "foo0(self: m.kwargs_and_defaults.KWClass, arg0: int, arg1: float) -> None"
    assert doc(m.KWClass.foo1) == \
        "foo1(self: m.kwargs_and_defaults.KWClass, x: int, y: float) -> None"


def test_named_arguments(msg):
    assert m.kw_func0(5, 10) == "x=5, y=10"

    assert m.kw_func1(5, 10) == "x=5, y=10"
    assert m.kw_func1(5, y=10) == "x=5, y=10"
    assert m.kw_func1(y=10, x=5) == "x=5, y=10"

    assert m.kw_func2() == "x=100, y=200"
    assert m.kw_func2(5) == "x=5, y=200"
    assert m.kw_func2(x=5) == "x=5, y=200"
    assert m.kw_func2(y=10) == "x=100, y=10"
    assert m.kw_func2(5, 10) == "x=5, y=10"
    assert m.kw_func2(x=5, y=10) == "x=5, y=10"

    with pytest.raises(TypeError) as excinfo:
        # noinspection PyArgumentList
        m.kw_func2(x=5, y=10, z=12)
    assert excinfo.match(
        r'(?s)^kw_func2\(\): incompatible.*Invoked with: kwargs: ((x=5|y=10|z=12)(, |$))' + '{3}$')

    assert m.kw_func4() == "{13 17}"
    assert m.kw_func4(myList=[1, 2, 3]) == "{1 2 3}"

    assert m.kw_func_udl(x=5, y=10) == "x=5, y=10"
    assert m.kw_func_udl_z(x=5) == "x=5, y=0"


def test_arg_and_kwargs():
    args = 'arg1_value', 'arg2_value', 3
    assert m.args_function(*args) == args

    args = 'a1', 'a2'
    kwargs = dict(arg3='a3', arg4=4)
    assert m.args_kwargs_function(*args, **kwargs) == (args, kwargs)


def test_mixed_args_and_kwargs(msg):
    mpa = m.mixed_plus_args
    mpk = m.mixed_plus_kwargs
    mpak = m.mixed_plus_args_kwargs
    mpakd = m.mixed_plus_args_kwargs_defaults

    assert mpa(1, 2.5, 4, 99.5, None) == (1, 2.5, (4, 99.5, None))
    assert mpa(1, 2.5) == (1, 2.5, ())
    with pytest.raises(TypeError) as excinfo:
        assert mpa(1)
    assert msg(excinfo.value) == """
        mixed_plus_args(): incompatible function arguments. The following argument types are supported:
            1. (arg0: int, arg1: float, *args) -> tuple

        Invoked with: 1
    """  # noqa: E501 line too long
    with pytest.raises(TypeError) as excinfo:
        assert mpa()
    assert msg(excinfo.value) == """
        mixed_plus_args(): incompatible function arguments. The following argument types are supported:
            1. (arg0: int, arg1: float, *args) -> tuple

        Invoked with:
    """  # noqa: E501 line too long

    assert mpk(-2, 3.5, pi=3.14159, e=2.71828) == (-2, 3.5, {'e': 2.71828, 'pi': 3.14159})
    assert mpak(7, 7.7, 7.77, 7.777, 7.7777, minusseven=-7) == (
        7, 7.7, (7.77, 7.777, 7.7777), {'minusseven': -7})
    assert mpakd() == (1, 3.14159, (), {})
    assert mpakd(3) == (3, 3.14159, (), {})
    assert mpakd(j=2.71828) == (1, 2.71828, (), {})
    assert mpakd(k=42) == (1, 3.14159, (), {'k': 42})
    assert mpakd(1, 1, 2, 3, 5, 8, then=13, followedby=21) == (
        1, 1, (2, 3, 5, 8), {'then': 13, 'followedby': 21})
    # Arguments specified both positionally and via kwargs should fail:
    with pytest.raises(TypeError) as excinfo:
        assert mpakd(1, i=1)
    assert msg(excinfo.value) == """
        mixed_plus_args_kwargs_defaults(): incompatible function arguments. The following argument types are supported:
            1. (i: int = 1, j: float = 3.14159, *args, **kwargs) -> tuple

        Invoked with: 1; kwargs: i=1
    """  # noqa: E501 line too long
    with pytest.raises(TypeError) as excinfo:
        assert mpakd(1, 2, j=1)
    assert msg(excinfo.value) == """
        mixed_plus_args_kwargs_defaults(): incompatible function arguments. The following argument types are supported:
            1. (i: int = 1, j: float = 3.14159, *args, **kwargs) -> tuple

        Invoked with: 1, 2; kwargs: j=1
    """  # noqa: E501 line too long


def test_args_refcount():
    """Issue/PR #1216 - py::args elements get double-inc_ref()ed when combined with regular
    arguments"""
    refcount = m.arg_refcount_h

    myval = 54321
    expected = refcount(myval)
    assert m.arg_refcount_h(myval) == expected
    assert m.arg_refcount_o(myval) == expected + 1
    assert m.arg_refcount_h(myval) == expected
    assert refcount(myval) == expected

    assert m.mixed_plus_args(1, 2.0, "a", myval) == (1, 2.0, ("a", myval))
    assert refcount(myval) == expected

    assert m.mixed_plus_kwargs(3, 4.0, a=1, b=myval) == (3, 4.0, {"a": 1, "b": myval})
    assert refcount(myval) == expected

    assert m.args_function(-1, myval) == (-1, myval)
    assert refcount(myval) == expected

    assert m.mixed_plus_args_kwargs(5, 6.0, myval, a=myval) == (5, 6.0, (myval,), {"a": myval})
    assert refcount(myval) == expected

    assert m.args_kwargs_function(7, 8, myval, a=1, b=myval) == \
        ((7, 8, myval), {"a": 1, "b": myval})
    assert refcount(myval) == expected

    exp3 = refcount(myval, myval, myval)
    assert m.args_refcount(myval, myval, myval) == (exp3, exp3, exp3)
    assert refcount(myval) == expected

    # This function takes the first arg as a `py::object` and the rest as a `py::args`.  Unlike the
    # previous case, when we have both positional and `py::args` we need to construct a new tuple
    # for the `py::args`; in the previous case, we could simply inc_ref and pass on Python's input
    # tuple without having to inc_ref the individual elements, but here we can't, hence the extra
    # refs.
    assert m.mixed_args_refcount(myval, myval, myval) == (exp3 + 3, exp3 + 3, exp3 + 3)
