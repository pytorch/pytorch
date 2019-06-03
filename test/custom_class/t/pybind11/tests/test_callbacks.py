import pytest
from pybind11_tests import callbacks as m


def test_callbacks():
    from functools import partial

    def func1():
        return "func1"

    def func2(a, b, c, d):
        return "func2", a, b, c, d

    def func3(a):
        return "func3({})".format(a)

    assert m.test_callback1(func1) == "func1"
    assert m.test_callback2(func2) == ("func2", "Hello", "x", True, 5)
    assert m.test_callback1(partial(func2, 1, 2, 3, 4)) == ("func2", 1, 2, 3, 4)
    assert m.test_callback1(partial(func3, "partial")) == "func3(partial)"
    assert m.test_callback3(lambda i: i + 1) == "func(43) = 44"

    f = m.test_callback4()
    assert f(43) == 44
    f = m.test_callback5()
    assert f(number=43) == 44


def test_bound_method_callback():
    # Bound Python method:
    class MyClass:
        def double(self, val):
            return 2 * val

    z = MyClass()
    assert m.test_callback3(z.double) == "func(43) = 86"

    z = m.CppBoundMethodTest()
    assert m.test_callback3(z.triple) == "func(43) = 129"


def test_keyword_args_and_generalized_unpacking():

    def f(*args, **kwargs):
        return args, kwargs

    assert m.test_tuple_unpacking(f) == (("positional", 1, 2, 3, 4, 5, 6), {})
    assert m.test_dict_unpacking(f) == (("positional", 1), {"key": "value", "a": 1, "b": 2})
    assert m.test_keyword_args(f) == ((), {"x": 10, "y": 20})
    assert m.test_unpacking_and_keywords1(f) == ((1, 2), {"c": 3, "d": 4})
    assert m.test_unpacking_and_keywords2(f) == (
        ("positional", 1, 2, 3, 4, 5),
        {"key": "value", "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )

    with pytest.raises(TypeError) as excinfo:
        m.test_unpacking_error1(f)
    assert "Got multiple values for keyword argument" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.test_unpacking_error2(f)
    assert "Got multiple values for keyword argument" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        m.test_arg_conversion_error1(f)
    assert "Unable to convert call argument" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        m.test_arg_conversion_error2(f)
    assert "Unable to convert call argument" in str(excinfo.value)


def test_lambda_closure_cleanup():
    m.test_cleanup()
    cstats = m.payload_cstats()
    assert cstats.alive() == 0
    assert cstats.copy_constructions == 1
    assert cstats.move_constructions >= 1


def test_cpp_function_roundtrip():
    """Test if passing a function pointer from C++ -> Python -> C++ yields the original pointer"""

    assert m.test_dummy_function(m.dummy_function) == "matches dummy_function: eval(1) = 2"
    assert (m.test_dummy_function(m.roundtrip(m.dummy_function)) ==
            "matches dummy_function: eval(1) = 2")
    assert m.roundtrip(None, expect_none=True) is None
    assert (m.test_dummy_function(lambda x: x + 2) ==
            "can't convert to function pointer: eval(1) = 3")

    with pytest.raises(TypeError) as excinfo:
        m.test_dummy_function(m.dummy_function2)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.test_dummy_function(lambda x, y: x + y)
    assert any(s in str(excinfo.value) for s in ("missing 1 required positional argument",
                                                 "takes exactly 2 arguments"))


def test_function_signatures(doc):
    assert doc(m.test_callback3) == "test_callback3(arg0: Callable[[int], int]) -> str"
    assert doc(m.test_callback4) == "test_callback4() -> Callable[[int], int]"


def test_movable_object():
    assert m.callback_with_movable(lambda _: None) is True
