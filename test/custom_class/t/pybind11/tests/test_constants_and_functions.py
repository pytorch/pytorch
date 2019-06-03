from pybind11_tests import constants_and_functions as m


def test_constants():
    assert m.some_constant == 14


def test_function_overloading():
    assert m.test_function() == "test_function()"
    assert m.test_function(7) == "test_function(7)"
    assert m.test_function(m.MyEnum.EFirstEntry) == "test_function(enum=1)"
    assert m.test_function(m.MyEnum.ESecondEntry) == "test_function(enum=2)"

    assert m.test_function() == "test_function()"
    assert m.test_function("abcd") == "test_function(char *)"
    assert m.test_function(1, 1.0) == "test_function(int, float)"
    assert m.test_function(1, 1.0) == "test_function(int, float)"
    assert m.test_function(2.0, 2) == "test_function(float, int)"


def test_bytes():
    assert m.print_bytes(m.return_bytes()) == "bytes[1 0 2 0]"


def test_exception_specifiers():
    c = m.C()
    assert c.m1(2) == 1
    assert c.m2(3) == 1
    assert c.m3(5) == 2
    assert c.m4(7) == 3
    assert c.m5(10) == 5
    assert c.m6(14) == 8
    assert c.m7(20) == 13
    assert c.m8(29) == 21

    assert m.f1(33) == 34
    assert m.f2(53) == 55
    assert m.f3(86) == 89
    assert m.f4(140) == 144
