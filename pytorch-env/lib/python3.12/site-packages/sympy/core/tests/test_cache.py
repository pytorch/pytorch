import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises

def test_cacheit_doc():
    @cacheit
    def testfn():
        "test docstring"
        pass

    assert testfn.__doc__ == "test docstring"
    assert testfn.__name__ == "testfn"

def test_cacheit_unhashable():
    @cacheit
    def testit(x):
        return x

    assert testit(1) == 1
    assert testit(1) == 1
    a = {}
    assert testit(a) == {}
    a[1] = 2
    assert testit(a) == {1: 2}

def test_cachit_exception():
    # Make sure the cache doesn't call functions multiple times when they
    # raise TypeError

    a = []

    @cacheit
    def testf(x):
        a.append(0)
        raise TypeError

    raises(TypeError, lambda: testf(1))
    assert len(a) == 1

    a.clear()
    # Unhashable type
    raises(TypeError, lambda: testf([]))
    assert len(a) == 1

    @cacheit
    def testf2(x):
        a.append(0)
        raise TypeError("Error")

    a.clear()
    raises(TypeError, lambda: testf2(1))
    assert len(a) == 1

    a.clear()
    # Unhashable type
    raises(TypeError, lambda: testf2([]))
    assert len(a) == 1

def test_cached_property():
    class A:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        @cached_property
        def prop(self):
            self.calls = self.calls + 1
            return self.value

    a = A(2)
    assert a.calls == 0
    assert a.prop == 2
    assert a.calls == 1
    assert a.prop == 2
    assert a.calls == 1
    b = A(None)
    assert b.prop == None


def test_lazy_function():
    module_name='xmlrpc.client'
    function_name = 'gzip_decode'
    lazy = lazy_function(module_name, function_name)
    assert lazy(b'') == b''
    assert module_name in sys.modules
    assert function_name in str(lazy)
    repr_lazy = repr(lazy)
    assert 'LazyFunction' in repr_lazy
    assert function_name in repr_lazy

    lazy = lazy_function('sympy.core.cache', 'cheap')
