from sympy.core.basic import Basic
from sympy.core.numbers import Rational
from sympy.core.singleton import S, Singleton

def test_Singleton():

    class MySingleton(Basic, metaclass=Singleton):
        pass

    MySingleton() # force instantiation
    assert MySingleton() is not Basic()
    assert MySingleton() is MySingleton()
    assert S.MySingleton is MySingleton()

    class MySingleton_sub(MySingleton):
        pass

    MySingleton_sub()
    assert MySingleton_sub() is not MySingleton()
    assert MySingleton_sub() is MySingleton_sub()

def test_singleton_redefinition():
    class TestSingleton(Basic, metaclass=Singleton):
        pass

    assert TestSingleton() is S.TestSingleton

    class TestSingleton(Basic, metaclass=Singleton):
        pass

    assert TestSingleton() is S.TestSingleton

def test_names_in_namespace():
    # Every singleton name should be accessible from the 'from sympy import *'
    # namespace in addition to the S object. However, it does not need to be
    # by the same name (e.g., oo instead of S.Infinity).

    # As a general rule, things should only be added to the singleton registry
    # if they are used often enough that code can benefit either from the
    # performance benefit of being able to use 'is' (this only matters in very
    # tight loops), or from the memory savings of having exactly one instance
    # (this matters for the numbers singletons, but very little else). The
    # singleton registry is already a bit overpopulated, and things cannot be
    # removed from it without breaking backwards compatibility. So if you got
    # here by adding something new to the singletons, ask yourself if it
    # really needs to be singletonized. Note that SymPy classes compare to one
    # another just fine, so Class() == Class() will give True even if each
    # Class() returns a new instance. Having unique instances is only
    # necessary for the above noted performance gains. It should not be needed
    # for any behavioral purposes.

    # If you determine that something really should be a singleton, it must be
    # accessible to sympify() without using 'S' (hence this test). Also, its
    # str printer should print a form that does not use S. This is because
    # sympify() disables attribute lookups by default for safety purposes.
    d = {}
    exec('from sympy import *', d)

    for name in dir(S) + list(S._classes_to_install):
        if name.startswith('_'):
            continue
        if name == 'register':
            continue
        if isinstance(getattr(S, name), Rational):
            continue
        if getattr(S, name).__module__.startswith('sympy.physics'):
            continue
        if name in ['MySingleton', 'MySingleton_sub', 'TestSingleton']:
            # From the tests above
            continue
        if name == 'NegativeInfinity':
            # Accessible by -oo
            continue

        # Use is here to ensure it is the exact same object
        assert any(getattr(S, name) is i for i in d.values()), name
