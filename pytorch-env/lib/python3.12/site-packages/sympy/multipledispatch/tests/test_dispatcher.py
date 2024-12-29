from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
                                         MethodDispatcher, halt_ordering,
                                         restart_ordering,
                                         ambiguity_register_error_ignore_dup)
from sympy.testing.pytest import raises, warns


def identity(x):
    return x


def inc(x):
    return x + 1


def dec(x):
    return x - 1


def test_dispatcher():
    f = Dispatcher('f')
    f.add((int,), inc)
    f.add((float,), dec)

    with warns(DeprecationWarning, test_stacklevel=False):
        assert f.resolve((int,)) == inc
    assert f.dispatch(int) is inc

    assert f(1) == 2
    assert f(1.0) == 0.0


def test_union_types():
    f = Dispatcher('f')
    f.register((int, float))(inc)

    assert f(1) == 2
    assert f(1.0) == 2.0


def test_dispatcher_as_decorator():
    f = Dispatcher('f')

    @f.register(int)
    def inc(x): # noqa:F811
        return x + 1

    @f.register(float) # noqa:F811
    def inc(x): # noqa:F811
        return x - 1

    assert f(1) == 2
    assert f(1.0) == 0.0


def test_register_instance_method():

    class Test:
        __init__ = MethodDispatcher('f')

        @__init__.register(list)
        def _init_list(self, data):
            self.data = data

        @__init__.register(object)
        def _init_obj(self, datum):
            self.data = [datum]

    a = Test(3)
    b = Test([3])
    assert a.data == b.data


def test_on_ambiguity():
    f = Dispatcher('f')

    def identity(x): return x

    ambiguities = [False]

    def on_ambiguity(dispatcher, amb):
        ambiguities[0] = True

    f.add((object, object), identity, on_ambiguity=on_ambiguity)
    assert not ambiguities[0]
    f.add((object, float), identity, on_ambiguity=on_ambiguity)
    assert not ambiguities[0]
    f.add((float, object), identity, on_ambiguity=on_ambiguity)
    assert ambiguities[0]


def test_raise_error_on_non_class():
    f = Dispatcher('f')
    assert raises(TypeError, lambda: f.add((1,), inc))


def test_docstring():

    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x + y

    def three(x, y):
        return x + y

    master_doc = 'Doc of the multimethod itself'

    f = Dispatcher('f', doc=master_doc)
    f.add((object, object), one)
    f.add((int, int), two)
    f.add((float, float), three)

    assert one.__doc__.strip() in f.__doc__
    assert two.__doc__.strip() in f.__doc__
    assert f.__doc__.find(one.__doc__.strip()) < \
        f.__doc__.find(two.__doc__.strip())
    assert 'object, object' in f.__doc__
    assert master_doc in f.__doc__


def test_help():
    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x + y

    def three(x, y):
        """ Docstring number three """
        return x + y

    master_doc = 'Doc of the multimethod itself'

    f = Dispatcher('f', doc=master_doc)
    f.add((object, object), one)
    f.add((int, int), two)
    f.add((float, float), three)

    assert f._help(1, 1) == two.__doc__
    assert f._help(1.0, 2.0) == three.__doc__


def test_source():
    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x - y

    master_doc = 'Doc of the multimethod itself'

    f = Dispatcher('f', doc=master_doc)
    f.add((int, int), one)
    f.add((float, float), two)

    assert 'x + y' in f._source(1, 1)
    assert 'x - y' in f._source(1.0, 1.0)


def test_source_raises_on_missing_function():
    f = Dispatcher('f')

    assert raises(TypeError, lambda: f.source(1))


def test_halt_method_resolution():
    g = [0]

    def on_ambiguity(a, b):
        g[0] += 1

    f = Dispatcher('f')

    halt_ordering()

    def func(*args):
        pass

    f.add((int, object), func)
    f.add((object, int), func)

    assert g == [0]

    restart_ordering(on_ambiguity=on_ambiguity)

    assert g == [1]

    assert set(f.ordering) == {(int, object), (object, int)}


def test_no_implementations():
    f = Dispatcher('f')
    assert raises(NotImplementedError, lambda: f('hello'))


def test_register_stacking():
    f = Dispatcher('f')

    @f.register(list)
    @f.register(tuple)
    def rev(x):
        return x[::-1]

    assert f((1, 2, 3)) == (3, 2, 1)
    assert f([1, 2, 3]) == [3, 2, 1]

    assert raises(NotImplementedError, lambda: f('hello'))
    assert rev('hello') == 'olleh'


def test_dispatch_method():
    f = Dispatcher('f')

    @f.register(list)
    def rev(x):
        return x[::-1]

    @f.register(int, int)
    def add(x, y):
        return x + y

    class MyList(list):
        pass

    assert f.dispatch(list) is rev
    assert f.dispatch(MyList) is rev
    assert f.dispatch(int, int) is add


def test_not_implemented():
    f = Dispatcher('f')

    @f.register(object)
    def _(x):
        return 'default'

    @f.register(int)
    def _(x):
        if x % 2 == 0:
            return 'even'
        else:
            raise MDNotImplementedError()

    assert f('hello') == 'default'  # default behavior
    assert f(2) == 'even'          # specialized behavior
    assert f(3) == 'default'       # fall bac to default behavior
    assert raises(NotImplementedError, lambda: f(1, 2))


def test_not_implemented_error():
    f = Dispatcher('f')

    @f.register(float)
    def _(a):
        raise MDNotImplementedError()

    assert raises(NotImplementedError, lambda: f(1.0))

def test_ambiguity_register_error_ignore_dup():
    f = Dispatcher('f')

    class A:
        pass
    class B(A):
        pass
    class C(A):
        pass

    # suppress warning for registering ambiguous signal
    f.add((A, B), lambda x,y: None, ambiguity_register_error_ignore_dup)
    f.add((B, A), lambda x,y: None, ambiguity_register_error_ignore_dup)
    f.add((A, C), lambda x,y: None, ambiguity_register_error_ignore_dup)
    f.add((C, A), lambda x,y: None, ambiguity_register_error_ignore_dup)

    # raises error if ambiguous signal is passed
    assert raises(NotImplementedError, lambda: f(B(), C()))
