"""
Important note on tests in this module - the Theano printing functions use a
global cache by default, which means that tests using it will modify global
state and thus not be independent from each other. Instead of using the "cache"
keyword argument each time, this module uses the theano_code_ and
theano_function_ functions defined below which default to using a new, empty
cache instead.
"""

import logging

from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy

theanologger = logging.getLogger('theano.configdefaults')
theanologger.setLevel(logging.CRITICAL)
theano = import_module('theano')
theanologger.setLevel(logging.WARNING)


if theano:
    import numpy as np
    ts = theano.scalar
    tt = theano.tensor
    xt, yt, zt = [tt.scalar(name, 'floatX') for name in 'xyz']
    Xt, Yt, Zt = [tt.tensor('floatX', (False, False), name=n) for n in 'XYZ']
else:
    #bin/test will not execute any tests now
    disabled = True

import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
        theano_function)


# Default set of matrix symbols for testing - make square so we can both
# multiply and perform elementwise operations between them.
X, Y, Z = [sy.MatrixSymbol(n, 4, 4) for n in 'XYZ']

# For testing AppliedUndef
f_t = sy.Function('f')(t)


def theano_code_(expr, **kwargs):
    """ Wrapper for theano_code that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return theano_code(expr, **kwargs)

def theano_function_(inputs, outputs, **kwargs):
    """ Wrapper for theano_function that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return theano_function(inputs, outputs, **kwargs)


def fgraph_of(*exprs):
    """ Transform SymPy expressions into Theano Computation.

    Parameters
    ==========
    exprs
        SymPy expressions

    Returns
    =======
    theano.gof.FunctionGraph
    """
    outs = list(map(theano_code_, exprs))
    ins = theano.gof.graph.inputs(outs)
    ins, outs = theano.gof.graph.clone(ins, outs)
    return theano.gof.FunctionGraph(ins, outs)


def theano_simplify(fgraph):
    """ Simplify a Theano Computation.

    Parameters
    ==========
    fgraph : theano.gof.FunctionGraph

    Returns
    =======
    theano.gof.FunctionGraph
    """
    mode = theano.compile.get_default_mode().excluding("fusion")
    fgraph = fgraph.clone()
    mode.optimizer.optimize(fgraph)
    return fgraph


def theq(a, b):
    """ Test two Theano objects for equality.

    Also accepts numeric types and lists/tuples of supported types.

    Note - debugprint() has a bug where it will accept numeric types but does
    not respect the "file" argument and in this case and instead prints the number
    to stdout and returns an empty string. This can lead to tests passing where
    they should fail because any two numbers will always compare as equal. To
    prevent this we treat numbers as a separate case.
    """
    numeric_types = (int, float, np.number)
    a_is_num = isinstance(a, numeric_types)
    b_is_num = isinstance(b, numeric_types)

    # Compare numeric types using regular equality
    if a_is_num or b_is_num:
        if not (a_is_num and b_is_num):
            return False

        return a == b

    # Compare sequences element-wise
    a_is_seq = isinstance(a, (tuple, list))
    b_is_seq = isinstance(b, (tuple, list))

    if a_is_seq or b_is_seq:
        if not (a_is_seq and b_is_seq) or type(a) != type(b):
            return False

        return list(map(theq, a)) == list(map(theq, b))

    # Otherwise, assume debugprint() can handle it
    astr = theano.printing.debugprint(a, file='str')
    bstr = theano.printing.debugprint(b, file='str')

    # Check for bug mentioned above
    for argname, argval, argstr in [('a', a, astr), ('b', b, bstr)]:
        if argstr == '':
            raise TypeError(
                'theano.printing.debugprint(%s) returned empty string '
                '(%s is instance of %r)'
                % (argname, argname, type(argval))
            )

    return astr == bstr


def test_example_symbols():
    """
    Check that the example symbols in this module print to their Theano
    equivalents, as many of the other tests depend on this.
    """
    assert theq(xt, theano_code_(x))
    assert theq(yt, theano_code_(y))
    assert theq(zt, theano_code_(z))
    assert theq(Xt, theano_code_(X))
    assert theq(Yt, theano_code_(Y))
    assert theq(Zt, theano_code_(Z))


def test_Symbol():
    """ Test printing a Symbol to a theano variable. """
    xx = theano_code_(x)
    assert isinstance(xx, (tt.TensorVariable, ts.ScalarVariable))
    assert xx.broadcastable == ()
    assert xx.name == x.name

    xx2 = theano_code_(x, broadcastables={x: (False,)})
    assert xx2.broadcastable == (False,)
    assert xx2.name == x.name

def test_MatrixSymbol():
    """ Test printing a MatrixSymbol to a theano variable. """
    XX = theano_code_(X)
    assert isinstance(XX, tt.TensorVariable)
    assert XX.broadcastable == (False, False)

@SKIP  # TODO - this is currently not checked but should be implemented
def test_MatrixSymbol_wrong_dims():
    """ Test MatrixSymbol with invalid broadcastable. """
    bcs = [(), (False,), (True,), (True, False), (False, True,), (True, True)]
    for bc in bcs:
        with raises(ValueError):
            theano_code_(X, broadcastables={X: bc})

def test_AppliedUndef():
    """ Test printing AppliedUndef instance, which works similarly to Symbol. """
    ftt = theano_code_(f_t)
    assert isinstance(ftt, tt.TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == 'f_t'


def test_add():
    expr = x + y
    comp = theano_code_(expr)
    assert comp.owner.op == theano.tensor.add

def test_trig():
    assert theq(theano_code_(sy.sin(x)), tt.sin(xt))
    assert theq(theano_code_(sy.tan(x)), tt.tan(xt))

def test_many():
    """ Test printing a complex expression with multiple symbols. """
    expr = sy.exp(x**2 + sy.cos(y)) * sy.log(2*z)
    comp = theano_code_(expr)
    expected = tt.exp(xt**2 + tt.cos(yt)) * tt.log(2*zt)
    assert theq(comp, expected)


def test_dtype():
    """ Test specifying specific data types through the dtype argument. """
    for dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
        assert theano_code_(x, dtypes={x: dtype}).type.dtype == dtype

    # "floatX" type
    assert theano_code_(x, dtypes={x: 'floatX'}).type.dtype in ('float32', 'float64')

    # Type promotion
    assert theano_code_(x + 1, dtypes={x: 'float32'}).type.dtype == 'float32'
    assert theano_code_(x + y, dtypes={x: 'float64', y: 'float32'}).type.dtype == 'float64'


def test_broadcastables():
    """ Test the "broadcastables" argument when printing symbol-like objects. """

    # No restrictions on shape
    for s in [x, f_t]:
        for bc in [(), (False,), (True,), (False, False), (True, False)]:
            assert theano_code_(s, broadcastables={s: bc}).broadcastable == bc

    # TODO - matrix broadcasting?

def test_broadcasting():
    """ Test "broadcastable" attribute after applying element-wise binary op. """

    expr = x + y

    cases = [
        [(), (), ()],
        [(False,), (False,), (False,)],
        [(True,), (False,), (False,)],
        [(False, True), (False, False), (False, False)],
        [(True, False), (False, False), (False, False)],
    ]

    for bc1, bc2, bc3 in cases:
        comp = theano_code_(expr, broadcastables={x: bc1, y: bc2})
        assert comp.broadcastable == bc3


def test_MatMul():
    expr = X*Y*Z
    expr_t = theano_code_(expr)
    assert isinstance(expr_t.owner.op, tt.Dot)
    assert theq(expr_t, Xt.dot(Yt).dot(Zt))

def test_Transpose():
    assert isinstance(theano_code_(X.T).owner.op, tt.DimShuffle)

def test_MatAdd():
    expr = X+Y+Z
    assert isinstance(theano_code_(expr).owner.op, tt.Elemwise)


def test_Rationals():
    assert theq(theano_code_(sy.Integer(2) / 3), tt.true_div(2, 3))
    assert theq(theano_code_(S.Half), tt.true_div(1, 2))

def test_Integers():
    assert theano_code_(sy.Integer(3)) == 3

def test_factorial():
    n = sy.Symbol('n')
    assert theano_code_(sy.factorial(n))

def test_Derivative():
    simp = lambda expr: theano_simplify(fgraph_of(expr))
    assert theq(simp(theano_code_(sy.Derivative(sy.sin(x), x, evaluate=False))),
                simp(theano.grad(tt.sin(xt), xt)))


def test_theano_function_simple():
    """ Test theano_function() with single output. """
    f = theano_function_([x, y], [x+y])
    assert f(2, 3) == 5

def test_theano_function_multi():
    """ Test theano_function() with multiple outputs. """
    f = theano_function_([x, y], [x+y, x-y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1

def test_theano_function_numpy():
    """ Test theano_function() vs Numpy implementation. """
    f = theano_function_([x, y], [x+y], dim=1,
                         dtypes={x: 'float64', y: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    f = theano_function_([x, y], [x+y], dtypes={x: 'float64', y: 'float64'},
                         dim=1)
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy) - 3*np.arange(3)) < 1e-9


def test_theano_function_matrix():
    m = sy.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])
    f = theano_function_([x, y, z], [m])
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = theano_function_([x, y, z], [m], scalar=True)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = theano_function_([x, y, z], [m, m])
    assert isinstance(f(1.0, 2.0, 3.0), type([]))
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[0], expected)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[1], expected)

def test_dim_handling():
    assert dim_handling([x], dim=2) == {x: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True),
                                                       y: (False, False)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}

def test_theano_function_kwargs():
    """
    Test passing additional kwargs from theano_function() to theano.function().
    """
    import numpy as np
    f = theano_function_([x, y, z], [x+y], dim=1, on_unused_input='ignore',
            dtypes={x: 'float64', y: 'float64', z: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4], [0, 0]) - np.asarray([4, 6])) < 1e-9

    f = theano_function_([x, y, z], [x+y],
                        dtypes={x: 'float64', y: 'float64', z: 'float64'},
                        dim=1, on_unused_input='ignore')
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    zz = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy, zz) - 3*np.arange(3)) < 1e-9

def test_theano_function_scalar():
    """ Test the "scalar" argument to theano_function(). """

    args = [
        ([x, y], [x + y], None, [0]),  # Single 0d output
        ([X, Y], [X + Y], None, [2]),  # Single 2d output
        ([x, y], [x + y], {x: 0, y: 1}, [1]),  # Single 1d output
        ([x, y], [x + y, x - y], None, [0, 0]),  # Two 0d outputs
        ([x, y, X, Y], [x + y, X + Y], None, [0, 2]),  # One 0d output, one 2d
    ]

    # Create and test functions with and without the scalar setting
    for inputs, outputs, in_dims, out_dims in args:
        for scalar in [False, True]:

            f = theano_function_(inputs, outputs, dims=in_dims, scalar=scalar)

            # Check the theano_function attribute is set whether wrapped or not
            assert isinstance(f.theano_function, theano.compile.function_module.Function)

            # Feed in inputs of the appropriate size and get outputs
            in_values = [
                np.ones([1 if bc else 5 for bc in i.type.broadcastable])
                for i in f.theano_function.input_storage
            ]
            out_values = f(*in_values)
            if not isinstance(out_values, list):
                out_values = [out_values]

            # Check output types and shapes
            assert len(out_dims) == len(out_values)
            for d, value in zip(out_dims, out_values):

                if scalar and d == 0:
                    # Should have been converted to a scalar value
                    assert isinstance(value, np.number)

                else:
                    # Otherwise should be an array
                    assert isinstance(value, np.ndarray)
                    assert value.ndim == d

def test_theano_function_bad_kwarg():
    """
    Passing an unknown keyword argument to theano_function() should raise an
    exception.
    """
    raises(Exception, lambda : theano_function_([x], [x+1], foobar=3))


def test_slice():
    assert theano_code_(slice(1, 2, 3)) == slice(1, 2, 3)

    def theq_slice(s1, s2):
        for attr in ['start', 'stop', 'step']:
            a1 = getattr(s1, attr)
            a2 = getattr(s2, attr)
            if a1 is None or a2 is None:
                if not (a1 is None or a2 is None):
                    return False
            elif not theq(a1, a2):
                return False
        return True

    dtypes = {x: 'int32', y: 'int32'}
    assert theq_slice(theano_code_(slice(x, y), dtypes=dtypes), slice(xt, yt))
    assert theq_slice(theano_code_(slice(1, x, 3), dtypes=dtypes), slice(1, xt, 3))

def test_MatrixSlice():
    from theano import Constant

    cache = {}

    n = sy.Symbol('n', integer=True)
    X = sy.MatrixSymbol('X', n, n)

    Y = X[1:2:3, 4:5:6]
    Yt = theano_code_(Y, cache=cache)

    s = ts.Scalar('int64')
    assert tuple(Yt.owner.op.idx_list) == (slice(s, s, s), slice(s, s, s))
    assert Yt.owner.inputs[0] == theano_code_(X, cache=cache)
    # == doesn't work in theano like it does in SymPy. You have to use
    # equals.
    assert all(Yt.owner.inputs[i].equals(Constant(s, i)) for i in range(1, 7))

    k = sy.Symbol('k')
    theano_code_(k, dtypes={k: 'int32'})
    start, stop, step = 4, k, 2
    Y = X[start:stop:step]
    Yt = theano_code_(Y, dtypes={n: 'int32', k: 'int32'})
    # assert Yt.owner.op.idx_list[0].stop == kt

def test_BlockMatrix():
    n = sy.Symbol('n', integer=True)
    A, B, C, D = [sy.MatrixSymbol(name, n, n) for name in 'ABCD']
    At, Bt, Ct, Dt = map(theano_code_, (A, B, C, D))
    Block = sy.BlockMatrix([[A, B], [C, D]])
    Blockt = theano_code_(Block)
    solutions = [tt.join(0, tt.join(1, At, Bt), tt.join(1, Ct, Dt)),
                 tt.join(1, tt.join(0, At, Ct), tt.join(0, Bt, Dt))]
    assert any(theq(Blockt, solution) for solution in solutions)

@SKIP
def test_BlockMatrix_Inverse_execution():
    k, n = 2, 4
    dtype = 'float32'
    A = sy.MatrixSymbol('A', n, k)
    B = sy.MatrixSymbol('B', n, n)
    inputs = A, B
    output = B.I*A

    cutsizes = {A: [(n//2, n//2), (k//2, k//2)],
                B: [(n//2, n//2), (n//2, n//2)]}
    cutinputs = [sy.blockcut(i, *cutsizes[i]) for i in inputs]
    cutoutput = output.subs(dict(zip(inputs, cutinputs)))

    dtypes = dict(zip(inputs, [dtype]*len(inputs)))
    f = theano_function_(inputs, [output], dtypes=dtypes, cache={})
    fblocked = theano_function_(inputs, [sy.block_collapse(cutoutput)],
                                dtypes=dtypes, cache={})

    ninputs = [np.random.rand(*x.shape).astype(dtype) for x in inputs]
    ninputs = [np.arange(n*k).reshape(A.shape).astype(dtype),
               np.eye(n).astype(dtype)]
    ninputs[1] += np.ones(B.shape)*1e-5

    assert np.allclose(f(*ninputs), fblocked(*ninputs), rtol=1e-5)

def test_DenseMatrix():
    t = sy.Symbol('theta')
    for MatrixType in [sy.Matrix, sy.ImmutableMatrix]:
        X = MatrixType([[sy.cos(t), -sy.sin(t)], [sy.sin(t), sy.cos(t)]])
        tX = theano_code_(X)
        assert isinstance(tX, tt.TensorVariable)
        assert tX.owner.op == tt.join_


def test_cache_basic():
    """ Test single symbol-like objects are cached when printed by themselves. """

    # Pairs of objects which should be considered equivalent with respect to caching
    pairs = [
        (x, sy.Symbol('x')),
        (X, sy.MatrixSymbol('X', *X.shape)),
        (f_t, sy.Function('f')(sy.Symbol('t'))),
    ]

    for s1, s2 in pairs:
        cache = {}
        st = theano_code_(s1, cache=cache)

        # Test hit with same instance
        assert theano_code_(s1, cache=cache) is st

        # Test miss with same instance but new cache
        assert theano_code_(s1, cache={}) is not st

        # Test hit with different but equivalent instance
        assert theano_code_(s2, cache=cache) is st

def test_global_cache():
    """ Test use of the global cache. """
    from sympy.printing.theanocode import global_cache

    backup = dict(global_cache)
    try:
        # Temporarily empty global cache
        global_cache.clear()

        for s in [x, X, f_t]:
            with warns_deprecated_sympy():
                st = theano_code(s)
                assert theano_code(s) is st

    finally:
        # Restore global cache
        global_cache.update(backup)

def test_cache_types_distinct():
    """
    Test that symbol-like objects of different types (Symbol, MatrixSymbol,
    AppliedUndef) are distinguished by the cache even if they have the same
    name.
    """
    symbols = [sy.Symbol('f_t'), sy.MatrixSymbol('f_t', 4, 4), f_t]

    cache = {}  # Single shared cache
    printed = {}

    for s in symbols:
        st = theano_code_(s, cache=cache)
        assert st not in printed.values()
        printed[s] = st

    # Check all printed objects are distinct
    assert len(set(map(id, printed.values()))) == len(symbols)

    # Check retrieving
    for s, st in printed.items():
        with warns_deprecated_sympy():
            assert theano_code(s, cache=cache) is st

def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    expr = sy.Add(x, x, evaluate=False)
    comp = theano_code_(expr)

    assert theq(comp, xt + xt)
    assert not theq(comp, xt + theano_code_(x))

def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    expr = x ** 2 + (y - sy.exp(x)) * sy.sin(z - x * y)
    symbol_names = {s.name for s in expr.free_symbols}
    expr_t = theano_code_(expr)

    # Iterate through variables in the Theano computational graph that the
    # printed expression depends on
    seen = set()
    for v in theano.gof.graph.ancestors([expr_t]):
        # Owner-less, non-constant variables should be our symbols
        if v.owner is None and not isinstance(v, theano.gof.graph.Constant):
            # Check it corresponds to a symbol and appears only once
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    # Check all were present
    assert seen == symbol_names


def test_Piecewise():
    # A piecewise linear
    expr = sy.Piecewise((0, x<0), (x, x<2), (1, True))  # ___/III
    result = theano_code_(expr)
    assert result.owner.op == tt.switch

    expected = tt.switch(xt<0, 0, tt.switch(xt<2, xt, 1))
    assert theq(result, expected)

    expr = sy.Piecewise((x, x < 0))
    result = theano_code_(expr)
    expected = tt.switch(xt < 0, xt, np.nan)
    assert theq(result, expected)

    expr = sy.Piecewise((0, sy.And(x>0, x<2)), \
        (x, sy.Or(x>2, x<0)))
    result = theano_code_(expr)
    expected = tt.switch(tt.and_(xt>0,xt<2), 0, \
        tt.switch(tt.or_(xt>2, xt<0), xt, np.nan))
    assert theq(result, expected)


def test_Relationals():
    assert theq(theano_code_(sy.Eq(x, y)), tt.eq(xt, yt))
    # assert theq(theano_code_(sy.Ne(x, y)), tt.neq(xt, yt))  # TODO - implement
    assert theq(theano_code_(x > y), xt > yt)
    assert theq(theano_code_(x < y), xt < yt)
    assert theq(theano_code_(x >= y), xt >= yt)
    assert theq(theano_code_(x <= y), xt <= yt)


def test_complexfunctions():
    with warns_deprecated_sympy():
        xt, yt = theano_code_(x, dtypes={x:'complex128'}), theano_code_(y, dtypes={y: 'complex128'})
    from sympy.functions.elementary.complexes import conjugate
    from theano.tensor import as_tensor_variable as atv
    from theano.tensor import complex as cplx
    with warns_deprecated_sympy():
        assert theq(theano_code_(y*conjugate(x)), yt*(xt.conj()))
        assert theq(theano_code_((1+2j)*x), xt*(atv(1.0)+atv(2.0)*cplx(0,1)))


def test_constantfunctions():
    with warns_deprecated_sympy():
        tf = theano_function_([],[1+1j])
    assert(tf()==1+1j)


def test_Exp1():
    """
    Test that exp(1) prints without error and evaluates close to SymPy's E
    """
    # sy.exp(1) should yield same instance of E as sy.E (singleton), but extra
    # check added for sanity
    e_a = sy.exp(1)
    e_b = sy.E

    np.testing.assert_allclose(float(e_a), np.e)
    np.testing.assert_allclose(float(e_b), np.e)

    e = theano_code_(e_a)
    np.testing.assert_allclose(float(e_a), e.eval())

    e = theano_code_(e_b)
    np.testing.assert_allclose(float(e_b), e.eval())
