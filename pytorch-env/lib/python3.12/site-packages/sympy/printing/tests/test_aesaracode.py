"""
Important note on tests in this module - the Aesara printing functions use a
global cache by default, which means that tests using it will modify global
state and thus not be independent from each other. Instead of using the "cache"
keyword argument each time, this module uses the aesara_code_ and
aesara_function_ functions defined below which default to using a new, empty
cache instead.
"""

import logging

from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP

from sympy.utilities.exceptions import ignore_warnings


aesaralogger = logging.getLogger('aesara.configdefaults')
aesaralogger.setLevel(logging.CRITICAL)
aesara = import_module('aesara')
aesaralogger.setLevel(logging.WARNING)


if aesara:
    import numpy as np
    aet = aesara.tensor
    from aesara.scalar.basic import ScalarType
    from aesara.graph.basic import Variable
    from aesara.tensor.var import TensorVariable
    from aesara.tensor.elemwise import Elemwise, DimShuffle
    from aesara.tensor.math import Dot

    from sympy.printing.aesaracode import true_divide

    xt, yt, zt = [aet.scalar(name, 'floatX') for name in 'xyz']
    Xt, Yt, Zt = [aet.tensor('floatX', (False, False), name=n) for n in 'XYZ']
else:
    #bin/test will not execute any tests now
    disabled = True

import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.aesaracode import (aesara_code, dim_handling,
        aesara_function)


# Default set of matrix symbols for testing - make square so we can both
# multiply and perform elementwise operations between them.
X, Y, Z = [sy.MatrixSymbol(n, 4, 4) for n in 'XYZ']

# For testing AppliedUndef
f_t = sy.Function('f')(t)


def aesara_code_(expr, **kwargs):
    """ Wrapper for aesara_code that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    return aesara_code(expr, **kwargs)

def aesara_function_(inputs, outputs, **kwargs):
    """ Wrapper for aesara_function that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    return aesara_function(inputs, outputs, **kwargs)


def fgraph_of(*exprs):
    """ Transform SymPy expressions into Aesara Computation.

    Parameters
    ==========
    exprs
        SymPy expressions

    Returns
    =======
    aesara.graph.fg.FunctionGraph
    """
    outs = list(map(aesara_code_, exprs))
    ins = list(aesara.graph.basic.graph_inputs(outs))
    ins, outs = aesara.graph.basic.clone(ins, outs)
    return aesara.graph.fg.FunctionGraph(ins, outs)


def aesara_simplify(fgraph):
    """ Simplify a Aesara Computation.

    Parameters
    ==========
    fgraph : aesara.graph.fg.FunctionGraph

    Returns
    =======
    aesara.graph.fg.FunctionGraph
    """
    mode = aesara.compile.get_default_mode().excluding("fusion")
    fgraph = fgraph.clone()
    mode.optimizer.rewrite(fgraph)
    return fgraph


def theq(a, b):
    """ Test two Aesara objects for equality.

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
    astr = aesara.printing.debugprint(a, file='str')
    bstr = aesara.printing.debugprint(b, file='str')

    # Check for bug mentioned above
    for argname, argval, argstr in [('a', a, astr), ('b', b, bstr)]:
        if argstr == '':
            raise TypeError(
                'aesara.printing.debugprint(%s) returned empty string '
                '(%s is instance of %r)'
                % (argname, argname, type(argval))
            )

    return astr == bstr


def test_example_symbols():
    """
    Check that the example symbols in this module print to their Aesara
    equivalents, as many of the other tests depend on this.
    """
    assert theq(xt, aesara_code_(x))
    assert theq(yt, aesara_code_(y))
    assert theq(zt, aesara_code_(z))
    assert theq(Xt, aesara_code_(X))
    assert theq(Yt, aesara_code_(Y))
    assert theq(Zt, aesara_code_(Z))


def test_Symbol():
    """ Test printing a Symbol to a aesara variable. """
    xx = aesara_code_(x)
    assert isinstance(xx, Variable)
    assert xx.broadcastable == ()
    assert xx.name == x.name

    xx2 = aesara_code_(x, broadcastables={x: (False,)})
    assert xx2.broadcastable == (False,)
    assert xx2.name == x.name

def test_MatrixSymbol():
    """ Test printing a MatrixSymbol to a aesara variable. """
    XX = aesara_code_(X)
    assert isinstance(XX, TensorVariable)
    assert XX.broadcastable == (False, False)

@SKIP  # TODO - this is currently not checked but should be implemented
def test_MatrixSymbol_wrong_dims():
    """ Test MatrixSymbol with invalid broadcastable. """
    bcs = [(), (False,), (True,), (True, False), (False, True,), (True, True)]
    for bc in bcs:
        with raises(ValueError):
            aesara_code_(X, broadcastables={X: bc})

def test_AppliedUndef():
    """ Test printing AppliedUndef instance, which works similarly to Symbol. """
    ftt = aesara_code_(f_t)
    assert isinstance(ftt, TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == 'f_t'


def test_add():
    expr = x + y
    comp = aesara_code_(expr)
    assert comp.owner.op == aesara.tensor.add

def test_trig():
    assert theq(aesara_code_(sy.sin(x)), aet.sin(xt))
    assert theq(aesara_code_(sy.tan(x)), aet.tan(xt))

def test_many():
    """ Test printing a complex expression with multiple symbols. """
    expr = sy.exp(x**2 + sy.cos(y)) * sy.log(2*z)
    comp = aesara_code_(expr)
    expected = aet.exp(xt**2 + aet.cos(yt)) * aet.log(2*zt)
    assert theq(comp, expected)


def test_dtype():
    """ Test specifying specific data types through the dtype argument. """
    for dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
        assert aesara_code_(x, dtypes={x: dtype}).type.dtype == dtype

    # "floatX" type
    assert aesara_code_(x, dtypes={x: 'floatX'}).type.dtype in ('float32', 'float64')

    # Type promotion
    assert aesara_code_(x + 1, dtypes={x: 'float32'}).type.dtype == 'float32'
    assert aesara_code_(x + y, dtypes={x: 'float64', y: 'float32'}).type.dtype == 'float64'


def test_broadcastables():
    """ Test the "broadcastables" argument when printing symbol-like objects. """

    # No restrictions on shape
    for s in [x, f_t]:
        for bc in [(), (False,), (True,), (False, False), (True, False)]:
            assert aesara_code_(s, broadcastables={s: bc}).broadcastable == bc

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
        comp = aesara_code_(expr, broadcastables={x: bc1, y: bc2})
        assert comp.broadcastable == bc3


def test_MatMul():
    expr = X*Y*Z
    expr_t = aesara_code_(expr)
    assert isinstance(expr_t.owner.op, Dot)
    assert theq(expr_t, Xt.dot(Yt).dot(Zt))

def test_Transpose():
    assert isinstance(aesara_code_(X.T).owner.op, DimShuffle)

def test_MatAdd():
    expr = X+Y+Z
    assert isinstance(aesara_code_(expr).owner.op, Elemwise)


def test_Rationals():
    assert theq(aesara_code_(sy.Integer(2) / 3), true_divide(2, 3))
    assert theq(aesara_code_(S.Half), true_divide(1, 2))

def test_Integers():
    assert aesara_code_(sy.Integer(3)) == 3

def test_factorial():
    n = sy.Symbol('n')
    assert aesara_code_(sy.factorial(n))

def test_Derivative():
    with ignore_warnings(UserWarning):
        simp = lambda expr: aesara_simplify(fgraph_of(expr))
        assert theq(simp(aesara_code_(sy.Derivative(sy.sin(x), x, evaluate=False))),
                    simp(aesara.grad(aet.sin(xt), xt)))


def test_aesara_function_simple():
    """ Test aesara_function() with single output. """
    f = aesara_function_([x, y], [x+y])
    assert f(2, 3) == 5

def test_aesara_function_multi():
    """ Test aesara_function() with multiple outputs. """
    f = aesara_function_([x, y], [x+y, x-y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1

def test_aesara_function_numpy():
    """ Test aesara_function() vs Numpy implementation. """
    f = aesara_function_([x, y], [x+y], dim=1,
                         dtypes={x: 'float64', y: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    f = aesara_function_([x, y], [x+y], dtypes={x: 'float64', y: 'float64'},
                         dim=1)
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy) - 3*np.arange(3)) < 1e-9


def test_aesara_function_matrix():
    m = sy.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])
    f = aesara_function_([x, y, z], [m])
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = aesara_function_([x, y, z], [m], scalar=True)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = aesara_function_([x, y, z], [m, m])
    assert isinstance(f(1.0, 2.0, 3.0), type([]))
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[0], expected)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[1], expected)

def test_dim_handling():
    assert dim_handling([x], dim=2) == {x: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True),
                                                       y: (False, False)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}

def test_aesara_function_kwargs():
    """
    Test passing additional kwargs from aesara_function() to aesara.function().
    """
    import numpy as np
    f = aesara_function_([x, y, z], [x+y], dim=1, on_unused_input='ignore',
            dtypes={x: 'float64', y: 'float64', z: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4], [0, 0]) - np.asarray([4, 6])) < 1e-9

    f = aesara_function_([x, y, z], [x+y],
                        dtypes={x: 'float64', y: 'float64', z: 'float64'},
                        dim=1, on_unused_input='ignore')
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    zz = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy, zz) - 3*np.arange(3)) < 1e-9

def test_aesara_function_scalar():
    """ Test the "scalar" argument to aesara_function(). """
    from aesara.compile.function.types import Function

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

            f = aesara_function_(inputs, outputs, dims=in_dims, scalar=scalar)

            # Check the aesara_function attribute is set whether wrapped or not
            assert isinstance(f.aesara_function, Function)

            # Feed in inputs of the appropriate size and get outputs
            in_values = [
                np.ones([1 if bc else 5 for bc in i.type.broadcastable])
                for i in f.aesara_function.input_storage
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

def test_aesara_function_bad_kwarg():
    """
    Passing an unknown keyword argument to aesara_function() should raise an
    exception.
    """
    raises(Exception, lambda : aesara_function_([x], [x+1], foobar=3))


def test_slice():
    assert aesara_code_(slice(1, 2, 3)) == slice(1, 2, 3)

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
    assert theq_slice(aesara_code_(slice(x, y), dtypes=dtypes), slice(xt, yt))
    assert theq_slice(aesara_code_(slice(1, x, 3), dtypes=dtypes), slice(1, xt, 3))

def test_MatrixSlice():
    cache = {}

    n = sy.Symbol('n', integer=True)
    X = sy.MatrixSymbol('X', n, n)

    Y = X[1:2:3, 4:5:6]
    Yt = aesara_code_(Y, cache=cache)

    s = ScalarType('int64')
    assert tuple(Yt.owner.op.idx_list) == (slice(s, s, s), slice(s, s, s))
    assert Yt.owner.inputs[0] == aesara_code_(X, cache=cache)
    # == doesn't work in Aesara like it does in SymPy. You have to use
    # equals.
    assert all(Yt.owner.inputs[i].data == i for i in range(1, 7))

    k = sy.Symbol('k')
    aesara_code_(k, dtypes={k: 'int32'})
    start, stop, step = 4, k, 2
    Y = X[start:stop:step]
    Yt = aesara_code_(Y, dtypes={n: 'int32', k: 'int32'})
    # assert Yt.owner.op.idx_list[0].stop == kt

def test_BlockMatrix():
    n = sy.Symbol('n', integer=True)
    A, B, C, D = [sy.MatrixSymbol(name, n, n) for name in 'ABCD']
    At, Bt, Ct, Dt = map(aesara_code_, (A, B, C, D))
    Block = sy.BlockMatrix([[A, B], [C, D]])
    Blockt = aesara_code_(Block)
    solutions = [aet.join(0, aet.join(1, At, Bt), aet.join(1, Ct, Dt)),
                 aet.join(1, aet.join(0, At, Ct), aet.join(0, Bt, Dt))]
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
    f = aesara_function_(inputs, [output], dtypes=dtypes, cache={})
    fblocked = aesara_function_(inputs, [sy.block_collapse(cutoutput)],
                                dtypes=dtypes, cache={})

    ninputs = [np.random.rand(*x.shape).astype(dtype) for x in inputs]
    ninputs = [np.arange(n*k).reshape(A.shape).astype(dtype),
               np.eye(n).astype(dtype)]
    ninputs[1] += np.ones(B.shape)*1e-5

    assert np.allclose(f(*ninputs), fblocked(*ninputs), rtol=1e-5)

def test_DenseMatrix():
    from aesara.tensor.basic import Join

    t = sy.Symbol('theta')
    for MatrixType in [sy.Matrix, sy.ImmutableMatrix]:
        X = MatrixType([[sy.cos(t), -sy.sin(t)], [sy.sin(t), sy.cos(t)]])
        tX = aesara_code_(X)
        assert isinstance(tX, TensorVariable)
        assert isinstance(tX.owner.op, Join)


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
        st = aesara_code_(s1, cache=cache)

        # Test hit with same instance
        assert aesara_code_(s1, cache=cache) is st

        # Test miss with same instance but new cache
        assert aesara_code_(s1, cache={}) is not st

        # Test hit with different but equivalent instance
        assert aesara_code_(s2, cache=cache) is st

def test_global_cache():
    """ Test use of the global cache. """
    from sympy.printing.aesaracode import global_cache

    backup = dict(global_cache)
    try:
        # Temporarily empty global cache
        global_cache.clear()

        for s in [x, X, f_t]:
            st = aesara_code(s)
            assert aesara_code(s) is st

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
        st = aesara_code_(s, cache=cache)
        assert st not in printed.values()
        printed[s] = st

    # Check all printed objects are distinct
    assert len(set(map(id, printed.values()))) == len(symbols)

    # Check retrieving
    for s, st in printed.items():
        assert aesara_code(s, cache=cache) is st

def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    expr = sy.Add(x, x, evaluate=False)
    comp = aesara_code_(expr)

    assert theq(comp, xt + xt)
    assert not theq(comp, xt + aesara_code_(x))

def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    expr = x ** 2 + (y - sy.exp(x)) * sy.sin(z - x * y)
    symbol_names = {s.name for s in expr.free_symbols}
    expr_t = aesara_code_(expr)

    # Iterate through variables in the Aesara computational graph that the
    # printed expression depends on
    seen = set()
    for v in aesara.graph.basic.ancestors([expr_t]):
        # Owner-less, non-constant variables should be our symbols
        if v.owner is None and not isinstance(v, aesara.graph.basic.Constant):
            # Check it corresponds to a symbol and appears only once
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    # Check all were present
    assert seen == symbol_names


def test_Piecewise():
    # A piecewise linear
    expr = sy.Piecewise((0, x<0), (x, x<2), (1, True))  # ___/III
    result = aesara_code_(expr)
    assert result.owner.op == aet.switch

    expected = aet.switch(xt<0, 0, aet.switch(xt<2, xt, 1))
    assert theq(result, expected)

    expr = sy.Piecewise((x, x < 0))
    result = aesara_code_(expr)
    expected = aet.switch(xt < 0, xt, np.nan)
    assert theq(result, expected)

    expr = sy.Piecewise((0, sy.And(x>0, x<2)), \
        (x, sy.Or(x>2, x<0)))
    result = aesara_code_(expr)
    expected = aet.switch(aet.and_(xt>0,xt<2), 0, \
        aet.switch(aet.or_(xt>2, xt<0), xt, np.nan))
    assert theq(result, expected)


def test_Relationals():
    assert theq(aesara_code_(sy.Eq(x, y)), aet.eq(xt, yt))
    # assert theq(aesara_code_(sy.Ne(x, y)), aet.neq(xt, yt))  # TODO - implement
    assert theq(aesara_code_(x > y), xt > yt)
    assert theq(aesara_code_(x < y), xt < yt)
    assert theq(aesara_code_(x >= y), xt >= yt)
    assert theq(aesara_code_(x <= y), xt <= yt)


def test_complexfunctions():
    dtypes = {x:'complex128', y:'complex128'}
    xt, yt = aesara_code(x, dtypes=dtypes), aesara_code(y, dtypes=dtypes)
    from sympy.functions.elementary.complexes import conjugate
    from aesara.tensor import as_tensor_variable as atv
    from aesara.tensor import complex as cplx
    assert theq(aesara_code(y*conjugate(x), dtypes=dtypes), yt*(xt.conj()))
    assert theq(aesara_code((1+2j)*x), xt*(atv(1.0)+atv(2.0)*cplx(0,1)))


def test_constantfunctions():
    tf = aesara_function([],[1+1j])
    assert(tf()==1+1j)
