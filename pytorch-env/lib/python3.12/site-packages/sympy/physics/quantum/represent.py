"""Logic for representing operators in state in various bases.

TODO:

* Get represent working with continuous hilbert spaces.
* Document default basis functionality.
"""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.integrals.integrals import integrate
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.matrixutils import flatten_scalar
from sympy.physics.quantum.state import KetBase, BraBase, StateBase
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state, state_to_operators

__all__ = [
    'represent',
    'rep_innerproduct',
    'rep_expectation',
    'integrate_result',
    'get_basis',
    'enumerate_states'
]

#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def _sympy_to_scalar(e):
    """Convert from a SymPy scalar to a Python scalar."""
    if isinstance(e, Expr):
        if e.is_Integer:
            return int(e)
        elif e.is_Float:
            return float(e)
        elif e.is_Rational:
            return float(e)
        elif e.is_Number or e.is_NumberSymbol or e == I:
            return complex(e)
    raise TypeError('Expected number, got: %r' % e)


def represent(expr, **options):
    """Represent the quantum expression in the given basis.

    In quantum mechanics abstract states and operators can be represented in
    various basis sets. Under this operation the follow transforms happen:

    * Ket -> column vector or function
    * Bra -> row vector of function
    * Operator -> matrix or differential operator

    This function is the top-level interface for this action.

    This function walks the SymPy expression tree looking for ``QExpr``
    instances that have a ``_represent`` method. This method is then called
    and the object is replaced by the representation returned by this method.
    By default, the ``_represent`` method will dispatch to other methods
    that handle the representation logic for a particular basis set. The
    naming convention for these methods is the following::

        def _represent_FooBasis(self, e, basis, **options)

    This function will have the logic for representing instances of its class
    in the basis set having a class named ``FooBasis``.

    Parameters
    ==========

    expr  : Expr
        The expression to represent.
    basis : Operator, basis set
        An object that contains the information about the basis set. If an
        operator is used, the basis is assumed to be the orthonormal
        eigenvectors of that operator. In general though, the basis argument
        can be any object that contains the basis set information.
    options : dict
        Key/value pairs of options that are passed to the underlying method
        that finds the representation. These options can be used to
        control how the representation is done. For example, this is where
        the size of the basis set would be set.

    Returns
    =======

    e : Expr
        The SymPy expression of the represented quantum expression.

    Examples
    ========

    Here we subclass ``Operator`` and ``Ket`` to create the z-spin operator
    and its spin 1/2 up eigenstate. By defining the ``_represent_SzOp``
    method, the ket can be represented in the z-spin basis.

    >>> from sympy.physics.quantum import Operator, represent, Ket
    >>> from sympy import Matrix

    >>> class SzUpKet(Ket):
    ...     def _represent_SzOp(self, basis, **options):
    ...         return Matrix([1,0])
    ...
    >>> class SzOp(Operator):
    ...     pass
    ...
    >>> sz = SzOp('Sz')
    >>> up = SzUpKet('up')
    >>> represent(up, basis=sz)
    Matrix([
    [1],
    [0]])

    Here we see an example of representations in a continuous
    basis. We see that the result of representing various combinations
    of cartesian position operators and kets give us continuous
    expressions involving DiracDelta functions.

    >>> from sympy.physics.quantum.cartesian import XOp, XKet, XBra
    >>> X = XOp()
    >>> x = XKet()
    >>> y = XBra('y')
    >>> represent(X*x)
    x*DiracDelta(x - x_2)
    >>> represent(X*x*y)
    x*DiracDelta(x - x_3)*DiracDelta(x_1 - y)

    """

    format = options.get('format', 'sympy')
    if format == 'numpy':
        import numpy as np
    if isinstance(expr, QExpr) and not isinstance(expr, OuterProduct):
        options['replace_none'] = False
        temp_basis = get_basis(expr, **options)
        if temp_basis is not None:
            options['basis'] = temp_basis
        try:
            return expr._represent(**options)
        except NotImplementedError as strerr:
            #If no _represent_FOO method exists, map to the
            #appropriate basis state and try
            #the other methods of representation
            options['replace_none'] = True

            if isinstance(expr, (KetBase, BraBase)):
                try:
                    return rep_innerproduct(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            elif isinstance(expr, Operator):
                try:
                    return rep_expectation(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            else:
                raise NotImplementedError(strerr)
    elif isinstance(expr, Add):
        result = represent(expr.args[0], **options)
        for args in expr.args[1:]:
            # scipy.sparse doesn't support += so we use plain = here.
            result = result + represent(args, **options)
        return result
    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        if format in ('numpy', 'scipy.sparse'):
            exp = _sympy_to_scalar(exp)
        base = represent(base, **options)
        # scipy.sparse doesn't support negative exponents
        # and warns when inverting a matrix in csr format.
        if format == 'scipy.sparse' and exp < 0:
            from scipy.sparse.linalg import inv
            exp = - exp
            base = inv(base.tocsc()).tocsr()
        if format == 'numpy':
            return np.linalg.matrix_power(base, exp)
        return base ** exp
    elif isinstance(expr, TensorProduct):
        new_args = [represent(arg, **options) for arg in expr.args]
        return TensorProduct(*new_args)
    elif isinstance(expr, Dagger):
        return Dagger(represent(expr.args[0], **options))
    elif isinstance(expr, Commutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) - Mul(B, A), **options)
    elif isinstance(expr, AntiCommutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) + Mul(B, A), **options)
    elif isinstance(expr, InnerProduct):
        return represent(Mul(expr.bra, expr.ket), **options)
    elif not isinstance(expr, (Mul, OuterProduct)):
        # For numpy and scipy.sparse, we can only handle numerical prefactors.
        if format in ('numpy', 'scipy.sparse'):
            return _sympy_to_scalar(expr)
        return expr

    if not isinstance(expr, (Mul, OuterProduct)):
        raise TypeError('Mul expected, got: %r' % expr)

    if "index" in options:
        options["index"] += 1
    else:
        options["index"] = 1

    if "unities" not in options:
        options["unities"] = []

    result = represent(expr.args[-1], **options)
    last_arg = expr.args[-1]

    for arg in reversed(expr.args[:-1]):
        if isinstance(last_arg, Operator):
            options["index"] += 1
            options["unities"].append(options["index"])
        elif isinstance(last_arg, BraBase) and isinstance(arg, KetBase):
            options["index"] += 1
        elif isinstance(last_arg, KetBase) and isinstance(arg, Operator):
            options["unities"].append(options["index"])
        elif isinstance(last_arg, KetBase) and isinstance(arg, BraBase):
            options["unities"].append(options["index"])

        next_arg = represent(arg, **options)
        if format == 'numpy' and isinstance(next_arg, np.ndarray):
            # Must use np.matmult to "matrix multiply" two np.ndarray
            result = np.matmul(next_arg, result)
        else:
            result = next_arg*result
        last_arg = arg

    # All three matrix formats create 1 by 1 matrices when inner products of
    # vectors are taken. In these cases, we simply return a scalar.
    result = flatten_scalar(result)

    result = integrate_result(expr, result, **options)

    return result


def rep_innerproduct(expr, **options):
    """
    Returns an innerproduct like representation (e.g. ``<x'|x>``) for the
    given state.

    Attempts to calculate inner product with a bra from the specified
    basis. Should only be passed an instance of KetBase or BraBase

    Parameters
    ==========

    expr : KetBase or BraBase
        The expression to be represented

    Examples
    ========

    >>> from sympy.physics.quantum.represent import rep_innerproduct
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> rep_innerproduct(XKet())
    DiracDelta(x - x_1)
    >>> rep_innerproduct(XKet(), basis=PxOp())
    sqrt(2)*exp(-I*px_1*x/hbar)/(2*sqrt(hbar)*sqrt(pi))
    >>> rep_innerproduct(PxKet(), basis=XOp())
    sqrt(2)*exp(I*px*x_1/hbar)/(2*sqrt(hbar)*sqrt(pi))

    """

    if not isinstance(expr, (KetBase, BraBase)):
        raise TypeError("expr passed is not a Bra or Ket")

    basis = get_basis(expr, **options)

    if not isinstance(basis, StateBase):
        raise NotImplementedError("Can't form this representation!")

    if "index" not in options:
        options["index"] = 1

    basis_kets = enumerate_states(basis, options["index"], 2)

    if isinstance(expr, BraBase):
        bra = expr
        ket = (basis_kets[1] if basis_kets[0].dual == expr else basis_kets[0])
    else:
        bra = (basis_kets[1].dual if basis_kets[0]
               == expr else basis_kets[0].dual)
        ket = expr

    prod = InnerProduct(bra, ket)
    result = prod.doit()

    format = options.get('format', 'sympy')
    return expr._format_represent(result, format)


def rep_expectation(expr, **options):
    """
    Returns an ``<x'|A|x>`` type representation for the given operator.

    Parameters
    ==========

    expr : Operator
        Operator to be represented in the specified basis

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp, PxKet
    >>> from sympy.physics.quantum.represent import rep_expectation
    >>> rep_expectation(XOp())
    x_1*DiracDelta(x_1 - x_2)
    >>> rep_expectation(XOp(), basis=PxOp())
    <px_2|*X*|px_1>
    >>> rep_expectation(XOp(), basis=PxKet())
    <px_2|*X*|px_1>

    """

    if "index" not in options:
        options["index"] = 1

    if not isinstance(expr, Operator):
        raise TypeError("The passed expression is not an operator")

    basis_state = get_basis(expr, **options)

    if basis_state is None or not isinstance(basis_state, StateBase):
        raise NotImplementedError("Could not get basis kets for this operator")

    basis_kets = enumerate_states(basis_state, options["index"], 2)

    bra = basis_kets[1].dual
    ket = basis_kets[0]

    return qapply(bra*expr*ket)


def integrate_result(orig_expr, result, **options):
    """
    Returns the result of integrating over any unities ``(|x><x|)`` in
    the given expression. Intended for integrating over the result of
    representations in continuous bases.

    This function integrates over any unities that may have been
    inserted into the quantum expression and returns the result.
    It uses the interval of the Hilbert space of the basis state
    passed to it in order to figure out the limits of integration.
    The unities option must be
    specified for this to work.

    Note: This is mostly used internally by represent(). Examples are
    given merely to show the use cases.

    Parameters
    ==========

    orig_expr : quantum expression
        The original expression which was to be represented

    result: Expr
        The resulting representation that we wish to integrate over

    Examples
    ========

    >>> from sympy import symbols, DiracDelta
    >>> from sympy.physics.quantum.represent import integrate_result
    >>> from sympy.physics.quantum.cartesian import XOp, XKet
    >>> x_ket = XKet()
    >>> X_op = XOp()
    >>> x, x_1, x_2 = symbols('x, x_1, x_2')
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2))
    x*DiracDelta(x - x_1)*DiracDelta(x_1 - x_2)
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2),
    ...     unities=[1])
    x*DiracDelta(x - x_2)

    """
    if not isinstance(result, Expr):
        return result

    options['replace_none'] = True
    if "basis" not in options:
        arg = orig_expr.args[-1]
        options["basis"] = get_basis(arg, **options)
    elif not isinstance(options["basis"], StateBase):
        options["basis"] = get_basis(orig_expr, **options)

    basis = options.pop("basis", None)

    if basis is None:
        return result

    unities = options.pop("unities", [])

    if len(unities) == 0:
        return result

    kets = enumerate_states(basis, unities)
    coords = [k.label[0] for k in kets]

    for coord in coords:
        if coord in result.free_symbols:
            #TODO: Add support for sets of operators
            basis_op = state_to_operators(basis)
            start = basis_op.hilbert_space.interval.start
            end = basis_op.hilbert_space.interval.end
            result = integrate(result, (coord, start, end))

    return result


def get_basis(expr, *, basis=None, replace_none=True, **options):
    """
    Returns a basis state instance corresponding to the basis specified in
    options=s. If no basis is specified, the function tries to form a default
    basis state of the given expression.

    There are three behaviors:

    1. The basis specified in options is already an instance of StateBase. If
       this is the case, it is simply returned. If the class is specified but
       not an instance, a default instance is returned.

    2. The basis specified is an operator or set of operators. If this
       is the case, the operator_to_state mapping method is used.

    3. No basis is specified. If expr is a state, then a default instance of
       its class is returned.  If expr is an operator, then it is mapped to the
       corresponding state.  If it is neither, then we cannot obtain the basis
       state.

    If the basis cannot be mapped, then it is not changed.

    This will be called from within represent, and represent will
    only pass QExpr's.

    TODO (?): Support for Muls and other types of expressions?

    Parameters
    ==========

    expr : Operator or StateBase
        Expression whose basis is sought

    Examples
    ========

    >>> from sympy.physics.quantum.represent import get_basis
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> x = XKet()
    >>> X = XOp()
    >>> get_basis(x)
    |x>
    >>> get_basis(X)
    |x>
    >>> get_basis(x, basis=PxOp())
    |px>
    >>> get_basis(x, basis=PxKet)
    |px>

    """

    if basis is None and not replace_none:
        return None

    if basis is None:
        if isinstance(expr, KetBase):
            return _make_default(expr.__class__)
        elif isinstance(expr, BraBase):
            return _make_default(expr.dual_class())
        elif isinstance(expr, Operator):
            state_inst = operators_to_state(expr)
            return (state_inst if state_inst is not None else None)
        else:
            return None
    elif (isinstance(basis, Operator) or
          (not isinstance(basis, StateBase) and issubclass(basis, Operator))):
        state = operators_to_state(basis)
        if state is None:
            return None
        elif isinstance(state, StateBase):
            return state
        else:
            return _make_default(state)
    elif isinstance(basis, StateBase):
        return basis
    elif issubclass(basis, StateBase):
        return _make_default(basis)
    else:
        return None


def _make_default(expr):
    # XXX: Catching TypeError like this is a bad way of distinguishing
    # instances from classes. The logic using this function should be
    # rewritten somehow.
    try:
        expr = expr()
    except TypeError:
        return expr

    return expr


def enumerate_states(*args, **options):
    """
    Returns instances of the given state with dummy indices appended

    Operates in two different modes:

    1. Two arguments are passed to it. The first is the base state which is to
       be indexed, and the second argument is a list of indices to append.

    2. Three arguments are passed. The first is again the base state to be
       indexed. The second is the start index for counting.  The final argument
       is the number of kets you wish to receive.

    Tries to call state._enumerate_state. If this fails, returns an empty list

    Parameters
    ==========

    args : list
        See list of operation modes above for explanation

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XBra, XKet
    >>> from sympy.physics.quantum.represent import enumerate_states
    >>> test = XKet('foo')
    >>> enumerate_states(test, 1, 3)
    [|foo_1>, |foo_2>, |foo_3>]
    >>> test2 = XBra('bar')
    >>> enumerate_states(test2, [4, 5, 10])
    [<bar_4|, <bar_5|, <bar_10|]

    """

    state = args[0]

    if len(args) not in (2, 3):
        raise NotImplementedError("Wrong number of arguments!")

    if not isinstance(state, StateBase):
        raise TypeError("First argument is not a state!")

    if len(args) == 3:
        num_states = args[2]
        options['start_index'] = args[1]
    else:
        num_states = len(args[1])
        options['index_list'] = args[1]

    try:
        ret = state._enumerate_state(num_states, **options)
    except NotImplementedError:
        ret = []

    return ret
