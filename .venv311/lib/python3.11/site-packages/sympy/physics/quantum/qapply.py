"""Logic for applying operators to states.

Todo:
* Sometimes the final result needs to be expanded, we should do this by hand.
"""

from sympy.concrete import Sum
from sympy.core.add import Add
from sympy.core.kind import NumberKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sympify import sympify, _sympify

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.state import State, KetBase, BraBase, Wavefunction
from sympy.physics.quantum.tensorproduct import TensorProduct

__all__ = [
    'qapply'
]


#-----------------------------------------------------------------------------
# Main code
#-----------------------------------------------------------------------------


def ip_doit_func(e):
    """Transform the inner products in an expression by calling ``.doit()``."""
    return e.replace(InnerProduct, lambda *args: InnerProduct(*args).doit())


def sum_doit_func(e):
    """Transform the sums in an expression by calling ``.doit()``."""
    return e.replace(Sum, lambda *args: Sum(*args).doit())


def qapply(e, **options):
    """Apply operators to states in a quantum expression.

    Parameters
    ==========

    e : Expr
        The expression containing operators and states. This expression tree
        will be walked to find operators acting on states symbolically.
    options : dict
        A dict of key/value pairs that determine how the operator actions
        are carried out.

        The following options are valid:

        * ``dagger``: try to apply Dagger operators to the left
          (default: False).
        * ``ip_doit``: call ``.doit()`` in inner products when they are
          encountered (default: True).
        * ``sum_doit``: call ``.doit()`` on sums when they are encountered
          (default: False). This is helpful for collapsing sums over Kronecker
          delta's that are created when calling ``qapply``.

    Returns
    =======

    e : Expr
        The original expression, but with the operators applied to states.

    Examples
    ========

        >>> from sympy.physics.quantum import qapply, Ket, Bra
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> A = k * b
        >>> A
        |k><b|
        >>> qapply(A * b.dual / (b * b.dual))
        |k>
        >>> qapply(k.dual * A / (k.dual * k))
        <b|
    """
    from sympy.physics.quantum.density import Density

    dagger = options.get('dagger', False)
    sum_doit = options.get('sum_doit', False)
    ip_doit = options.get('ip_doit', True)

    e = _sympify(e)

    # Using the kind API here helps us to narrow what types of expressions
    # we call ``ip_doit_func`` on.
    if e.kind == NumberKind:
        return ip_doit_func(e) if ip_doit else e

    # This may be a bit aggressive but ensures that everything gets expanded
    # to its simplest form before trying to apply operators. This includes
    # things like (A+B+C)*|a> and A*(|a>+|b>) and all Commutators and
    # TensorProducts. The only problem with this is that if we can't apply
    # all the Operators, we have just expanded everything.
    # TODO: don't expand the scalars in front of each Mul.
    e = e.expand(commutator=True, tensorproduct=True)

    # If we just have a raw ket, return it.
    if isinstance(e, KetBase):
        return e

    # We have an Add(a, b, c, ...) and compute
    # Add(qapply(a), qapply(b), ...)
    elif isinstance(e, Add):
        result = 0
        for arg in e.args:
            result += qapply(arg, **options)
        return result.expand()

    # For a Density operator call qapply on its state
    elif isinstance(e, Density):
        new_args = [(qapply(state, **options), prob) for (state,
                     prob) in e.args]
        return Density(*new_args)

    # For a raw TensorProduct, call qapply on its args.
    elif isinstance(e, TensorProduct):
        return TensorProduct(*[qapply(t, **options) for t in e.args])

    # For a Sum, call qapply on its function.
    elif isinstance(e, Sum):
        result = Sum(qapply(e.function, **options), *e.limits)
        result = sum_doit_func(result) if sum_doit else result
        return result

    # For a Pow, call qapply on its base.
    elif isinstance(e, Pow):
        return qapply(e.base, **options)**e.exp

    # We have a Mul where there might be actual operators to apply to kets.
    elif isinstance(e, Mul):
        c_part, nc_part = e.args_cnc()
        c_mul = Mul(*c_part)
        nc_mul = Mul(*nc_part)
        if not nc_part: # If we only have a commuting part, just return it.
            result = c_mul
        elif isinstance(nc_mul, Mul):
            result = c_mul*qapply_Mul(nc_mul, **options)
        else:
            result = c_mul*qapply(nc_mul, **options)
        if result == e and dagger:
            result = Dagger(qapply_Mul(Dagger(e), **options))
        result = ip_doit_func(result) if ip_doit else result
        result = sum_doit_func(result) if sum_doit else result
        return result

    # In all other cases (State, Operator, Pow, Commutator, InnerProduct,
    # OuterProduct) we won't ever have operators to apply to kets.
    else:
        return e


def qapply_Mul(e, **options):

    args = list(e.args)
    extra = S.One
    result = None

    # If we only have 0 or 1 args, we have nothing to do and return.
    if len(args) <= 1 or not isinstance(e, Mul):
        return e
    rhs = args.pop()
    lhs = args.pop()

    # Make sure we have two non-commutative objects before proceeding.
    if (not isinstance(rhs, Wavefunction) and sympify(rhs).is_commutative) or \
            (not isinstance(lhs, Wavefunction) and sympify(lhs).is_commutative):
        return e

    # For a Pow with an integer exponent, apply one of them and reduce the
    # exponent by one.
    if isinstance(lhs, Pow) and lhs.exp.is_Integer:
        args.append(lhs.base**(lhs.exp - 1))
        lhs = lhs.base

    # Pull OuterProduct apart
    if isinstance(lhs, OuterProduct):
        args.append(lhs.ket)
        lhs = lhs.bra

    if isinstance(rhs, OuterProduct):
        extra = rhs.bra # Append to the right of the result
        rhs = rhs.ket

    # Call .doit() on Commutator/AntiCommutator.
    if isinstance(lhs, (Commutator, AntiCommutator)):
        comm = lhs.doit()
        if isinstance(comm, Add):
            return qapply(
                e.func(*(args + [comm.args[0], rhs])) +
                e.func(*(args + [comm.args[1], rhs])),
                **options
            )*extra
        else:
            return qapply(e.func(*args)*comm*rhs, **options)*extra

    # Apply tensor products of operators to states
    if isinstance(lhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in lhs.args) and \
            isinstance(rhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in rhs.args) and \
            len(lhs.args) == len(rhs.args):
        result = TensorProduct(*[qapply(lhs.args[n]*rhs.args[n], **options) for n in range(len(lhs.args))]).expand(tensorproduct=True)
        return qapply_Mul(e.func(*args), **options)*result*extra

    # For Sums, move the Sum to the right.
    if isinstance(rhs, Sum):
        if isinstance(lhs, Sum):
            if set(lhs.variables).intersection(set(rhs.variables)):
                raise ValueError('Duplicated dummy indices in separate sums in qapply.')
            limits = lhs.limits + rhs.limits
            result = Sum(qapply(lhs.function*rhs.function, **options), *limits)
            return qapply_Mul(e.func(*args)*result, **options)
        else:
            result = Sum(qapply(lhs*rhs.function, **options), *rhs.limits)
            return qapply_Mul(e.func(*args)*result, **options)

    if isinstance(lhs, Sum):
        result = Sum(qapply(lhs.function*rhs, **options), *lhs.limits)
        return qapply_Mul(e.func(*args)*result, **options)

    # Now try to actually apply the operator and build an inner product.
    _apply = getattr(lhs, '_apply_operator', None)
    if _apply is not None:
        try:
            result = _apply(rhs, **options)
        except NotImplementedError:
            result = None
    else:
        result = None

    if result is None:
        _apply_right = getattr(rhs, '_apply_from_right_to', None)
        if _apply_right is not None:
            try:
                result = _apply_right(lhs, **options)
            except NotImplementedError:
                result = None

    if result is None:
        if isinstance(lhs, BraBase) and isinstance(rhs, KetBase):
            result = InnerProduct(lhs, rhs)

    # TODO: I may need to expand before returning the final result.
    if isinstance(result, (int, complex, float)):
        return _sympify(result)
    elif result is None:
        if len(args) == 0:
            # We had two args to begin with so args=[].
            return e
        else:
            return qapply_Mul(e.func(*(args + [lhs])), **options)*rhs*extra
    elif isinstance(result, InnerProduct):
        return result*qapply_Mul(e.func(*args), **options)*extra
    else:  # result is a scalar times a Mul, Add or TensorProduct
        return qapply(e.func(*args)*result, **options)*extra
