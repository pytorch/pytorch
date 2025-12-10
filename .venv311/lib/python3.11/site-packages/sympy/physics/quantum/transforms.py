"""Transforms that are always applied to quantum expressions.

This module uses the kind and _constructor_postprocessor_mapping APIs
to transform different combinations of Operators, Bras, and Kets into
Inner/Outer/TensorProducts. These transformations are registered
with the postprocessing API of core classes like `Mul` and `Pow` and
are always applied to any expression involving Bras, Kets, and
Operators. This API replaces the custom `__mul__` and `__pow__`
methods of the quantum classes, which were found to be inconsistent.

THIS IS EXPERIMENTAL.
"""
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.multipledispatch.dispatcher import (
    Dispatcher, ambiguity_register_error_ignore_dup
)
from sympy.utilities.misc import debug

from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.kind import KetKind, BraKind, OperatorKind
from sympy.physics.quantum.operator import (
    OuterProduct, IdentityOperator, Operator
)
from sympy.physics.quantum.state import BraBase, KetBase, StateBase
from sympy.physics.quantum.tensorproduct import TensorProduct


#-----------------------------------------------------------------------------
# Multipledispatch based transformed for Mul and Pow
#-----------------------------------------------------------------------------

_transform_state_pair = Dispatcher('_transform_state_pair')
"""Transform a pair of expression in a Mul to their canonical form.

All functions that are registered with this dispatcher need to take
two inputs and return either tuple of transformed outputs, or None if no
transform is applied. The output tuple is inserted into the right place
of the ``Mul`` that is being put into canonical form. It works something like
the following:

``Mul(a, b, c, d, e, f) -> Mul(*(_transform_state_pair(a, b) + (c, d, e, f))))``

The transforms here are always applied when quantum objects are multiplied.

THIS IS EXPERIMENTAL.

However, users of ``sympy.physics.quantum`` can import this dispatcher and
register their own transforms to control the canonical form of products
of quantum expressions.
"""

@_transform_state_pair.register(Expr, Expr)
def _transform_expr(a, b):
    """Default transformer that does nothing for base types."""
    return None


# The identity times anything is the anything.
_transform_state_pair.add(
    (IdentityOperator, Expr),
    lambda x, y: (y,),
    on_ambiguity=ambiguity_register_error_ignore_dup
)
_transform_state_pair.add(
    (Expr, IdentityOperator),
    lambda x, y: (x,),
    on_ambiguity=ambiguity_register_error_ignore_dup
)
_transform_state_pair.add(
    (IdentityOperator, IdentityOperator),
    lambda x, y: S.One,
    on_ambiguity=ambiguity_register_error_ignore_dup
)

@_transform_state_pair.register(BraBase, KetBase)
def _transform_bra_ket(a, b):
    """Transform a bra*ket -> InnerProduct(bra, ket)."""
    return (InnerProduct(a, b),)

@_transform_state_pair.register(KetBase, BraBase)
def _transform_ket_bra(a, b):
    """Transform a keT*bra -> OuterProduct(ket, bra)."""
    return (OuterProduct(a, b),)

@_transform_state_pair.register(KetBase, KetBase)
def _transform_ket_ket(a, b):
    """Raise a TypeError if a user tries to multiply two kets.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    raise TypeError(
        'Multiplication of two kets is not allowed. Use TensorProduct instead.'
    )

@_transform_state_pair.register(BraBase, BraBase)
def _transform_bra_bra(a, b):
    """Raise a TypeError if a user tries to multiply two bras.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    raise TypeError(
        'Multiplication of two bras is not allowed. Use TensorProduct instead.'
    )

@_transform_state_pair.register(OuterProduct, KetBase)
def _transform_op_ket(a, b):
    return (InnerProduct(a.bra, b), a.ket)

@_transform_state_pair.register(BraBase, OuterProduct)
def _transform_bra_op(a, b):
    return (InnerProduct(a, b.ket), b.bra)

@_transform_state_pair.register(TensorProduct, KetBase)
def _transform_tp_ket(a, b):
    """Raise a TypeError if a user tries to multiply TensorProduct(*kets)*ket.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if a.kind == KetKind:
        raise TypeError(
            'Multiplication of TensorProduct(*kets)*ket is invalid.'
        )

@_transform_state_pair.register(KetBase, TensorProduct)
def _transform_ket_tp(a, b):
    """Raise a TypeError if a user tries to multiply ket*TensorProduct(*kets).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if b.kind == KetKind:
        raise TypeError(
            'Multiplication of ket*TensorProduct(*kets) is invalid.'
        )

@_transform_state_pair.register(TensorProduct, BraBase)
def _transform_tp_bra(a, b):
    """Raise a TypeError if a user tries to multiply TensorProduct(*bras)*bra.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if a.kind == BraKind:
        raise TypeError(
            'Multiplication of TensorProduct(*bras)*bra is invalid.'
        )

@_transform_state_pair.register(BraBase, TensorProduct)
def _transform_bra_tp(a, b):
    """Raise a TypeError if a user tries to multiply bra*TensorProduct(*bras).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if b.kind == BraKind:
        raise TypeError(
            'Multiplication of bra*TensorProduct(*bras) is invalid.'
        )

@_transform_state_pair.register(TensorProduct, TensorProduct)
def _transform_tp_tp(a, b):
    """Combine a product of tensor products if their number of args matches."""
    debug('_transform_tp_tp', a, b)
    if len(a.args) == len(b.args):
        if a.kind == BraKind and b.kind == KetKind:
            return tuple([InnerProduct(i, j) for (i, j) in zip(a.args, b.args)])
        else:
            return (TensorProduct(*(i*j for (i, j) in zip(a.args, b.args))), )

@_transform_state_pair.register(OuterProduct, OuterProduct)
def _transform_op_op(a, b):
    """Extract an inner produt from a product of outer products."""
    return (InnerProduct(a.bra, b.ket), OuterProduct(a.ket, b.bra))


#-----------------------------------------------------------------------------
# Postprocessing transforms for Mul and Pow
#-----------------------------------------------------------------------------


def _postprocess_state_mul(expr):
    """Transform a ``Mul`` of quantum expressions into canonical form.

    This function is registered ``_constructor_postprocessor_mapping`` as a
    transformer for ``Mul``. This means that every time a quantum expression
    is multiplied, this function will be called to transform it into canonical
    form as defined by the binary functions registered with
    ``_transform_state_pair``.

    The algorithm of this function is as follows. It walks the args
    of the input ``Mul`` from left to right and calls ``_transform_state_pair``
    on every overlapping pair of args. Each time ``_transform_state_pair``
    is called it can return a tuple of items or None. If None, the pair isn't
    transformed. If a tuple, then the last element of the tuple goes back into
    the args to be transformed again and the others are extended onto the result
    args list.

    The algorithm can be visualized in the following table:

    step   result                                 args
    ============================================================================
    #0     []                                     [a, b, c, d, e, f]
    #1     []                                     [T(a,b), c, d, e, f]
    #2     [T(a,b)[:-1]]                          [T(a,b)[-1], c, d, e, f]
    #3     [T(a,b)[:-1]]                          [T(T(a,b)[-1], c), d, e, f]
    #4     [T(a,b)[:-1], T(T(a,b)[-1], c)[:-1]]   [T(T(T(a,b)[-1], c)[-1], d), e, f]
    #5     ...

    One limitation of the current implementation is that we assume that only the
    last item of the transformed tuple goes back into the args to be transformed
    again. These seems to handle the cases needed for Mul. However, we may need
    to extend the algorithm to have the entire tuple go back into the args for
    further transformation.
    """
    args = list(expr.args)
    result = []

    # Continue as long as we have at least 2 elements
    while len(args) > 1:
        # Get first two elements
        first = args.pop(0)
        second = args[0]  # Look at second element without popping yet

        transformed = _transform_state_pair(first, second)

        if transformed is None:
            # If transform returns None, append first element
            result.append(first)
        else:
            # This item was transformed, pop and discard
            args.pop(0)
            # The last item goes back to be transformed again
            args.insert(0, transformed[-1])
            # All other items go directly into the result
            result.extend(transformed[:-1])

    # Append any remaining element
    if args:
        result.append(args[0])

    return Mul._from_args(result, is_commutative=False)


def _postprocess_state_pow(expr):
    """Handle bras and kets raised to powers.

    Under ``*`` multiplication this is invalid. Users should use a
    TensorProduct instead.
    """
    base, exp = expr.as_base_exp()
    if base.kind == KetKind or base.kind == BraKind:
        raise TypeError(
            'A bra or ket to a power is invalid, use TensorProduct instead.'
        )


def _postprocess_tp_pow(expr):
    """Handle TensorProduct(*operators)**(positive integer).

    This handles a tensor product of operators, to an integer power.
    The power here is interpreted as regular multiplication, not
    tensor product exponentiation. The form of exponentiation performed
    here leaves the space and dimension of the object the same.

    This operation does not make sense for tensor product's of states.
    """
    base, exp = expr.as_base_exp()
    debug('_postprocess_tp_pow: ', base, exp, expr.args)
    if isinstance(base, TensorProduct) and exp.is_integer and exp.is_positive and base.kind == OperatorKind:
        new_args = [a**exp for a in base.args]
        return TensorProduct(*new_args)


#-----------------------------------------------------------------------------
# Register the transformers with Basic._constructor_postprocessor_mapping
#-----------------------------------------------------------------------------


Basic._constructor_postprocessor_mapping[StateBase] = {
    "Mul": [_postprocess_state_mul],
    "Pow": [_postprocess_state_pow]
}

Basic._constructor_postprocessor_mapping[TensorProduct] = {
    "Mul": [_postprocess_state_mul],
    "Pow": [_postprocess_tp_pow]
}

Basic._constructor_postprocessor_mapping[Operator] = {
    "Mul": [_postprocess_state_mul]
}
