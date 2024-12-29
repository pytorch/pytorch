"""
This module contains query handlers responsible for Matrices queries:
Square, Symmetric, Invertible etc.
"""

from sympy.logic.boolalg import conjuncts
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import test_closed_group
from sympy.matrices import MatrixBase
from sympy.matrices.expressions import (BlockMatrix, BlockDiagMatrix, Determinant,
    DiagMatrix, DiagonalMatrix, HadamardProduct, Identity, Inverse, MatAdd, MatMul,
    MatPow, MatrixExpr, MatrixSlice, MatrixSymbol, OneMatrix, Trace, Transpose,
    ZeroMatrix)
from sympy.matrices.expressions.blockmatrix import reblock_2x2
from sympy.matrices.expressions.factorizations import Factorization
from sympy.matrices.expressions.fourier import DFT
from sympy.core.logic import fuzzy_and
from sympy.utilities.iterables import sift
from sympy.core import Basic

from ..predicates.matrices import (SquarePredicate, SymmetricPredicate,
    InvertiblePredicate, OrthogonalPredicate, UnitaryPredicate,
    FullRankPredicate, PositiveDefinitePredicate, UpperTriangularPredicate,
    LowerTriangularPredicate, DiagonalPredicate, IntegerElementsPredicate,
    RealElementsPredicate, ComplexElementsPredicate)


def _Factorization(predicate, expr, assumptions):
    if predicate in expr.predicates:
        return True


# SquarePredicate

@SquarePredicate.register(MatrixExpr)
def _(expr, assumptions):
    return expr.shape[0] == expr.shape[1]


# SymmetricPredicate

@SymmetricPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if all(ask(Q.symmetric(arg), assumptions) for arg in mmul.args):
        return True
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if len(mmul.args) >= 2 and mmul.args[0] == mmul.args[-1].T:
        if len(mmul.args) == 2:
            return True
        return ask(Q.symmetric(MatMul(*mmul.args[1:-1])), assumptions)

@SymmetricPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.symmetric(base), assumptions)
    return None

@SymmetricPredicate.register(MatAdd)
def _(expr, assumptions):
    return all(ask(Q.symmetric(arg), assumptions) for arg in expr.args)

@SymmetricPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if Q.symmetric(expr) in conjuncts(assumptions):
        return True

@SymmetricPredicate.register_many(OneMatrix, ZeroMatrix)
def _(expr, assumptions):
    return ask(Q.square(expr), assumptions)

@SymmetricPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.symmetric(expr.arg), assumptions)

@SymmetricPredicate.register(MatrixSlice)
def _(expr, assumptions):
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if not expr.on_diag:
        return None
    else:
        return ask(Q.symmetric(expr.parent), assumptions)

@SymmetricPredicate.register(Identity)
def _(expr, assumptions):
    return True


# InvertiblePredicate

@InvertiblePredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if all(ask(Q.invertible(arg), assumptions) for arg in mmul.args):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@InvertiblePredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    if exp.is_negative == False:
        return ask(Q.invertible(base), assumptions)
    return None

@InvertiblePredicate.register(MatAdd)
def _(expr, assumptions):
    return None

@InvertiblePredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if Q.invertible(expr) in conjuncts(assumptions):
        return True

@InvertiblePredicate.register_many(Identity, Inverse)
def _(expr, assumptions):
    return True

@InvertiblePredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@InvertiblePredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@InvertiblePredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.invertible(expr.arg), assumptions)

@InvertiblePredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.invertible(expr.parent), assumptions)

@InvertiblePredicate.register(MatrixBase)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    return expr.rank() == expr.rows

@InvertiblePredicate.register(MatrixExpr)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    return None

@InvertiblePredicate.register(BlockMatrix)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if expr.blockshape == (1, 1):
        return ask(Q.invertible(expr.blocks[0, 0]), assumptions)
    expr = reblock_2x2(expr)
    if expr.blockshape == (2, 2):
        [[A, B], [C, D]] = expr.blocks.tolist()
        if ask(Q.invertible(A), assumptions) == True:
            invertible = ask(Q.invertible(D - C * A.I * B), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(B), assumptions) == True:
            invertible = ask(Q.invertible(C - D * B.I * A), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(C), assumptions) == True:
            invertible = ask(Q.invertible(B - A * C.I * D), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(D), assumptions) == True:
            invertible = ask(Q.invertible(A - B * D.I * C), assumptions)
            if invertible is not None:
                return invertible
    return None

@InvertiblePredicate.register(BlockDiagMatrix)
def _(expr, assumptions):
    if expr.rowblocksizes != expr.colblocksizes:
        return None
    return fuzzy_and([ask(Q.invertible(a), assumptions) for a in expr.diag])


# OrthogonalPredicate

@OrthogonalPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.orthogonal(arg), assumptions) for arg in mmul.args) and
            factor == 1):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@OrthogonalPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp:
        return ask(Q.orthogonal(base), assumptions)
    return None

@OrthogonalPredicate.register(MatAdd)
def _(expr, assumptions):
    if (len(expr.args) == 1 and
            ask(Q.orthogonal(expr.args[0]), assumptions)):
        return True

@OrthogonalPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if (not expr.is_square or
                    ask(Q.invertible(expr), assumptions) is False):
        return False
    if Q.orthogonal(expr) in conjuncts(assumptions):
        return True

@OrthogonalPredicate.register(Identity)
def _(expr, assumptions):
    return True

@OrthogonalPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@OrthogonalPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.orthogonal(expr.arg), assumptions)

@OrthogonalPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.orthogonal(expr.parent), assumptions)

@OrthogonalPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.orthogonal, expr, assumptions)


# UnitaryPredicate

@UnitaryPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.unitary(arg), assumptions) for arg in mmul.args) and
            abs(factor) == 1):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@UnitaryPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp:
        return ask(Q.unitary(base), assumptions)
    return None

@UnitaryPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if (not expr.is_square or
                    ask(Q.invertible(expr), assumptions) is False):
        return False
    if Q.unitary(expr) in conjuncts(assumptions):
        return True

@UnitaryPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.unitary(expr.arg), assumptions)

@UnitaryPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.unitary(expr.parent), assumptions)

@UnitaryPredicate.register_many(DFT, Identity)
def _(expr, assumptions):
    return True

@UnitaryPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@UnitaryPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.unitary, expr, assumptions)


# FullRankPredicate

@FullRankPredicate.register(MatMul)
def _(expr, assumptions):
    if all(ask(Q.fullrank(arg), assumptions) for arg in expr.args):
        return True

@FullRankPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp and ask(~Q.negative(exp), assumptions):
        return ask(Q.fullrank(base), assumptions)
    return None

@FullRankPredicate.register(Identity)
def _(expr, assumptions):
    return True

@FullRankPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@FullRankPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@FullRankPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.fullrank(expr.arg), assumptions)

@FullRankPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if ask(Q.orthogonal(expr.parent), assumptions):
        return True


# PositiveDefinitePredicate

@PositiveDefinitePredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.positive_definite(arg), assumptions)
            for arg in mmul.args) and factor > 0):
        return True
    if (len(mmul.args) >= 2
            and mmul.args[0] == mmul.args[-1].T
            and ask(Q.fullrank(mmul.args[0]), assumptions)):
        return ask(Q.positive_definite(
            MatMul(*mmul.args[1:-1])), assumptions)

@PositiveDefinitePredicate.register(MatPow)
def _(expr, assumptions):
    # a power of a positive definite matrix is positive definite
    if ask(Q.positive_definite(expr.args[0]), assumptions):
        return True

@PositiveDefinitePredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.positive_definite(arg), assumptions)
            for arg in expr.args):
        return True

@PositiveDefinitePredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if Q.positive_definite(expr) in conjuncts(assumptions):
        return True

@PositiveDefinitePredicate.register(Identity)
def _(expr, assumptions):
    return True

@PositiveDefinitePredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@PositiveDefinitePredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@PositiveDefinitePredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.positive_definite(expr.arg), assumptions)

@PositiveDefinitePredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.positive_definite(expr.parent), assumptions)


# UpperTriangularPredicate

@UpperTriangularPredicate.register(MatMul)
def _(expr, assumptions):
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.upper_triangular(m), assumptions) for m in matrices):
        return True

@UpperTriangularPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.upper_triangular(arg), assumptions) for arg in expr.args):
        return True

@UpperTriangularPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.upper_triangular(base), assumptions)
    return None

@UpperTriangularPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if Q.upper_triangular(expr) in conjuncts(assumptions):
        return True

@UpperTriangularPredicate.register_many(Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@UpperTriangularPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@UpperTriangularPredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.lower_triangular(expr.arg), assumptions)

@UpperTriangularPredicate.register(Inverse)
def _(expr, assumptions):
    return ask(Q.upper_triangular(expr.arg), assumptions)

@UpperTriangularPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.upper_triangular(expr.parent), assumptions)

@UpperTriangularPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.upper_triangular, expr, assumptions)

# LowerTriangularPredicate

@LowerTriangularPredicate.register(MatMul)
def _(expr, assumptions):
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.lower_triangular(m), assumptions) for m in matrices):
        return True

@LowerTriangularPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.lower_triangular(arg), assumptions) for arg in expr.args):
        return True

@LowerTriangularPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.lower_triangular(base), assumptions)
    return None

@LowerTriangularPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if Q.lower_triangular(expr) in conjuncts(assumptions):
        return True

@LowerTriangularPredicate.register_many(Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@LowerTriangularPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@LowerTriangularPredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.upper_triangular(expr.arg), assumptions)

@LowerTriangularPredicate.register(Inverse)
def _(expr, assumptions):
    return ask(Q.lower_triangular(expr.arg), assumptions)

@LowerTriangularPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.lower_triangular(expr.parent), assumptions)

@LowerTriangularPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.lower_triangular, expr, assumptions)


# DiagonalPredicate

def _is_empty_or_1x1(expr):
    return expr.shape in ((0, 0), (1, 1))

@DiagonalPredicate.register(MatMul)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.diagonal(m), assumptions) for m in matrices):
        return True

@DiagonalPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.diagonal(base), assumptions)
    return None

@DiagonalPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.diagonal(arg), assumptions) for arg in expr.args):
        return True

@DiagonalPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    if Q.diagonal(expr) in conjuncts(assumptions):
        return True

@DiagonalPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@DiagonalPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.diagonal(expr.arg), assumptions)

@DiagonalPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    if not expr.on_diag:
        return None
    else:
        return ask(Q.diagonal(expr.parent), assumptions)

@DiagonalPredicate.register_many(DiagonalMatrix, DiagMatrix, Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@DiagonalPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.diagonal, expr, assumptions)


# IntegerElementsPredicate

def BM_elements(predicate, expr, assumptions):
    """ Block Matrix elements. """
    return all(ask(predicate(b), assumptions) for b in expr.blocks)

def MS_elements(predicate, expr, assumptions):
    """ Matrix Slice elements. """
    return ask(predicate(expr.parent), assumptions)

def MatMul_elements(matrix_predicate, scalar_predicate, expr, assumptions):
    d = sift(expr.args, lambda x: isinstance(x, MatrixExpr))
    factors, matrices = d[False], d[True]
    return fuzzy_and([
        test_closed_group(Basic(*factors), assumptions, scalar_predicate),
        test_closed_group(Basic(*matrices), assumptions, matrix_predicate)])


@IntegerElementsPredicate.register_many(Determinant, HadamardProduct, MatAdd,
    Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.integer_elements)

@IntegerElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    if exp.is_negative == False:
        return ask(Q.integer_elements(base), assumptions)
    return None

@IntegerElementsPredicate.register_many(Identity, OneMatrix, ZeroMatrix)
def _(expr, assumptions):
    return True

@IntegerElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.integer_elements, Q.integer, expr, assumptions)

@IntegerElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.integer_elements, expr, assumptions)

@IntegerElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.integer_elements, expr, assumptions)


# RealElementsPredicate

@RealElementsPredicate.register_many(Determinant, Factorization, HadamardProduct,
    MatAdd, Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.real_elements)

@RealElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.real_elements(base), assumptions)
    return None

@RealElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.real_elements, Q.real, expr, assumptions)

@RealElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.real_elements, expr, assumptions)

@RealElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.real_elements, expr, assumptions)


# ComplexElementsPredicate

@ComplexElementsPredicate.register_many(Determinant, Factorization, HadamardProduct,
    Inverse, MatAdd, Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.complex_elements)

@ComplexElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.complex_elements(base), assumptions)
    return None

@ComplexElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.complex_elements, Q.complex, expr, assumptions)

@ComplexElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.complex_elements, expr, assumptions)

@ComplexElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.complex_elements, expr, assumptions)

@ComplexElementsPredicate.register(DFT)
def _(expr, assumptions):
    return True
