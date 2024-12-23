"""

Routines for computing eigenvectors with DomainMatrix.

"""
from sympy.core.symbol import Dummy

from ..agca.extensions import FiniteExtension
from ..factortools import dup_factor_list
from ..polyroots import roots
from ..polytools import Poly
from ..rootoftools import CRootOf

from .domainmatrix import DomainMatrix


def dom_eigenvects(A, l=Dummy('lambda')):
    charpoly = A.charpoly()
    rows, cols = A.shape
    domain = A.domain
    _, factors = dup_factor_list(charpoly, domain)

    rational_eigenvects = []
    algebraic_eigenvects = []
    for base, exp in factors:
        if len(base) == 2:
            field = domain
            eigenval = -base[1] / base[0]

            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            basis = (A - EE).nullspace(divide_last=True)
            rational_eigenvects.append((field, eigenval, exp, basis))
        else:
            minpoly = Poly.from_list(base, l, domain=domain)
            field = FiniteExtension(minpoly)
            eigenval = field(l)

            AA_items = [
                [Poly.from_list([item], l, domain=domain).rep for item in row]
                for row in A.rep.to_ddm()]
            AA_items = [[field(item) for item in row] for row in AA_items]
            AA = DomainMatrix(AA_items, (rows, cols), field)
            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            basis = (AA - EE).nullspace(divide_last=True)
            algebraic_eigenvects.append((field, minpoly, exp, basis))

    return rational_eigenvects, algebraic_eigenvects


def dom_eigenvects_to_sympy(
    rational_eigenvects, algebraic_eigenvects,
    Matrix, **kwargs
):
    result = []

    for field, eigenvalue, multiplicity, eigenvects in rational_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        eigenvalue = field.to_sympy(eigenvalue)
        new_eigenvects = [
            Matrix([field.to_sympy(x) for x in vect])
            for vect in eigenvects]
        result.append((eigenvalue, multiplicity, new_eigenvects))

    for field, minpoly, multiplicity, eigenvects in algebraic_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        l = minpoly.gens[0]

        eigenvects = [[field.to_sympy(x) for x in vect] for vect in eigenvects]

        degree = minpoly.degree()
        minpoly = minpoly.as_expr()
        eigenvals = roots(minpoly, l, **kwargs)
        if len(eigenvals) != degree:
            eigenvals = [CRootOf(minpoly, l, idx) for idx in range(degree)]

        for eigenvalue in eigenvals:
            new_eigenvects = [
                Matrix([x.subs(l, eigenvalue) for x in vect])
                for vect in eigenvects]
            result.append((eigenvalue, multiplicity, new_eigenvects))

    return result
