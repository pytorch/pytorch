from types import FunctionType
from collections import Counter

from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps

from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd

from .exceptions import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot

from .utilities import _iszero, _simplify


__doctest_requires__ = {
    ('_is_indefinite',
     '_is_negative_definite',
     '_is_negative_semidefinite',
     '_is_positive_definite',
     '_is_positive_semidefinite'): ['matplotlib'],
}


def _eigenvals_eigenvects_mpmath(M):
    norm2 = lambda v: mp.sqrt(sum(i**2 for i in v))

    v1 = None
    prec = max(x._prec for x in M.atoms(Float))
    eps = 2**-prec

    while prec < DEFAULT_MAXPREC:
        with workprec(prec):
            A = mp.matrix(M.evalf(n=prec_to_dps(prec)))
            E, ER = mp.eig(A)
            v2 = norm2([i for e in E for i in (mp.re(e), mp.im(e))])
            if v1 is not None and mp.fabs(v1 - v2) < eps:
                return E, ER
            v1 = v2
        prec *= 2

    # we get here because the next step would have taken us
    # past MAXPREC or because we never took a step; in case
    # of the latter, we refuse to send back a solution since
    # it would not have been verified; we also resist taking
    # a small step to arrive exactly at MAXPREC since then
    # the two calculations might be artificially close.
    raise PrecisionExhausted


def _eigenvals_mpmath(M, multiple=False):
    """Compute eigenvalues using mpmath"""
    E, _ = _eigenvals_eigenvects_mpmath(M)
    result = [_sympify(x) for x in E]
    if multiple:
        return result
    return dict(Counter(result))


def _eigenvects_mpmath(M):
    E, ER = _eigenvals_eigenvects_mpmath(M)
    result = []
    for i in range(M.rows):
        eigenval = _sympify(E[i])
        eigenvect = _sympify(ER[:, i])
        result.append((eigenval, 1, [eigenvect]))

    return result


# This function is a candidate for caching if it gets implemented for matrices.
def _eigenvals(
    M, error_when_incomplete=True, *, simplify=False, multiple=False,
    rational=False, **flags):
    r"""Compute eigenvalues of the matrix.

    Parameters
    ==========

    error_when_incomplete : bool, optional
        If it is set to ``True``, it will raise an error if not all
        eigenvalues are computed. This is caused by ``roots`` not returning
        a full list of eigenvalues.

    simplify : bool or function, optional
        If it is set to ``True``, it attempts to return the most
        simplified form of expressions returned by applying default
        simplification method in every routine.

        If it is set to ``False``, it will skip simplification in this
        particular routine to save computation resources.

        If a function is passed to, it will attempt to apply
        the particular function as simplification method.

    rational : bool, optional
        If it is set to ``True``, every floating point numbers would be
        replaced with rationals before computation. It can solve some
        issues of ``roots`` routine not working well with floats.

    multiple : bool, optional
        If it is set to ``True``, the result will be in the form of a
        list.

        If it is set to ``False``, the result will be in the form of a
        dictionary.

    Returns
    =======

    eigs : list or dict
        Eigenvalues of a matrix. The return format would be specified by
        the key ``multiple``.

    Raises
    ======

    MatrixError
        If not enough roots had got computed.

    NonSquareMatrixError
        If attempted to compute eigenvalues from a non-square matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1])
    >>> M.eigenvals()
    {-1: 1, 0: 1, 2: 1}

    See Also
    ========

    MatrixBase.charpoly
    eigenvects

    Notes
    =====

    Eigenvalues of a matrix $A$ can be computed by solving a matrix
    equation $\det(A - \lambda I) = 0$

    It's not always possible to return radical solutions for
    eigenvalues for matrices larger than $4, 4$ shape due to
    Abel-Ruffini theorem.

    If there is no radical solution is found for the eigenvalue,
    it may return eigenvalues in the form of
    :class:`sympy.polys.rootoftools.ComplexRootOf`.
    """
    if not M:
        if multiple:
            return []
        return {}

    if not M.is_square:
        raise NonSquareMatrixError("{} must be a square matrix.".format(M))

    if M._rep.domain not in (ZZ, QQ):
        # Skip this check for ZZ/QQ because it can be slow
        if all(x.is_number for x in M) and M.has(Float):
            return _eigenvals_mpmath(M, multiple=multiple)

    if rational:
        from sympy.simplify import nsimplify
        M = M.applyfunc(
            lambda x: nsimplify(x, rational=True) if x.has(Float) else x)

    if multiple:
        return _eigenvals_list(
            M, error_when_incomplete=error_when_incomplete, simplify=simplify,
            **flags)
    return _eigenvals_dict(
        M, error_when_incomplete=error_when_incomplete, simplify=simplify,
        **flags)


eigenvals_error_message = \
"It is not always possible to express the eigenvalues of a matrix " + \
"of size 5x5 or higher in radicals. " + \
"We have CRootOf, but domains other than the rationals are not " + \
"currently supported. " + \
"If there are no symbols in the matrix, " + \
"it should still be possible to compute numeric approximations " + \
"of the eigenvalues using " + \
"M.evalf().eigenvals() or M.charpoly().nroots()."


def _eigenvals_list(
    M, error_when_incomplete=True, simplify=False, **flags):
    iblocks = M.strongly_connected_components()
    all_eigs = []
    is_dom = M._rep.domain in (ZZ, QQ)
    for b in iblocks:

        # Fast path for a 1x1 block:
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            all_eigs.append(val)
            continue

        block = M[b, b]

        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()

        eigs = roots(charpoly, multiple=True, **flags)

        if len(eigs) != block.rows:
            try:
                eigs = charpoly.all_roots(multiple=True)
            except NotImplementedError:
                if error_when_incomplete:
                    raise MatrixError(eigenvals_error_message)
                else:
                    eigs = []

        all_eigs += eigs

    if not simplify:
        return all_eigs
    if not isinstance(simplify, FunctionType):
        simplify = _simplify
    return [simplify(value) for value in all_eigs]


def _eigenvals_dict(
    M, error_when_incomplete=True, simplify=False, **flags):
    iblocks = M.strongly_connected_components()
    all_eigs = {}
    is_dom = M._rep.domain in (ZZ, QQ)
    for b in iblocks:

        # Fast path for a 1x1 block:
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            all_eigs[val] = all_eigs.get(val, 0) + 1
            continue

        block = M[b, b]

        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()

        eigs = roots(charpoly, multiple=False, **flags)

        if sum(eigs.values()) != block.rows:
            try:
                eigs = dict(charpoly.all_roots(multiple=False))
            except NotImplementedError:
                if error_when_incomplete:
                    raise MatrixError(eigenvals_error_message)
                else:
                    eigs = {}

        for k, v in eigs.items():
            if k in all_eigs:
                all_eigs[k] += v
            else:
                all_eigs[k] = v

    if not simplify:
        return all_eigs
    if not isinstance(simplify, FunctionType):
        simplify = _simplify
    return {simplify(key): value for key, value in all_eigs.items()}


def _eigenspace(M, eigenval, iszerofunc=_iszero, simplify=False):
    """Get a basis for the eigenspace for a particular eigenvalue"""
    m   = M - M.eye(M.rows) * eigenval
    ret = m.nullspace(iszerofunc=iszerofunc)

    # The nullspace for a real eigenvalue should be non-trivial.
    # If we didn't find an eigenvector, try once more a little harder
    if len(ret) == 0 and simplify:
        ret = m.nullspace(iszerofunc=iszerofunc, simplify=True)
    if len(ret) == 0:
        raise NotImplementedError(
            "Can't evaluate eigenvector for eigenvalue {}".format(eigenval))
    return ret


def _eigenvects_DOM(M, **kwargs):
    DOM = DomainMatrix.from_Matrix(M, field=True, extension=True)
    DOM = DOM.to_dense()

    if DOM.domain != EX:
        rational, algebraic = dom_eigenvects(DOM)
        eigenvects = dom_eigenvects_to_sympy(
            rational, algebraic, M.__class__, **kwargs)
        eigenvects = sorted(eigenvects, key=lambda x: default_sort_key(x[0]))

        return eigenvects
    return None


def _eigenvects_sympy(M, iszerofunc, simplify=True, **flags):
    eigenvals = M.eigenvals(rational=False, **flags)

    # Make sure that we have all roots in radical form
    for x in eigenvals:
        if x.has(CRootOf):
            raise MatrixError(
                "Eigenvector computation is not implemented if the matrix have "
                "eigenvalues in CRootOf form")

    eigenvals = sorted(eigenvals.items(), key=default_sort_key)
    ret = []
    for val, mult in eigenvals:
        vects = _eigenspace(M, val, iszerofunc=iszerofunc, simplify=simplify)
        ret.append((val, mult, vects))
    return ret


# This functions is a candidate for caching if it gets implemented for matrices.
def _eigenvects(M, error_when_incomplete=True, iszerofunc=_iszero, *, chop=False, **flags):
    """Compute eigenvectors of the matrix.

    Parameters
    ==========

    error_when_incomplete : bool, optional
        Raise an error when not all eigenvalues are computed. This is
        caused by ``roots`` not returning a full list of eigenvalues.

    iszerofunc : function, optional
        Specifies a zero testing function to be used in ``rref``.

        Default value is ``_iszero``, which uses SymPy's naive and fast
        default assumption handler.

        It can also accept any user-specified zero testing function, if it
        is formatted as a function which accepts a single symbolic argument
        and returns ``True`` if it is tested as zero and ``False`` if it
        is tested as non-zero, and ``None`` if it is undecidable.

    simplify : bool or function, optional
        If ``True``, ``as_content_primitive()`` will be used to tidy up
        normalization artifacts.

        It will also be used by the ``nullspace`` routine.

    chop : bool or positive number, optional
        If the matrix contains any Floats, they will be changed to Rationals
        for computation purposes, but the answers will be returned after
        being evaluated with evalf. The ``chop`` flag is passed to ``evalf``.
        When ``chop=True`` a default precision will be used; a number will
        be interpreted as the desired level of precision.

    Returns
    =======

    ret : [(eigenval, multiplicity, eigenspace), ...]
        A ragged list containing tuples of data obtained by ``eigenvals``
        and ``nullspace``.

        ``eigenspace`` is a list containing the ``eigenvector`` for each
        eigenvalue.

        ``eigenvector`` is a vector in the form of a ``Matrix``. e.g.
        a vector of length 3 is returned as ``Matrix([a_1, a_2, a_3])``.

    Raises
    ======

    NotImplementedError
        If failed to compute nullspace.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1])
    >>> M.eigenvects()
    [(-1, 1, [Matrix([
    [-1],
    [ 1],
    [ 0]])]), (0, 1, [Matrix([
    [ 0],
    [-1],
    [ 1]])]), (2, 1, [Matrix([
    [2/3],
    [1/3],
    [  1]])])]

    See Also
    ========

    eigenvals
    MatrixBase.nullspace
    """
    simplify = flags.get('simplify', True)
    primitive = flags.get('simplify', False)
    flags.pop('simplify', None)  # remove this if it's there
    flags.pop('multiple', None)  # remove this if it's there

    if not isinstance(simplify, FunctionType):
        simpfunc = _simplify if simplify else lambda x: x

    has_floats = M.has(Float)
    if has_floats:
        if all(x.is_number for x in M):
            return _eigenvects_mpmath(M)
        from sympy.simplify import nsimplify
        M = M.applyfunc(lambda x: nsimplify(x, rational=True))

    ret = _eigenvects_DOM(M)
    if ret is None:
        ret = _eigenvects_sympy(M, iszerofunc, simplify=simplify, **flags)

    if primitive:
        # if the primitive flag is set, get rid of any common
        # integer denominators
        def denom_clean(l):
            return [(v / gcd(list(v))).applyfunc(simpfunc) for v in l]

        ret = [(val, mult, denom_clean(es)) for val, mult, es in ret]

    if has_floats:
        # if we had floats to start with, turn the eigenvectors to floats
        ret = [(val.evalf(chop=chop), mult, [v.evalf(chop=chop) for v in es])
                for val, mult, es in ret]

    return ret


def _is_diagonalizable_with_eigen(M, reals_only=False):
    """See _is_diagonalizable. This function returns the bool along with the
    eigenvectors to avoid calculating them again in functions like
    ``diagonalize``."""

    if not M.is_square:
        return False, []

    eigenvecs = M.eigenvects(simplify=True)

    for val, mult, basis in eigenvecs:
        if reals_only and not val.is_real: # if we have a complex eigenvalue
            return False, eigenvecs

        if mult != len(basis): # if the geometric multiplicity doesn't equal the algebraic
            return False, eigenvecs

    return True, eigenvecs

def _is_diagonalizable(M, reals_only=False, **kwargs):
    """Returns ``True`` if a matrix is diagonalizable.

    Parameters
    ==========

    reals_only : bool, optional
        If ``True``, it tests whether the matrix can be diagonalized
        to contain only real numbers on the diagonal.


        If ``False``, it tests whether the matrix can be diagonalized
        at all, even with numbers that may not be real.

    Examples
    ========

    Example of a diagonalizable matrix:

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 0], [0, 3, 0], [2, -4, 2]])
    >>> M.is_diagonalizable()
    True

    Example of a non-diagonalizable matrix:

    >>> M = Matrix([[0, 1], [0, 0]])
    >>> M.is_diagonalizable()
    False

    Example of a matrix that is diagonalized in terms of non-real entries:

    >>> M = Matrix([[0, 1], [-1, 0]])
    >>> M.is_diagonalizable(reals_only=False)
    True
    >>> M.is_diagonalizable(reals_only=True)
    False

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.is_diagonal
    diagonalize
    """
    if not M.is_square:
        return False

    if all(e.is_real for e in M) and M.is_symmetric():
        return True

    if all(e.is_complex for e in M) and M.is_hermitian:
        return True

    return _is_diagonalizable_with_eigen(M, reals_only=reals_only)[0]


#G&VL, Matrix Computations, Algo 5.4.2
def _householder_vector(x):
    if not x.cols == 1:
        raise ValueError("Input must be a column matrix")
    v = x.copy()
    v_plus = x.copy()
    v_minus = x.copy()
    q = x[0, 0] / abs(x[0, 0])
    norm_x = x.norm()
    v_plus[0, 0] = x[0, 0] + q * norm_x
    v_minus[0, 0] = x[0, 0] - q * norm_x
    if x[1:, 0].norm() == 0:
        bet = 0
        v[0, 0] = 1
    else:
        if v_plus.norm() <= v_minus.norm():
            v = v_plus
        else:
            v = v_minus
        v = v / v[0]
        bet = 2 / (v.norm() ** 2)
    return v, bet


def _bidiagonal_decmp_hholder(M):
    m = M.rows
    n = M.cols
    A = M.as_mutable()
    U, V = A.eye(m), A.eye(n)
    for i in range(min(m, n)):
        v, bet = _householder_vector(A[i:, i])
        hh_mat = A.eye(m - i) - bet * v * v.H
        A[i:, i:] = hh_mat * A[i:, i:]
        temp = A.eye(m)
        temp[i:, i:] = hh_mat
        U = U * temp
        if i + 1 <= n - 2:
            v, bet = _householder_vector(A[i, i+1:].T)
            hh_mat = A.eye(n - i - 1) - bet * v * v.H
            A[i:, i+1:] = A[i:, i+1:] * hh_mat
            temp = A.eye(n)
            temp[i+1:, i+1:] = hh_mat
            V = temp * V
    return U, A, V


def _eval_bidiag_hholder(M):
    m = M.rows
    n = M.cols
    A = M.as_mutable()
    for i in range(min(m, n)):
        v, bet = _householder_vector(A[i:, i])
        hh_mat = A.eye(m-i) - bet * v * v.H
        A[i:, i:] = hh_mat * A[i:, i:]
        if i + 1 <= n - 2:
            v, bet = _householder_vector(A[i, i+1:].T)
            hh_mat = A.eye(n - i - 1) - bet * v * v.H
            A[i:, i+1:] = A[i:, i+1:] * hh_mat
    return A


def _bidiagonal_decomposition(M, upper=True):
    """
    Returns $(U,B,V.H)$ for

    $$A = UBV^{H}$$

    where $A$ is the input matrix, and $B$ is its Bidiagonalized form

    Note: Bidiagonal Computation can hang for symbolic matrices.

    Parameters
    ==========

    upper : bool. Whether to do upper bidiagnalization or lower.
                True for upper and False for lower.

    References
    ==========

    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition
    .. [2] Complex Matrix Bidiagonalization, https://github.com/vslobody/Householder-Bidiagonalization

    """

    if not isinstance(upper, bool):
        raise ValueError("upper must be a boolean")

    if upper:
        return _bidiagonal_decmp_hholder(M)

    X = _bidiagonal_decmp_hholder(M.H)
    return X[2].H, X[1].H, X[0].H


def _bidiagonalize(M, upper=True):
    """
    Returns $B$, the Bidiagonalized form of the input matrix.

    Note: Bidiagonal Computation can hang for symbolic matrices.

    Parameters
    ==========

    upper : bool. Whether to do upper bidiagnalization or lower.
                True for upper and False for lower.

    References
    ==========

    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition
    .. [2] Complex Matrix Bidiagonalization : https://github.com/vslobody/Householder-Bidiagonalization

    """

    if not isinstance(upper, bool):
        raise ValueError("upper must be a boolean")

    if upper:
        return _eval_bidiag_hholder(M)
    return _eval_bidiag_hholder(M.H).H


def _diagonalize(M, reals_only=False, sort=False, normalize=False):
    """
    Return (P, D), where D is diagonal and

        D = P^-1 * M * P

    where M is current matrix.

    Parameters
    ==========

    reals_only : bool. Whether to throw an error if complex numbers are need
                    to diagonalize. (Default: False)

    sort : bool. Sort the eigenvalues along the diagonal. (Default: False)

    normalize : bool. If True, normalize the columns of P. (Default: False)

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    >>> M
    Matrix([
    [1,  2, 0],
    [0,  3, 0],
    [2, -4, 2]])
    >>> (P, D) = M.diagonalize()
    >>> D
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])
    >>> P
    Matrix([
    [-1, 0, -1],
    [ 0, 0, -1],
    [ 2, 1,  2]])
    >>> P.inv() * M * P
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.is_diagonal
    is_diagonalizable
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    is_diagonalizable, eigenvecs = _is_diagonalizable_with_eigen(M,
                reals_only=reals_only)

    if not is_diagonalizable:
        raise MatrixError("Matrix is not diagonalizable")

    if sort:
        eigenvecs = sorted(eigenvecs, key=default_sort_key)

    p_cols, diag = [], []

    for val, mult, basis in eigenvecs:
        diag   += [val] * mult
        p_cols += basis

    if normalize:
        p_cols = [v / v.norm() for v in p_cols]

    return M.hstack(*p_cols), M.diag(*diag)


def _fuzzy_positive_definite(M):
    positive_diagonals = M._has_positive_diagonals()
    if positive_diagonals is False:
        return False

    if positive_diagonals and M.is_strongly_diagonally_dominant:
        return True

    return None


def _fuzzy_positive_semidefinite(M):
    nonnegative_diagonals = M._has_nonnegative_diagonals()
    if nonnegative_diagonals is False:
        return False

    if nonnegative_diagonals and M.is_weakly_diagonally_dominant:
        return True

    return None


def _is_positive_definite(M):
    if not M.is_hermitian:
        if not M.is_square:
            return False
        M = M + M.H

    fuzzy = _fuzzy_positive_definite(M)
    if fuzzy is not None:
        return fuzzy

    return _is_positive_definite_GE(M)


def _is_positive_semidefinite(M):
    if not M.is_hermitian:
        if not M.is_square:
            return False
        M = M + M.H

    fuzzy = _fuzzy_positive_semidefinite(M)
    if fuzzy is not None:
        return fuzzy

    return _is_positive_semidefinite_cholesky(M)


def _is_negative_definite(M):
    return _is_positive_definite(-M)


def _is_negative_semidefinite(M):
    return _is_positive_semidefinite(-M)


def _is_indefinite(M):
    if M.is_hermitian:
        eigen = M.eigenvals()
        args1        = [x.is_positive for x in eigen.keys()]
        any_positive = fuzzy_or(args1)
        args2        = [x.is_negative for x in eigen.keys()]
        any_negative = fuzzy_or(args2)

        return fuzzy_and([any_positive, any_negative])

    elif M.is_square:
        return (M + M.H).is_indefinite

    return False


def _is_positive_definite_GE(M):
    """A division-free gaussian elimination method for testing
    positive-definiteness."""
    M = M.as_mutable()
    size = M.rows

    for i in range(size):
        is_positive = M[i, i].is_positive
        if is_positive is not True:
            return is_positive
        for j in range(i+1, size):
            M[j, i+1:] = M[i, i] * M[j, i+1:] - M[j, i] * M[i, i+1:]
    return True


def _is_positive_semidefinite_cholesky(M):
    """Uses Cholesky factorization with complete pivoting

    References
    ==========

    .. [1] http://eprints.ma.man.ac.uk/1199/1/covered/MIMS_ep2008_116.pdf

    .. [2] https://www.value-at-risk.net/cholesky-factorization/
    """
    M = M.as_mutable()
    for k in range(M.rows):
        diags = [M[i, i] for i in range(k, M.rows)]
        pivot, pivot_val, nonzero, _ = _find_reasonable_pivot(diags)

        if nonzero:
            return None

        if pivot is None:
            for i in range(k+1, M.rows):
                for j in range(k, M.cols):
                    iszero = M[i, j].is_zero
                    if iszero is None:
                        return None
                    elif iszero is False:
                        return False
            return True

        if M[k, k].is_negative or pivot_val.is_negative:
            return False
        elif not (M[k, k].is_nonnegative and pivot_val.is_nonnegative):
            return None

        if pivot > 0:
            M.col_swap(k, k+pivot)
            M.row_swap(k, k+pivot)

        M[k, k] = sqrt(M[k, k])
        M[k, k+1:] /= M[k, k]
        M[k+1:, k+1:] -= M[k, k+1:].H * M[k, k+1:]

    return M[-1, -1].is_nonnegative


_doc_positive_definite = \
    r"""Finds out the definiteness of a matrix.

    Explanation
    ===========

    A square real matrix $A$ is:

    - A positive definite matrix if $x^T A x > 0$
      for all non-zero real vectors $x$.
    - A positive semidefinite matrix if $x^T A x \geq 0$
      for all non-zero real vectors $x$.
    - A negative definite matrix if $x^T A x < 0$
      for all non-zero real vectors $x$.
    - A negative semidefinite matrix if $x^T A x \leq 0$
      for all non-zero real vectors $x$.
    - An indefinite matrix if there exists non-zero real vectors
      $x, y$ with $x^T A x > 0 > y^T A y$.

    A square complex matrix $A$ is:

    - A positive definite matrix if $\text{re}(x^H A x) > 0$
      for all non-zero complex vectors $x$.
    - A positive semidefinite matrix if $\text{re}(x^H A x) \geq 0$
      for all non-zero complex vectors $x$.
    - A negative definite matrix if $\text{re}(x^H A x) < 0$
      for all non-zero complex vectors $x$.
    - A negative semidefinite matrix if $\text{re}(x^H A x) \leq 0$
      for all non-zero complex vectors $x$.
    - An indefinite matrix if there exists non-zero complex vectors
      $x, y$ with $\text{re}(x^H A x) > 0 > \text{re}(y^H A y)$.

    A matrix need not be symmetric or hermitian to be positive definite.

    - A real non-symmetric matrix is positive definite if and only if
      $\frac{A + A^T}{2}$ is positive definite.
    - A complex non-hermitian matrix is positive definite if and only if
      $\frac{A + A^H}{2}$ is positive definite.

    And this extension can apply for all the definitions above.

    However, for complex cases, you can restrict the definition of
    $\text{re}(x^H A x) > 0$ to $x^H A x > 0$ and require the matrix
    to be hermitian.
    But we do not present this restriction for computation because you
    can check ``M.is_hermitian`` independently with this and use
    the same procedure.

    Examples
    ========

    An example of symmetric positive definite matrix:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import Matrix, symbols
        >>> from sympy.plotting import plot3d
        >>> a, b = symbols('a b')
        >>> x = Matrix([a, b])

        >>> A = Matrix([[1, 0], [0, 1]])
        >>> A.is_positive_definite
        True
        >>> A.is_positive_semidefinite
        True

        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))

    An example of symmetric positive semidefinite matrix:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> A = Matrix([[1, -1], [-1, 1]])
        >>> A.is_positive_definite
        False
        >>> A.is_positive_semidefinite
        True

        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))

    An example of symmetric negative definite matrix:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> A = Matrix([[-1, 0], [0, -1]])
        >>> A.is_negative_definite
        True
        >>> A.is_negative_semidefinite
        True
        >>> A.is_indefinite
        False

        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))

    An example of symmetric indefinite matrix:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> A = Matrix([[1, 2], [2, -1]])
        >>> A.is_indefinite
        True

        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))

    An example of non-symmetric positive definite matrix.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> A = Matrix([[1, 2], [-2, 1]])
        >>> A.is_positive_definite
        True
        >>> A.is_positive_semidefinite
        True

        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))

    Notes
    =====

    Although some people trivialize the definition of positive definite
    matrices only for symmetric or hermitian matrices, this restriction
    is not correct because it does not classify all instances of
    positive definite matrices from the definition $x^T A x > 0$ or
    $\text{re}(x^H A x) > 0$.

    For instance, ``Matrix([[1, 2], [-2, 1]])`` presented in
    the example above is an example of real positive definite matrix
    that is not symmetric.

    However, since the following formula holds true;

    .. math::
        \text{re}(x^H A x) > 0 \iff
        \text{re}(x^H \frac{A + A^H}{2} x) > 0

    We can classify all positive definite matrices that may or may not
    be symmetric or hermitian by transforming the matrix to
    $\frac{A + A^T}{2}$ or $\frac{A + A^H}{2}$
    (which is guaranteed to be always real symmetric or complex
    hermitian) and we can defer most of the studies to symmetric or
    hermitian positive definite matrices.

    But it is a different problem for the existence of Cholesky
    decomposition. Because even though a non symmetric or a non
    hermitian matrix can be positive definite, Cholesky or LDL
    decomposition does not exist because the decompositions require the
    matrix to be symmetric or hermitian.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Eigenvalues

    .. [2] https://mathworld.wolfram.com/PositiveDefiniteMatrix.html

    .. [3] Johnson, C. R. "Positive Definite Matrices." Amer.
        Math. Monthly 77, 259-264 1970.
    """

_is_positive_definite.__doc__     = _doc_positive_definite
_is_positive_semidefinite.__doc__ = _doc_positive_definite
_is_negative_definite.__doc__     = _doc_positive_definite
_is_negative_semidefinite.__doc__ = _doc_positive_definite
_is_indefinite.__doc__            = _doc_positive_definite


def _jordan_form(M, calc_transform=True, *, chop=False):
    """Return $(P, J)$ where $J$ is a Jordan block
    matrix and $P$ is a matrix such that $M = P J P^{-1}$

    Parameters
    ==========

    calc_transform : bool
        If ``False``, then only $J$ is returned.

    chop : bool
        All matrices are converted to exact types when computing
        eigenvalues and eigenvectors.  As a result, there may be
        approximation errors.  If ``chop==True``, these errors
        will be truncated.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[ 6,  5, -2, -3], [-3, -1,  3,  3], [ 2,  1, -2, -3], [-1,  1,  5,  5]])
    >>> P, J = M.jordan_form()
    >>> J
    Matrix([
    [2, 1, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 1],
    [0, 0, 0, 2]])

    See Also
    ========

    jordan_block
    """

    if not M.is_square:
        raise NonSquareMatrixError("Only square matrices have Jordan forms")

    mat        = M
    has_floats = M.has(Float)

    if has_floats:
        try:
            max_prec = max(term._prec for term in M.values() if isinstance(term, Float))
        except ValueError:
            # if no term in the matrix is explicitly a Float calling max()
            # will throw a error so setting max_prec to default value of 53
            max_prec = 53

        # setting minimum max_dps to 15 to prevent loss of precision in
        # matrix containing non evaluated expressions
        max_dps = max(prec_to_dps(max_prec), 15)

    def restore_floats(*args):
        """If ``has_floats`` is `True`, cast all ``args`` as
        matrices of floats."""

        if has_floats:
            args = [m.evalf(n=max_dps, chop=chop) for m in args]
        if len(args) == 1:
            return args[0]

        return args

    # cache calculations for some speedup
    mat_cache = {}

    def eig_mat(val, pow):
        """Cache computations of ``(M - val*I)**pow`` for quick
        retrieval"""

        if (val, pow) in mat_cache:
            return mat_cache[(val, pow)]

        if (val, pow - 1) in mat_cache:
            mat_cache[(val, pow)] = mat_cache[(val, pow - 1)].multiply(
                    mat_cache[(val, 1)], dotprodsimp=None)
        else:
            mat_cache[(val, pow)] = (mat - val*M.eye(M.rows)).pow(pow)

        return mat_cache[(val, pow)]

    # helper functions
    def nullity_chain(val, algebraic_multiplicity):
        """Calculate the sequence  [0, nullity(E), nullity(E**2), ...]
        until it is constant where ``E = M - val*I``"""

        # mat.rank() is faster than computing the null space,
        # so use the rank-nullity theorem
        cols    = M.cols
        ret     = [0]
        nullity = cols - eig_mat(val, 1).rank()
        i       = 2

        while nullity != ret[-1]:
            ret.append(nullity)

            if nullity == algebraic_multiplicity:
                break

            nullity  = cols - eig_mat(val, i).rank()
            i       += 1

            # Due to issues like #7146 and #15872, SymPy sometimes
            # gives the wrong rank. In this case, raise an error
            # instead of returning an incorrect matrix
            if nullity < ret[-1] or nullity > algebraic_multiplicity:
                raise MatrixError(
                    "SymPy had encountered an inconsistent "
                    "result while computing Jordan block: "
                    "{}".format(M))

        return ret

    def blocks_from_nullity_chain(d):
        """Return a list of the size of each Jordan block.
        If d_n is the nullity of E**n, then the number
        of Jordan blocks of size n is

            2*d_n - d_(n-1) - d_(n+1)"""

        # d[0] is always the number of columns, so skip past it
        mid = [2*d[n] - d[n - 1] - d[n + 1] for n in range(1, len(d) - 1)]
        # d is assumed to plateau with "d[ len(d) ] == d[-1]", so
        # 2*d_n - d_(n-1) - d_(n+1) == d_n - d_(n-1)
        end = [d[-1] - d[-2]] if len(d) > 1 else [d[0]]

        return mid + end

    def pick_vec(small_basis, big_basis):
        """Picks a vector from big_basis that isn't in
        the subspace spanned by small_basis"""

        if len(small_basis) == 0:
            return big_basis[0]

        for v in big_basis:
            _, pivots = M.hstack(*(small_basis + [v])).echelon_form(
                    with_pivots=True)

            if pivots[-1] == len(small_basis):
                return v

    # roots doesn't like Floats, so replace them with Rationals
    if has_floats:
        from sympy.simplify import nsimplify
        mat = mat.applyfunc(lambda x: nsimplify(x, rational=True))

    # first calculate the jordan block structure
    eigs = mat.eigenvals()

    # Make sure that we have all roots in radical form
    for x in eigs:
        if x.has(CRootOf):
            raise MatrixError(
                "Jordan normal form is not implemented if the matrix have "
                "eigenvalues in CRootOf form")

    # most matrices have distinct eigenvalues
    # and so are diagonalizable.  In this case, don't
    # do extra work!
    if len(eigs.keys()) == mat.cols:
        blocks     = sorted(eigs.keys(), key=default_sort_key)
        jordan_mat = mat.diag(*blocks)

        if not calc_transform:
            return restore_floats(jordan_mat)

        jordan_basis = [eig_mat(eig, 1).nullspace()[0]
                for eig in blocks]
        basis_mat    = mat.hstack(*jordan_basis)

        return restore_floats(basis_mat, jordan_mat)

    block_structure = []

    for eig in sorted(eigs.keys(), key=default_sort_key):
        algebraic_multiplicity = eigs[eig]
        chain = nullity_chain(eig, algebraic_multiplicity)
        block_sizes = blocks_from_nullity_chain(chain)

        # if block_sizes =       = [a, b, c, ...], then the number of
        # Jordan blocks of size 1 is a, of size 2 is b, etc.
        # create an array that has (eig, block_size) with one
        # entry for each block
        size_nums = [(i+1, num) for i, num in enumerate(block_sizes)]

        # we expect larger Jordan blocks to come earlier
        size_nums.reverse()

        block_structure.extend(
            [(eig, size) for size, num in size_nums for _ in range(num)])

    jordan_form_size = sum(size for eig, size in block_structure)

    if jordan_form_size != M.rows:
        raise MatrixError(
            "SymPy had encountered an inconsistent result while "
            "computing Jordan block. : {}".format(M))

    blocks     = (mat.jordan_block(size=size, eigenvalue=eig) for eig, size in block_structure)
    jordan_mat = mat.diag(*blocks)

    if not calc_transform:
        return restore_floats(jordan_mat)

    # For each generalized eigenspace, calculate a basis.
    # We start by looking for a vector in null( (A - eig*I)**n )
    # which isn't in null( (A - eig*I)**(n-1) ) where n is
    # the size of the Jordan block
    #
    # Ideally we'd just loop through block_structure and
    # compute each generalized eigenspace.  However, this
    # causes a lot of unneeded computation.  Instead, we
    # go through the eigenvalues separately, since we know
    # their generalized eigenspaces must have bases that
    # are linearly independent.
    jordan_basis = []

    for eig in sorted(eigs.keys(), key=default_sort_key):
        eig_basis = []

        for block_eig, size in block_structure:
            if block_eig != eig:
                continue

            null_big   = (eig_mat(eig, size)).nullspace()
            null_small = (eig_mat(eig, size - 1)).nullspace()

            # we want to pick something that is in the big basis
            # and not the small, but also something that is independent
            # of any other generalized eigenvectors from a different
            # generalized eigenspace sharing the same eigenvalue.
            vec      = pick_vec(null_small + eig_basis, null_big)
            new_vecs = [eig_mat(eig, i).multiply(vec, dotprodsimp=None)
                    for i in range(size)]

            eig_basis.extend(new_vecs)
            jordan_basis.extend(reversed(new_vecs))

    basis_mat = mat.hstack(*jordan_basis)

    return restore_floats(basis_mat, jordan_mat)


def _left_eigenvects(M, **flags):
    """Returns left eigenvectors and eigenvalues.

    This function returns the list of triples (eigenval, multiplicity,
    basis) for the left eigenvectors. Options are the same as for
    eigenvects(), i.e. the ``**flags`` arguments gets passed directly to
    eigenvects().

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
    >>> M.eigenvects()
    [(-1, 1, [Matrix([
    [-1],
    [ 1],
    [ 0]])]), (0, 1, [Matrix([
    [ 0],
    [-1],
    [ 1]])]), (2, 1, [Matrix([
    [2/3],
    [1/3],
    [  1]])])]
    >>> M.left_eigenvects()
    [(-1, 1, [Matrix([[-2, 1, 1]])]), (0, 1, [Matrix([[-1, -1, 1]])]), (2,
    1, [Matrix([[1, 1, 1]])])]

    """

    eigs = M.transpose().eigenvects(**flags)

    return [(val, mult, [l.transpose() for l in basis]) for val, mult, basis in eigs]


def _singular_values(M):
    """Compute the singular values of a Matrix

    Examples
    ========

    >>> from sympy import Matrix, Symbol
    >>> x = Symbol('x', real=True)
    >>> M = Matrix([[0, 1, 0], [0, x, 0], [-1, 0, 0]])
    >>> M.singular_values()
    [sqrt(x**2 + 1), 1, 0]

    See Also
    ========

    condition_number
    """

    if M.rows >= M.cols:
        valmultpairs = M.H.multiply(M).eigenvals()
    else:
        valmultpairs = M.multiply(M.H).eigenvals()

    # Expands result from eigenvals into a simple list
    vals = []

    for k, v in valmultpairs.items():
        vals += [sqrt(k)] * v  # dangerous! same k in several spots!

    # Pad with zeros if singular values are computed in reverse way,
    # to give consistent format.
    if len(vals) < M.cols:
        vals += [M.zero] * (M.cols - len(vals))

    # sort them in descending order
    vals.sort(reverse=True, key=default_sort_key)

    return vals
