from collections import defaultdict

from operator import index as index_

from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, GF, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent, as_int

from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .matrixbase import classof, MatrixBase
from .kind import MatrixKind


class RepMatrix(MatrixBase):
    """Matrix implementation based on DomainMatrix as an internal representation.

    The RepMatrix class is a superclass for Matrix, ImmutableMatrix,
    SparseMatrix and ImmutableSparseMatrix which are the main usable matrix
    classes in SymPy. Most methods on this class are simply forwarded to
    DomainMatrix.
    """

    #
    # MatrixBase is the common superclass for all of the usable explicit matrix
    # classes in SymPy. The idea is that MatrixBase is an abstract class though
    # and that subclasses will implement the lower-level methods.
    #
    # RepMatrix is a subclass of MatrixBase that uses DomainMatrix as an
    # internal representation and delegates lower-level methods to
    # DomainMatrix. All of SymPy's standard explicit matrix classes subclass
    # RepMatrix and so use DomainMatrix internally.
    #
    # A RepMatrix uses an internal DomainMatrix with the domain set to ZZ, QQ
    # or EXRAW. The EXRAW domain is equivalent to the previous implementation
    # of Matrix that used Expr for the elements. The ZZ and QQ domains are used
    # when applicable just because they are compatible with the previous
    # implementation but are much more efficient. Other domains such as QQ[x]
    # are not used because they differ from Expr in some way (e.g. automatic
    # expansion of powers and products).
    #

    _rep: DomainMatrix

    def __eq__(self, other):
        # Skip sympify for mutable matrices...
        if not isinstance(other, RepMatrix):
            try:
                other = _sympify(other)
            except SympifyError:
                return NotImplemented
            if not isinstance(other, RepMatrix):
                return NotImplemented

        return self._rep.unify_eq(other._rep)

    def to_DM(self, domain=None, **kwargs):
        """Convert to a :class:`~.DomainMatrix`.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> M.to_DM()
        DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

        The :meth:`DomainMatrix.to_Matrix` method can be used to convert back:

        >>> M.to_DM().to_Matrix() == M
        True

        The domain can be given explicitly or otherwise it will be chosen by
        :func:`construct_domain`. Any keyword arguments (besides ``domain``)
        are passed to :func:`construct_domain`:

        >>> from sympy import QQ, symbols
        >>> x = symbols('x')
        >>> M = Matrix([[x, 1], [1, x]])
        >>> M
        Matrix([
        [x, 1],
        [1, x]])
        >>> M.to_DM().domain
        ZZ[x]
        >>> M.to_DM(field=True).domain
        ZZ(x)
        >>> M.to_DM(domain=QQ[x]).domain
        QQ[x]

        See Also
        ========

        DomainMatrix
        DomainMatrix.to_Matrix
        DomainMatrix.convert_to
        DomainMatrix.choose_domain
        construct_domain
        """
        if domain is not None:
            if kwargs:
                raise TypeError("Options cannot be used with domain parameter")
            return self._rep.convert_to(domain)

        rep = self._rep
        dom = rep.domain

        # If the internal DomainMatrix is already ZZ or QQ then we can maybe
        # bypass calling construct_domain or performing any conversions. Some
        # kwargs might affect this though e.g. field=True (not sure if there
        # are others).
        if not kwargs:
            if dom.is_ZZ:
                return rep.copy()
            elif dom.is_QQ:
                # All elements might be integers
                try:
                    return rep.convert_to(ZZ)
                except CoercionFailed:
                    pass
                return rep.copy()

        # Let construct_domain choose a domain
        rep_dom = rep.choose_domain(**kwargs)

        # XXX: There should be an option to construct_domain to choose EXRAW
        # instead of EX. At least converting to EX does not initially trigger
        # EX.simplify which is what we want here but should probably be
        # considered a bug in EX. Perhaps also this could be handled in
        # DomainMatrix.choose_domain rather than here...
        if rep_dom.domain.is_EX:
            rep_dom = rep_dom.convert_to(EXRAW)

        return rep_dom

    @classmethod
    def _unify_element_sympy(cls, rep, element):
        domain = rep.domain
        element = _sympify(element)

        if domain != EXRAW:
            # The domain can only be ZZ, QQ or EXRAW
            if element.is_Integer:
                new_domain = domain
            elif element.is_Rational:
                new_domain = QQ
            else:
                new_domain = EXRAW

            # XXX: This converts the domain for all elements in the matrix
            # which can be slow. This happens e.g. if __setitem__ changes one
            # element to something that does not fit in the domain
            if new_domain != domain:
                rep = rep.convert_to(new_domain)
                domain = new_domain

            if domain != EXRAW:
                element = new_domain.from_sympy(element)

        if domain == EXRAW and not isinstance(element, Expr):
            sympy_deprecation_warning(
                """
                non-Expr objects in a Matrix is deprecated. Matrix represents
                a mathematical matrix. To represent a container of non-numeric
                entities, Use a list of lists, TableForm, NumPy array, or some
                other data structure instead.
                """,
                deprecated_since_version="1.9",
                active_deprecations_target="deprecated-non-expr-in-matrix",
                stacklevel=4,
            )

        return rep, element

    @classmethod
    def _dod_to_DomainMatrix(cls, rows, cols, dod, types):

        if not all(issubclass(typ, Expr) for typ in types):
            sympy_deprecation_warning(
                """
                non-Expr objects in a Matrix is deprecated. Matrix represents
                a mathematical matrix. To represent a container of non-numeric
                entities, Use a list of lists, TableForm, NumPy array, or some
                other data structure instead.
                """,
                deprecated_since_version="1.9",
                active_deprecations_target="deprecated-non-expr-in-matrix",
                stacklevel=6,
            )

        rep = DomainMatrix(dod, (rows, cols), EXRAW)

        if all(issubclass(typ, Rational) for typ in types):
            if all(issubclass(typ, Integer) for typ in types):
                rep = rep.convert_to(ZZ)
            else:
                rep = rep.convert_to(QQ)

        return rep

    @classmethod
    def _flat_list_to_DomainMatrix(cls, rows, cols, flat_list):

        elements_dod = defaultdict(dict)
        for n, element in enumerate(flat_list):
            if element != 0:
                i, j = divmod(n, cols)
                elements_dod[i][j] = element

        types = set(map(type, flat_list))

        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    @classmethod
    def _smat_to_DomainMatrix(cls, rows, cols, smat):

        elements_dod = defaultdict(dict)
        for (i, j), element in smat.items():
            if element != 0:
                elements_dod[i][j] = element

        types = set(map(type, smat.values()))

        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    def flat(self):
        return self._rep.to_sympy().to_list_flat()

    def _eval_tolist(self):
        return self._rep.to_sympy().to_list()

    def _eval_todok(self):
        return self._rep.to_sympy().to_dok()

    @classmethod
    def _eval_from_dok(cls, rows, cols, dok):
        return cls._fromrep(cls._smat_to_DomainMatrix(rows, cols, dok))

    def _eval_values(self):
        return list(self._eval_iter_values())

    def _eval_iter_values(self):
        rep = self._rep
        K = rep.domain
        values = rep.iter_values()
        if not K.is_EXRAW:
            values = map(K.to_sympy, values)
        return values

    def _eval_iter_items(self):
        rep = self._rep
        K = rep.domain
        to_sympy = K.to_sympy
        items = rep.iter_items()
        if not K.is_EXRAW:
            items = ((i, to_sympy(v)) for i, v in items)
        return items

    def copy(self):
        return self._fromrep(self._rep.copy())

    @property
    def kind(self) -> MatrixKind:
        domain = self._rep.domain
        element_kind: Kind
        if domain in (ZZ, QQ):
            element_kind = NumberKind
        elif domain == EXRAW:
            kinds = {e.kind for e in self.values()}
            if len(kinds) == 1:
                [element_kind] = kinds
            else:
                element_kind = UndefinedKind
        else: # pragma: no cover
            raise RuntimeError("Domain should only be ZZ, QQ or EXRAW")
        return MatrixKind(element_kind)

    def _eval_has(self, *patterns):
        # if the matrix has any zeros, see if S.Zero
        # has the pattern.  If _smat is full length,
        # the matrix has no zeros.
        zhas = False
        dok = self.todok()
        if len(dok) != self.rows*self.cols:
            zhas = S.Zero.has(*patterns)
        return zhas or any(value.has(*patterns) for value in dok.values())

    def _eval_is_Identity(self):
        if not all(self[i, i] == 1 for i in range(self.rows)):
            return False
        return len(self.todok()) == self.rows

    def _eval_is_symmetric(self, simpfunc):
        diff = (self - self.T).applyfunc(simpfunc)
        return len(diff.values()) == 0

    def _eval_transpose(self):
        """Returns the transposed SparseMatrix of this SparseMatrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.T
        Matrix([
        [1, 3],
        [2, 4]])
        """
        return self._fromrep(self._rep.transpose())

    def _eval_col_join(self, other):
        return self._fromrep(self._rep.vstack(other._rep))

    def _eval_row_join(self, other):
        return self._fromrep(self._rep.hstack(other._rep))

    def _eval_extract(self, rowsList, colsList):
        return self._fromrep(self._rep.extract(rowsList, colsList))

    def __getitem__(self, key):
        return _getitem_RepMatrix(self, key)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        rep = DomainMatrix.zeros((rows, cols), ZZ)
        return cls._fromrep(rep)

    @classmethod
    def _eval_eye(cls, rows, cols):
        rep = DomainMatrix.eye((rows, cols), ZZ)
        return cls._fromrep(rep)

    def _eval_add(self, other):
        return classof(self, other)._fromrep(self._rep + other._rep)

    def _eval_matrix_mul(self, other):
        return classof(self, other)._fromrep(self._rep * other._rep)

    def _eval_matrix_mul_elementwise(self, other):
        selfrep, otherrep = self._rep.unify(other._rep)
        newrep = selfrep.mul_elementwise(otherrep)
        return classof(self, other)._fromrep(newrep)

    def _eval_scalar_mul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.scalarmul(other))

    def _eval_scalar_rmul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.rscalarmul(other))

    def _eval_Abs(self):
        return self._fromrep(self._rep.applyfunc(abs))

    def _eval_conjugate(self):
        rep = self._rep
        domain = rep.domain
        if domain in (ZZ, QQ):
            return self.copy()
        else:
            return self._fromrep(rep.applyfunc(lambda e: e.conjugate()))

    def equals(self, other, failing_expression=False):
        """Applies ``equals`` to corresponding elements of the matrices,
        trying to prove that the elements are equivalent, returning True
        if they are, False if any pair is not, and None (or the first
        failing expression if failing_expression is True) if it cannot
        be decided if the expressions are equivalent or not. This is, in
        general, an expensive operation.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> A = Matrix([x*(x - 1), 0])
        >>> B = Matrix([x**2 - x, 0])
        >>> A == B
        False
        >>> A.simplify() == B.simplify()
        True
        >>> A.equals(B)
        True
        >>> A.equals(2)
        False

        See Also
        ========
        sympy.core.expr.Expr.equals
        """
        if self.shape != getattr(other, 'shape', None):
            return False

        rv = True
        for i in range(self.rows):
            for j in range(self.cols):
                ans = self[i, j].equals(other[i, j], failing_expression)
                if ans is False:
                    return False
                elif ans is not True and rv is True:
                    rv = ans
        return rv

    def inv_mod(M, m):
        r"""
        Returns the inverse of the integer matrix ``M`` modulo ``m``.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.inv_mod(5)
        Matrix([
        [3, 1],
        [4, 2]])
        >>> A.inv_mod(3)
        Matrix([
        [1, 1],
        [0, 1]])

        """

        if not M.is_square:
            raise NonSquareMatrixError()

        try:
            m = as_int(m)
        except ValueError:
            raise TypeError("inv_mod: modulus m must be an integer")

        K = GF(m, symmetric=False)

        try:
            dM = M.to_DM(K)
        except CoercionFailed:
            raise ValueError("inv_mod: matrix entries must be integers")

        try:
            dMi = dM.inv()
        except DMNonInvertibleMatrixError as exc:
            msg = f'Matrix is not invertible (mod {m})'
            raise NonInvertibleMatrixError(msg) from exc

        return dMi.to_Matrix()

    def lll(self, delta=0.75):
        """LLL-reduced basis for the rowspace of a matrix of integers.

        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.

        The implementation is provided by :class:`~DomainMatrix`. See
        :meth:`~DomainMatrix.lll` for more details.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0, 0, 0, -20160],
        ...             [0, 1, 0, 0, 33768],
        ...             [0, 0, 1, 0, 39578],
        ...             [0, 0, 0, 1, 47757]])
        >>> M.lll()
        Matrix([
        [ 10, -3,  -2,  8,  -4],
        [  3, -9,   8,  1, -11],
        [ -3, 13,  -9, -3,  -9],
        [-12, -7, -11,  9,  -1]])

        See Also
        ========

        lll_transform
        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
        """
        delta = QQ.from_sympy(_sympify(delta))
        dM = self._rep.convert_to(ZZ)
        basis = dM.lll(delta=delta)
        return self._fromrep(basis)

    def lll_transform(self, delta=0.75):
        """LLL-reduced basis and transformation matrix.

        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.

        The implementation is provided by :class:`~DomainMatrix`. See
        :meth:`~DomainMatrix.lll_transform` for more details.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0, 0, 0, -20160],
        ...             [0, 1, 0, 0, 33768],
        ...             [0, 0, 1, 0, 39578],
        ...             [0, 0, 0, 1, 47757]])
        >>> B, T = M.lll_transform()
        >>> B
        Matrix([
        [ 10, -3,  -2,  8,  -4],
        [  3, -9,   8,  1, -11],
        [ -3, 13,  -9, -3,  -9],
        [-12, -7, -11,  9,  -1]])
        >>> T
        Matrix([
        [ 10, -3,  -2,  8],
        [  3, -9,   8,  1],
        [ -3, 13,  -9, -3],
        [-12, -7, -11,  9]])

        The transformation matrix maps the original basis to the LLL-reduced
        basis:

        >>> T * M == B
        True

        See Also
        ========

        lll
        sympy.polys.matrices.domainmatrix.DomainMatrix.lll_transform
        """
        delta = QQ.from_sympy(_sympify(delta))
        dM = self._rep.convert_to(ZZ)
        basis, transform = dM.lll_transform(delta=delta)
        B = self._fromrep(basis)
        T = self._fromrep(transform)
        return B, T


class MutableRepMatrix(RepMatrix):
    """Mutable matrix based on DomainMatrix as the internal representation"""

    #
    # MutableRepMatrix is a subclass of RepMatrix that adds/overrides methods
    # to make the instances mutable. MutableRepMatrix is a superclass for both
    # MutableDenseMatrix and MutableSparseMatrix.
    #

    is_zero = False

    def __new__(cls, *args, **kwargs):
        return cls._new(*args, **kwargs)

    @classmethod
    def _new(cls, *args, copy=True, **kwargs):
        if copy is False:
            # The input was rows, cols, [list].
            # It should be used directly without creating a copy.
            if len(args) != 3:
                raise TypeError("'copy=False' requires a matrix be initialized as rows,cols,[list]")
            rows, cols, flat_list = args
        else:
            rows, cols, flat_list = cls._handle_creation_inputs(*args, **kwargs)
            flat_list = list(flat_list) # create a shallow copy

        rep = cls._flat_list_to_DomainMatrix(rows, cols, flat_list)

        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        obj = super().__new__(cls)
        obj.rows, obj.cols = rep.shape
        obj._rep = rep
        return obj

    def copy(self):
        return self._fromrep(self._rep.copy())

    def as_mutable(self):
        return self.copy()

    def __setitem__(self, key, value):
        """

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        rv = self._setitem(key, value)
        if rv is not None:
            i, j, value = rv
            self._rep, value = self._unify_element_sympy(self._rep, value)
            self._rep.rep.setitem(i, j, value)

    def _eval_col_del(self, col):
        self._rep = DomainMatrix.hstack(self._rep[:,:col], self._rep[:,col+1:])
        self.cols -= 1

    def _eval_row_del(self, row):
        self._rep = DomainMatrix.vstack(self._rep[:row,:], self._rep[row+1:, :])
        self.rows -= 1

    def _eval_col_insert(self, col, other):
        other = self._new(other)
        return self.hstack(self[:,:col], other, self[:,col:])

    def _eval_row_insert(self, row, other):
        other = self._new(other)
        return self.vstack(self[:row,:], other, self[row:,:])

    def col_op(self, j, f):
        """In-place operation on col j using two-arg functor whose args are
        interpreted as (self[i, j], i).

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.col_op(1, lambda v, i: v + 2*M[i, 0]); M
        Matrix([
        [1, 2, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        col
        row_op
        """
        for i in range(self.rows):
            self[i, j] = f(self[i, j], i)

    def col_swap(self, i, j):
        """Swap the two given columns of the matrix in-place.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 0], [1, 0]])
        >>> M
        Matrix([
        [1, 0],
        [1, 0]])
        >>> M.col_swap(0, 1)
        >>> M
        Matrix([
        [0, 1],
        [0, 1]])

        See Also
        ========

        col
        row_swap
        """
        for k in range(0, self.rows):
            self[k, i], self[k, j] = self[k, j], self[k, i]

    def row_op(self, i, f):
        """In-place operation on row ``i`` using two-arg functor whose args are
        interpreted as ``(self[i, j], j)``.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_op(1, lambda v, j: v + 2*M[0, j]); M
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        row
        zip_row_op
        col_op

        """
        for j in range(self.cols):
            self[i, j] = f(self[i, j], j)

    #The next three methods give direct support for the most common row operations inplace.
    def row_mult(self,i,factor):
        """Multiply the given row by the given factor in-place.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_mult(1,7); M
        Matrix([
        [1, 0, 0],
        [0, 7, 0],
        [0, 0, 1]])

        """
        for j in range(self.cols):
            self[i,j] *= factor

    def row_add(self,s,t,k):
        """Add k times row s (source) to row t (target) in place.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.row_add(0, 2,3); M
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [3, 0, 1]])
        """

        for j in range(self.cols):
            self[t,j] += k*self[s,j]

    def row_swap(self, i, j):
        """Swap the two given rows of the matrix in-place.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[0, 1], [1, 0]])
        >>> M
        Matrix([
        [0, 1],
        [1, 0]])
        >>> M.row_swap(0, 1)
        >>> M
        Matrix([
        [1, 0],
        [0, 1]])

        See Also
        ========

        row
        col_swap
        """
        for k in range(0, self.cols):
            self[i, k], self[j, k] = self[j, k], self[i, k]

    def zip_row_op(self, i, k, f):
        """In-place operation on row ``i`` using two-arg functor whose args are
        interpreted as ``(self[i, j], self[k, j])``.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.zip_row_op(1, 0, lambda v, u: v + 2*u); M
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        row
        row_op
        col_op

        """
        for j in range(self.cols):
            self[i, j] = f(self[i, j], self[k, j])

    def copyin_list(self, key, value):
        """Copy in elements from a list.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : iterable
            The iterable to copy values from.

        Examples
        ========

        >>> from sympy import eye
        >>> I = eye(3)
        >>> I[:2, 0] = [1, 2] # col
        >>> I
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])
        >>> I[1, :2] = [[3, 4]]
        >>> I
        Matrix([
        [1, 0, 0],
        [3, 4, 0],
        [0, 0, 1]])

        See Also
        ========

        copyin_matrix
        """
        if not is_sequence(value):
            raise TypeError("`value` must be an ordered iterable, not %s." % type(value))
        return self.copyin_matrix(key, type(self)(value))

    def copyin_matrix(self, key, value):
        """Copy in values from a matrix into the given bounds.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : Matrix
            The matrix to copy values from.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> M = Matrix([[0, 1], [2, 3], [4, 5]])
        >>> I = eye(3)
        >>> I[:3, :2] = M
        >>> I
        Matrix([
        [0, 1, 0],
        [2, 3, 0],
        [4, 5, 1]])
        >>> I[0, 1] = M
        >>> I
        Matrix([
        [0, 0, 1],
        [2, 2, 3],
        [4, 4, 5]])

        See Also
        ========

        copyin_list
        """
        rlo, rhi, clo, chi = self.key2bounds(key)
        shape = value.shape
        dr, dc = rhi - rlo, chi - clo
        if shape != (dr, dc):
            raise ShapeError(filldedent("The Matrix `value` doesn't have the "
                                        "same dimensions "
                                        "as the in sub-Matrix given by `key`."))

        for i in range(value.rows):
            for j in range(value.cols):
                self[i + rlo, j + clo] = value[i, j]

    def fill(self, value):
        """Fill self with the given value.

        Notes
        =====

        Unless many values are going to be deleted (i.e. set to zero)
        this will create a matrix that is slower than a dense matrix in
        operations.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> M = SparseMatrix.zeros(3); M
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
        >>> M.fill(1); M
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

        See Also
        ========

        zeros
        ones
        """
        value = _sympify(value)
        if not value:
            self._rep = DomainMatrix.zeros(self.shape, EXRAW)
        else:
            elements_dod = {i: dict.fromkeys(range(self.cols), value) for i in range(self.rows)}
            self._rep = DomainMatrix(elements_dod, self.shape, EXRAW)


def _getitem_RepMatrix(self, key):
    """Return portion of self defined by key. If the key involves a slice
    then a list will be returned (if key is a single slice) or a matrix
    (if key was a tuple involving a slice).

    Examples
    ========

    >>> from sympy import Matrix, I
    >>> m = Matrix([
    ... [1, 2 + I],
    ... [3, 4    ]])

    If the key is a tuple that does not involve a slice then that element
    is returned:

    >>> m[1, 0]
    3

    When a tuple key involves a slice, a matrix is returned. Here, the
    first column is selected (all rows, column 0):

    >>> m[:, 0]
    Matrix([
    [1],
    [3]])

    If the slice is not a tuple then it selects from the underlying
    list of elements that are arranged in row order and a list is
    returned if a slice is involved:

    >>> m[0]
    1
    >>> m[::2]
    [1, 3]
    """
    if isinstance(key, tuple):
        i, j = key
        try:
            return self._rep.getitem_sympy(index_(i), index_(j))
        except (TypeError, IndexError):
            if (isinstance(i, Expr) and not i.is_number) or (isinstance(j, Expr) and not j.is_number):
                if ((j < 0) is True) or ((j >= self.shape[1]) is True) or\
                   ((i < 0) is True) or ((i >= self.shape[0]) is True):
                    raise ValueError("index out of boundary")
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)

            if isinstance(i, slice):
                i = range(self.rows)[i]
            elif is_sequence(i):
                pass
            else:
                i = [i]
            if isinstance(j, slice):
                j = range(self.cols)[j]
            elif is_sequence(j):
                pass
            else:
                j = [j]
            return self.extract(i, j)

    else:
        # Index/slice like a flattened list
        rows, cols = self.shape

        # Raise the appropriate exception:
        if not rows * cols:
            return [][key]

        rep = self._rep.rep
        domain = rep.domain
        is_slice = isinstance(key, slice)

        if is_slice:
            values = [rep.getitem(*divmod(n, cols)) for n in range(rows * cols)[key]]
        else:
            values = [rep.getitem(*divmod(index_(key), cols))]

        if domain != EXRAW:
            to_sympy = domain.to_sympy
            values = [to_sympy(val) for val in values]

        if is_slice:
            return values
        else:
            return values[0]
