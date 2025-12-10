"""

Module for the DDM class.

The DDM class is an internal representation used by DomainMatrix. The letters
DDM stand for Dense Domain Matrix. A DDM instance represents a matrix using
elements from a polynomial Domain (e.g. ZZ, QQ, ...) in a dense-matrix
representation.

Basic usage:

    >>> from sympy import ZZ, QQ
    >>> from sympy.polys.matrices.ddm import DDM
    >>> A = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> A.shape
    (2, 2)
    >>> A
    [[0, 1], [-1, 0]]
    >>> type(A)
    <class 'sympy.polys.matrices.ddm.DDM'>
    >>> A @ A
    [[-1, 0], [0, -1]]

The ddm_* functions are designed to operate on DDM as well as on an ordinary
list of lists:

    >>> from sympy.polys.matrices.dense import ddm_idet
    >>> ddm_idet(A, QQ)
    1
    >>> ddm_idet([[0, 1], [-1, 0]], QQ)
    1
    >>> A
    [[-1, 0], [0, -1]]

Note that ddm_idet modifies the input matrix in-place. It is recommended to
use the DDM.det method as a friendlier interface to this instead which takes
care of copying the matrix:

    >>> B = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> B.det()
    1

Normally DDM would not be used directly and is just part of the internal
representation of DomainMatrix which adds further functionality including e.g.
unifying domains.

The dense format used by DDM is a list of lists of elements e.g. the 2x2
identity matrix is like [[1, 0], [0, 1]]. The DDM class itself is a subclass
of list and its list items are plain lists. Elements are accessed as e.g.
ddm[i][j] where ddm[i] gives the ith row and ddm[i][j] gets the element in the
jth column of that row. Subclassing list makes e.g. iteration and indexing
very efficient. We do not override __getitem__ because it would lose that
benefit.

The core routines are implemented by the ddm_* functions defined in dense.py.
Those functions are intended to be able to operate on a raw list-of-lists
representation of matrices with most functions operating in-place. The DDM
class takes care of copying etc and also stores a Domain object associated
with its elements. This makes it possible to implement things like A + B with
domain checking and also shape checking so that the list of lists
representation is friendlier.

"""
from itertools import chain

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from .exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMShapeError,
)

from sympy.polys.domains import QQ

from .dense import (
        ddm_transpose,
        ddm_iadd,
        ddm_isub,
        ddm_ineg,
        ddm_imul,
        ddm_irmul,
        ddm_imatmul,
        ddm_irref,
        ddm_irref_den,
        ddm_idet,
        ddm_iinv,
        ddm_ilu_split,
        ddm_ilu_solve,
        ddm_berk,
        )

from .lll import ddm_lll, ddm_lll_transform


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DDM.to_dfm', 'DDM.to_dfm_or_ddm']


class DDM(list):
    """Dense matrix based on polys domain elements

    This is a list subclass and is a wrapper for a list of lists that supports
    basic matrix arithmetic +, -, *, **.
    """

    fmt = 'dense'
    is_DFM = False
    is_DDM = True

    def __init__(self, rowslist, shape, domain):
        if not (isinstance(rowslist, list) and all(type(row) is list for row in rowslist)):
            raise DMBadInputError("rowslist must be a list of lists")
        m, n = shape
        if len(rowslist) != m or any(len(row) != n for row in rowslist):
            raise DMBadInputError("Inconsistent row-list/shape")

        super().__init__([i.copy() for i in rowslist])
        self.shape = (m, n)
        self.rows = m
        self.cols = n
        self.domain = domain

    def getitem(self, i, j):
        return self[i][j]

    def setitem(self, i, j, value):
        self[i][j] = value

    def extract_slice(self, slice1, slice2):
        ddm = [row[slice2] for row in self[slice1]]
        rows = len(ddm)
        cols = len(ddm[0]) if ddm else len(range(self.shape[1])[slice2])
        return DDM(ddm, (rows, cols), self.domain)

    def extract(self, rows, cols):
        ddm = []
        for i in rows:
            rowi = self[i]
            ddm.append([rowi[j] for j in cols])
        return DDM(ddm, (len(rows), len(cols)), self.domain)

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """
        Create a :class:`DDM` from a list of lists.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        >>> A
        [[0, 1], [-1, 0]]
        >>> A == DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        True

        See Also
        ========

        from_list_flat
        """
        return cls(rowslist, shape, domain)

    @classmethod
    def from_ddm(cls, other):
        return other.copy()

    def to_list(self):
        """
        Convert to a list of lists.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list()
        [[1, 2], [3, 4]]

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list
        """
        return [row[:] for row in self]

    def to_list_flat(self):
        """
        Convert to a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list_flat
        """
        flat = []
        for row in self:
            flat.extend(row)
        return flat

    @classmethod
    def from_list_flat(cls, flat, shape, domain):
        """
        Create a :class:`DDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list_flat([1, 2, 3, 4], (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_list_flat
        """
        assert type(flat) is list
        rows, cols = shape
        if not (len(flat) == rows*cols):
            raise DMBadInputError("Inconsistent flat-list shape")
        lol = [flat[i*cols:(i+1)*cols] for i in range(rows)]
        return cls(lol, shape, domain)

    def flatiter(self):
        return chain.from_iterable(self)

    def flat(self):
        items = []
        for row in self:
            items.extend(row)
        return items

    def to_flat_nz(self):
        """
        Convert to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a matrix using :meth:`from_flat_nz`. Zero elements are
        included in the list but that may change in the future.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        sympy.polys.matrices.sdm.SDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        return self.to_sdm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`DDM` after calling :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        to_flat_nz
        sympy.polys.matrices.sdm.SDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        return SDM.from_flat_nz(elements, data, domain).to_ddm()

    def to_dod(self):
        """
        Convert to a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.sdm.SDM.to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        dod = {}
        for i, row in enumerate(self):
            row = {j:e for j, e in enumerate(row) if e}
            if row:
                dod[i] = row
        return dod

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dod = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> A = DDM.from_dod(dod, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dod
        sympy.polys.matrices.sdm.SDM.from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dod
        """
        rows, cols = shape
        lol = [[domain.zero] * cols for _ in range(rows)]
        for i, row in dod.items():
            for j, element in row.items():
                lol[i][j] = element
        return DDM(lol, shape, domain)

    def to_dok(self):
        """
        Convert :class:`DDM` to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dok()
        {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}

        See Also
        ========

        from_dok
        sympy.polys.matrices.sdm.SDM.to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dok
        """
        dok = {}
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    dok[i, j] = element
        return dok

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dok = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        >>> A = DDM.from_dok(dok, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dok
        sympy.polys.matrices.sdm.SDM.from_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dok
        """
        rows, cols = shape
        lol = [[domain.zero] * cols for _ in range(rows)]
        for (i, j), element in dok.items():
            lol[i][j] = element
        return DDM(lol, shape, domain)

    def iter_values(self):
        """
        Iterate over the non-zero values of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_values
        """
        for row in self:
            yield from filter(None, row)

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    yield (i, j), element

    def to_ddm(self):
        """
        Convert to a :class:`DDM`.

        This just returns ``self`` but exists to parallel the corresponding
        method in other matrix types like :class:`~.SDM`.

        See Also
        ========

        to_sdm
        to_dfm
        to_dfm_or_ddm
        sympy.polys.matrices.sdm.SDM.to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_ddm
        """
        return self

    def to_sdm(self):
        """
        Convert to a :class:`~.SDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_sdm()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> type(A.to_sdm())
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        SDM
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        return SDM.from_list(self, self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Convert to :class:`~.DDM` to :class:`~.DFM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        DFM
        sympy.polys.matrices._dfm.DFM.to_ddm
        """
        return DFM(list(self), self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Convert to :class:`~.DFM` if possible or otherwise return self.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm_or_ddm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_dfm
        to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        if DFM._supports_domain(self.domain):
            return self.to_dfm()
        return self

    def convert_to(self, K):
        Kold = self.domain
        if K == Kold:
            return self.copy()
        rows = [[K.convert_from(e, Kold) for e in row] for row in self]
        return DDM(rows, self.shape, K)

    def __str__(self):
        rowsstr = ['[%s]' % ', '.join(map(str, row)) for row in self]
        return '[%s]' % ', '.join(rowsstr)

    def __repr__(self):
        cls = type(self).__name__
        rows = list.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    def __eq__(self, other):
        if not isinstance(other, DDM):
            return False
        return (super().__eq__(other) and self.domain == other.domain)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def zeros(cls, shape, domain):
        z = domain.zero
        m, n = shape
        rowslist = [[z] * n for _ in range(m)]
        return DDM(rowslist, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        one = domain.one
        m, n = shape
        rowlist = [[one] * n for _ in range(m)]
        return DDM(rowlist, shape, domain)

    @classmethod
    def eye(cls, size, domain):
        if isinstance(size, tuple):
            m, n = size
        elif isinstance(size, int):
            m = n = size
        one = domain.one
        ddm = cls.zeros((m, n), domain)
        for i in range(min(m, n)):
            ddm[i][i] = one
        return ddm

    def copy(self):
        copyrows = [row[:] for row in self]
        return DDM(copyrows, self.shape, self.domain)

    def transpose(self):
        rows, cols = self.shape
        if rows:
            ddmT = ddm_transpose(self)
        else:
            ddmT = [[]] * cols
        return DDM(ddmT, (cols, rows), self.domain)

    def __add__(a, b):
        if not isinstance(b, DDM):
            return NotImplemented
        return a.add(b)

    def __sub__(a, b):
        if not isinstance(b, DDM):
            return NotImplemented
        return a.sub(b)

    def __neg__(a):
        return a.neg()

    def __mul__(a, b):
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __matmul__(a, b):
        if isinstance(b, DDM):
            return a.matmul(b)
        else:
            return NotImplemented

    @classmethod
    def _check(cls, a, op, b, ashape, bshape):
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)

    def add(a, b):
        """a + b"""
        a._check(a, '+', b, a.shape, b.shape)
        c = a.copy()
        ddm_iadd(c, b)
        return c

    def sub(a, b):
        """a - b"""
        a._check(a, '-', b, a.shape, b.shape)
        c = a.copy()
        ddm_isub(c, b)
        return c

    def neg(a):
        """-a"""
        b = a.copy()
        ddm_ineg(b)
        return b

    def mul(a, b):
        c = a.copy()
        ddm_imul(c, b)
        return c

    def rmul(a, b):
        c = a.copy()
        ddm_irmul(c, b)
        return c

    def matmul(a, b):
        """a @ b (matrix product)"""
        m, o = a.shape
        o2, n = b.shape
        a._check(a, '*', b, o, o2)
        c = a.zeros((m, n), a.domain)
        ddm_imatmul(c, a, b)
        return c

    def mul_elementwise(a, b):
        assert a.shape == b.shape
        assert a.domain == b.domain
        c = [[aij * bij for aij, bij in zip(ai, bi)] for ai, bi in zip(a, b)]
        return DDM(c, a.shape, a.domain)

    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        [[1, 2, 5, 6], [3, 4, 7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]
        """
        Anew = list(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain

            cols += Bkcols

            for i, Bki in enumerate(Bk):
                Anew[i].extend(Bki)

        return DDM(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        """Vertically stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        """
        Anew = list(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain

            rows += Bkrows

            Anew.extend(Bk.copy())

        return DDM(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        elements = [list(map(func, row)) for row in self]
        return DDM(elements, self.shape, domain)

    def nnz(a):
        """Number of non-zero entries in :py:class:`~.DDM` matrix.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        return sum(sum(map(bool, row)) for row in a)

    def scc(a):
        """Strongly connected components of a square matrix *a*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> A = DDM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc

        """
        return a.to_sdm().scc()

    @classmethod
    def diag(cls, values, domain):
        """Returns a square diagonal matrix with *values* on the diagonal.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> DDM.diag([ZZ(1), ZZ(2), ZZ(3)], ZZ)
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.diag
        """
        return SDM.diag(values, domain).to_ddm()

    def rref(a):
        """Reduced-row echelon form of a and list of pivots.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref
            The underlying algorithm.
        """
        b = a.copy()
        K = a.domain
        partial_pivot = K.is_RealField or K.is_ComplexField
        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
        return b, pivots

    def rref_den(a):
        """Reduced-row echelon form of a with denominator and list of pivots

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref_den
            The underlying algorithm.
        """
        b = a.copy()
        K = a.domain
        denom, pivots = ddm_irref_den(b, K)
        return b, denom, pivots

    def nullspace(a):
        """Returns a basis for the nullspace of a.

        The domain of the matrix must be a field.

        See Also
        ========

        rref
        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
        """
        rref, pivots = a.rref()
        return rref.nullspace_from_rref(pivots)

    def nullspace_from_rref(a, pivots=None):
        """Compute the nullspace of a matrix from its rref.

        The domain of the matrix can be any domain.

        Returns a tuple (basis, nonpivots).

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher level interface to this function.
        """
        m, n = a.shape
        K = a.domain

        if pivots is None:
            pivots = []
            last_pivot = -1
            for i in range(m):
                ai = a[i]
                for j in range(last_pivot+1, n):
                    if ai[j]:
                        last_pivot = j
                        pivots.append(j)
                        break

        if not pivots:
            return (a.eye(n, K), list(range(n)))

        # After rref the pivots are all one but after rref_den they may not be.
        pivot_val = a[0][pivots[0]]

        basis = []
        nonpivots = []
        for i in range(n):
            if i in pivots:
                continue
            nonpivots.append(i)
            vec = [pivot_val if i == j else K.zero for j in range(n)]
            for ii, jj in enumerate(pivots):
                vec[jj] -= a[ii][i]
            basis.append(vec)

        basis_ddm = DDM(basis, (len(basis), n), K)

        return (basis_ddm, nonpivots)

    def particular(a):
        return a.to_sdm().particular().to_ddm()

    def det(a):
        """Determinant of a"""
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        b = a.copy()
        K = b.domain
        deta = ddm_idet(b, K)
        return deta

    def inv(a):
        """Inverse of a"""
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        ainv = a.copy()
        K = a.domain
        ddm_iinv(ainv, a, K)
        return ainv

    def lu(a):
        """L, U decomposition of a"""
        m, n = a.shape
        K = a.domain

        U = a.copy()
        L = a.eye(m, K)
        swaps = ddm_ilu_split(L, U, K)

        return L, U, swaps

    def _fflu(self):
        """
        Private method for Phase 1 of fraction-free LU decomposition.
        Performs row operations and elimination to compute U and permutation indices.

        Returns:
            LU : decomposition as a single matrix.
            perm (list): Permutation indices for row swaps.
        """
        rows, cols = self.shape
        K = self.domain

        LU = self.copy()
        perm = list(range(rows))
        rank = 0

        for j in range(min(rows, cols)):
            # Skip columns where all entries are zero
            if all(LU[i][j] == K.zero for i in range(rows)):
                continue

            # Find the first non-zero pivot in the current column
            pivot_row = -1
            for i in range(rank, rows):
                if LU[i][j] != K.zero:
                    pivot_row = i
                    break

            # If no pivot is found, skip column
            if pivot_row == -1:
                continue

            # Swap rows to bring the pivot to the current rank
            if pivot_row != rank:
                LU[rank], LU[pivot_row] = LU[pivot_row], LU[rank]
                perm[rank], perm[pivot_row] = perm[pivot_row], perm[rank]

            # Found pivot - (Gauss-Bareiss elimination)
            pivot = LU[rank][j]
            for i in range(rank + 1, rows):
                multiplier = LU[i][j]
                # Denominator is previous pivot or 1
                denominator = LU[rank - 1][rank - 1] if rank > 0 else K.one
                for k in range(j + 1, cols):
                    LU[i][k] = K.exquo(pivot * LU[i][k] - LU[rank][k] * multiplier, denominator)
                # Keep the multiplier for L matrix
                LU[i][j] = multiplier
            rank += 1

        return LU, perm

    def fflu(self):
        """
        Fraction-free LU decomposition of DDM.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.fflu
            The higher-level interface to this function.
        """
        rows, cols = self.shape
        K = self.domain

        # Phase 1: Perform row operations and get permutation
        U, perm = self._fflu()

        # Phase 2: Construct P, L, D matrices
        # Create P from permutation
        P = self.zeros((rows, rows), K)
        for i, pi in enumerate(perm):
            P[i][pi] = K.one

        # Create L matrix
        L = self.zeros((rows, rows), K)
        i = j = 0
        while i < rows and j < cols:
            if U[i][j] != K.zero:
                # Found non-zero pivot
                # Diagonal entry is the pivot
                L[i][i] = U[i][j]
                for l in range(i + 1, rows):
                    # Off-diagonal entries are the multipliers
                    L[l][i] = U[l][j]
                    # zero out the entries in U
                    U[l][j] = K.zero
                i += 1
            j += 1

        # Fill remaining diagonal of L with ones
        for i in range(i, rows):
            L[i][i] = K.one

        # Create D matrix - using FLINT's approach with accumulator
        D = self.zeros((rows, rows), K)
        if rows >= 1:
            D[0][0] = L[0][0]
        di = K.one
        for i in range(1, rows):
            # Accumulate product of pivots
            di = L[i - 1][i - 1] * L[i][i]
            D[i][i] = di

        return P, L, D, U

    def qr(self):
        """
        QR decomposition for DDM.

        Returns:
            - Q: Orthogonal matrix as a DDM.
            - R: Upper triangular matrix as a DDM.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.qr
            The higher-level interface to this function.
        """
        rows, cols = self.shape
        K = self.domain
        Q = self.copy()
        R = self.zeros((min(rows, cols), cols), K)

        # Check that the domain is a field
        if not K.is_Field:
            raise DMDomainError("QR decomposition requires a field (e.g. QQ).")

        dot_cols = lambda i, j: K.sum(Q[k][i] * Q[k][j] for k in range(rows))

        for j in range(cols):
            for i in range(min(j, rows)):
                dot_ii = dot_cols(i, i)
                if dot_ii != K.zero:
                    R[i][j] = dot_cols(i, j) / dot_ii
                    for k in range(rows):
                        Q[k][j] -= R[i][j] * Q[k][i]

            if j < rows:
                dot_jj = dot_cols(j, j)
                if dot_jj != K.zero:
                    R[j][j] = K.one

        Q = Q.extract(range(rows), range(min(rows, cols)))

        return Q, R

    def lu_solve(a, b):
        """x where a*x = b"""
        m, n = a.shape
        m2, o = b.shape
        a._check(a, 'lu_solve', b, m, m2)
        if not a.domain.is_Field:
            raise DMDomainError("lu_solve requires a field")

        L, U, swaps = a.lu()
        x = a.zeros((n, o), a.domain)
        ddm_ilu_solve(x, L, U, swaps, b)
        return x

    def charpoly(a):
        """Coefficients of characteristic polynomial of a"""
        K = a.domain
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Charpoly of non-square matrix")
        vec = ddm_berk(a, K)
        coeffs = [vec[i][0] for i in range(n+1)]
        return coeffs

    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
        zero = self.domain.zero
        return all(Mij == zero for Mij in self.flatiter())

    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        zero = self.domain.zero
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[:i])

    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        zero = self.domain.zero
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[i+1:])

    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned even if
        the matrix is not square.
        """
        return self.is_upper() and self.is_lower()

    def diagonal(self):
        """
        Returns a list of the elements from the diagonal of the matrix.
        """
        m, n = self.shape
        return [self[i][i] for i in range(min(m, n))]

    def lll(A, delta=QQ(3, 4)):
        return ddm_lll(A, delta=delta)

    def lll_transform(A, delta=QQ(3, 4)):
        return ddm_lll_transform(A, delta=delta)


from .sdm import SDM
from .dfm import DFM
