"""

Module for the SDM class.

"""

from operator import add, neg, pos, sub, mul
from collections import defaultdict

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import _strongly_connected_components

from .exceptions import DMBadInputError, DMDomainError, DMShapeError

from sympy.polys.domains import QQ

from .ddm import DDM


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['SDM.to_dfm', 'SDM.to_dfm_or_ddm']


class SDM(dict):
    r"""Sparse matrix based on polys domain elements

    This is a dict subclass and is a wrapper for a dict of dicts that supports
    basic matrix arithmetic +, -, *, **.


    In order to create a new :py:class:`~.SDM`, a dict
    of dicts mapping non-zero elements to their
    corresponding row and column in the matrix is needed.

    We also need to specify the shape and :py:class:`~.Domain`
    of our :py:class:`~.SDM` object.

    We declare a 2x2 :py:class:`~.SDM` matrix belonging
    to QQ domain as shown below.
    The 2x2 Matrix in the example is

    .. math::
           A = \left[\begin{array}{ccc}
                0 & \frac{1}{2} \\
                0 & 0 \end{array} \right]


    >>> from sympy.polys.matrices.sdm import SDM
    >>> from sympy import QQ
    >>> elemsdict = {0:{1:QQ(1, 2)}}
    >>> A = SDM(elemsdict, (2, 2), QQ)
    >>> A
    {0: {1: 1/2}}

    We can manipulate :py:class:`~.SDM` the same way
    as a Matrix class

    >>> from sympy import ZZ
    >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
    >>> A + B
    {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

    Multiplication

    >>> A*B
    {0: {1: 8}, 1: {0: 3}}
    >>> A*ZZ(2)
    {0: {1: 4}, 1: {0: 2}}

    """

    fmt = 'sparse'
    is_DFM = False
    is_DDM = False

    def __init__(self, elemsdict, shape, domain):
        super().__init__(elemsdict)
        self.shape = self.rows, self.cols = m, n = shape
        self.domain = domain

        if not all(0 <= r < m for r in self):
            raise DMBadInputError("Row out of range")
        if not all(0 <= c < n for row in self.values() for c in row):
            raise DMBadInputError("Column out of range")

    def getitem(self, i, j):
        try:
            return self[i][j]
        except KeyError:
            m, n = self.shape
            if -m <= i < m and -n <= j < n:
                try:
                    return self[i % m][j % n]
                except KeyError:
                    return self.domain.zero
            else:
                raise IndexError("index out of range")

    def setitem(self, i, j, value):
        m, n = self.shape
        if not (-m <= i < m and -n <= j < n):
            raise IndexError("index out of range")
        i, j = i % m, j % n
        if value:
            try:
                self[i][j] = value
            except KeyError:
                self[i] = {j: value}
        else:
            rowi = self.get(i, None)
            if rowi is not None:
                try:
                    del rowi[j]
                except KeyError:
                    pass
                else:
                    if not rowi:
                        del self[i]

    def extract_slice(self, slice1, slice2):
        m, n = self.shape
        ri = range(m)[slice1]
        ci = range(n)[slice2]

        sdm = {}
        for i, row in self.items():
            if i in ri:
                row = {ci.index(j): e for j, e in row.items() if j in ci}
                if row:
                    sdm[ri.index(i)] = row

        return self.new(sdm, (len(ri), len(ci)), self.domain)

    def extract(self, rows, cols):
        if not (self and rows and cols):
            return self.zeros((len(rows), len(cols)), self.domain)

        m, n = self.shape
        if not (-m <= min(rows) <= max(rows) < m):
            raise IndexError('Row index out of range')
        if not (-n <= min(cols) <= max(cols) < n):
            raise IndexError('Column index out of range')

        # rows and cols can contain duplicates e.g. M[[1, 2, 2], [0, 1]]
        # Build a map from row/col in self to list of rows/cols in output
        rowmap = defaultdict(list)
        colmap = defaultdict(list)
        for i2, i1 in enumerate(rows):
            rowmap[i1 % m].append(i2)
        for j2, j1 in enumerate(cols):
            colmap[j1 % n].append(j2)

        # Used to efficiently skip zero rows/cols
        rowset = set(rowmap)
        colset = set(colmap)

        sdm1 = self
        sdm2 = {}
        for i1 in rowset & sdm1.keys():
            row1 = sdm1[i1]
            row2 = {}
            for j1 in colset & row1.keys():
                row1_j1 = row1[j1]
                for j2 in colmap[j1]:
                    row2[j2] = row1_j1
            if row2:
                for i2 in rowmap[i1]:
                    sdm2[i2] = row2.copy()

        return self.new(sdm2, (len(rows), len(cols)), self.domain)

    def __str__(self):
        rowsstr = []
        for i, row in self.items():
            elemsstr = ', '.join('%s: %s' % (j, elem) for j, elem in row.items())
            rowsstr.append('%s: {%s}' % (i, elemsstr))
        return '{%s}' % ', '.join(rowsstr)

    def __repr__(self):
        cls = type(self).__name__
        rows = dict.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    @classmethod
    def new(cls, sdm, shape, domain):
        """

        Parameters
        ==========

        sdm: A dict of dicts for non-zero elements in SDM
        shape: tuple representing dimension of SDM
        domain: Represents :py:class:`~.Domain` of SDM

        Returns
        =======

        An :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1: QQ(2)}}
        >>> A = SDM.new(elemsdict, (2, 2), QQ)
        >>> A
        {0: {1: 2}}

        """
        return cls(sdm, shape, domain)

    def copy(A):
        """
        Returns the copy of a :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> B = A.copy()
        >>> B
        {0: {1: 2}, 1: {}}

        """
        Ac = {i: Ai.copy() for i, Ai in A.items()}
        return A.new(Ac, A.shape, A.domain)

    @classmethod
    def from_list(cls, ddm, shape, domain):
        """
        Create :py:class:`~.SDM` object from a list of lists.

        Parameters
        ==========

        ddm:
            list of lists containing domain elements
        shape:
            Dimensions of :py:class:`~.SDM` matrix
        domain:
            Represents :py:class:`~.Domain` of :py:class:`~.SDM` object

        Returns
        =======

        :py:class:`~.SDM` containing elements of ddm

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = [[QQ(1, 2), QQ(0)], [QQ(0), QQ(3, 4)]]
        >>> A = SDM.from_list(ddm, (2, 2), QQ)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}

        See Also
        ========

        to_list
        from_list_flat
        from_dok
        from_ddm
        """

        m, n = shape
        if not (len(ddm) == m and all(len(row) == n for row in ddm)):
            raise DMBadInputError("Inconsistent row-list/shape")
        getrow = lambda i: {j:ddm[i][j] for j in range(n) if ddm[i][j]}
        irows = ((i, getrow(i)) for i in range(m))
        sdm = {i: row for i, row in irows if row}
        return cls(sdm, shape, domain)

    @classmethod
    def from_ddm(cls, ddm):
        """
        Create :py:class:`~.SDM` from a :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = DDM( [[QQ(1, 2), 0], [0, QQ(3, 4)]], (2, 2), QQ)
        >>> A = SDM.from_ddm(ddm)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}
        >>> SDM.from_ddm(ddm).to_ddm() == ddm
        True

        See Also
        ========

        to_ddm
        from_list
        from_list_flat
        from_dok
        """
        return cls.from_list(ddm, ddm.shape, ddm.domain)

    def to_list(M):
        """
        Convert a :py:class:`~.SDM` object to a list of lists.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> A.to_list()
        [[0, 2], [0, 0]]


        """
        m, n = M.shape
        zero = M.domain.zero
        ddm = [[zero] * n for _ in range(m)]
        for i, row in M.items():
            for j, e in row.items():
                ddm[i][j] = e
        return ddm

    def to_list_flat(M):
        """
        Convert :py:class:`~.SDM` to a flat list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_list_flat()
        [0, 2, 3, 0]
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        from_list_flat
        to_list
        to_dok
        to_ddm
        """
        m, n = M.shape
        zero = M.domain.zero
        flat = [zero] * (m * n)
        for i, row in M.items():
            for j, e in row.items():
                flat[i*n + j] = e
        return flat

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :py:class:`~.SDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.from_list_flat([QQ(0), QQ(2), QQ(0), QQ(0)], (2, 2), QQ)
        >>> A
        {0: {1: 2}}
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        from_list
        from_dok
        from_ddm
        """
        m, n = shape
        if len(elements) != m * n:
            raise DMBadInputError("Inconsistent flat-list shape")
        sdm = defaultdict(dict)
        for inj, element in enumerate(elements):
            if element:
                i, j = divmod(inj, n)
                sdm[i][j] = element
        return cls(sdm, shape, domain)

    def to_flat_nz(M):
        """
        Convert :class:`SDM` to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a modified matrix with elements in the same positions using
        :meth:`from_flat_nz`. Zero elements are omitted from the list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [2, 3]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        to_list_flat
        sympy.polys.matrices.ddm.DDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        dok = M.to_dok()
        indices = tuple(dok)
        elements = list(dok.values())
        data = (indices, M.shape)
        return elements, data

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`~.SDM` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        from_list_flat
        sympy.polys.matrices.ddm.DDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        indices, shape = data
        dok = dict(zip(indices, elements))
        return cls.from_dok(dok, shape, domain)

    def to_dod(M):
        """
        Convert to dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dod()
        {0: {1: 2}, 1: {0: 3}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        return {i: row.copy() for i, row in M.items()}

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dod = {0: {1: QQ(2)}, 1: {0: QQ(3)}}
        >>> A = SDM.from_dod(dod, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dod(A.to_dod(), A.shape, A.domain)
        True

        See Also
        ========

        to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        sdm = defaultdict(dict)
        for i, row in dod.items():
            for j, e in row.items():
                if e:
                    sdm[i][j] = e
        return cls(sdm, shape, domain)

    def to_dok(M):
        """
        Convert to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dok()
        {(0, 1): 2, (1, 0): 3}

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_ddm
        """
        return {(i, j): e for i, row in M.items() for j, e in row.items()}

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dok = {(0, 1): QQ(2), (1, 0): QQ(3)}
        >>> A = SDM.from_dok(dok, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        to_dok
        from_list
        from_list_flat
        from_ddm
        """
        sdm = defaultdict(dict)
        for (i, j), e in dok.items():
            if e:
                sdm[i][j] = e
        return cls(sdm, shape, domain)

    def iter_values(M):
        """
        Iterate over the nonzero values of a :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_values())
        [2, 3]

        """
        for row in M.values():
            yield from row.values()

    def iter_items(M):
        """
        Iterate over indices and values of the nonzero elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 1), 2), ((1, 0), 3)]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        for i, row in M.items():
            for j, e in row.items():
                yield (i, j), e

    def to_ddm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_ddm()
        [[0, 2], [0, 0]]

        """
        return DDM(M.to_list(), M.shape, M.domain)

    def to_sdm(M):
        """
        Convert to :py:class:`~.SDM` format (returns self).
        """
        return M

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DFM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm()
        [[0, 2], [0, 0]]

        See Also
        ========

        to_ddm
        to_dfm_or_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm
        """
        return M.to_ddm().to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(M):
        """
        Convert to :py:class:`~.DFM` if possible, else :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[0, 2], [0, 0]]
        >>> type(A.to_dfm_or_ddm())  # depends on the ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dfm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        return M.to_ddm().to_dfm_or_ddm()

    @classmethod
    def zeros(cls, shape, domain):
        r"""

        Returns a :py:class:`~.SDM` of size shape,
        belonging to the specified domain

        In the example below we declare a matrix A where,

        .. math::
            A := \left[\begin{array}{ccc}
            0 & 0 & 0 \\
            0 & 0 & 0 \end{array} \right]

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.zeros((2, 3), QQ)
        >>> A
        {}

        """
        return cls({}, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        one = domain.one
        m, n = shape
        row = dict(zip(range(n), [one]*n))
        sdm = {i: row.copy() for i in range(m)}
        return cls(sdm, shape, domain)

    @classmethod
    def eye(cls, shape, domain):
        """

        Returns a identity :py:class:`~.SDM` matrix of dimensions
        size x size, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> I = SDM.eye((2, 2), QQ)
        >>> I
        {0: {0: 1}, 1: {1: 1}}

        """
        if isinstance(shape, int):
            rows, cols = shape, shape
        else:
            rows, cols = shape
        one = domain.one
        sdm = {i: {i: one} for i in range(min(rows, cols))}
        return cls(sdm, (rows, cols), domain)

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        if shape is None:
            shape = (len(diagonal), len(diagonal))
        sdm = {i: {i: v} for i, v in enumerate(diagonal) if v}
        return cls(sdm, shape, domain)

    def transpose(M):
        """

        Returns the transpose of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.transpose()
        {1: {0: 2}}

        """
        MT = sdm_transpose(M)
        return M.new(MT, M.shape[::-1], M.domain)

    def __add__(A, B):
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s + %s" % (A.shape, B.shape))
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s - %s" % (A.shape, B.shape))
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, SDM):
            return A.matmul(B)
        elif B in A.domain:
            return A.mul(B)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if b in a.domain:
            return a.rmul(b)
        else:
            return NotImplemented

    def matmul(A, B):
        """
        Performs matrix multiplication of two SDM matrices

        Parameters
        ==========

        A, B: SDM to multiply

        Returns
        =======

        SDM
            SDM after multiplication

        Raises
        ======

        DomainError
            If domain of A does not match
            with that of B

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0:ZZ(2), 1:ZZ(3)}, 1:{0:ZZ(4)}}, (2, 2), ZZ)
        >>> A.matmul(B)
        {0: {0: 8}, 1: {0: 2, 1: 3}}

        """
        if A.domain != B.domain:
            raise DMDomainError
        m, n = A.shape
        n2, o = B.shape
        if n != n2:
            raise DMShapeError
        C = sdm_matmul(A, B, A.domain, m, o)
        return A.new(C, (m, o), A.domain)

    def mul(A, b):
        """
        Multiplies each element of A with a scalar b

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.mul(ZZ(3))
        {0: {1: 6}, 1: {0: 3}}

        """
        Csdm = unop_dict(A, lambda aij: aij*b)
        return A.new(Csdm, A.shape, A.domain)

    def rmul(A, b):
        Csdm = unop_dict(A, lambda aij: b*aij)
        return A.new(Csdm, A.shape, A.domain)

    def mul_elementwise(A, B):
        if A.domain != B.domain:
            raise DMDomainError
        if A.shape != B.shape:
            raise DMShapeError
        zero = A.domain.zero
        fzero = lambda e: zero
        Csdm = binop_dict(A, B, mul, fzero, fzero)
        return A.new(Csdm, A.shape, A.domain)

    def add(A, B):
        """

        Adds two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.add(B)
        {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

        """
        Csdm = binop_dict(A, B, add, pos, pos)
        return A.new(Csdm, A.shape, A.domain)

    def sub(A, B):
        """

        Subtracts two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.sub(B)
        {0: {0: -3, 1: 2}, 1: {0: 1, 1: -4}}

        """
        Csdm = binop_dict(A, B, sub, pos, neg)
        return A.new(Csdm, A.shape, A.domain)

    def neg(A):
        """

        Returns the negative of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.neg()
        {0: {1: -2}, 1: {0: -1}}

        """
        Csdm = unop_dict(A, neg)
        return A.new(Csdm, A.shape, A.domain)

    def convert_to(A, K):
        """
        Converts the :py:class:`~.Domain` of a :py:class:`~.SDM` matrix to K

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.convert_to(QQ)
        {0: {1: 2}, 1: {0: 1}}

        """
        Kold = A.domain
        if K == Kold:
            return A.copy()
        Ak = unop_dict(A, lambda e: K.convert_from(e, Kold))
        return A.new(Ak, A.shape, K)

    def nnz(A):
        """Number of non-zero elements in the :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.nnz()
        2

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        return sum(map(len, A.values()))

    def scc(A):
        """Strongly connected components of a square matrix *A*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0: ZZ(2)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc
        """
        rows, cols = A.shape
        assert rows == cols
        V = range(rows)
        Emap = {v: list(A.get(v, [])) for v in V}
        return _strongly_connected_components(V, Emap)

    def rref(A):
        """

        Returns reduced-row echelon form and list of pivots for the :py:class:`~.SDM`

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref()
        ({0: {0: 1, 1: 2}}, [0])

        """
        B, pivots, _ = sdm_irref(A)
        return A.new(B, A.shape, A.domain), pivots

    def rref_den(A):
        """

        Returns reduced-row echelon form (RREF) with denominator and pivots.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref_den()
        ({0: {0: 1, 1: 2}}, 1, [0])

        """
        K = A.domain
        A_rref_sdm, denom, pivots = sdm_rref_den(A, K)
        A_rref = A.new(A_rref_sdm, A.shape, A.domain)
        return A_rref, denom, pivots

    def inv(A):
        """

        Returns inverse of a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.inv()
        {0: {0: -2, 1: 1}, 1: {0: 3/2, 1: -1/2}}

        """
        return A.to_dfm_or_ddm().inv().to_sdm()

    def det(A):
        """
        Returns determinant of A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.det()
        -2

        """
        # It would be better to have a sparse implementation of det for use
        # with very sparse matrices. Extremely sparse matrices probably just
        # have determinant zero and we could probably detect that very quickly.
        # In the meantime, we convert to a dense matrix and use ddm_idet.
        #
        # If GROUND_TYPES=flint though then we will use Flint's implementation
        # if possible (dfm).
        return A.to_dfm_or_ddm().det()

    def lu(A):
        """

        Returns LU decomposition for a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.lu()
        ({0: {0: 1}, 1: {0: 3, 1: 1}}, {0: {0: 1, 1: 2}, 1: {1: -2}}, [])

        """
        L, U, swaps = A.to_ddm().lu()
        return A.from_ddm(L), A.from_ddm(U), swaps

    def lu_solve(A, b):
        """

        Uses LU decomposition to solve Ax = b,

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)
        >>> A.lu_solve(b)
        {1: {0: 1/2}}

        """
        return A.from_ddm(A.to_ddm().lu_solve(b.to_ddm()))

    def nullspace(A):
        """
        Nullspace of a :py:class:`~.SDM` matrix A.

        The domain of the matrix must be a field.

        It is better to use the :meth:`~.DomainMatrix.nullspace` method rather
        than this method which is otherwise no longer used.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A.nullspace()
        ({0: {0: -2, 1: 1}}, [1])


        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The preferred way to get the nullspace of a matrix.

        """
        ncols = A.shape[1]
        one = A.domain.one
        B, pivots, nzcols = sdm_irref(A)
        K, nonpivots = sdm_nullspace_from_rref(B, one, ncols, pivots, nzcols)
        K = dict(enumerate(K))
        shape = (len(K), ncols)
        return A.new(K, shape, A.domain), nonpivots

    def nullspace_from_rref(A, pivots=None):
        """
        Returns nullspace for a :py:class:`~.SDM` matrix ``A`` in RREF.

        The domain of the matrix can be any domain.

        The matrix must already be in reduced row echelon form (RREF).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A_rref, pivots = A.rref()
        >>> A_null, nonpivots = A_rref.nullspace_from_rref(pivots)
        >>> A_null
        {0: {0: -2, 1: 1}}
        >>> pivots
        [0]
        >>> nonpivots
        [1]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher-level function that would usually be called instead of
            calling this one directly.

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace_from_rref
            The higher-level direct equivalent of this function.

        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
            The equivalent function for dense :py:class:`~.DDM` matrices.

        """
        m, n = A.shape
        K = A.domain

        if pivots is None:
            pivots = sorted(map(min, A.values()))

        if not pivots:
            return A.eye((n, n), K), list(range(n))
        elif len(pivots) == n:
            return A.zeros((0, n), K), []

        # In fraction-free RREF the nonzero entry inserted for the pivots is
        # not necessarily 1.
        pivot_val = A[0][pivots[0]]
        assert not K.is_zero(pivot_val)

        pivots_set = set(pivots)

        # Loop once over all nonzero entries making a map from column indices
        # to the nonzero entries in that column along with the row index of the
        # nonzero entry. This is basically the transpose of the matrix.
        nonzero_cols = defaultdict(list)
        for i, Ai in A.items():
            for j, Aij in Ai.items():
                nonzero_cols[j].append((i, Aij))

        # Usually in SDM we want to avoid looping over the dimensions of the
        # matrix because it is optimised to support extremely sparse matrices.
        # Here in nullspace though every zero column becomes a nonzero column
        # so we need to loop once over the columns at least (range(n)) rather
        # than just the nonzero entries of the matrix. We can still avoid
        # an inner loop over the rows though by using the nonzero_cols map.
        basis = []
        nonpivots = []
        for j in range(n):
            if j in pivots_set:
                continue
            nonpivots.append(j)

            vec = {j: pivot_val}
            for ip, Aij in nonzero_cols[j]:
                vec[pivots[ip]] = -Aij

            basis.append(vec)

        sdm = dict(enumerate(basis))
        A_null = A.new(sdm, (len(basis), n), K)

        return (A_null, nonpivots)

    def particular(A):
        ncols = A.shape[1]
        B, pivots, nzcols = sdm_irref(A)
        P = sdm_particular_from_rref(B, ncols, pivots)
        rep = {0:P} if P else {}
        return A.new(rep, (1, ncols-1), A.domain)

    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.hstack(B)
        {0: {0: 1, 1: 2, 2: 5, 3: 6}, 1: {0: 3, 1: 4, 2: 7, 3: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.hstack(B, C)
        {0: {0: 1, 1: 2, 2: 5, 3: 6, 4: 9, 5: 10}, 1: {0: 3, 1: 4, 2: 7, 3: 8, 4: 11, 5: 12}}
        """
        Anew = dict(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain

            for i, Bki in Bk.items():
                Ai = Anew.get(i, None)
                if Ai is None:
                    Anew[i] = Ai = {}
                for j, Bkij in Bki.items():
                    Ai[j + cols] = Bkij
            cols += Bkcols

        return A.new(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        """Vertically stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.vstack(B)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.vstack(B, C)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}, 4: {0: 9, 1: 10}, 5: {0: 11, 1: 12}}
        """
        Anew = dict(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain

            for i, Bki in Bk.items():
                Anew[i + rows] = Bki
            rows += Bkrows

        return A.new(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        sdm = {i: {j: func(e) for j, e in row.items()} for i, row in self.items()}
        return self.new(sdm, self.shape, domain)

    def charpoly(A):
        """
        Returns the coefficients of the characteristic polynomial
        of the :py:class:`~.SDM` matrix. These elements will be domain elements.
        The domain of the elements will be same as domain of the :py:class:`~.SDM`.

        Examples
        ========

        >>> from sympy import QQ, Symbol
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy.polys import Poly
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.charpoly()
        [1, -5, -2]

        We can create a polynomial using the
        coefficients using :py:class:`~.Poly`

        >>> x = Symbol('x')
        >>> p = Poly(A.charpoly(), x, domain=A.domain)
        >>> p
        Poly(x**2 - 5*x - 2, x, domain='QQ')

        """
        K = A.domain
        n, _ = A.shape
        pdict = sdm_berk(A, n, K)
        plist = [K.zero] * (n + 1)
        for i, pi in pdict.items():
            plist[i] = pi
        return plist

    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
        return not self

    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i <= j for i, row in self.items() for j in row)

    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i >= j for i, row in self.items() for j in row)

    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned
        even if the matrix is not square.
        """
        return all(i == j for i, row in self.items() for j in row)

    def diagonal(self):
        """
        Returns the diagonal of the matrix as a list.
        """
        m, n = self.shape
        zero = self.domain.zero
        return [row.get(i, zero) for i, row in self.items() if i < n]

    def lll(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis for the :py:class:`~.SDM` matrix.
        """
        return A.to_dfm_or_ddm().lll(delta=delta).to_sdm()

    def lll_transform(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis and transformation matrix.
        """
        reduced, transform = A.to_dfm_or_ddm().lll_transform(delta=delta)
        return reduced.to_sdm(), transform.to_sdm()


def binop_dict(A, B, fab, fa, fb):
    Anz, Bnz = set(A), set(B)
    C = {}

    for i in Anz & Bnz:
        Ai, Bi = A[i], B[i]
        Ci = {}
        Anzi, Bnzi = set(Ai), set(Bi)
        for j in Anzi & Bnzi:
            Cij = fab(Ai[j], Bi[j])
            if Cij:
                Ci[j] = Cij
        for j in Anzi - Bnzi:
            Cij = fa(Ai[j])
            if Cij:
                Ci[j] = Cij
        for j in Bnzi - Anzi:
            Cij = fb(Bi[j])
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    for i in Anz - Bnz:
        Ai = A[i]
        Ci = {}
        for j, Aij in Ai.items():
            Cij = fa(Aij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    for i in Bnz - Anz:
        Bi = B[i]
        Ci = {}
        for j, Bij in Bi.items():
            Cij = fb(Bij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    return C


def unop_dict(A, f):
    B = {}
    for i, Ai in A.items():
        Bi = {}
        for j, Aij in Ai.items():
            Bij = f(Aij)
            if Bij:
                Bi[j] = Bij
        if Bi:
            B[i] = Bi
    return B


def sdm_transpose(M):
    MT = {}
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            try:
                MT[j][i] = Mij
            except KeyError:
                MT[j] = {i: Mij}
    return MT


def sdm_dotvec(A, B, K):
    return K.sum(A[j] * B[j] for j in A.keys() & B.keys())


def sdm_matvecmul(A, B, K):
    C = {}
    for i, Ai in A.items():
        Ci = sdm_dotvec(Ai, B, K)
        if Ci:
            C[i] = Ci
    return C


def sdm_matmul(A, B, K, m, o):
    #
    # Should be fast if A and B are very sparse.
    # Consider e.g. A = B = eye(1000).
    #
    # The idea here is that we compute C = A*B in terms of the rows of C and
    # B since the dict of dicts representation naturally stores the matrix as
    # rows. The ith row of C (Ci) is equal to the sum of Aik * Bk where Bk is
    # the kth row of B. The algorithm below loops over each nonzero element
    # Aik of A and if the corresponding row Bj is nonzero then we do
    #    Ci += Aik * Bk.
    # To make this more efficient we don't need to loop over all elements Aik.
    # Instead for each row Ai we compute the intersection of the nonzero
    # columns in Ai with the nonzero rows in B. That gives the k such that
    # Aik and Bk are both nonzero. In Python the intersection of two sets
    # of int can be computed very efficiently.
    #
    if K.is_EXRAW:
        return sdm_matmul_exraw(A, B, K, m, o)

    C = {}
    B_knz = set(B)
    for i, Ai in A.items():
        Ci = {}
        Ai_knz = set(Ai)
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            for j, Bkj in B[k].items():
                Cij = Ci.get(j, None)
                if Cij is not None:
                    Cij = Cij + Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
                    else:
                        Ci.pop(j)
                else:
                    Cij = Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
        if Ci:
            C[i] = Ci
    return C


def sdm_matmul_exraw(A, B, K, m, o):
    #
    # Like sdm_matmul above except that:
    #
    # - Handles cases like 0*oo -> nan (sdm_matmul skips multipication by zero)
    # - Uses K.sum (Add(*items)) for efficient addition of Expr
    #
    zero = K.zero
    C = {}
    B_knz = set(B)
    for i, Ai in A.items():
        Ci_list = defaultdict(list)
        Ai_knz = set(Ai)

        # Nonzero row/column pair
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            if zero * Aik == zero:
                # This is the main inner loop:
                for j, Bkj in B[k].items():
                    Ci_list[j].append(Aik * Bkj)
            else:
                for j in range(o):
                    Ci_list[j].append(Aik * B[k].get(j, zero))

        # Zero row in B, check for infinities in A
        for k in Ai_knz - B_knz:
            zAik = zero * Ai[k]
            if zAik != zero:
                for j in range(o):
                    Ci_list[j].append(zAik)

        # Add terms using K.sum (Add(*terms)) for efficiency
        Ci = {}
        for j, Cij_list in Ci_list.items():
            Cij = K.sum(Cij_list)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    # Find all infinities in B
    for k, Bk in B.items():
        for j, Bkj in Bk.items():
            if zero * Bkj != zero:
                for i in range(m):
                    Aik = A.get(i, {}).get(k, zero)
                    # If Aik is not zero then this was handled above
                    if Aik == zero:
                        Ci = C.get(i, {})
                        Cij = Ci.get(j, zero) + Aik * Bkj
                        if Cij != zero:
                            Ci[j] = Cij
                        else:  # pragma: no cover
                            # Not sure how we could get here but let's raise an
                            # exception just in case.
                            raise RuntimeError
                        C[i] = Ci

    return C


def sdm_irref(A):
    """RREF and pivots of a sparse matrix *A*.

    Compute the reduced row echelon form (RREF) of the matrix *A* and return a
    list of the pivot columns. This routine does not work in place and leaves
    the original matrix *A* unmodified.

    The domain of the matrix must be a field.

    Examples
    ========

    This routine works with a dict of dicts sparse representation of a matrix:

    >>> from sympy import QQ
    >>> from sympy.polys.matrices.sdm import sdm_irref
    >>> A = {0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}
    >>> Arref, pivots, _ = sdm_irref(A)
    >>> Arref
    {0: {0: 1}, 1: {1: 1}}
    >>> pivots
    [0, 1]

    The analogous calculation with :py:class:`~.MutableDenseMatrix` would be

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> Mrref, pivots = M.rref()
    >>> Mrref
    Matrix([
    [1, 0],
    [0, 1]])
    >>> pivots
    (0, 1)

    Notes
    =====

    The cost of this algorithm is determined purely by the nonzero elements of
    the matrix. No part of the cost of any step in this algorithm depends on
    the number of rows or columns in the matrix. No step depends even on the
    number of nonzero rows apart from the primary loop over those rows. The
    implementation is much faster than ddm_rref for sparse matrices. In fact
    at the time of writing it is also (slightly) faster than the dense
    implementation even if the input is a fully dense matrix so it seems to be
    faster in all cases.

    The elements of the matrix should support exact division with ``/``. For
    example elements of any domain that is a field (e.g. ``QQ``) should be
    fine. No attempt is made to handle inexact arithmetic.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The higher-level function that would normally be used to call this
        routine.
    sympy.polys.matrices.dense.ddm_irref
        The dense equivalent of this routine.
    sdm_rref_den
        Fraction-free version of this routine.
    """
    #
    # Any zeros in the matrix are not stored at all so an element is zero if
    # its row dict has no index at that key. A row is entirely zero if its
    # row index is not in the outer dict. Since rref reorders the rows and
    # removes zero rows we can completely discard the row indices. The first
    # step then copies the row dicts into a list sorted by the index of the
    # first nonzero column in each row.
    #
    # The algorithm then processes each row Ai one at a time. Previously seen
    # rows are used to cancel their pivot columns from Ai. Then a pivot from
    # Ai is chosen and is cancelled from all previously seen rows. At this
    # point Ai joins the previously seen rows. Once all rows are seen all
    # elimination has occurred and the rows are sorted by pivot column index.
    #
    # The previously seen rows are stored in two separate groups. The reduced
    # group consists of all rows that have been reduced to a single nonzero
    # element (the pivot). There is no need to attempt any further reduction
    # with these. Rows that still have other nonzeros need to be considered
    # when Ai is cancelled from the previously seen rows.
    #
    # A dict nonzerocolumns is used to map from a column index to a set of
    # previously seen rows that still have a nonzero element in that column.
    # This means that we can cancel the pivot from Ai into the previously seen
    # rows without needing to loop over each row that might have a zero in
    # that column.
    #

    # Row dicts sorted by index of first nonzero column
    # (Maybe sorting is not needed/useful.)
    Arows = sorted((Ai.copy() for Ai in A.values()), key=min)

    # Each processed row has an associated pivot column.
    # pivot_row_map maps from the pivot column index to the row dict.
    # This means that we can represent a set of rows purely as a set of their
    # pivot indices.
    pivot_row_map = {}

    # Set of pivot indices for rows that are fully reduced to a single nonzero.
    reduced_pivots = set()

    # Set of pivot indices for rows not fully reduced
    nonreduced_pivots = set()

    # Map from column index to a set of pivot indices representing the rows
    # that have a nonzero at that column.
    nonzero_columns = defaultdict(set)

    while Arows:
        # Select pivot element and row
        Ai = Arows.pop()

        # Nonzero columns from fully reduced pivot rows can be removed
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced_pivots}

        # Others require full row cancellation
        for j in nonreduced_pivots & set(Ai):
            Aj = pivot_row_map[j]
            Aij = Ai[j]
            Ainz = set(Ai)
            Ajnz = set(Aj)
            for k in Ajnz - Ainz:
                Ai[k] = - Aij * Aj[k]
            Ai.pop(j)
            Ainz.remove(j)
            for k in Ajnz & Ainz:
                Aik = Ai[k] - Aij * Aj[k]
                if Aik:
                    Ai[k] = Aik
                else:
                    Ai.pop(k)

        # We have now cancelled previously seen pivots from Ai.
        # If it is zero then discard it.
        if not Ai:
            continue

        # Choose a pivot from Ai:
        j = min(Ai)
        Aij = Ai[j]
        pivot_row_map[j] = Ai
        Ainz = set(Ai)

        # Normalise the pivot row to make the pivot 1.
        #
        # This approach is slow for some domains. Cross cancellation might be
        # better for e.g. QQ(x) with division delayed to the final steps.
        Aijinv = Aij**-1
        for l in Ai:
            Ai[l] *= Aijinv

        # Use Aij to cancel column j from all previously seen rows
        for k in nonzero_columns.pop(j, ()):
            Ak = pivot_row_map[k]
            Akj = Ak[j]
            Aknz = set(Ak)
            for l in Ainz - Aknz:
                Ak[l] = - Akj * Ai[l]
                nonzero_columns[l].add(k)
            Ak.pop(j)
            Aknz.remove(j)
            for l in Ainz & Aknz:
                Akl = Ak[l] - Akj * Ai[l]
                if Akl:
                    Ak[l] = Akl
                else:
                    # Drop nonzero elements
                    Ak.pop(l)
                    if l != j:
                        nonzero_columns[l].remove(k)
            if len(Ak) == 1:
                reduced_pivots.add(k)
                nonreduced_pivots.remove(k)

        if len(Ai) == 1:
            reduced_pivots.add(j)
        else:
            nonreduced_pivots.add(j)
            for l in Ai:
                if l != j:
                    nonzero_columns[l].add(j)

    # All done!
    pivots = sorted(reduced_pivots | nonreduced_pivots)
    pivot2row = {p: n for n, p in enumerate(pivots)}
    nonzero_columns = {c: {pivot2row[p] for p in s} for c, s in nonzero_columns.items()}
    rows = [pivot_row_map[i] for i in pivots]
    rref = dict(enumerate(rows))
    return rref, pivots, nonzero_columns


def sdm_rref_den(A, K):
    """
    Return the reduced row echelon form (RREF) of A with denominator.

    The RREF is computed using fraction-free Gauss-Jordan elimination.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic. This implementation is also
    optimized for sparse matrices.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.sdm import sdm_rref_den
    >>> from sympy.polys.domains import ZZ
    >>> A = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}
    >>> A_rref, den, pivots = sdm_rref_den(A, ZZ)
    >>> A_rref
    {0: {0: -2}, 1: {1: -2}}
    >>> den
    -2
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher-level interface to ``sdm_rref_den`` that would usually be used
        instead of calling this function directly.
    sympy.polys.matrices.sdm.sdm_rref_den
        The ``SDM`` method that uses this function.
    sdm_irref
        Computes RREF using field division.
    ddm_irref_den
        The dense version of this algorithm.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
    #
    # We represent each row of the matrix as a dict mapping column indices to
    # nonzero elements. We will build the RREF matrix starting from an empty
    # matrix and appending one row at a time. At each step we will have the
    # RREF of the rows we have processed so far.
    #
    # Our representation of the RREF divides it into three parts:
    #
    # 1. Fully reduced rows having only a single nonzero element (the pivot).
    # 2. Partially reduced rows having nonzeros after the pivot.
    # 3. The current denominator and divisor.
    #
    # For example if the incremental RREF might be:
    #
    #   [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #   [0, 0, 2, 0, 0, 0, 7, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
    #
    # Here the second row is partially reduced and the other rows are fully
    # reduced. The denominator would be 2 in this case. We distinguish the
    # fully reduced rows because we can handle them more efficiently when
    # adding a new row.
    #
    # When adding a new row we need to multiply it by the current denominator.
    # Then we reduce the new row by cross cancellation with the previous rows.
    # Then if it is not reduced to zero we take its leading entry as the new
    # pivot, cross cancel the new row from the previous rows and update the
    # denominator. In the fraction-free version this last step requires
    # multiplying and dividing the whole matrix by the new pivot and the
    # current divisor. The advantage of building the RREF one row at a time is
    # that in the sparse case we only need to work with the relatively sparse
    # upper rows of the matrix. The simple version of FFGJ in [1] would
    # multiply and divide all the dense lower rows at each step.

    # Handle the trivial cases.
    if not A:
        return ({}, K.one, [])
    elif len(A) == 1:
        Ai, = A.values()
        j = min(Ai)
        Aij = Ai[j]
        return ({0: Ai.copy()}, Aij, [j])

    # For inexact domains like RR[x] we use quo and discard the remainder.
    # Maybe it would be better for K.exquo to do this automatically.
    if K.is_Exact:
        exquo = K.exquo
    else:
        exquo = K.quo

    # Make sure we have the rows in order to make this deterministic from the
    # outset.
    _, rows_in_order = zip(*sorted(A.items()))

    col_to_row_reduced = {}
    col_to_row_unreduced = {}
    reduced = col_to_row_reduced.keys()
    unreduced = col_to_row_unreduced.keys()

    # Our representation of the RREF so far.
    A_rref_rows = []
    denom = None
    divisor = None

    # The rows that remain to be added to the RREF. These are sorted by the
    # column index of their leading entry. Note that sorted() is stable so the
    # previous sort by unique row index is still needed to make this
    # deterministic (there may be multiple rows with the same leading column).
    A_rows = sorted(rows_in_order, key=min)

    for Ai in A_rows:

        # All fully reduced columns can be immediately discarded.
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced}

        # We need to multiply the new row by the current denominator to bring
        # it into the same scale as the previous rows and then cross-cancel to
        # reduce it wrt the previous unreduced rows. All pivots in the previous
        # rows are equal to denom so the coefficients we need to make a linear
        # combination of the previous rows to cancel into the new row are just
        # the ones that are already in the new row *before* we multiply by
        # denom. We compute that linear combination first and then multiply the
        # new row by denom before subtraction.
        Ai_cancel = {}

        for j in unreduced & Ai.keys():
            # Remove the pivot column from the new row since it would become
            # zero anyway.
            Aij = Ai.pop(j)

            Aj = A_rref_rows[col_to_row_unreduced[j]]

            for k, Ajk in Aj.items():
                Aik_cancel = Ai_cancel.get(k)
                if Aik_cancel is None:
                    Ai_cancel[k] = Aij * Ajk
                else:
                    Aik_cancel = Aik_cancel + Aij * Ajk
                    if Aik_cancel:
                        Ai_cancel[k] = Aik_cancel
                    else:
                        Ai_cancel.pop(k)

        # Multiply the new row by the current denominator and subtract.
        Ai_nz = set(Ai)
        Ai_cancel_nz = set(Ai_cancel)

        d = denom or K.one

        for k in Ai_cancel_nz - Ai_nz:
            Ai[k] = -Ai_cancel[k]

        for k in Ai_nz - Ai_cancel_nz:
            Ai[k] = Ai[k] * d

        for k in Ai_cancel_nz & Ai_nz:
            Aik = Ai[k] * d - Ai_cancel[k]
            if Aik:
                Ai[k] = Aik
            else:
                Ai.pop(k)

        # Now Ai has the same scale as the other rows and is reduced wrt the
        # unreduced rows.

        # If the row is reduced to zero then discard it.
        if not Ai:
            continue

        # Choose a pivot for this row.
        j = min(Ai)
        Aij = Ai.pop(j)

        # Cross cancel the unreduced rows by the new row.
        #     a[k][l] = (a[i][j]*a[k][l] - a[k][j]*a[i][l]) / divisor
        for pk, k in list(col_to_row_unreduced.items()):

            Ak = A_rref_rows[k]

            if j not in Ak:
                # This row is already reduced wrt the new row but we need to
                # bring it to the same scale as the new denominator. This step
                # is not needed in sdm_irref.
                for l, Akl in Ak.items():
                    Akl = Akl * Aij
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                continue

            Akj = Ak.pop(j)
            Ai_nz = set(Ai)
            Ak_nz = set(Ak)

            for l in Ai_nz - Ak_nz:
                Ak[l] = - Akj * Ai[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            # This loop also not needed in sdm_irref.
            for l in Ak_nz - Ai_nz:
                Ak[l] = Aij * Ak[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            for l in Ai_nz & Ak_nz:
                Akl = Aij * Ak[l] - Akj * Ai[l]
                if Akl:
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                else:
                    Ak.pop(l)

            if not Ak:
                col_to_row_unreduced.pop(pk)
                col_to_row_reduced[pk] = k

        i = len(A_rref_rows)
        A_rref_rows.append(Ai)
        if Ai:
            col_to_row_unreduced[j] = i
        else:
            col_to_row_reduced[j] = i

        # Update the denominator.
        if not K.is_one(Aij):
            if denom is None:
                denom = Aij
            else:
                denom *= Aij

        if divisor is not None:
            denom = exquo(denom, divisor)

        # Update the divisor.
        divisor = denom

    if denom is None:
        denom = K.one

    # Sort the rows by their leading column index.
    col_to_row = {**col_to_row_reduced, **col_to_row_unreduced}
    row_to_col = {i: j for j, i in col_to_row.items()}
    A_rref_rows_col = [(row_to_col[i], Ai) for i, Ai in enumerate(A_rref_rows)]
    pivots, A_rref = zip(*sorted(A_rref_rows_col))
    pivots = list(pivots)

    # Insert the pivot values
    for i, Ai in enumerate(A_rref):
        Ai[pivots[i]] = denom

    A_rref_sdm = dict(enumerate(A_rref))

    return A_rref_sdm, denom, pivots


def sdm_nullspace_from_rref(A, one, ncols, pivots, nonzero_cols):
    """Get nullspace from A which is in RREF"""
    nonpivots = sorted(set(range(ncols)) - set(pivots))

    K = []
    for j in nonpivots:
        Kj = {j:one}
        for i in nonzero_cols.get(j, ()):
            Kj[pivots[i]] = -A[i][j]
        K.append(Kj)

    return K, nonpivots


def sdm_particular_from_rref(A, ncols, pivots):
    """Get a particular solution from A which is in RREF"""
    P = {}
    for i, j in enumerate(pivots):
        Ain = A[i].get(ncols-1, None)
        if Ain is not None:
            P[j] = Ain / A[i][j]
    return P


def sdm_berk(M, n, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring. This implementation is for sparse
    matrices represented in a dict-of-dicts format (like :class:`SDM`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.sdm import sdm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    >>> sdm_berk(M, 2, ZZ)
    {0: 1, 1: -5, 2: -2}
    >>> Matrix([[1, 2], [3, 4]]).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.
    sympy.polys.matrices.dense.ddm_berk
        The dense version of this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    zero = K.zero
    one = K.one

    if n == 0:
        return {0: one}
    elif n == 1:
        pdict = {0: one}
        if M00 := M.get(0, {}).get(0, zero):
            pdict[1] = -M00

    # M = [[a, R],
    #      [C, A]]
    a, R, C, A = K.zero, {}, {}, defaultdict(dict)
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            if i and j:
                A[i-1][j-1] = Mij
            elif i:
                C[i-1] = Mij
            elif j:
                R[j-1] = Mij
            else:
                a = Mij

    # T = [       1,      0,    0,  0, 0, ... ]
    #     [      -a,      1,    0,  0, 0, ... ]
    #     [    -R*C,     -a,    1,  0, 0, ... ]
    #     [  -R*A*C,   -R*C,   -a,  1, 0, ... ]
    #     [-R*A^2*C, -R*A*C, -R*C, -a, 1, ... ]
    #     [ ...                               ]
    # T is (n+1) x n
    #
    # In the sparse case we might have A^m*C = 0 for some m making T banded
    # rather than triangular so we just compute the nonzero entries of the
    # first column rather than constructing the matrix explicitly.

    AnC = C
    RC = sdm_dotvec(R, C, K)

    Tvals = [one, -a, -RC]
    for i in range(3, n+1):
        AnC = sdm_matvecmul(A, AnC, K)
        if not AnC:
            break
        RAnC = sdm_dotvec(R, AnC, K)
        Tvals.append(-RAnC)

    # Strip trailing zeros
    while Tvals and not Tvals[-1]:
        Tvals.pop()

    q = sdm_berk(A, n-1, K)

    # This would be the explicit multiplication T*q but we can do better:
    #
    #  T = {}
    #  for i in range(n+1):
    #      Ti = {}
    #      for j in range(max(0, i-len(Tvals)+1), min(i+1, n)):
    #          Ti[j] = Tvals[i-j]
    #      T[i] = Ti
    #  Tq = sdm_matvecmul(T, q, K)
    #
    # In the sparse case q might be mostly zero. We know that T[i,j] is nonzero
    # for i <= j < i + len(Tvals) so if q does not have a nonzero entry in that
    # range then Tq[j] must be zero. We exploit this potential banded
    # structure and the potential sparsity of q to compute Tq more efficiently.

    Tvals = Tvals[::-1]

    Tq = {}

    for i in range(min(q), min(max(q)+len(Tvals), n+1)):
        Ti = dict(enumerate(Tvals, i-len(Tvals)+1))
        if Tqi := sdm_dotvec(Ti, q, K):
            Tq[i] = Tqi

    return Tq
