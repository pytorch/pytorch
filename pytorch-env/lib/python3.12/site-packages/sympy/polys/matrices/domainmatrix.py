"""

Module for the DomainMatrix class.

A DomainMatrix represents a matrix with elements that are in a particular
Domain. Each DomainMatrix internally wraps a DDM which is used for the
lower-level operations. The idea is that the DomainMatrix class provides the
convenience routines for converting between Expr and the poly domains as well
as unifying matrices with different domains.

"""
from collections import Counter
from functools import reduce
from typing import Union as tUnion, Tuple as tTuple

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from sympy.core.sympify import _sympify

from ..domains import Domain

from ..constructor import construct_domain

from .exceptions import (
    DMFormatError,
    DMBadInputError,
    DMShapeError,
    DMDomainError,
    DMNotAField,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError
)

from .domainscalar import DomainScalar

from sympy.polys.domains import ZZ, EXRAW, QQ

from sympy.polys.densearith import dup_mul
from sympy.polys.densebasic import dup_convert
from sympy.polys.densetools import (
    dup_mul_ground,
    dup_quo_ground,
    dup_content,
    dup_clear_denoms,
    dup_primitive,
    dup_transform,
)
from sympy.polys.factortools import dup_factor_list
from sympy.polys.polyutils import _sort_factors

from .ddm import DDM

from .sdm import SDM

from .dfm import DFM

from .rref import _dm_rref, _dm_rref_den


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DomainMatrix.to_dfm', 'DomainMatrix.to_dfm_or_ddm']
else:
    __doctest_skip__ = ['DomainMatrix.from_list']


def DM(rows, domain):
    """Convenient alias for DomainMatrix.from_list

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DM
    >>> DM([[1, 2], [3, 4]], ZZ)
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DomainMatrix.from_list
    """
    return DomainMatrix.from_list(rows, domain)


class DomainMatrix:
    r"""
    Associate Matrix with :py:class:`~.Domain`

    Explanation
    ===========

    DomainMatrix uses :py:class:`~.Domain` for its internal representation
    which makes it faster than the SymPy Matrix class (currently) for many
    common operations, but this advantage makes it not entirely compatible
    with Matrix. DomainMatrix are analogous to numpy arrays with "dtype".
    In the DomainMatrix, each element has a domain such as :ref:`ZZ`
    or  :ref:`QQ(a)`.


    Examples
    ========

    Creating a DomainMatrix from the existing Matrix class:

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices import DomainMatrix
    >>> Matrix1 = Matrix([
    ...    [1, 2],
    ...    [3, 4]])
    >>> A = DomainMatrix.from_Matrix(Matrix1)
    >>> A
    DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    Directly forming a DomainMatrix:

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> A = DomainMatrix([
    ...    [ZZ(1), ZZ(2)],
    ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> A
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DDM
    SDM
    Domain
    Poly

    """
    rep: tUnion[SDM, DDM, DFM]
    shape: tTuple[int, int]
    domain: Domain

    def __new__(cls, rows, shape, domain, *, fmt=None):
        """
        Creates a :py:class:`~.DomainMatrix`.

        Parameters
        ==========

        rows : Represents elements of DomainMatrix as list of lists
        shape : Represents dimension of DomainMatrix
        domain : Represents :py:class:`~.Domain` of DomainMatrix

        Raises
        ======

        TypeError
            If any of rows, shape and domain are not provided

        """
        if isinstance(rows, (DDM, SDM, DFM)):
            raise TypeError("Use from_rep to initialise from SDM/DDM")
        elif isinstance(rows, list):
            rep = DDM(rows, shape, domain)
        elif isinstance(rows, dict):
            rep = SDM(rows, shape, domain)
        else:
            msg = "Input should be list-of-lists or dict-of-dicts"
            raise TypeError(msg)

        if fmt is not None:
            if fmt == 'sparse':
                rep = rep.to_sdm()
            elif fmt == 'dense':
                rep = rep.to_ddm()
            else:
                raise ValueError("fmt should be 'sparse' or 'dense'")

        # Use python-flint for dense matrices if possible
        if rep.fmt == 'dense' and DFM._supports_domain(domain):
            rep = rep.to_dfm()

        return cls.from_rep(rep)

    def __reduce__(self):
        rep = self.rep
        if rep.fmt == 'dense':
            arg = self.to_list()
        elif rep.fmt == 'sparse':
            arg = dict(rep)
        else:
            raise RuntimeError # pragma: no cover
        args = (arg, rep.shape, rep.domain)
        return (self.__class__, args)

    def __getitem__(self, key):
        i, j = key
        m, n = self.shape
        if not (isinstance(i, slice) or isinstance(j, slice)):
            return DomainScalar(self.rep.getitem(i, j), self.domain)

        if not isinstance(i, slice):
            if not -m <= i < m:
                raise IndexError("Row index out of range")
            i = i % m
            i = slice(i, i+1)
        if not isinstance(j, slice):
            if not -n <= j < n:
                raise IndexError("Column index out of range")
            j = j % n
            j = slice(j, j+1)

        return self.from_rep(self.rep.extract_slice(i, j))

    def getitem_sympy(self, i, j):
        return self.domain.to_sympy(self.rep.getitem(i, j))

    def extract(self, rowslist, colslist):
        return self.from_rep(self.rep.extract(rowslist, colslist))

    def __setitem__(self, key, value):
        i, j = key
        if not self.domain.of_type(value):
            raise TypeError
        if isinstance(i, int) and isinstance(j, int):
            self.rep.setitem(i, j, value)
        else:
            raise NotImplementedError

    @classmethod
    def from_rep(cls, rep):
        """Create a new DomainMatrix efficiently from DDM/SDM.

        Examples
        ========

        Create a :py:class:`~.DomainMatrix` with an dense internal
        representation as :py:class:`~.DDM`:

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.ddm import DDM
        >>> drep = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

        Create a :py:class:`~.DomainMatrix` with a sparse internal
        representation as :py:class:`~.SDM`:

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import ZZ
        >>> drep = SDM({0:{1:ZZ(1)},1:{0:ZZ(2)}}, (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)

        Parameters
        ==========

        rep: SDM or DDM
            The internal sparse or dense representation of the matrix.

        Returns
        =======

        DomainMatrix
            A :py:class:`~.DomainMatrix` wrapping *rep*.

        Notes
        =====

        This takes ownership of rep as its internal representation. If rep is
        being mutated elsewhere then a copy should be provided to
        ``from_rep``. Only minimal verification or checking is done on *rep*
        as this is supposed to be an efficient internal routine.

        """
        if not (isinstance(rep, (DDM, SDM)) or (DFM is not None and isinstance(rep, DFM))):
            raise TypeError("rep should be of type DDM or SDM")
        self = super().__new__(cls)
        self.rep = rep
        self.shape = rep.shape
        self.domain = rep.domain
        return self

    @classmethod
    @doctest_depends_on(ground_types=['python', 'gmpy'])
    def from_list(cls, rows, domain):
        r"""
        Convert a list of lists into a DomainMatrix

        Parameters
        ==========

        rows: list of lists
            Each element of the inner lists should be either the single arg,
            or tuple of args, that would be passed to the domain constructor
            in order to form an element of the domain. See examples.

        Returns
        =======

        DomainMatrix containing elements defined in rows

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import FF, QQ, ZZ
        >>> A = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], ZZ)
        >>> A
        DomainMatrix([[1, 0, 1], [0, 0, 1]], (2, 3), ZZ)
        >>> B = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], FF(7))
        >>> B
        DomainMatrix([[1 mod 7, 0 mod 7, 1 mod 7], [0 mod 7, 0 mod 7, 1 mod 7]], (2, 3), GF(7))
        >>> C = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
        >>> C
        DomainMatrix([[1/2, 3], [1/4, 5]], (2, 2), QQ)

        See Also
        ========

        from_list_sympy

        """
        nrows = len(rows)
        ncols = 0 if not nrows else len(rows[0])
        conv = lambda e: domain(*e) if isinstance(e, tuple) else domain(e)
        domain_rows = [[conv(e) for e in row] for row in rows]
        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
        r"""
        Convert a list of lists of Expr into a DomainMatrix using construct_domain

        Parameters
        ==========

        nrows: number of rows
        ncols: number of columns
        rows: list of lists

        Returns
        =======

        DomainMatrix containing elements of rows

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x, y, z
        >>> A = DomainMatrix.from_list_sympy(1, 3, [[x, y, z]])
        >>> A
        DomainMatrix([[x, y, z]], (1, 3), ZZ[x,y,z])

        See Also
        ========

        sympy.polys.constructor.construct_domain, from_dict_sympy

        """
        assert len(rows) == nrows
        assert all(len(row) == ncols for row in rows)

        items_sympy = [_sympify(item) for row in rows for item in row]

        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        domain_rows = [[items_domain[ncols*r + c] for c in range(ncols)] for r in range(nrows)]

        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_dict_sympy(cls, nrows, ncols, elemsdict, **kwargs):
        """

        Parameters
        ==========

        nrows: number of rows
        ncols: number of cols
        elemsdict: dict of dicts containing non-zero elements of the DomainMatrix

        Returns
        =======

        DomainMatrix containing elements of elemsdict

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x,y,z
        >>> elemsdict = {0: {0:x}, 1:{1: y}, 2: {2: z}}
        >>> A = DomainMatrix.from_dict_sympy(3, 3, elemsdict)
        >>> A
        DomainMatrix({0: {0: x}, 1: {1: y}, 2: {2: z}}, (3, 3), ZZ[x,y,z])

        See Also
        ========

        from_list_sympy

        """
        if not all(0 <= r < nrows for r in elemsdict):
            raise DMBadInputError("Row out of range")
        if not all(0 <= c < ncols for row in elemsdict.values() for c in row):
            raise DMBadInputError("Column out of range")

        items_sympy = [_sympify(item) for row in elemsdict.values() for item in row.values()]
        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        idx = 0
        items_dict = {}
        for i, row in elemsdict.items():
            items_dict[i] = {}
            for j in row:
                items_dict[i][j] = items_domain[idx]
                idx += 1

        return DomainMatrix(items_dict, (nrows, ncols), domain)

    @classmethod
    def from_Matrix(cls, M, fmt='sparse',**kwargs):
        r"""
        Convert Matrix to DomainMatrix

        Parameters
        ==========

        M: Matrix

        Returns
        =======

        Returns DomainMatrix with identical elements as M

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = Matrix([
        ...    [1.0, 3.4],
        ...    [2.4, 1]])
        >>> A = DomainMatrix.from_Matrix(M)
        >>> A
        DomainMatrix({0: {0: 1.0, 1: 3.4}, 1: {0: 2.4, 1: 1.0}}, (2, 2), RR)

        We can keep internal representation as ddm using fmt='dense'
        >>> from sympy import Matrix, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')
        >>> A.rep
        [[1/2, 3/4], [0, 0]]

        See Also
        ========

        Matrix

        """
        if fmt == 'dense':
            return cls.from_list_sympy(*M.shape, M.tolist(), **kwargs)

        return cls.from_dict_sympy(*M.shape, M.todod(), **kwargs)

    @classmethod
    def get_domain(cls, items_sympy, **kwargs):
        K, items_K = construct_domain(items_sympy, **kwargs)
        return K, items_K

    def choose_domain(self, **opts):
        """Convert to a domain found by :func:`~.construct_domain`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[1, 2], [3, 4]], ZZ)
        >>> M
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> M.choose_domain(field=True)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        >>> from sympy.abc import x
        >>> M = DM([[1, x], [x**2, x**3]], ZZ[x])
        >>> M.choose_domain(field=True).domain
        ZZ(x)

        Keyword arguments are passed to :func:`~.construct_domain`.

        See Also
        ========

        construct_domain
        convert_to
        """
        elements, data = self.to_sympy().to_flat_nz()
        dom, elements_dom = construct_domain(elements, **opts)
        return self.from_flat_nz(elements_dom, data, dom)

    def copy(self):
        return self.from_rep(self.rep.copy())

    def convert_to(self, K):
        r"""
        Change the domain of DomainMatrix to desired domain or field

        Parameters
        ==========

        K : Represents the desired domain or field.
            Alternatively, ``None`` may be passed, in which case this method
            just returns a copy of this DomainMatrix.

        Returns
        =======

        DomainMatrix
            DomainMatrix with the desired domain or field

        Examples
        ========

        >>> from sympy import ZZ, ZZ_I
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.convert_to(ZZ_I)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ_I)

        """
        if K == self.domain:
            return self.copy()

        rep = self.rep

        # The DFM, DDM and SDM types do not do any implicit conversions so we
        # manage switching between DDM and DFM here.
        if rep.is_DFM and not DFM._supports_domain(K):
            rep_K = rep.to_ddm().convert_to(K)
        elif rep.is_DDM and DFM._supports_domain(K):
            rep_K = rep.convert_to(K).to_dfm()
        else:
            rep_K = rep.convert_to(K)

        return self.from_rep(rep_K)

    def to_sympy(self):
        return self.convert_to(EXRAW)

    def to_field(self):
        r"""
        Returns a DomainMatrix with the appropriate field

        Returns
        =======

        DomainMatrix
            DomainMatrix with the appropriate field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_field()
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        """
        K = self.domain.get_field()
        return self.convert_to(K)

    def to_sparse(self):
        """
        Return a sparse DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> A.rep
        [[1, 0], [0, 2]]
        >>> B = A.to_sparse()
        >>> B.rep
        {0: {0: 1}, 1: {1: 2}}
        """
        if self.rep.fmt == 'sparse':
            return self

        return self.from_rep(self.rep.to_sdm())

    def to_dense(self):
        """
        Return a dense DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> A.rep
        {0: {0: 1}, 1: {1: 2}}
        >>> B = A.to_dense()
        >>> B.rep
        [[1, 0], [0, 2]]

        """
        rep = self.rep

        if rep.fmt == 'dense':
            return self

        return self.from_rep(rep.to_dfm_or_ddm())

    def to_ddm(self):
        """
        Return a :class:`~.DDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> ddm = A.to_ddm()
        >>> ddm
        [[1, 0], [0, 2]]
        >>> type(ddm)
        <class 'sympy.polys.matrices.ddm.DDM'>

        See Also
        ========

        to_sdm
        to_dense
        sympy.polys.matrices.ddm.DDM.to_sdm
        """
        return self.rep.to_ddm()

    def to_sdm(self):
        """
        Return a :class:`~.SDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> sdm = A.to_sdm()
        >>> sdm
        {0: {0: 1}, 1: {1: 2}}
        >>> type(sdm)
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        to_ddm
        to_sparse
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        return self.rep.to_sdm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Return a :class:`~.DFM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dense
        DFM
        """
        return self.rep.to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Return a :class:`~.DFM` or :class:`~.DDM` representation of *self*.

        Explanation
        ===========

        The :class:`~.DFM` representation can only be used if the ground types
        are ``flint`` and the ground domain is supported by ``python-flint``.
        This method will return a :class:`~.DFM` representation if possible,
        but will return a :class:`~.DDM` representation otherwise.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm_or_ddm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)  # Depends on the ground domain and ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm: Always return a :class:`~.DDM` representation.
        to_dfm: Returns a :class:`~.DFM` representation or raise an error.
        to_dense: Convert internally to a :class:`~.DFM` or :class:`~.DDM`
        DFM: The :class:`~.DFM` dense FLINT matrix representation.
        DDM: The Python :class:`~.DDM` dense domain matrix representation.
        """
        return self.rep.to_dfm_or_ddm()

    @classmethod
    def _unify_domain(cls, *matrices):
        """Convert matrices to a common domain"""
        domains = {matrix.domain for matrix in matrices}
        if len(domains) == 1:
            return matrices
        domain = reduce(lambda x, y: x.unify(y), domains)
        return tuple(matrix.convert_to(domain) for matrix in matrices)

    @classmethod
    def _unify_fmt(cls, *matrices, fmt=None):
        """Convert matrices to the same format.

        If all matrices have the same format, then return unmodified.
        Otherwise convert both to the preferred format given as *fmt* which
        should be 'dense' or 'sparse'.
        """
        formats = {matrix.rep.fmt for matrix in matrices}
        if len(formats) == 1:
            return matrices
        if fmt == 'sparse':
            return tuple(matrix.to_sparse() for matrix in matrices)
        elif fmt == 'dense':
            return tuple(matrix.to_dense() for matrix in matrices)
        else:
            raise ValueError("fmt should be 'sparse' or 'dense'")

    def unify(self, *others, fmt=None):
        """
        Unifies the domains and the format of self and other
        matrices.

        Parameters
        ==========

        others : DomainMatrix

        fmt: string 'dense', 'sparse' or `None` (default)
            The preferred format to convert to if self and other are not
            already in the same format. If `None` or not specified then no
            conversion if performed.

        Returns
        =======

        Tuple[DomainMatrix]
            Matrices with unified domain and format

        Examples
        ========

        Unify the domain of DomainMatrix that have different domains:

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix([[QQ(1, 2), QQ(2)]], (1, 2), QQ)
        >>> Aq, Bq = A.unify(B)
        >>> Aq
        DomainMatrix([[1, 2]], (1, 2), QQ)
        >>> Bq
        DomainMatrix([[1/2, 2]], (1, 2), QQ)

        Unify the format (dense or sparse):

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix({0:{0: ZZ(1)}}, (2, 2), ZZ)
        >>> B.rep
        {0: {0: 1}}

        >>> A2, B2 = A.unify(B, fmt='dense')
        >>> B2.rep
        [[1, 0], [0, 0]]

        See Also
        ========

        convert_to, to_dense, to_sparse

        """
        matrices = (self,) + others
        matrices = DomainMatrix._unify_domain(*matrices)
        if fmt is not None:
            matrices = DomainMatrix._unify_fmt(*matrices, fmt=fmt)
        return matrices

    def to_Matrix(self):
        r"""
        Convert DomainMatrix to Matrix

        Returns
        =======

        Matrix
            MutableDenseMatrix for the DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_Matrix()
        Matrix([
            [1, 2],
            [3, 4]])

        See Also
        ========

        from_Matrix

        """
        from sympy.matrices.dense import MutableDenseMatrix

        # XXX: If the internal representation of RepMatrix changes then this
        # might need to be changed also.
        if self.domain in (ZZ, QQ, EXRAW):
            if self.rep.fmt == "sparse":
                rep = self.copy()
            else:
                rep = self.to_sparse()
        else:
            rep = self.convert_to(EXRAW).to_sparse()

        return MutableDenseMatrix._fromrep(rep)

    def to_list(self):
        """
        Convert :class:`DomainMatrix` to list of lists.

        See Also
        ========

        from_list
        to_list_flat
        to_flat_nz
        to_dok
        """
        return self.rep.to_list()

    def to_list_flat(self):
        """
        Convert :class:`DomainMatrix` to flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]

        See Also
        ========

        from_list_flat
        to_list
        to_flat_nz
        to_dok
        """
        return self.rep.to_list_flat()

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :class:`DomainMatrix` from flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> element_list = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
        >>> A = DomainMatrix.from_list_flat(element_list, (2, 2), ZZ)
        >>> A
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        """
        ddm = DDM.from_list_flat(elements, shape, domain)
        return cls.from_rep(ddm.to_dfm_or_ddm())

    def to_flat_nz(self):
        """
        Convert :class:`DomainMatrix` to list of nonzero elements and data.

        Explanation
        ===========

        Returns a tuple ``(elements, data)`` where ``elements`` is a list of
        elements of the matrix with zeros possibly excluded. The matrix can be
        reconstructed by passing these to :meth:`from_flat_nz`. The idea is to
        be able to modify a flat list of the elements and then create a new
        matrix of the same shape with the modified elements in the same
        positions.

        The format of ``data`` differs depending on whether the underlying
        representation is dense or sparse but either way it represents the
        positions of the elements in the list in a way that
        :meth:`from_flat_nz` can use to reconstruct the matrix. The
        :meth:`from_flat_nz` method should be called on the same
        :class:`DomainMatrix` that was used to call :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        Create a matrix with the elements doubled:

        >>> elements_doubled = [2*x for x in elements]
        >>> A2 = A.from_flat_nz(elements_doubled, data, A.domain)
        >>> A2 == 2*A
        True

        See Also
        ========

        from_flat_nz
        """
        return self.rep.to_flat_nz()

    def from_flat_nz(self, elements, data, domain):
        """
        Reconstruct :class:`DomainMatrix` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        """
        rep = self.rep.from_flat_nz(elements, data, domain)
        return self.from_rep(rep)

    def to_dod(self):
        """
        Convert :class:`DomainMatrix` to dictionary of dictionaries (dod) format.

        Explanation
        ===========

        Returns a dictionary of dictionaries representing the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2), ZZ(0)], [ZZ(3), ZZ(0), ZZ(4)]], ZZ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 2: 4}}
        >>> A.to_sparse() == A.from_dod(A.to_dod(), A.shape, A.domain)
        True
        >>> A == A.from_dod_like(A.to_dod())
        True

        See Also
        ========

        from_dod
        from_dod_like
        to_dok
        to_list
        to_list_flat
        to_flat_nz
        sympy.matrices.matrixbase.MatrixBase.todod
        """
        return self.rep.to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create sparse :class:`DomainMatrix` from dict of dict (dod) format.

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod_like
        """
        return cls.from_rep(SDM.from_dod(dod, shape, domain))

    def from_dod_like(self, dod, domain=None):
        """
        Create :class:`DomainMatrix` like ``self`` from dict of dict (dod) format.

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod
        """
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.from_dod(dod, self.shape, domain))

    def to_dok(self):
        """
        Convert :class:`DomainMatrix` to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(0)],
        ...    [ZZ(0), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_dok()
        {(0, 0): 1, (1, 1): 4}

        The matrix can be reconstructed by calling :meth:`from_dok` although
        the reconstructed matrix will always be in sparse format:

        >>> A.to_sparse() == A.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_flat_nz
        """
        return self.rep.to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :class:`DomainMatrix` from dictionary of keys (dok) format.

        See :meth:`to_dok` for explanation.

        See Also
        ========

        to_dok
        """
        return cls.from_rep(SDM.from_dok(dok, shape, domain))

    def iter_values(self):
        """
        Iterate over nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.matrices.matrixbase.MatrixBase.iter_values
        """
        return self.rep.iter_values()

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.matrices.matrixbase.MatrixBase.iter_items
        """
        return self.rep.iter_items()

    def nnz(self):
        """
        Number of nonzero elements in the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[1, 0], [0, 4]], ZZ)
        >>> A.nnz()
        2
        """
        return self.rep.nnz()

    def __repr__(self):
        return 'DomainMatrix(%s, %r, %r)' % (str(self.rep), self.shape, self.domain)

    def transpose(self):
        """Matrix transpose of ``self``"""
        return self.from_rep(self.rep.transpose())

    def flat(self):
        rows, cols = self.shape
        return [self[i,j].element for i in range(rows) for j in range(cols)]

    @property
    def is_zero_matrix(self):
        return self.rep.is_zero_matrix()

    @property
    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        return self.rep.is_upper()

    @property
    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        return self.rep.is_lower()

    @property
    def is_diagonal(self):
        """
        True if the matrix is diagonal.

        Can return true for non-square matrices. A matrix is diagonal if
        ``M[i,j] == 0`` whenever ``i != j``.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], ZZ)
        >>> M.is_diagonal
        True

        See Also
        ========

        is_upper
        is_lower
        is_square
        diagonal
        """
        return self.rep.is_diagonal()

    def diagonal(self):
        """
        Get the diagonal entries of the matrix as a list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> M.diagonal()
        [1, 4]

        See Also
        ========

        is_diagonal
        diag
        """
        return self.rep.diagonal()

    @property
    def is_square(self):
        """
        True if the matrix is square.
        """
        return self.shape[0] == self.shape[1]

    def rank(self):
        rref, pivots = self.rref()
        return len(pivots)

    def hstack(A, *B):
        r"""Horizontally stack the given matrices.

        Parameters
        ==========

        B: DomainMatrix
            Matrices to stack horizontally.

        Returns
        =======

        DomainMatrix
            DomainMatrix by stacking horizontally.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        DomainMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], (2, 4), ZZ)

        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        DomainMatrix([[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]], (2, 6), ZZ)

        See Also
        ========

        unify
        """
        A, *B = A.unify(*B, fmt=A.rep.fmt)
        return DomainMatrix.from_rep(A.rep.hstack(*(Bk.rep for Bk in B)))

    def vstack(A, *B):
        r"""Vertically stack the given matrices.

        Parameters
        ==========

        B: DomainMatrix
            Matrices to stack vertically.

        Returns
        =======

        DomainMatrix
            DomainMatrix by stacking vertically.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8]], (4, 2), ZZ)

        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], (6, 2), ZZ)

        See Also
        ========

        unify
        """
        A, *B = A.unify(*B, fmt='dense')
        return DomainMatrix.from_rep(A.rep.vstack(*(Bk.rep for Bk in B)))

    def applyfunc(self, func, domain=None):
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.applyfunc(func, domain))

    def __add__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        A, B = A.unify(B, fmt='dense')
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        A, B = A.unify(B, fmt='dense')
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, DomainMatrix):
            A, B = A.unify(B, fmt='dense')
            return A.matmul(B)
        elif B in A.domain:
            return A.scalarmul(B)
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.scalarmul(B.element)
        else:
            return NotImplemented

    def __rmul__(A, B):
        if B in A.domain:
            return A.rscalarmul(B)
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.rscalarmul(B.element)
        else:
            return NotImplemented

    def __pow__(A, n):
        """A ** n"""
        if not isinstance(n, int):
            return NotImplemented
        return A.pow(n)

    def _check(a, op, b, ashape, bshape):
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)
        if a.rep.fmt != b.rep.fmt:
            msg = "Format mismatch: %s %s %s" % (a.rep.fmt, op, b.rep.fmt)
            raise DMFormatError(msg)
        if type(a.rep) != type(b.rep):
            msg = "Type mismatch: %s %s %s" % (type(a.rep), op, type(b.rep))
            raise DMFormatError(msg)

    def add(A, B):
        r"""
        Adds two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to add

        Returns
        =======

        DomainMatrix
            DomainMatrix after Addition

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.add(B)
        DomainMatrix([[5, 5], [5, 5]], (2, 2), ZZ)

        See Also
        ========

        sub, matmul

        """
        A._check('+', B, A.shape, B.shape)
        return A.from_rep(A.rep.add(B.rep))


    def sub(A, B):
        r"""
        Subtracts two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to subtract

        Returns
        =======

        DomainMatrix
            DomainMatrix after Subtraction

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.sub(B)
        DomainMatrix([[-3, -1], [1, 3]], (2, 2), ZZ)

        See Also
        ========

        add, matmul

        """
        A._check('-', B, A.shape, B.shape)
        return A.from_rep(A.rep.sub(B.rep))

    def neg(A):
        r"""
        Returns the negative of DomainMatrix

        Parameters
        ==========

        A : Represents a DomainMatrix

        Returns
        =======

        DomainMatrix
            DomainMatrix after Negation

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.neg()
        DomainMatrix([[-1, -2], [-3, -4]], (2, 2), ZZ)

        """
        return A.from_rep(A.rep.neg())

    def mul(A, b):
        r"""
        Performs term by term multiplication for the second DomainMatrix
        w.r.t first DomainMatrix. Returns a DomainMatrix whose rows are
        list of DomainMatrix matrices created after term by term multiplication.

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to multiply term-wise

        Returns
        =======

        DomainMatrix
            DomainMatrix after term by term multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> b = ZZ(2)

        >>> A.mul(b)
        DomainMatrix([[2, 4], [6, 8]], (2, 2), ZZ)

        See Also
        ========

        matmul

        """
        return A.from_rep(A.rep.mul(b))

    def rmul(A, b):
        return A.from_rep(A.rep.rmul(b))

    def matmul(A, B):
        r"""
        Performs matrix multiplication of two DomainMatrix matrices

        Parameters
        ==========

        A, B: DomainMatrix
            to multiply

        Returns
        =======

        DomainMatrix
            DomainMatrix after multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.matmul(B)
        DomainMatrix([[1, 3], [3, 7]], (2, 2), ZZ)

        See Also
        ========

        mul, pow, add, sub

        """

        A._check('*', B, A.shape[1], B.shape[0])
        return A.from_rep(A.rep.matmul(B.rep))

    def _scalarmul(A, lamda, reverse):
        if lamda == A.domain.zero:
            return DomainMatrix.zeros(A.shape, A.domain)
        elif lamda == A.domain.one:
            return A.copy()
        elif reverse:
            return A.rmul(lamda)
        else:
            return A.mul(lamda)

    def scalarmul(A, lamda):
        return A._scalarmul(lamda, reverse=False)

    def rscalarmul(A, lamda):
        return A._scalarmul(lamda, reverse=True)

    def mul_elementwise(A, B):
        assert A.domain == B.domain
        return A.from_rep(A.rep.mul_elementwise(B.rep))

    def __truediv__(A, lamda):
        """ Method for Scalar Division"""
        if isinstance(lamda, int) or ZZ.of_type(lamda):
            lamda = DomainScalar(ZZ(lamda), ZZ)
        elif A.domain.is_Field and lamda in A.domain:
            K = A.domain
            lamda = DomainScalar(K.convert(lamda), K)

        if not isinstance(lamda, DomainScalar):
            return NotImplemented

        A, lamda = A.to_field().unify(lamda)
        if lamda.element == lamda.domain.zero:
            raise ZeroDivisionError
        if lamda.element == lamda.domain.one:
            return A

        return A.mul(1 / lamda.element)

    def pow(A, n):
        r"""
        Computes A**n

        Parameters
        ==========

        A : DomainMatrix

        n : exponent for A

        Returns
        =======

        DomainMatrix
            DomainMatrix on computing A**n

        Raises
        ======

        NotImplementedError
            if n is negative.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.pow(2)
        DomainMatrix([[1, 2], [0, 1]], (2, 2), ZZ)

        See Also
        ========

        matmul

        """
        nrows, ncols = A.shape
        if nrows != ncols:
            raise DMNonSquareMatrixError('Power of a nonsquare matrix')
        if n < 0:
            raise NotImplementedError('Negative powers')
        elif n == 0:
            return A.eye(nrows, A.domain)
        elif n == 1:
            return A
        elif n % 2 == 1:
            return A * A**(n - 1)
        else:
            sqrtAn = A ** (n // 2)
            return sqrtAn * sqrtAn

    def scc(self):
        """Compute the strongly connected components of a DomainMatrix

        Explanation
        ===========

        A square matrix can be considered as the adjacency matrix for a
        directed graph where the row and column indices are the vertices. In
        this graph if there is an edge from vertex ``i`` to vertex ``j`` if
        ``M[i, j]`` is nonzero. This routine computes the strongly connected
        components of that graph which are subsets of the rows and columns that
        are connected by some nonzero element of the matrix. The strongly
        connected components are useful because many operations such as the
        determinant can be computed by working with the submatrices
        corresponding to each component.

        Examples
        ========

        Find the strongly connected components of a matrix:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = DomainMatrix([[ZZ(1), ZZ(0), ZZ(2)],
        ...                   [ZZ(0), ZZ(3), ZZ(0)],
        ...                   [ZZ(4), ZZ(6), ZZ(5)]], (3, 3), ZZ)
        >>> M.scc()
        [[1], [0, 2]]

        Compute the determinant from the components:

        >>> MM = M.to_Matrix()
        >>> MM
        Matrix([
        [1, 0, 2],
        [0, 3, 0],
        [4, 6, 5]])
        >>> MM[[1], [1]]
        Matrix([[3]])
        >>> MM[[0, 2], [0, 2]]
        Matrix([
        [1, 2],
        [4, 5]])
        >>> MM.det()
        -9
        >>> MM[[1], [1]].det() * MM[[0, 2], [0, 2]].det()
        -9

        The components are given in reverse topological order and represent a
        permutation of the rows and columns that will bring the matrix into
        block lower-triangular form:

        >>> MM[[1, 0, 2], [1, 0, 2]]
        Matrix([
        [3, 0, 0],
        [0, 1, 2],
        [6, 4, 5]])

        Returns
        =======

        List of lists of integers
            Each list represents a strongly connected component.

        See also
        ========

        sympy.matrices.matrixbase.MatrixBase.strongly_connected_components
        sympy.utilities.iterables.strongly_connected_components

        """
        if not self.is_square:
            raise DMNonSquareMatrixError('Matrix must be square for scc')

        return self.rep.scc()

    def clear_denoms(self, convert=False):
        """
        Clear denominators, but keep the domain unchanged.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3)], [(1,4), (1,5)]], QQ)
        >>> den, Anum = A.clear_denoms()
        >>> den.to_sympy()
        60
        >>> Anum.to_Matrix()
        Matrix([
        [30, 20],
        [15, 12]])
        >>> den * A == Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms()[1].domain
        QQ
        >>> A.clear_denoms(convert=True)[1].domain
        ZZ

        The denominator is always in the associated ring:

        >>> A.clear_denoms()[0].domain
        ZZ
        >>> A.domain.get_ring()
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms_rowwise
        """
        elems0, data = self.to_flat_nz()

        K0 = self.domain
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        den, elems1 = dup_clear_denoms(elems0, K0, K1, convert=convert)

        if convert:
            Kden, Knum = K1, K1
        else:
            Kden, Knum = K1, K0

        den = DomainScalar(den, Kden)
        num = self.from_flat_nz(elems1, data, Knum)

        return den, num

    def clear_denoms_rowwise(self, convert=False):
        """
        Clear denominators from each row of the matrix.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3), (1,4)], [(1,5), (1,6), (1,7)]], QQ)
        >>> den, Anum = A.clear_denoms_rowwise()
        >>> den.to_Matrix()
        Matrix([
        [12,   0],
        [ 0, 210]])
        >>> Anum.to_Matrix()
        Matrix([
        [ 6,  4,  3],
        [42, 35, 30]])

        The denominator matrix is a diagonal matrix with the denominators of
        each row on the diagonal. The invariants are:

        >>> den * A == Anum
        True
        >>> A == den.to_field().inv() * Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms_rowwise()[1].domain
        QQ
        >>> A.clear_denoms_rowwise(convert=True)[1].domain
        ZZ

        The domain of the denominator matrix is the associated ring:

        >>> A.clear_denoms_rowwise()[0].domain
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms
        """
        dod = self.to_dod()

        K0 = self.domain
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        diagonals = [K0.one] * self.shape[0]
        dod_num = {}
        for i, rowi in dod.items():
            indices, elems = zip(*rowi.items())
            den, elems_num = dup_clear_denoms(elems, K0, K1, convert=convert)
            rowi_num = dict(zip(indices, elems_num))
            diagonals[i] = den
            dod_num[i] = rowi_num

        if convert:
            Kden, Knum = K1, K1
        else:
            Kden, Knum = K1, K0

        den = self.diag(diagonals, Kden)
        num = self.from_dod_like(dod_num, Knum)

        return den, num

    def cancel_denom(self, denom):
        """
        Cancel factors between a matrix and a denominator.

        Returns a matrix and denominator on lowest terms.

        Requires ``gcd`` in the ground domain.

        Methods like :meth:`solve_den`, :meth:`inv_den` and :meth:`rref_den`
        return a matrix and denominator but not necessarily on lowest terms.
        Reduction to lowest terms without fractions can be performed with
        :meth:`cancel_denom`.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 2, 0],
        ...         [0, 2, 2],
        ...         [0, 0, 2]], ZZ)
        >>> Minv, den = M.inv_den()
        >>> Minv.to_Matrix()
        Matrix([
        [1, -1,  1],
        [0,  1, -1],
        [0,  0,  1]])
        >>> den
        2
        >>> Minv_reduced, den_reduced = Minv.cancel_denom(den)
        >>> Minv_reduced.to_Matrix()
        Matrix([
        [1, -1,  1],
        [0,  1, -1],
        [0,  0,  1]])
        >>> den_reduced
        2
        >>> Minv_reduced.to_field() / den_reduced == Minv.to_field() / den
        True

        The denominator is made canonical with respect to units (e.g. a
        negative denominator is made positive):

        >>> M = DM([[2, 2, 0]], ZZ)
        >>> den = ZZ(-4)
        >>> M.cancel_denom(den)
        (DomainMatrix([[-1, -1, 0]], (1, 3), ZZ), 2)

        Any factor common to _all_ elements will be cancelled but there can
        still be factors in common between _some_ elements of the matrix and
        the denominator. To cancel factors between each element and the
        denominator, use :meth:`cancel_denom_elementwise` or otherwise convert
        to a field and use division:

        >>> M = DM([[4, 6]], ZZ)
        >>> den = ZZ(12)
        >>> M.cancel_denom(den)
        (DomainMatrix([[2, 3]], (1, 2), ZZ), 6)
        >>> numers, denoms = M.cancel_denom_elementwise(den)
        >>> numers
        DomainMatrix([[1, 1]], (1, 2), ZZ)
        >>> denoms
        DomainMatrix([[3, 2]], (1, 2), ZZ)
        >>> M.to_field() / den
        DomainMatrix([[1/3, 1/2]], (1, 2), QQ)

        See Also
        ========

        solve_den
        inv_den
        rref_den
        cancel_denom_elementwise
        """
        M = self
        K = self.domain

        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            return (M.copy(), denom)

        elements, data = M.to_flat_nz()

        # First canonicalize the denominator (e.g. multiply by -1).
        if K.is_negative(denom):
            u = -K.one
        else:
            u = K.canonical_unit(denom)

        # Often after e.g. solve_den the denominator will be much more
        # complicated than the elements of the numerator. Hopefully it will be
        # quicker to find the gcd of the numerator and if there is no content
        # then we do not need to look at the denominator at all.
        content = dup_content(elements, K)
        common = K.gcd(content, denom)

        if not K.is_one(content):

            common = K.gcd(content, denom)

            if not K.is_one(common):
                elements = dup_quo_ground(elements, common, K)
                denom = K.quo(denom, common)

        if not K.is_one(u):
            elements = dup_mul_ground(elements, u, K)
            denom = u * denom
        elif K.is_one(common):
            return (M.copy(), denom)

        M_cancelled = M.from_flat_nz(elements, data, K)

        return M_cancelled, denom

    def cancel_denom_elementwise(self, denom):
        """
        Cancel factors between the elements of a matrix and a denominator.

        Returns a matrix of numerators and matrix of denominators.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 3], [4, 12]], ZZ)
        >>> denom = ZZ(6)
        >>> numers, denoms = M.cancel_denom_elementwise(denom)
        >>> numers.to_Matrix()
        Matrix([
        [1, 1],
        [2, 2]])
        >>> denoms.to_Matrix()
        Matrix([
        [3, 2],
        [3, 1]])
        >>> M_frac = (M.to_field() / denom).to_Matrix()
        >>> M_frac
        Matrix([
        [1/3, 1/2],
        [2/3,   2]])
        >>> denoms_inverted = denoms.to_Matrix().applyfunc(lambda e: 1/e)
        >>> numers.to_Matrix().multiply_elementwise(denoms_inverted) == M_frac
        True

        Use :meth:`cancel_denom` to cancel factors between the matrix and the
        denominator while preserving the form of a matrix with a scalar
        denominator.

        See Also
        ========

        cancel_denom
        """
        K = self.domain
        M = self

        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            M_numers = M.copy()
            M_denoms = M.ones(M.shape, M.domain)
            return (M_numers, M_denoms)

        elements, data = M.to_flat_nz()

        cofactors = [K.cofactors(numer, denom) for numer in elements]
        gcds, numers, denoms = zip(*cofactors)

        M_numers = M.from_flat_nz(list(numers), data, K)
        M_denoms = M.from_flat_nz(list(denoms), data, K)

        return (M_numers, M_denoms)

    def content(self):
        """
        Return the gcd of the elements of the matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> M.content()
        2

        See Also
        ========

        primitive
        cancel_denom
        """
        K = self.domain
        elements, _ = self.to_flat_nz()
        return dup_content(elements, K)

    def primitive(self):
        """
        Factor out gcd of the elements of a matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> content, M_primitive = M.primitive()
        >>> content
        2
        >>> M_primitive
        DomainMatrix([[1, 2], [2, 6]], (2, 2), ZZ)
        >>> content * M_primitive == M
        True
        >>> M_primitive.content() == ZZ(1)
        True

        See Also
        ========

        content
        cancel_denom
        """
        K = self.domain
        elements, data = self.to_flat_nz()
        content, prims = dup_primitive(elements, K)
        M_primitive = self.from_flat_nz(prims, data, K)
        return content, M_primitive

    def rref(self, *, method='auto'):
        r"""
        Returns reduced-row echelon form (RREF) and list of pivots.

        If the domain is not a field then it will be converted to a field. See
        :meth:`rref_den` for the fraction-free version of this routine that
        returns RREF with denominator instead.

        The domain must either be a field or have an associated fraction field
        (see :meth:`to_field`).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)

        >>> rref_matrix, rref_pivots = A.rref()
        >>> rref_matrix
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> rref_pivots
        (0, 1, 2)

        Parameters
        ==========

        method : str, optional (default: 'auto')
            The method to use to compute the RREF. The default is ``'auto'``,
            which will attempt to choose the fastest method. The other options
            are:

            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with
              division. If the domain is not a field then it will be converted
              to a field with :meth:`to_field` first and RREF will be computed
              by inverting the pivot elements in each row. This is most
              efficient for very sparse matrices or for matrices whose elements
              have complex denominators.

            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan
              elimination. Elimination is performed using exact division
              (``exquo``) to control the growth of the coefficients. In this
              case the current domain is always used for elimination but if
              the domain is not a field then it will be converted to a field
              at the end and divided by the denominator. This is most efficient
              for dense matrices or for matrices with simple denominators.

            - ``A.rref(method='CD')`` clears the denominators before using
              fraction-free Gauss-Jordan elimination in the assoicated ring.
              This is most efficient for dense matrices with very simple
              denominators.

            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and
              ``A.rref(method='CD_dense')`` are the same as the above methods
              except that the dense implementations of the algorithms are used.
              By default ``A.rref(method='auto')`` will usually choose the
              sparse implementations for RREF.

            Regardless of which algorithm is used the returned matrix will
            always have the same format (sparse or dense) as the input and its
            domain will always be the field of fractions of the input domain.

        Returns
        =======

        (DomainMatrix, list)
            reduced-row echelon form and list of pivots for the DomainMatrix

        See Also
        ========

        rref_den
            RREF with denominator
        sympy.polys.matrices.sdm.sdm_irref
            Sparse implementation of ``method='GJ'``.
        sympy.polys.matrices.sdm.sdm_rref_den
            Sparse implementation of ``method='FF'`` and ``method='CD'``.
        sympy.polys.matrices.dense.ddm_irref
            Dense implementation of ``method='GJ'``.
        sympy.polys.matrices.dense.ddm_irref_den
            Dense implementation of ``method='FF'`` and ``method='CD'``.
        clear_denoms
            Clear denominators from a matrix, used by ``method='CD'`` and
            by ``method='GJ'`` when the original domain is not a field.

        """
        return _dm_rref(self, method=method)

    def rref_den(self, *, method='auto', keep_domain=True):
        r"""
        Returns reduced-row echelon form with denominator and list of pivots.

        Requires exact division in the ground domain (``exquo``).

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)

        >>> A_rref, denom, pivots = A.rref_den()
        >>> A_rref
        DomainMatrix([[6, 0, 0], [0, 6, 0], [0, 0, 6]], (3, 3), ZZ)
        >>> denom
        6
        >>> pivots
        (0, 1, 2)
        >>> A_rref.to_field() / denom
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> A_rref.to_field() / denom == A.convert_to(QQ).rref()[0]
        True

        Parameters
        ==========

        method : str, optional (default: 'auto')
            The method to use to compute the RREF. The default is ``'auto'``,
            which will attempt to choose the fastest method. The other options
            are:

            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan
              elimination. Elimination is performed using exact division
              (``exquo``) to control the growth of the coefficients. In this
              case the current domain is always used for elimination and the
              result is always returned as a matrix over the current domain.
              This is most efficient for dense matrices or for matrices with
              simple denominators.

            - ``A.rref(method='CD')`` clears denominators before using
              fraction-free Gauss-Jordan elimination in the assoicated ring.
              The result will be converted back to the original domain unless
              ``keep_domain=False`` is passed in which case the result will be
              over the ring used for elimination. This is most efficient for
              dense matrices with very simple denominators.

            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with
              division. If the domain is not a field then it will be converted
              to a field with :meth:`to_field` first and RREF will be computed
              by inverting the pivot elements in each row. The result is
              converted back to the original domain by clearing denominators
              unless ``keep_domain=False`` is passed in which case the result
              will be over the field used for elimination. This is most
              efficient for very sparse matrices or for matrices whose elements
              have complex denominators.

            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and
              ``A.rref(method='CD_dense')`` are the same as the above methods
              except that the dense implementations of the algorithms are used.
              By default ``A.rref(method='auto')`` will usually choose the
              sparse implementations for RREF.

            Regardless of which algorithm is used the returned matrix will
            always have the same format (sparse or dense) as the input and if
            ``keep_domain=True`` its domain will always be the same as the
            input.

        keep_domain : bool, optional
            If True (the default), the domain of the returned matrix and
            denominator are the same as the domain of the input matrix. If
            False, the domain of the returned matrix might be changed to an
            associated ring or field if the algorithm used a different domain.
            This is useful for efficiency if the caller does not need the
            result to be in the original domain e.g. it avoids clearing
            denominators in the case of ``A.rref(method='GJ')``.

        Returns
        =======

        (DomainMatrix, scalar, list)
            Reduced-row echelon form, denominator and list of pivot indices.

        See Also
        ========

        rref
            RREF without denominator for field domains.
        sympy.polys.matrices.sdm.sdm_irref
            Sparse implementation of ``method='GJ'``.
        sympy.polys.matrices.sdm.sdm_rref_den
            Sparse implementation of ``method='FF'`` and ``method='CD'``.
        sympy.polys.matrices.dense.ddm_irref
            Dense implementation of ``method='GJ'``.
        sympy.polys.matrices.dense.ddm_irref_den
            Dense implementation of ``method='FF'`` and ``method='CD'``.
        clear_denoms
            Clear denominators from a matrix, used by ``method='CD'``.

        """
        return _dm_rref_den(self, method=method, keep_domain=keep_domain)

    def columnspace(self):
        r"""
        Returns the columnspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The columns of this matrix form a basis for the columnspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.columnspace()
        DomainMatrix([[1], [2]], (2, 1), QQ)

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        rref, pivots = self.rref()
        rows, cols = self.shape
        return self.extract(range(rows), pivots)

    def rowspace(self):
        r"""
        Returns the rowspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The rows of this matrix form a basis for the rowspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.rowspace()
        DomainMatrix([[1, -1]], (1, 2), QQ)

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        rref, pivots = self.rref()
        rows, cols = self.shape
        return self.extract(range(len(pivots)), range(cols))

    def nullspace(self, divide_last=False):
        r"""
        Returns the nullspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The rows of this matrix form a basis for the nullspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([
        ...    [QQ(2), QQ(-2)],
        ...    [QQ(4), QQ(-4)]], QQ)
        >>> A.nullspace()
        DomainMatrix([[1, 1]], (1, 2), QQ)

        The returned matrix is a basis for the nullspace:

        >>> A_null = A.nullspace().transpose()
        >>> A * A_null
        DomainMatrix([[0], [0]], (2, 1), QQ)
        >>> rows, cols = A.shape
        >>> nullity = rows - A.rank()
        >>> A_null.shape == (cols, nullity)
        True

        Nullspace can also be computed for non-field rings. If the ring is not
        a field then division is not used. Setting ``divide_last`` to True will
        raise an error in this case:

        >>> from sympy import ZZ
        >>> B = DM([[6, -3],
        ...         [4, -2]], ZZ)
        >>> B.nullspace()
        DomainMatrix([[3, 6]], (1, 2), ZZ)
        >>> B.nullspace(divide_last=True)
        Traceback (most recent call last):
        ...
        DMNotAField: Cannot normalize vectors over a non-field

        Over a ring with ``gcd`` defined the nullspace can potentially be
        reduced with :meth:`primitive`:

        >>> B.nullspace().primitive()
        (3, DomainMatrix([[1, 2]], (1, 2), ZZ))

        A matrix over a ring can often be normalized by converting it to a
        field but it is often a bad idea to do so:

        >>> from sympy.abc import a, b, c
        >>> from sympy import Matrix
        >>> M = Matrix([[        a*b,       b + c,        c],
        ...             [      a - b,         b*c,     c**2],
        ...             [a*b + a - b, b*c + b + c, c**2 + c]])
        >>> M.to_DM().domain
        ZZ[a,b,c]
        >>> M.to_DM().nullspace().to_Matrix().transpose()
        Matrix([
        [                             c**3],
        [            -a*b*c**2 + a*c - b*c],
        [a*b**2*c - a*b - a*c + b**2 + b*c]])

        The unnormalized form here is nicer than the normalized form that
        spreads a large denominator throughout the matrix:

        >>> M.to_DM().to_field().nullspace(divide_last=True).to_Matrix().transpose()
        Matrix([
        [                   c**3/(a*b**2*c - a*b - a*c + b**2 + b*c)],
        [(-a*b*c**2 + a*c - b*c)/(a*b**2*c - a*b - a*c + b**2 + b*c)],
        [                                                          1]])

        Parameters
        ==========

        divide_last : bool, optional
            If False (the default), the vectors are not normalized and the RREF
            is computed using :meth:`rref_den` and the denominator is
            discarded. If True, then each row is divided by its final element;
            the domain must be a field in this case.

        See Also
        ========

        nullspace_from_rref
        rref
        rref_den
        rowspace
        """
        A = self
        K = A.domain

        if divide_last and not K.is_Field:
            raise DMNotAField("Cannot normalize vectors over a non-field")

        if divide_last:
            A_rref, pivots = A.rref()
        else:
            A_rref, den, pivots = A.rref_den()

            # Ensure that the sign is canonical before discarding the
            # denominator. Then M.nullspace().primitive() is canonical.
            u = K.canonical_unit(den)
            if u != K.one:
                A_rref *= u

        A_null = A_rref.nullspace_from_rref(pivots)

        return A_null

    def nullspace_from_rref(self, pivots=None):
        """
        Compute nullspace from rref and pivots.

        The domain of the matrix can be any domain.

        The matrix must be in reduced row echelon form already. Otherwise the
        result will be incorrect. Use :meth:`rref` or :meth:`rref_den` first
        to get the reduced row echelon form or use :meth:`nullspace` instead.

        See Also
        ========

        nullspace
        rref
        rref_den
        sympy.polys.matrices.sdm.SDM.nullspace_from_rref
        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
        """
        null_rep, nonpivots = self.rep.nullspace_from_rref(pivots)
        return self.from_rep(null_rep)

    def inv(self):
        r"""
        Finds the inverse of the DomainMatrix if exists

        Returns
        =======

        DomainMatrix
            DomainMatrix after inverse

        Raises
        ======

        ValueError
            If the domain of DomainMatrix not a Field

        DMNonSquareMatrixError
            If the DomainMatrix is not a not Square DomainMatrix

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> A.inv()
        DomainMatrix([[2/3, 1/3, 1/6], [1/3, 2/3, 1/3], [0, 0, 1/2]], (3, 3), QQ)

        See Also
        ========

        neg

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        m, n = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        inv = self.rep.inv()
        return self.from_rep(inv)

    def det(self):
        r"""
        Returns the determinant of a square :class:`DomainMatrix`.

        Returns
        =======

        determinant: DomainElement
            Determinant of the matrix.

        Raises
        ======

        ValueError
            If the domain of DomainMatrix is not a Field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.det()
        -2

        """
        m, n = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        return self.rep.det()

    def adj_det(self):
        """
        Adjugate and determinant of a square :class:`DomainMatrix`.

        Returns
        =======

        (adjugate, determinant) : (DomainMatrix, DomainScalar)
            The adjugate matrix and determinant of this matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([
        ...     [ZZ(1), ZZ(2)],
        ...     [ZZ(3), ZZ(4)]], ZZ)
        >>> adjA, detA = A.adj_det()
        >>> adjA
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)
        >>> detA
        -2

        See Also
        ========

        adjugate
            Returns only the adjugate matrix.
        det
            Returns only the determinant.
        inv_den
            Returns a matrix/denominator pair representing the inverse matrix
            but perhaps differing from the adjugate and determinant by a common
            factor.
        """
        m, n = self.shape
        I_m = self.eye((m, m), self.domain)
        adjA, detA = self.solve_den_charpoly(I_m, check=False)
        if self.rep.fmt == "dense":
            adjA = adjA.to_dense()
        return adjA, detA

    def adjugate(self):
        """
        Adjugate of a square :class:`DomainMatrix`.

        The adjugate matrix is the transpose of the cofactor matrix and is
        related to the inverse by::

            adj(A) = det(A) * A.inv()

        Unlike the inverse matrix the adjugate matrix can be computed and
        expressed without division or fractions in the ground domain.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> A.adjugate()
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)

        Returns
        =======

        DomainMatrix
            The adjugate matrix of this matrix with the same domain.

        See Also
        ========

        adj_det
        """
        adjA, detA = self.adj_det()
        return adjA

    def inv_den(self, method=None):
        """
        Return the inverse as a :class:`DomainMatrix` with denominator.

        Returns
        =======

        (inv, den) : (:class:`DomainMatrix`, :class:`~.DomainElement`)
            The inverse matrix and its denominator.

        This is more or less equivalent to :meth:`adj_det` except that ``inv``
        and ``den`` are not guaranteed to be the adjugate and inverse. The
        ratio ``inv/den`` is equivalent to ``adj/det`` but some factors
        might be cancelled between ``inv`` and ``den``. In simple cases this
        might just be a minus sign so that ``(inv, den) == (-adj, -det)`` but
        factors more complicated than ``-1`` can also be cancelled.
        Cancellation is not guaranteed to be complete so ``inv`` and ``den``
        may not be on lowest terms. The denominator ``den`` will be zero if and
        only if the determinant is zero.

        If the actual adjugate and determinant are needed, use :meth:`adj_det`
        instead. If the intention is to compute the inverse matrix or solve a
        system of equations then :meth:`inv_den` is more efficient.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)
        >>> Ainv, den = A.inv_den()
        >>> den
        6
        >>> Ainv
        DomainMatrix([[4, 2, 1], [2, 4, 2], [0, 0, 3]], (3, 3), ZZ)
        >>> A * Ainv == den * A.eye(A.shape, A.domain).to_dense()
        True

        Parameters
        ==========

        method : str, optional
            The method to use to compute the inverse. Can be one of ``None``,
            ``'rref'`` or ``'charpoly'``. If ``None`` then the method is
            chosen automatically (see :meth:`solve_den` for details).

        See Also
        ========

        inv
        det
        adj_det
        solve_den
        """
        I = self.eye(self.shape, self.domain)
        return self.solve_den(I, method=method)

    def solve_den(self, b, method=None):
        """
        Solve matrix equation $Ax = b$ without fractions in the ground domain.

        Examples
        ========

        Solve a matrix equation over the integers:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, xden = A.solve_den(b)
        >>> xden
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == xden * b
        True

        Solve a matrix equation over a polynomial ring:

        >>> from sympy import ZZ
        >>> from sympy.abc import x, y, z, a, b
        >>> R = ZZ[x, y, z, a, b]
        >>> M = DM([[x*y, x*z], [y*z, x*z]], R)
        >>> b = DM([[a], [b]], R)
        >>> M.to_Matrix()
        Matrix([
        [x*y, x*z],
        [y*z, x*z]])
        >>> b.to_Matrix()
        Matrix([
        [a],
        [b]])
        >>> xnum, xden = M.solve_den(b)
        >>> xden
        x**2*y*z - x*y*z**2
        >>> xnum.to_Matrix()
        Matrix([
        [ a*x*z - b*x*z],
        [-a*y*z + b*x*y]])
        >>> M * xnum == xden * b
        True

        The solution can be expressed over a fraction field which will cancel
        gcds between the denominator and the elements of the numerator:

        >>> xsol = xnum.to_field() / xden
        >>> xsol.to_Matrix()
        Matrix([
        [           (a - b)/(x*y - y*z)],
        [(-a*z + b*x)/(x**2*z - x*z**2)]])
        >>> (M * xsol).to_Matrix() == b.to_Matrix()
        True

        When solving a large system of equations this cancellation step might
        be a lot slower than :func:`solve_den` itself. The solution can also be
        expressed as a ``Matrix`` without attempting any polynomial
        cancellation between the numerator and denominator giving a less
        simplified result more quickly:

        >>> xsol_uncancelled = xnum.to_Matrix() / xnum.domain.to_sympy(xden)
        >>> xsol_uncancelled
        Matrix([
        [ (a*x*z - b*x*z)/(x**2*y*z - x*y*z**2)],
        [(-a*y*z + b*x*y)/(x**2*y*z - x*y*z**2)]])
        >>> from sympy import cancel
        >>> cancel(xsol_uncancelled) == xsol.to_Matrix()
        True

        Parameters
        ==========

        self : :class:`DomainMatrix`
            The ``m x n`` matrix $A$ in the equation $Ax = b$. Underdetermined
            systems are not supported so ``m >= n``: $A$ should be square or
            have more rows than columns.
        b : :class:`DomainMatrix`
            The ``n x m`` matrix $b$ for the rhs.
        cp : list of :class:`~.DomainElement`, optional
            The characteristic polynomial of the matrix $A$. If not given, it
            will be computed using :meth:`charpoly`.
        method: str, optional
            The method to use for solving the system. Can be one of ``None``,
            ``'charpoly'`` or ``'rref'``. If ``None`` (the default) then the
            method will be chosen automatically.

            The ``charpoly`` method uses :meth:`solve_den_charpoly` and can
            only be used if the matrix is square. This method is division free
            and can be used with any domain.

            The ``rref`` method is fraction free but requires exact division
            in the ground domain (``exquo``). This is also suitable for most
            domains. This method can be used with overdetermined systems (more
            equations than unknowns) but not underdetermined systems as a
            unique solution is sought.

        Returns
        =======

        (xnum, xden) : (DomainMatrix, DomainElement)
            The solution of the equation $Ax = b$ as a pair consisting of an
            ``n x m`` matrix numerator ``xnum`` and a scalar denominator
            ``xden``.

        The solution $x$ is given by ``x = xnum / xden``. The division free
        invariant is ``A * xnum == xden * b``. If $A$ is square then the
        denominator ``xden`` will be a divisor of the determinant $det(A)$.

        Raises
        ======

        DMNonInvertibleMatrixError
            If the system $Ax = b$ does not have a unique solution.

        See Also
        ========

        solve_den_charpoly
        solve_den_rref
        inv_den
        """
        m, n = self.shape
        bm, bn = b.shape

        if m != bm:
            raise DMShapeError("Matrix equation shape mismatch.")

        if method is None:
            method = 'rref'
        elif method == 'charpoly' and m != n:
            raise DMNonSquareMatrixError("method='charpoly' requires a square matrix.")

        if method == 'charpoly':
            xnum, xden = self.solve_den_charpoly(b)
        elif method == 'rref':
            xnum, xden = self.solve_den_rref(b)
        else:
            raise DMBadInputError("method should be 'rref' or 'charpoly'")

        return xnum, xden

    def solve_den_rref(self, b):
        """
        Solve matrix equation $Ax = b$ using fraction-free RREF

        Solves the matrix equation $Ax = b$ for $x$ and returns the solution
        as a numerator/denominator pair.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, xden = A.solve_den_rref(b)
        >>> xden
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == xden * b
        True

        See Also
        ========

        solve_den
        solve_den_charpoly
        """
        A = self
        m, n = A.shape
        bm, bn = b.shape

        if m != bm:
            raise DMShapeError("Matrix equation shape mismatch.")

        if m < n:
            raise DMShapeError("Underdetermined matrix equation.")

        Aaug = A.hstack(b)
        Aaug_rref, denom, pivots = Aaug.rref_den()

        # XXX: We check here if there are pivots after the last column. If
        # there were than it possibly means that rref_den performed some
        # unnecessary elimination. It would be better if rref methods had a
        # parameter indicating how many columns should be used for elimination.
        if len(pivots) != n or pivots and pivots[-1] >= n:
            raise DMNonInvertibleMatrixError("Non-unique solution.")

        xnum = Aaug_rref[:n, n:]
        xden = denom

        return xnum, xden

    def solve_den_charpoly(self, b, cp=None, check=True):
        """
        Solve matrix equation $Ax = b$ using the characteristic polynomial.

        This method solves the square matrix equation $Ax = b$ for $x$ using
        the characteristic polynomial without any division or fractions in the
        ground domain.

        Examples
        ========

        Solve a matrix equation over the integers:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, detA = A.solve_den_charpoly(b)
        >>> detA
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == detA * b
        True

        Parameters
        ==========

        self : DomainMatrix
            The ``n x n`` matrix `A` in the equation `Ax = b`. Must be square
            and invertible.
        b : DomainMatrix
            The ``n x m`` matrix `b` for the rhs.
        cp : list, optional
            The characteristic polynomial of the matrix `A` if known. If not
            given, it will be computed using :meth:`charpoly`.
        check : bool, optional
            If ``True`` (the default) check that the determinant is not zero
            and raise an error if it is. If ``False`` then if the determinant
            is zero the return value will be equal to ``(A.adjugate()*b, 0)``.

        Returns
        =======

        (xnum, detA) : (DomainMatrix, DomainElement)
            The solution of the equation `Ax = b` as a matrix numerator and
            scalar denominator pair. The denominator is equal to the
            determinant of `A` and the numerator is ``adj(A)*b``.

        The solution $x$ is given by ``x = xnum / detA``. The division free
        invariant is ``A * xnum == detA * b``.

        If ``b`` is the identity matrix, then ``xnum`` is the adjugate matrix
        and we have ``A * adj(A) == detA * I``.

        See Also
        ========

        solve_den
            Main frontend for solving matrix equations with denominator.
        solve_den_rref
            Solve matrix equations using fraction-free RREF.
        inv_den
            Invert a matrix using the characteristic polynomial.
        """
        A, b = self.unify(b)
        m, n = self.shape
        mb, nb = b.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if mb != m:
            raise DMShapeError("Matrix and vector must have the same number of rows")

        f, detA = self.adj_poly_det(cp=cp)

        if check and not detA:
            raise DMNonInvertibleMatrixError("Matrix is not invertible")

        # Compute adj(A)*b = det(A)*inv(A)*b using Horner's method without
        # constructing inv(A) explicitly.
        adjA_b = self.eval_poly_mul(f, b)

        return (adjA_b, detA)

    def adj_poly_det(self, cp=None):
        """
        Return the polynomial $p$ such that $p(A) = adj(A)$ and also the
        determinant of $A$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p, detA = A.adj_poly_det()
        >>> p
        [-1, 5]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), QQ)
        >>> p[0]*A**1 + p[1]*A**0 == p_A
        True
        >>> p_A == A.adjugate()
        True
        >>> A * A.adjugate() == detA * A.eye(A.shape, A.domain).to_dense()
        True

        See Also
        ========

        adjugate
        eval_poly
        adj_det
        """

        # Cayley-Hamilton says that a matrix satisfies its own minimal
        # polynomial
        #
        #   p[0]*A^n + p[1]*A^(n-1) + ... + p[n]*I = 0
        #
        # with p[0]=1 and p[n]=(-1)^n*det(A) or
        #
        #   det(A)*I = -(-1)^n*(p[0]*A^(n-1) + p[1]*A^(n-2) + ... + p[n-1]*A).
        #
        # Define a new polynomial f with f[i] = -(-1)^n*p[i] for i=0..n-1. Then
        #
        #   det(A)*I = f[0]*A^n + f[1]*A^(n-1) + ... + f[n-1]*A.
        #
        # Multiplying on the right by inv(A) gives
        #
        #   det(A)*inv(A) = f[0]*A^(n-1) + f[1]*A^(n-2) + ... + f[n-1].
        #
        # So adj(A) = det(A)*inv(A) = f(A)

        A = self
        m, n = self.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if cp is None:
            cp = A.charpoly()

        if len(cp) % 2:
            # n is even
            detA = cp[-1]
            f = [-cpi for cpi in cp[:-1]]
        else:
            # n is odd
            detA = -cp[-1]
            f = cp[:-1]

        return f, detA

    def eval_poly(self, p):
        """
        Evaluate polynomial function of a matrix $p(A)$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[12, 14], [21, 33]], (2, 2), QQ)
        >>> p_A == p[0]*A**2 + p[1]*A + p[2]*A**0
        True

        See Also
        ========

        eval_poly_mul
        """
        A = self
        m, n = A.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if not p:
            return self.zeros(self.shape, self.domain)
        elif len(p) == 1:
            return p[0] * self.eye(self.shape, self.domain)

        # Evaluate p(A) using Horner's method:
        # XXX: Use Paterson-Stockmeyer method?
        I = A.eye(A.shape, A.domain)
        p_A = p[0] * I
        for pi in p[1:]:
            p_A = A*p_A + pi*I

        return p_A

    def eval_poly_mul(self, p, B):
        r"""
        Evaluate polynomial matrix product $p(A) \times B$.

        Evaluate the polynomial matrix product $p(A) \times B$ using Horner's
        method without creating the matrix $p(A)$ explicitly. If $B$ is a
        column matrix then this method will only use matrix-vector multiplies
        and no matrix-matrix multiplies are needed.

        If $B$ is square or wide or if $A$ can be represented in a simpler
        domain than $B$ then it might be faster to evaluate $p(A)$ explicitly
        (see :func:`eval_poly`) and then multiply with $B$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> b = DM([[QQ(5)], [QQ(6)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A_b = A.eval_poly_mul(p, b)
        >>> p_A_b
        DomainMatrix([[144], [303]], (2, 1), QQ)
        >>> p_A_b == p[0]*A**2*b + p[1]*A*b + p[2]*b
        True
        >>> A.eval_poly_mul(p, b) == A.eval_poly(p)*b
        True

        See Also
        ========

        eval_poly
        solve_den_charpoly
        """
        A = self
        m, n = A.shape
        mb, nb = B.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if mb != n:
            raise DMShapeError("Matrices are not aligned")

        if A.domain != B.domain:
            raise DMDomainError("Matrices must have the same domain")

        # Given a polynomial p(x) = p[0]*x^n + p[1]*x^(n-1) + ... + p[n-1]
        # and matrices A and B we want to find
        #
        #   p(A)*B = p[0]*A^n*B + p[1]*A^(n-1)*B + ... + p[n-1]*B
        #
        # Factoring out A term by term we get
        #
        #   p(A)*B = A*(...A*(A*(A*(p[0]*B) + p[1]*B) + p[2]*B) + ...) + p[n-1]*B
        #
        # where each pair of brackets represents one iteration of the loop
        # below starting from the innermost p[0]*B. If B is a column matrix
        # then products like A*(...) are matrix-vector multiplies and products
        # like p[i]*B are scalar-vector multiplies so there are no
        # matrix-matrix multiplies.

        if not p:
            return B.zeros(B.shape, B.domain, fmt=B.rep.fmt)

        p_A_B = p[0]*B

        for p_i in p[1:]:
            p_A_B = A*p_A_B + p_i*B

        return p_A_B

    def lu(self):
        r"""
        Returns Lower and Upper decomposition of the DomainMatrix

        Returns
        =======

        (L, U, exchange)
            L, U are Lower and Upper decomposition of the DomainMatrix,
            exchange is the list of indices of rows exchanged in the
            decomposition.

        Raises
        ======

        ValueError
            If the domain of DomainMatrix not a Field

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> L, U, exchange = A.lu()
        >>> L
        DomainMatrix([[1, 0], [2, 1]], (2, 2), QQ)
        >>> U
        DomainMatrix([[1, -1], [0, 0]], (2, 2), QQ)
        >>> exchange
        []

        See Also
        ========

        lu_solve

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        L, U, swaps = self.rep.lu()
        return self.from_rep(L), self.from_rep(U), swaps

    def lu_solve(self, rhs):
        r"""
        Solver for DomainMatrix x in the A*x = B

        Parameters
        ==========

        rhs : DomainMatrix B

        Returns
        =======

        DomainMatrix
            x in A*x = B

        Raises
        ======

        DMShapeError
            If the DomainMatrix A and rhs have different number of rows

        ValueError
            If the domain of DomainMatrix A not a Field

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(2)],
        ...    [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> B = DomainMatrix([
        ...    [QQ(1), QQ(1)],
        ...    [QQ(0), QQ(1)]], (2, 2), QQ)

        >>> A.lu_solve(B)
        DomainMatrix([[-2, -1], [3/2, 1]], (2, 2), QQ)

        See Also
        ========

        lu

        """
        if self.shape[0] != rhs.shape[0]:
            raise DMShapeError("Shape")
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        sol = self.rep.lu_solve(rhs.rep)
        return self.from_rep(sol)

    def _solve(A, b):
        # XXX: Not sure about this method or its signature. It is just created
        # because it is needed by the holonomic module.
        if A.shape[0] != b.shape[0]:
            raise DMShapeError("Shape")
        if A.domain != b.domain or not A.domain.is_Field:
            raise DMNotAField('Not a field')
        Aaug = A.hstack(b)
        Arref, pivots = Aaug.rref()
        particular = Arref.from_rep(Arref.rep.particular())
        nullspace_rep, nonpivots = Arref[:,:-1].rep.nullspace()
        nullspace = Arref.from_rep(nullspace_rep)
        return particular, nullspace

    def charpoly(self):
        r"""
        Characteristic polynomial of a square matrix.

        Computes the characteristic polynomial in a fully expanded form using
        division free arithmetic. If a factorization of the characteristic
        polynomial is needed then it is more efficient to call
        :meth:`charpoly_factor_list` than calling :meth:`charpoly` and then
        factorizing the result.

        Returns
        =======

        list: list of DomainElement
            coefficients of the characteristic polynomial

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.charpoly()
        [1, -5, -2]

        See Also
        ========

        charpoly_factor_list
            Compute the factorisation of the characteristic polynomial.
        charpoly_factor_blocks
            A partial factorisation of the characteristic polynomial that can
            be computed more efficiently than either the full factorisation or
            the fully expanded polynomial.
        """
        M = self
        K = M.domain

        factors = M.charpoly_factor_blocks()

        cp = [K.one]

        for f, mult in factors:
            for _ in range(mult):
                cp = dup_mul(cp, f, K)

        return cp

    def charpoly_factor_list(self):
        """
        Full factorization of the characteristic polynomial.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        Compute the factorization of the characteristic polynomial:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the unfactorized characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        The same calculations with ``Matrix``:

        >>> M.to_Matrix().charpoly().as_expr()
        lambda**4 - 25*lambda**3 + 203*lambda**2 - 495*lambda - 324
        >>> M.to_Matrix().charpoly().as_expr().factor()
        (lambda - 9)**2*(lambda**2 - 7*lambda - 4)

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A full factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Expanded form of the characteristic polynomial.
        charpoly_factor_blocks
            A partial factorisation of the characteristic polynomial that can
            be computed more efficiently.
        """
        M = self
        K = M.domain

        # It is more efficient to start from the partial factorization provided
        # for free by M.charpoly_factor_blocks than the expanded M.charpoly.
        factors = M.charpoly_factor_blocks()

        factors_irreducible = []

        for factor_i, mult_i in factors:

            _, factors_list = dup_factor_list(factor_i, K)

            for factor_j, mult_j in factors_list:
                factors_irreducible.append((factor_j, mult_i * mult_j))

        return _collect_factors(factors_irreducible)

    def charpoly_factor_blocks(self):
        """
        Partial factorisation of the characteristic polynomial.

        This factorisation arises from a block structure of the matrix (if any)
        and so the factors are not guaranteed to be irreducible. The
        :meth:`charpoly_factor_blocks` method is the most efficient way to get
        a representation of the characteristic polynomial but the result is
        neither fully expanded nor fully factored.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        This computes a partial factorization using only the block structure of
        the matrix to reveal factors:

        >>> M.charpoly_factor_blocks()
        [([1, -18, 81], 1), ([1, -7, -4], 1)]

        These factors correspond to the two diagonal blocks in the matrix:

        >>> DM([[6, -1], [9, 12]], ZZ).charpoly()
        [1, -18, 81]
        >>> DM([[1, 2], [5, 6]], ZZ).charpoly()
        [1, -7, -4]

        Use :meth:`charpoly_factor_list` to get a complete factorization into
        irreducibles:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the expanded characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A partial factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Compute the fully expanded characteristic polynomial.
        charpoly_factor_list
            Compute a full factorization of the characteristic polynomial.
        """
        M = self

        if not M.is_square:
            raise DMNonSquareMatrixError("not square")

        # scc returns indices that permute the matrix into block triangular
        # form and can extract the diagonal blocks. M.charpoly() is equal to
        # the product of the diagonal block charpolys.
        components = M.scc()

        block_factors = []

        for indices in components:
            block = M.extract(indices, indices)
            block_factors.append((block.charpoly_base(), 1))

        return _collect_factors(block_factors)

    def charpoly_base(self):
        """
        Base case for :meth:`charpoly_factor_blocks` after block decomposition.

        This method is used internally by :meth:`charpoly_factor_blocks` as the
        base case for computing the characteristic polynomial of a block. It is
        more efficient to call :meth:`charpoly_factor_blocks`, :meth:`charpoly`
        or :meth:`charpoly_factor_list` rather than call this method directly.

        This will use either the dense or the sparse implementation depending
        on the sparsity of the matrix and will clear denominators if possible
        before calling :meth:`charpoly_berk` to compute the characteristic
        polynomial using the Berkowitz algorithm.

        See Also
        ========

        charpoly
        charpoly_factor_list
        charpoly_factor_blocks
        charpoly_berk
        """
        M = self
        K = M.domain

        # It seems that the sparse implementation is always faster for random
        # matrices with fewer than 50% non-zero entries. This does not seem to
        # depend on domain, size, bit count etc.
        density = self.nnz() / self.shape[0]**2
        if density < 0.5:
            M = M.to_sparse()
        else:
            M = M.to_dense()

        # Clearing denominators is always more efficient if it can be done.
        # Doing it here after block decomposition is good because each block
        # might have a smaller denominator. However it might be better for
        # charpoly and charpoly_factor_list to restore the denominators only at
        # the very end so that they can call e.g. dup_factor_list before
        # restoring the denominators. The methods would need to be changed to
        # return (poly, denom) pairs to make that work though.
        clear_denoms = K.is_Field and K.has_assoc_Ring

        if clear_denoms:
            clear_denoms = True
            d, M = M.clear_denoms(convert=True)
            d = d.element
            K_f = K
            K_r = M.domain

        # Berkowitz algorithm over K_r.
        cp = M.charpoly_berk()

        if clear_denoms:
            # Restore the denominator in the charpoly over K_f.
            #
            # If M = N/d then p_M(x) = p_N(x*d)/d^n.
            cp = dup_convert(cp, K_r, K_f)
            p = [K_f.one, K_f.zero]
            q = [K_f.one/d]
            cp = dup_transform(cp, p, q, K_f)

        return cp

    def charpoly_berk(self):
        """Compute the characteristic polynomial using the Berkowitz algorithm.

        This method directly calls the underlying implementation of the
        Berkowitz algorithm (:meth:`sympy.polys.matrices.dense.ddm_berk` or
        :meth:`sympy.polys.matrices.sdm.sdm_berk`).

        This is used by :meth:`charpoly` and other methods as the base case for
        for computing the characteristic polynomial. However those methods will
        apply other optimizations such as block decomposition, clearing
        denominators and converting between dense and sparse representations
        before calling this method. It is more efficient to call those methods
        instead of this one but this method is provided for direct access to
        the Berkowitz algorithm.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import QQ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], QQ)
        >>> M.charpoly_berk()
        [1, -25, 203, -495, -324]

        See Also
        ========

        charpoly
        charpoly_base
        charpoly_factor_list
        charpoly_factor_blocks
        sympy.polys.matrices.dense.ddm_berk
        sympy.polys.matrices.sdm.sdm_berk
        """
        return self.rep.charpoly()

    @classmethod
    def eye(cls, shape, domain):
        r"""
        Return identity matrix of size n or shape (m, n).

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.eye(3, QQ)
        DomainMatrix({0: {0: 1}, 1: {1: 1}, 2: {2: 1}}, (3, 3), QQ)

        """
        if isinstance(shape, int):
            shape = (shape, shape)
        return cls.from_rep(SDM.eye(shape, domain))

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        r"""
        Return diagonal matrix with entries from ``diagonal``.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> DomainMatrix.diag([ZZ(5), ZZ(6)], ZZ)
        DomainMatrix({0: {0: 5}, 1: {1: 6}}, (2, 2), ZZ)

        """
        if shape is None:
            N = len(diagonal)
            shape = (N, N)
        return cls.from_rep(SDM.diag(diagonal, domain, shape))

    @classmethod
    def zeros(cls, shape, domain, *, fmt='sparse'):
        """Returns a zero DomainMatrix of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.zeros((2, 3), QQ)
        DomainMatrix({}, (2, 3), QQ)

        """
        return cls.from_rep(SDM.zeros(shape, domain))

    @classmethod
    def ones(cls, shape, domain):
        """Returns a DomainMatrix of 1s, of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.ones((2,3), QQ)
        DomainMatrix([[1, 1, 1], [1, 1, 1]], (2, 3), QQ)

        """
        return cls.from_rep(DDM.ones(shape, domain).to_dfm_or_ddm())

    def __eq__(A, B):
        r"""
        Checks for two DomainMatrix matrices to be equal or not

        Parameters
        ==========

        A, B: DomainMatrix
            to check equality

        Returns
        =======

        Boolean
            True for equal, else False

        Raises
        ======

        NotImplementedError
            If B is not a DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.__eq__(A)
        True
        >>> A.__eq__(B)
        False

        """
        if not isinstance(A, type(B)):
            return NotImplemented
        return A.domain == B.domain and A.rep == B.rep

    def unify_eq(A, B):
        if A.shape != B.shape:
            return False
        if A.domain != B.domain:
            A, B = A.unify(B)
        return A == B

    def lll(A, delta=QQ(3, 4)):
        """
        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.
        See [1]_ and [2]_.

        Parameters
        ==========

        delta : QQ, optional
            The Lovász parameter. Must be in the interval (0.25, 1), with larger
            values producing a more reduced basis. The default is 0.75 for
            historical reasons.

        Returns
        =======

        The reduced basis as a DomainMatrix over ZZ.

        Throws
        ======

        DMValueError: if delta is not in the range (0.25, 1)
        DMShapeError: if the matrix is not of shape (m, n) with m <= n
        DMDomainError: if the matrix domain is not ZZ
        DMRankError: if the matrix contains linearly dependent rows

        Examples
        ========

        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.polys.matrices import DM
        >>> x = DM([[1, 0, 0, 0, -20160],
        ...         [0, 1, 0, 0, 33768],
        ...         [0, 0, 1, 0, 39578],
        ...         [0, 0, 0, 1, 47757]], ZZ)
        >>> y = DM([[10, -3, -2, 8, -4],
        ...         [3, -9, 8, 1, -11],
        ...         [-3, 13, -9, -3, -9],
        ...         [-12, -7, -11, 9, -1]], ZZ)
        >>> assert x.lll(delta=QQ(5, 6)) == y

        Notes
        =====

        The implementation is derived from the Maple code given in Figures 4.3
        and 4.4 of [3]_ (pp.68-69). It uses the efficient method of only calculating
        state updates as they are required.

        See also
        ========

        lll_transform

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm
        .. [2] https://web.archive.org/web/20221029115428/https://web.cs.elte.hu/~lovasz/scans/lll.pdf
        .. [3] Murray R. Bremner, "Lattice Basis Reduction: An Introduction to the LLL Algorithm and Its Applications"

        """
        return DomainMatrix.from_rep(A.rep.lll(delta=delta))

    def lll_transform(A, delta=QQ(3, 4)):
        """
        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm
        and returns the reduced basis and transformation matrix.

        Explanation
        ===========

        Parameters, algorithm and basis are the same as for :meth:`lll` except that
        the return value is a tuple `(B, T)` with `B` the reduced basis and
        `T` a transformation matrix. The original basis `A` is transformed to
        `B` with `T*A == B`. If only `B` is needed then :meth:`lll` should be
        used as it is a little faster.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.polys.matrices import DM
        >>> X = DM([[1, 0, 0, 0, -20160],
        ...         [0, 1, 0, 0, 33768],
        ...         [0, 0, 1, 0, 39578],
        ...         [0, 0, 0, 1, 47757]], ZZ)
        >>> B, T = X.lll_transform(delta=QQ(5, 6))
        >>> T * X == B
        True

        See also
        ========

        lll

        """
        reduced, transform = A.rep.lll_transform(delta=delta)
        return DomainMatrix.from_rep(reduced), DomainMatrix.from_rep(transform)


def _collect_factors(factors_list):
    """
    Collect repeating factors and sort.

    >>> from sympy.polys.matrices.domainmatrix import _collect_factors
    >>> _collect_factors([([1, 2], 2), ([1, 4], 3), ([1, 2], 5)])
    [([1, 4], 3), ([1, 2], 7)]
    """
    factors = Counter()
    for factor, exponent in factors_list:
        factors[tuple(factor)] += exponent

    factors_list = [(list(f), e) for f, e in factors.items()]

    return _sort_factors(factors_list)
