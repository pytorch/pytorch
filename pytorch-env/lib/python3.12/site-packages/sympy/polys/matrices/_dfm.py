#
# sympy.polys.matrices.dfm
#
# This modules defines the DFM class which is a wrapper for dense flint
# matrices as found in python-flint.
#
# As of python-flint 0.4.1 matrices over the following domains can be supported
# by python-flint:
#
#   ZZ: flint.fmpz_mat
#   QQ: flint.fmpq_mat
#   GF(p): flint.nmod_mat (p prime and p < ~2**62)
#
# The underlying flint library has many more domains, but these are not yet
# supported by python-flint.
#
# The DFM class is a wrapper for the flint matrices and provides a common
# interface for all supported domains that is interchangeable with the DDM
# and SDM classes so that DomainMatrix can be used with any as its internal
# matrix representation.
#

# TODO:
#
# Implement the following methods that are provided by python-flint:
#
# - hnf (Hermite normal form)
# - snf (Smith normal form)
# - minpoly
# - is_hnf
# - is_snf
# - rank
#
# The other types DDM and SDM do not have these methods and the algorithms
# for hnf, snf and rank are already implemented. Algorithms for minpoly,
# is_hnf and is_snf would need to be added.
#
# Add more methods to python-flint to expose more of Flint's functionality
# and also to make some of the above methods simpler or more efficient e.g.
# slicing, fancy indexing etc.

from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import import_module
from sympy.utilities.decorator import doctest_depends_on

from sympy.polys.domains import ZZ, QQ

from .exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError,
    DMRankError,
    DMShapeError,
    DMValueError,
)


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['*']


flint = import_module('flint')


__all__ = ['DFM']


@doctest_depends_on(ground_types=['flint'])
class DFM:
    """
    Dense FLINT matrix. This class is a wrapper for matrices from python-flint.

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.matrices.dfm import DFM
    >>> dfm = DFM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> dfm
    [[1, 2], [3, 4]]
    >>> dfm.rep
    [1, 2]
    [3, 4]
    >>> type(dfm.rep)  # doctest: +SKIP
    <class 'flint._flint.fmpz_mat'>

    Usually, the DFM class is not instantiated directly, but is created as the
    internal representation of :class:`~.DomainMatrix`. When
    `SYMPY_GROUND_TYPES` is set to `flint` and `python-flint` is installed, the
    :class:`DFM` class is used automatically as the internal representation of
    :class:`~.DomainMatrix` in dense format if the domain is supported by
    python-flint.

    >>> from sympy.polys.matrices.domainmatrix import DM
    >>> dM = DM([[1, 2], [3, 4]], ZZ)
    >>> dM.rep
    [[1, 2], [3, 4]]

    A :class:`~.DomainMatrix` can be converted to :class:`DFM` by calling the
    :meth:`to_dfm` method:

    >>> dM.to_dfm()
    [[1, 2], [3, 4]]

    """

    fmt = 'dense'
    is_DFM = True
    is_DDM = False

    def __new__(cls, rowslist, shape, domain):
        """Construct from a nested list."""
        flint_mat = cls._get_flint_func(domain)

        if 0 not in shape:
            try:
                rep = flint_mat(rowslist)
            except (ValueError, TypeError):
                raise DMBadInputError(f"Input should be a list of list of {domain}")
        else:
            rep = flint_mat(*shape)

        return cls._new(rep, shape, domain)

    @classmethod
    def _new(cls, rep, shape, domain):
        """Internal constructor from a flint matrix."""
        cls._check(rep, shape, domain)
        obj = object.__new__(cls)
        obj.rep = rep
        obj.shape = obj.rows, obj.cols = shape
        obj.domain = domain
        return obj

    def _new_rep(self, rep):
        """Create a new DFM with the same shape and domain but a new rep."""
        return self._new(rep, self.shape, self.domain)

    @classmethod
    def _check(cls, rep, shape, domain):
        repshape = (rep.nrows(), rep.ncols())
        if repshape != shape:
            raise DMBadInputError("Shape of rep does not match shape of DFM")
        if domain == ZZ and not isinstance(rep, flint.fmpz_mat):
            raise RuntimeError("Rep is not a flint.fmpz_mat")
        elif domain == QQ and not isinstance(rep, flint.fmpq_mat):
            raise RuntimeError("Rep is not a flint.fmpq_mat")
        elif domain not in (ZZ, QQ):
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @classmethod
    def _supports_domain(cls, domain):
        """Return True if the given domain is supported by DFM."""
        return domain in (ZZ, QQ)

    @classmethod
    def _get_flint_func(cls, domain):
        """Return the flint matrix class for the given domain."""
        if domain == ZZ:
            return flint.fmpz_mat
        elif domain == QQ:
            return flint.fmpq_mat
        else:
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @property
    def _func(self):
        """Callable to create a flint matrix of the same domain."""
        return self._get_flint_func(self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return str(self.to_ddm())

    def __repr__(self):
        """Return ``repr(self)``."""
        return f'DFM{repr(self.to_ddm())[3:]}'

    def __eq__(self, other):
        """Return ``self == other``."""
        if not isinstance(other, DFM):
            return NotImplemented
        # Compare domains first because we do *not* want matrices with
        # different domains to be equal but e.g. a flint fmpz_mat and fmpq_mat
        # with the same entries will compare equal.
        return self.domain == other.domain and self.rep == other.rep

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """Construct from a nested list."""
        return cls(rowslist, shape, domain)

    def to_list(self):
        """Convert to a nested list."""
        return self.rep.tolist()

    def copy(self):
        """Return a copy of self."""
        return self._new_rep(self._func(self.rep))

    def to_ddm(self):
        """Convert to a DDM."""
        return DDM.from_list(self.to_list(), self.shape, self.domain)

    def to_sdm(self):
        """Convert to a SDM."""
        return SDM.from_list(self.to_list(), self.shape, self.domain)

    def to_dfm(self):
        """Return self."""
        return self

    def to_dfm_or_ddm(self):
        """
        Convert to a :class:`DFM`.

        This :class:`DFM` method exists to parallel the :class:`~.DDM` and
        :class:`~.SDM` methods. For :class:`DFM` it will always return self.

        See Also
        ========

        to_ddm
        to_sdm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        return self

    @classmethod
    def from_ddm(cls, ddm):
        """Convert from a DDM."""
        return cls.from_list(ddm.to_list(), ddm.shape, ddm.domain)

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """Inverse of :meth:`to_list_flat`."""
        func = cls._get_flint_func(domain)
        try:
            rep = func(*shape, elements)
        except ValueError:
            raise DMBadInputError(f"Incorrect number of elements for shape {shape}")
        except TypeError:
            raise DMBadInputError(f"Input should be a list of {domain}")
        return cls(rep, shape, domain)

    def to_list_flat(self):
        """Convert to a flat list."""
        return self.rep.entries()

    def to_flat_nz(self):
        """Convert to a flat list of non-zeros."""
        return self.to_ddm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """Inverse of :meth:`to_flat_nz`."""
        return DDM.from_flat_nz(elements, data, domain).to_dfm()

    def to_dod(self):
        """Convert to a DOD."""
        return self.to_ddm().to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """Inverse of :meth:`to_dod`."""
        return DDM.from_dod(dod, shape, domain).to_dfm()

    def to_dok(self):
        """Convert to a DOK."""
        return self.to_ddm().to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """Inverse of :math:`to_dod`."""
        return DDM.from_dok(dok, shape, domain).to_dfm()

    def iter_values(self):
        """Iterater over the non-zero values of the matrix."""
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                if repij:
                    yield rep[i, j]

    def iter_items(self):
        """Iterate over indices and values of nonzero elements of the matrix."""
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                if repij:
                    yield ((i, j), repij)

    def convert_to(self, domain):
        """Convert to a new domain."""
        if domain == self.domain:
            return self.copy()
        elif domain == QQ and self.domain == ZZ:
            return self._new(flint.fmpq_mat(self.rep), self.shape, domain)
        elif domain == ZZ and self.domain == QQ:
            # XXX: python-flint has no fmpz_mat.from_fmpq_mat
            return self.to_ddm().convert_to(domain).to_dfm()
        else:
            # It is the callers responsibility to convert to DDM before calling
            # this method if the domain is not supported by DFM.
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    def getitem(self, i, j):
        """Get the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            return self.rep[i, j]
        except ValueError:
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    def setitem(self, i, j, value):
        """Set the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            self.rep[i, j] = value
        except ValueError:
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    def _extract(self, i_indices, j_indices):
        """Extract a submatrix with no checking."""
        # Indices must be positive and in range.
        M = self.rep
        lol = [[M[i, j] for j in j_indices] for i in i_indices]
        shape = (len(i_indices), len(j_indices))
        return self.from_list(lol, shape, self.domain)

    def extract(self, rowslist, colslist):
        """Extract a submatrix."""
        # XXX: flint matrices do not support fancy indexing or negative indices
        #
        # Check and convert negative indices before calling _extract.
        m, n = self.shape

        new_rows = []
        new_cols = []

        for i in rowslist:
            if i < 0:
                i_pos = i + m
            else:
                i_pos = i
            if not 0 <= i_pos < m:
                raise IndexError(f"Invalid row index {i} for Matrix of shape {self.shape}")
            new_rows.append(i_pos)

        for j in colslist:
            if j < 0:
                j_pos = j + n
            else:
                j_pos = j
            if not 0 <= j_pos < n:
                raise IndexError(f"Invalid column index {j} for Matrix of shape {self.shape}")
            new_cols.append(j_pos)

        return self._extract(new_rows, new_cols)

    def extract_slice(self, rowslice, colslice):
        """Slice a DFM."""
        # XXX: flint matrices do not support slicing
        m, n = self.shape
        i_indices = range(m)[rowslice]
        j_indices = range(n)[colslice]
        return self._extract(i_indices, j_indices)

    def neg(self):
        """Negate a DFM matrix."""
        return self._new_rep(-self.rep)

    def add(self, other):
        """Add two DFM matrices."""
        return self._new_rep(self.rep + other.rep)

    def sub(self, other):
        """Subtract two DFM matrices."""
        return self._new_rep(self.rep - other.rep)

    def mul(self, other):
        """Multiply a DFM matrix from the right by a scalar."""
        return self._new_rep(self.rep * other)

    def rmul(self, other):
        """Multiply a DFM matrix from the left by a scalar."""
        return self._new_rep(other * self.rep)

    def mul_elementwise(self, other):
        """Elementwise multiplication of two DFM matrices."""
        # XXX: flint matrices do not support elementwise multiplication
        return self.to_ddm().mul_elementwise(other.to_ddm()).to_dfm()

    def matmul(self, other):
        """Multiply two DFM matrices."""
        shape = (self.rows, other.cols)
        return self._new(self.rep * other.rep, shape, self.domain)

    # XXX: For the most part DomainMatrix does not expect DDM, SDM, or DFM to
    # have arithmetic operators defined. The only exception is negation.
    # Perhaps that should be removed.

    def __neg__(self):
        """Negate a DFM matrix."""
        return self.neg()

    @classmethod
    def zeros(cls, shape, domain):
        """Return a zero DFM matrix."""
        func = cls._get_flint_func(domain)
        return cls._new(func(*shape), shape, domain)

    # XXX: flint matrices do not have anything like ones or eye
    # In the methods below we convert to DDM and then back to DFM which is
    # probably about as efficient as implementing these methods directly.

    @classmethod
    def ones(cls, shape, domain):
        """Return a one DFM matrix."""
        # XXX: flint matrices do not have anything like ones
        return DDM.ones(shape, domain).to_dfm()

    @classmethod
    def eye(cls, n, domain):
        """Return the identity matrix of size n."""
        # XXX: flint matrices do not have anything like eye
        return DDM.eye(n, domain).to_dfm()

    @classmethod
    def diag(cls, elements, domain):
        """Return a diagonal matrix."""
        return DDM.diag(elements, domain).to_dfm()

    def applyfunc(self, func, domain):
        """Apply a function to each entry of a DFM matrix."""
        return self.to_ddm().applyfunc(func, domain).to_dfm()

    def transpose(self):
        """Transpose a DFM matrix."""
        return self._new(self.rep.transpose(), (self.cols, self.rows), self.domain)

    def hstack(self, *others):
        """Horizontally stack matrices."""
        return self.to_ddm().hstack(*[o.to_ddm() for o in others]).to_dfm()

    def vstack(self, *others):
        """Vertically stack matrices."""
        return self.to_ddm().vstack(*[o.to_ddm() for o in others]).to_dfm()

    def diagonal(self):
        """Return the diagonal of a DFM matrix."""
        M = self.rep
        m, n = self.shape
        return [M[i, i] for i in range(min(m, n))]

    def is_upper(self):
        """Return ``True`` if the matrix is upper triangular."""
        M = self.rep
        for i in range(self.rows):
            for j in range(i):
                if M[i, j]:
                    return False
        return True

    def is_lower(self):
        """Return ``True`` if the matrix is lower triangular."""
        M = self.rep
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if M[i, j]:
                    return False
        return True

    def is_diagonal(self):
        """Return ``True`` if the matrix is diagonal."""
        return self.is_upper() and self.is_lower()

    def is_zero_matrix(self):
        """Return ``True`` if the matrix is the zero matrix."""
        M = self.rep
        for i in range(self.rows):
            for j in range(self.cols):
                if M[i, j]:
                    return False
        return True

    def nnz(self):
        """Return the number of non-zero elements in the matrix."""
        return self.to_ddm().nnz()

    def scc(self):
        """Return the strongly connected components of the matrix."""
        return self.to_ddm().scc()

    @doctest_depends_on(ground_types='flint')
    def det(self):
        """
        Compute the determinant of the matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm()
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.det()
        -2

        Notes
        =====

        Calls the ``.det()`` method of the underlying FLINT matrix.

        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_det`` or
        ``fmpq_mat_det`` respectively.

        At the time of writing the implementation of ``fmpz_mat_det`` uses one
        of several algorithms depending on the size of the matrix and bit size
        of the entries. The algorithms used are:

        - Cofactor for very small (up to 4x4) matrices.
        - Bareiss for small (up to 25x25) matrices.
        - Modular algorithms for larger matrices (up to 60x60) or for larger
          matrices with large bit sizes.
        - Modular "accelerated" for larger matrices (60x60 upwards) if the bit
          size is smaller than the dimensions of the matrix.

        The implementation of ``fmpq_mat_det`` clears denominators from each
        row (not the whole matrix) and then calls ``fmpz_mat_det`` and divides
        by the product of the denominators.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.det
            Higher level interface to compute the determinant of a matrix.
        """
        # XXX: At least the first three algorithms described above should also
        # be implemented in the pure Python DDM and SDM classes which at the
        # time of writng just use Bareiss for all matrices and domains.
        # Probably in Python the thresholds would be different though.
        return self.rep.det()

    @doctest_depends_on(ground_types='flint')
    def charpoly(self):
        """
        Compute the characteristic polynomial of the matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm()  # need ground types = 'flint'
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.charpoly()
        [1, -5, -2]

        Notes
        =====

        Calls the ``.charpoly()`` method of the underlying FLINT matrix.

        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_charpoly`` or
        ``fmpq_mat_charpoly`` respectively.

        At the time of writing the implementation of ``fmpq_mat_charpoly``
        clears a denominator from the whole matrix and then calls
        ``fmpz_mat_charpoly``. The coefficients of the characteristic
        polynomial are then multiplied by powers of the denominator.

        The ``fmpz_mat_charpoly`` method uses a modular algorithm with CRT
        reconstruction. The modular algorithm uses ``nmod_mat_charpoly`` which
        uses Berkowitz for small matrices and non-prime moduli or otherwise
        the Danilevsky method.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
            Higher level interface to compute the characteristic polynomial of
            a matrix.
        """
        # FLINT polynomial coefficients are in reverse order compared to SymPy.
        return self.rep.charpoly().coeffs()[::-1]

    @doctest_depends_on(ground_types='flint')
    def inv(self):
        """
        Compute the inverse of a matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix, QQ
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.inv()
        [[-2, 1], [3/2, -1/2]]
        >>> dfm.matmul(dfm.inv())
        [[1, 0], [0, 1]]

        Notes
        =====

        Calls the ``.inv()`` method of the underlying FLINT matrix.

        For now this will raise an error if the domain is :ref:`ZZ` but will
        use the FLINT method for :ref:`QQ`.

        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_inv`` and
        ``fmpq_mat_inv`` respectively. The ``fmpz_mat_inv`` method computes an
        inverse with denominator. This is implemented by calling
        ``fmpz_mat_solve`` (see notes in :meth:`lu_solve` about the algorithm).

        The ``fmpq_mat_inv`` method clears denominators from each row and then
        multiplies those into the rhs identity matrix before calling
        ``fmpz_mat_solve``.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.inv
            Higher level method for computing the inverse of a matrix.
        """
        # TODO: Implement similar algorithms for DDM and SDM.
        #
        # XXX: The flint fmpz_mat and fmpq_mat inv methods both return fmpq_mat
        # by default. The fmpz_mat method has an optional argument to return
        # fmpz_mat instead for unimodular matrices.
        #
        # The convention in DomainMatrix is to raise an error if the matrix is
        # not over a field regardless of whether the matrix is invertible over
        # its domain or over any associated field. Maybe DomainMatrix.inv
        # should be changed to always return a matrix over an associated field
        # except with a unimodular argument for returning an inverse over a
        # ring if possible.
        #
        # For now we follow the existing DomainMatrix convention...
        K = self.domain
        m, n = self.shape

        if m != n:
            raise DMNonSquareMatrixError("cannot invert a non-square matrix")

        if K == ZZ:
            raise DMDomainError("field expected, got %s" % K)
        elif K == QQ:
            try:
                return self._new_rep(self.rep.inv())
            except ZeroDivisionError:
                raise DMNonInvertibleMatrixError("matrix is not invertible")
        else:
            # If more domains are added for DFM then we will need to consider
            # what happens here.
            raise NotImplementedError("DFM.inv() is not implemented for %s" % K)

    def lu(self):
        """Return the LU decomposition of the matrix."""
        L, U, swaps = self.to_ddm().lu()
        return L.to_dfm(), U.to_dfm(), swaps

    # XXX: The lu_solve function should be renamed to solve. Whether or not it
    # uses an LU decomposition is an implementation detail. A method called
    # lu_solve would make sense for a situation in which an LU decomposition is
    # reused several times to solve iwth different rhs but that would imply a
    # different call signature.
    #
    # The underlying python-flint method has an algorithm= argument so we could
    # use that and have e.g. solve_lu and solve_modular or perhaps also a
    # method= argument to choose between the two. Flint itself has more
    # possible algorithms to choose from than are exposed by python-flint.

    @doctest_depends_on(ground_types='flint')
    def lu_solve(self, rhs):
        """
        Solve a matrix equation using FLINT.

        Examples
        ========

        >>> from sympy import Matrix, QQ
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
        >>> dfm
        [[1, 2], [3, 4]]
        >>> rhs = Matrix([1, 2]).to_DM().to_dfm().convert_to(QQ)
        >>> dfm.lu_solve(rhs)
        [[0], [1/2]]

        Notes
        =====

        Calls the ``.solve()`` method of the underlying FLINT matrix.

        For now this will raise an error if the domain is :ref:`ZZ` but will
        use the FLINT method for :ref:`QQ`.

        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_solve``
        and ``fmpq_mat_solve`` respectively. The ``fmpq_mat_solve`` method
        uses one of two algorithms:

        - For small matrices (<25 rows) it clears denominators between the
          matrix and rhs and uses ``fmpz_mat_solve``.
        - For larger matrices it uses ``fmpq_mat_solve_dixon`` which is a
          modular approach with CRT reconstruction over :ref:`QQ`.

        The ``fmpz_mat_solve`` method uses one of four algorithms:

        - For very small (<= 3x3) matrices it uses a Cramer's rule.
        - For small (<= 15x15) matrices it uses a fraction-free LU solve.
        - Otherwise it uses either Dixon or another multimodular approach.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve
            Higher level interface to solve a matrix equation.
        """
        if not self.domain == rhs.domain:
            raise DMDomainError("Domains must match: %s != %s" % (self.domain, rhs.domain))

        # XXX: As for inv we should consider whether to return a matrix over
        # over an associated field or attempt to find a solution in the ring.
        # For now we follow the existing DomainMatrix convention...
        if not self.domain.is_Field:
            raise DMDomainError("Field expected, got %s" % self.domain)

        m, n = self.shape
        j, k = rhs.shape
        if m != j:
            raise DMShapeError("Matrix size mismatch: %s * %s vs %s * %s" % (m, n, j, k))
        sol_shape = (n, k)

        # XXX: The Flint solve method only handles square matrices. Probably
        # Flint has functions that could be used to solve non-square systems
        # but they are not exposed in python-flint yet. Alternatively we could
        # put something here using the features that are available like rref.
        if m != n:
            return self.to_ddm().lu_solve(rhs.to_ddm()).to_dfm()

        try:
            sol = self.rep.solve(rhs.rep)
        except ZeroDivisionError:
            raise DMNonInvertibleMatrixError("Matrix det == 0; not invertible.")

        return self._new(sol, sol_shape, self.domain)

    def nullspace(self):
        """Return a basis for the nullspace of the matrix."""
        # Code to compute nullspace using flint:
        #
        # V, nullity = self.rep.nullspace()
        # V_dfm = self._new_rep(V)._extract(range(self.rows), range(nullity))
        #
        # XXX: That gives the nullspace but does not give us nonpivots. So we
        # use the slower DDM method anyway. It would be better to change the
        # signature of the nullspace method to not return nonpivots.
        #
        # XXX: Also python-flint exposes a nullspace method for fmpz_mat but
        # not for fmpq_mat. This is the reverse of the situation for DDM etc
        # which only allow nullspace over a field. The nullspace method for
        # DDM, SDM etc should be changed to allow nullspace over ZZ as well.
        # The DomainMatrix nullspace method does allow the domain to be a ring
        # but does not directly call the lower-level nullspace methods and uses
        # rref_den instead. Nullspace methods should also be added to all
        # matrix types in python-flint.
        ddm, nonpivots = self.to_ddm().nullspace()
        return ddm.to_dfm(), nonpivots

    def nullspace_from_rref(self, pivots=None):
        """Return a basis for the nullspace of the matrix."""
        # XXX: Use the flint nullspace method!!!
        sdm, nonpivots = self.to_sdm().nullspace_from_rref(pivots=pivots)
        return sdm.to_dfm(), nonpivots

    def particular(self):
        """Return a particular solution to the system."""
        return self.to_ddm().particular().to_dfm()

    def _lll(self, transform=False, delta=0.99, eta=0.51, rep='zbasis', gram='approx'):
        """Call the fmpz_mat.lll() method but check rank to avoid segfaults."""

        # XXX: There are tests that pass e.g. QQ(5,6) for delta. That fails
        # with a TypeError in flint because if QQ is fmpq then conversion with
        # float fails. We handle that here but there are two better fixes:
        #
        # - Make python-flint's fmpq convert with float(x)
        # - Change the tests because delta should just be a float.

        def to_float(x):
            if QQ.of_type(x):
                return float(x.numerator) / float(x.denominator)
            else:
                return float(x)

        delta = to_float(delta)
        eta = to_float(eta)

        if not 0.25 < delta < 1:
            raise DMValueError("delta must be between 0.25 and 1")

        # XXX: The flint lll method segfaults if the matrix is not full rank.
        m, n = self.shape
        if self.rep.rank() != m:
            raise DMRankError("Matrix must have full row rank for Flint LLL.")

        # Actually call the flint method.
        return self.rep.lll(transform=transform, delta=delta, eta=eta, rep=rep, gram=gram)

    @doctest_depends_on(ground_types='flint')
    def lll(self, delta=0.75):
        """Compute LLL-reduced basis using FLINT.

        See :meth:`lll_transform` for more information.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> M.to_DM().to_dfm().lll()
        [[2, 1, 0], [-1, 1, 3]]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
            Higher level interface to compute LLL-reduced basis.
        lll_transform
            Compute LLL-reduced basis and transform matrix.
        """
        if self.domain != ZZ:
            raise DMDomainError("ZZ expected, got %s" % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError("Matrix must not have more rows than columns.")

        rep = self._lll(delta=delta)
        return self._new_rep(rep)

    @doctest_depends_on(ground_types='flint')
    def lll_transform(self, delta=0.75):
        """Compute LLL-reduced basis and transform using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6]]).to_DM().to_dfm()
        >>> M_lll, T = M.lll_transform()
        >>> M_lll
        [[2, 1, 0], [-1, 1, 3]]
        >>> T
        [[-2, 1], [3, -1]]
        >>> T.matmul(M) == M_lll
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
            Higher level interface to compute LLL-reduced basis.
        lll
            Compute LLL-reduced basis without transform matrix.
        """
        if self.domain != ZZ:
            raise DMDomainError("ZZ expected, got %s" % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError("Matrix must not have more rows than columns.")

        rep, T = self._lll(transform=True, delta=delta)
        basis = self._new_rep(rep)
        T_dfm = self._new(T, (self.rows, self.rows), self.domain)
        return basis, T_dfm


# Avoid circular imports
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.ddm import SDM
