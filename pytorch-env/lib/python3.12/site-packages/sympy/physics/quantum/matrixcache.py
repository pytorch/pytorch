"""A cache for storing small matrices in multiple formats."""

from sympy.core.numbers import (I, Rational, pi)
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix

from sympy.physics.quantum.matrixutils import (
    to_sympy, to_numpy, to_scipy_sparse
)


class MatrixCache:
    """A cache for small matrices in different formats.

    This class takes small matrices in the standard ``sympy.Matrix`` format,
    and then converts these to both ``numpy.matrix`` and
    ``scipy.sparse.csr_matrix`` matrices. These matrices are then stored for
    future recovery.
    """

    def __init__(self, dtype='complex'):
        self._cache = {}
        self.dtype = dtype

    def cache_matrix(self, name, m):
        """Cache a matrix by its name.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        m : list of lists
            The raw matrix data as a SymPy Matrix.
        """
        try:
            self._sympy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._numpy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._scipy_sparse_matrix(name, m)
        except ImportError:
            pass

    def get_matrix(self, name, format):
        """Get a cached matrix by name and format.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        format : str
            The format desired ('sympy', 'numpy', 'scipy.sparse')
        """
        m = self._cache.get((name, format))
        if m is not None:
            return m
        raise NotImplementedError(
            'Matrix with name %s and format %s is not available.' %
            (name, format)
        )

    def _store_matrix(self, name, format, m):
        self._cache[(name, format)] = m

    def _sympy_matrix(self, name, m):
        self._store_matrix(name, 'sympy', to_sympy(m))

    def _numpy_matrix(self, name, m):
        m = to_numpy(m, dtype=self.dtype)
        self._store_matrix(name, 'numpy', m)

    def _scipy_sparse_matrix(self, name, m):
        # TODO: explore different sparse formats. But sparse.kron will use
        # coo in most cases, so we use that here.
        m = to_scipy_sparse(m, dtype=self.dtype)
        self._store_matrix(name, 'scipy.sparse', m)


sqrt2_inv = Pow(2, Rational(-1, 2), evaluate=False)

# Save the common matrices that we will need
matrix_cache = MatrixCache()
matrix_cache.cache_matrix('eye2', Matrix([[1, 0], [0, 1]]))
matrix_cache.cache_matrix('op11', Matrix([[0, 0], [0, 1]]))  # |1><1|
matrix_cache.cache_matrix('op00', Matrix([[1, 0], [0, 0]]))  # |0><0|
matrix_cache.cache_matrix('op10', Matrix([[0, 0], [1, 0]]))  # |1><0|
matrix_cache.cache_matrix('op01', Matrix([[0, 1], [0, 0]]))  # |0><1|
matrix_cache.cache_matrix('X', Matrix([[0, 1], [1, 0]]))
matrix_cache.cache_matrix('Y', Matrix([[0, -I], [I, 0]]))
matrix_cache.cache_matrix('Z', Matrix([[1, 0], [0, -1]]))
matrix_cache.cache_matrix('S', Matrix([[1, 0], [0, I]]))
matrix_cache.cache_matrix('T', Matrix([[1, 0], [0, exp(I*pi/4)]]))
matrix_cache.cache_matrix('H', sqrt2_inv*Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('Hsqrt2', Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix(
    'SWAP', Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
matrix_cache.cache_matrix('ZX', sqrt2_inv*Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('ZY', Matrix([[I, 0], [0, -I]]))
