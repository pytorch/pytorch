from sympy.core.containers import Dict
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int, filldedent

from .sparse import MutableSparseMatrix as SparseMatrix


def _doktocsr(dok):
    """Converts a sparse matrix to Compressed Sparse Row (CSR) format.

    Parameters
    ==========

    A : contains non-zero elements sorted by key (row, column)
    JA : JA[i] is the column corresponding to A[i]
    IA : IA[i] contains the index in A for the first non-zero element
        of row[i]. Thus IA[i+1] - IA[i] gives number of non-zero
        elements row[i]. The length of IA is always 1 more than the
        number of rows in the matrix.

    Examples
    ========

    >>> from sympy.matrices.sparsetools import _doktocsr
    >>> from sympy import SparseMatrix, diag
    >>> m = SparseMatrix(diag(1, 2, 3))
    >>> m[2, 0] = -1
    >>> _doktocsr(m)
    [[1, 2, -1, 3], [0, 1, 0, 2], [0, 1, 2, 4], [3, 3]]

    """
    row, JA, A = [list(i) for i in zip(*dok.row_list())]
    IA = [0]*((row[0] if row else 0) + 1)
    for i, r in enumerate(row):
        IA.extend([i]*(r - row[i - 1]))  # if i = 0 nothing is extended
    IA.extend([len(A)]*(dok.rows - len(IA) + 1))
    shape = [dok.rows, dok.cols]
    return [A, JA, IA, shape]


def _csrtodok(csr):
    """Converts a CSR representation to DOK representation.

    Examples
    ========

    >>> from sympy.matrices.sparsetools import _csrtodok
    >>> _csrtodok([[5, 8, 3, 6], [0, 1, 2, 1], [0, 0, 2, 3, 4], [4, 3]])
    Matrix([
    [0, 0, 0],
    [5, 8, 0],
    [0, 0, 3],
    [0, 6, 0]])

    """
    smat = {}
    A, JA, IA, shape = csr
    for i in range(len(IA) - 1):
        indices = slice(IA[i], IA[i + 1])
        for l, m in zip(A[indices], JA[indices]):
            smat[i, m] = l
    return SparseMatrix(*shape, smat)


def banded(*args, **kwargs):
    """Returns a SparseMatrix from the given dictionary describing
    the diagonals of the matrix. The keys are positive for upper
    diagonals and negative for those below the main diagonal. The
    values may be:

    * expressions or single-argument functions,

    * lists or tuples of values,

    * matrices

    Unless dimensions are given, the size of the returned matrix will
    be large enough to contain the largest non-zero value provided.

    kwargs
    ======

    rows : rows of the resulting matrix; computed if
           not given.

    cols : columns of the resulting matrix; computed if
           not given.

    Examples
    ========

    >>> from sympy import banded, ones, Matrix
    >>> from sympy.abc import x

    If explicit values are given in tuples,
    the matrix will autosize to contain all values, otherwise
    a single value is filled onto the entire diagonal:

    >>> banded({1: (1, 2, 3), -1: (4, 5, 6), 0: x})
    Matrix([
    [x, 1, 0, 0],
    [4, x, 2, 0],
    [0, 5, x, 3],
    [0, 0, 6, x]])

    A function accepting a single argument can be used to fill the
    diagonal as a function of diagonal index (which starts at 0).
    The size (or shape) of the matrix must be given to obtain more
    than a 1x1 matrix:

    >>> s = lambda d: (1 + d)**2
    >>> banded(5, {0: s, 2: s, -2: 2})
    Matrix([
    [1, 0, 1,  0,  0],
    [0, 4, 0,  4,  0],
    [2, 0, 9,  0,  9],
    [0, 2, 0, 16,  0],
    [0, 0, 2,  0, 25]])

    The diagonal of matrices placed on a diagonal will coincide
    with the indicated diagonal:

    >>> vert = Matrix([1, 2, 3])
    >>> banded({0: vert}, cols=3)
    Matrix([
    [1, 0, 0],
    [2, 1, 0],
    [3, 2, 1],
    [0, 3, 2],
    [0, 0, 3]])

    >>> banded(4, {0: ones(2)})
    Matrix([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]])

    Errors are raised if the designated size will not hold
    all values an integral number of times. Here, the rows
    are designated as odd (but an even number is required to
    hold the off-diagonal 2x2 ones):

    >>> banded({0: 2, 1: ones(2)}, rows=5)
    Traceback (most recent call last):
    ...
    ValueError:
    sequence does not fit an integral number of times in the matrix

    And here, an even number of rows is given...but the square
    matrix has an even number of columns, too. As we saw
    in the previous example, an odd number is required:

    >>> banded(4, {0: 2, 1: ones(2)})  # trying to make 4x4 and cols must be odd
    Traceback (most recent call last):
    ...
    ValueError:
    sequence does not fit an integral number of times in the matrix

    A way around having to count rows is to enclosing matrix elements
    in a tuple and indicate the desired number of them to the right:

    >>> banded({0: 2, 2: (ones(2),)*3})
    Matrix([
    [2, 0, 1, 1, 0, 0, 0, 0],
    [0, 2, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 1, 0, 0],
    [0, 0, 0, 2, 1, 1, 0, 0],
    [0, 0, 0, 0, 2, 0, 1, 1],
    [0, 0, 0, 0, 0, 2, 1, 1]])

    An error will be raised if more than one value
    is written to a given entry. Here, the ones overlap
    with the main diagonal if they are placed on the
    first diagonal:

    >>> banded({0: (2,)*5, 1: (ones(2),)*3})
    Traceback (most recent call last):
    ...
    ValueError: collision at (1, 1)

    By placing a 0 at the bottom left of the 2x2 matrix of
    ones, the collision is avoided:

    >>> u2 = Matrix([
    ... [1, 1],
    ... [0, 1]])
    >>> banded({0: [2]*5, 1: [u2]*3})
    Matrix([
    [2, 1, 1, 0, 0, 0, 0],
    [0, 2, 1, 0, 0, 0, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 1]])
    """
    try:
        if len(args) not in (1, 2, 3):
            raise TypeError
        if not isinstance(args[-1], (dict, Dict)):
            raise TypeError
        if len(args) == 1:
            rows = kwargs.get('rows', None)
            cols = kwargs.get('cols', None)
            if rows is not None:
                rows = as_int(rows)
            if cols is not None:
                cols = as_int(cols)
        elif len(args) == 2:
            rows = cols = as_int(args[0])
        else:
            rows, cols = map(as_int, args[:2])
        # fails with ValueError if any keys are not ints
        _ = all(as_int(k) for k in args[-1])
    except (ValueError, TypeError):
        raise TypeError(filldedent(
            '''unrecognized input to banded:
            expecting [[row,] col,] {int: value}'''))
    def rc(d):
        # return row,col coord of diagonal start
        r = -d if d < 0 else 0
        c = 0 if r else d
        return r, c
    smat = {}
    undone = []
    tba = Dummy()
    # first handle objects with size
    for d, v in args[-1].items():
        r, c = rc(d)
        # note: only list and tuple are recognized since this
        # will allow other Basic objects like Tuple
        # into the matrix if so desired
        if isinstance(v, (list, tuple)):
            extra = 0
            for i, vi in enumerate(v):
                i += extra
                if is_sequence(vi):
                    vi = SparseMatrix(vi)
                    smat[r + i, c + i] = vi
                    extra += min(vi.shape) - 1
                else:
                    smat[r + i, c + i] = vi
        elif is_sequence(v):
            v = SparseMatrix(v)
            rv, cv = v.shape
            if rows and cols:
                nr, xr = divmod(rows - r, rv)
                nc, xc = divmod(cols - c, cv)
                x = xr or xc
                do = min(nr, nc)
            elif rows:
                do, x = divmod(rows - r, rv)
            elif cols:
                do, x = divmod(cols - c, cv)
            else:
                do = 1
                x = 0
            if x:
                raise ValueError(filldedent('''
                    sequence does not fit an integral number of times
                    in the matrix'''))
            j = min(v.shape)
            for i in range(do):
                smat[r, c] = v
                r += j
                c += j
        elif v:
            smat[r, c] = tba
            undone.append((d, v))
    s = SparseMatrix(None, smat)  # to expand matrices
    smat = s.todok()
    # check for dim errors here
    if rows is not None and rows < s.rows:
        raise ValueError('Designated rows %s < needed %s' % (rows, s.rows))
    if cols is not None and cols < s.cols:
        raise ValueError('Designated cols %s < needed %s' % (cols, s.cols))
    if rows is cols is None:
        rows = s.rows
        cols = s.cols
    elif rows is not None and cols is None:
        cols = max(rows, s.cols)
    elif cols is not None and rows is None:
        rows = max(cols, s.rows)
    def update(i, j, v):
        # update smat and make sure there are
        # no collisions
        if v:
            if (i, j) in smat and smat[i, j] not in (tba, v):
                raise ValueError('collision at %s' % ((i, j),))
            smat[i, j] = v
    if undone:
        for d, vi in undone:
            r, c = rc(d)
            v = vi if callable(vi) else lambda _: vi
            i = 0
            while r + i < rows and c + i < cols:
                update(r + i, c + i, v(i))
                i += 1
    return SparseMatrix(rows, cols, smat)
