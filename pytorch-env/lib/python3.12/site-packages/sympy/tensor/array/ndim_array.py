from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable

import itertools
from collections.abc import Iterable


class ArrayKind(Kind):
    """
    Kind for N-dimensional array in SymPy.

    This kind represents the multidimensional array that algebraic
    operations are defined. Basic class for this kind is ``NDimArray``,
    but any expression representing the array can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is :obj:NumberKind `<sympy.core.kind.NumberKind>`,
        which means that the array contains only numbers.

    Examples
    ========

    Any instance of array class has ``ArrayKind``.

    >>> from sympy import NDimArray
    >>> NDimArray([1,2,3]).kind
    ArrayKind(NumberKind)

    Although expressions representing an array may be not instance of
    array class, it will have ``ArrayKind`` as well.

    >>> from sympy import Integral
    >>> from sympy.tensor.array import NDimArray
    >>> from sympy.abc import x
    >>> intA = Integral(NDimArray([1,2,3]), x)
    >>> isinstance(intA, NDimArray)
    False
    >>> intA.kind
    ArrayKind(NumberKind)

    Use ``isinstance()`` to check for ``ArrayKind` without specifying
    the element kind. Use ``is`` with specifying the element kind.

    >>> from sympy.tensor.array import ArrayKind
    >>> from sympy.core import NumberKind
    >>> boolA = NDimArray([True, False])
    >>> isinstance(boolA.kind, ArrayKind)
    True
    >>> boolA.kind is ArrayKind(NumberKind)
    False

    See Also
    ========

    shape : Function to return the shape of objects with ``MatrixKind``.

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "ArrayKind(%s)" % self.element_kind

    @classmethod
    def _union(cls, kinds) -> 'ArrayKind':
        elem_kinds = {e.kind for e in kinds}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return ArrayKind(elemkind)


class NDimArray(Printable):
    """N-dimensional array.

    Examples
    ========

    Create an N-dim array of zeros:

    >>> from sympy import MutableDenseNDimArray
    >>> a = MutableDenseNDimArray.zeros(2, 3, 4)
    >>> a
    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    Create an N-dim array from a list;

    >>> a = MutableDenseNDimArray([[2, 3], [4, 5]])
    >>> a
    [[2, 3], [4, 5]]

    >>> b = MutableDenseNDimArray([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    >>> b
    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]

    Create an N-dim array from a flat list with dimension shape:

    >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
    >>> a
    [[1, 2, 3], [4, 5, 6]]

    Create an N-dim array from a matrix:

    >>> from sympy import Matrix
    >>> a = Matrix([[1,2],[3,4]])
    >>> a
    Matrix([
    [1, 2],
    [3, 4]])
    >>> b = MutableDenseNDimArray(a)
    >>> b
    [[1, 2], [3, 4]]

    Arithmetic operations on N-dim arrays

    >>> a = MutableDenseNDimArray([1, 1, 1, 1], (2, 2))
    >>> b = MutableDenseNDimArray([4, 4, 4, 4], (2, 2))
    >>> c = a + b
    >>> c
    [[5, 5], [5, 5]]
    >>> a - b
    [[-3, -3], [-3, -3]]

    """

    _diff_wrt = True
    is_scalar = False

    def __new__(cls, iterable, shape=None, **kwargs):
        from sympy.tensor.array import ImmutableDenseNDimArray
        return ImmutableDenseNDimArray(iterable, shape, **kwargs)

    def __getitem__(self, index):
        raise NotImplementedError("A subclass of NDimArray should implement __getitem__")

    def _parse_index(self, index):
        if isinstance(index, (SYMPY_INTS, Integer)):
            if index >= self._loop_size:
                raise ValueError("Only a tuple index is accepted")
            return index

        if self._loop_size == 0:
            raise ValueError("Index not valid with an empty array")

        if len(index) != self._rank:
            raise ValueError('Wrong number of array axes')

        real_index = 0
        # check if input index can exist in current indexing
        for i in range(self._rank):
            if (index[i] >= self.shape[i]) or (index[i] < -self.shape[i]):
                raise ValueError('Index ' + str(index) + ' out of border')
            if index[i] < 0:
                real_index += 1
            real_index = real_index*self.shape[i] + index[i]

        return real_index

    def _get_tuple_index(self, integer_index):
        index = []
        for sh in reversed(self.shape):
            index.append(integer_index % sh)
            integer_index //= sh
        index.reverse()
        return tuple(index)

    def _check_symbolic_index(self, index):
        # Check if any index is symbolic:
        tuple_index = (index if isinstance(index, tuple) else (index,))
        if any((isinstance(i, Expr) and (not i.is_number)) for i in tuple_index):
            for i, nth_dim in zip(tuple_index, self.shape):
                if ((i < 0) == True) or ((i >= nth_dim) == True):
                    raise ValueError("index out of range")
            from sympy.tensor import Indexed
            return Indexed(self, *tuple_index)
        return None

    def _setter_iterable_check(self, value):
        from sympy.matrices.matrixbase import MatrixBase
        if isinstance(value, (Iterable, MatrixBase, NDimArray)):
            raise NotImplementedError

    @classmethod
    def _scan_iterable_shape(cls, iterable):
        def f(pointer):
            if not isinstance(pointer, Iterable):
                return [pointer], ()

            if len(pointer) == 0:
                return [], (0,)

            result = []
            elems, shapes = zip(*[f(i) for i in pointer])
            if len(set(shapes)) != 1:
                raise ValueError("could not determine shape unambiguously")
            for i in elems:
                result.extend(i)
            return result, (len(shapes),)+shapes[0]

        return f(iterable)

    @classmethod
    def _handle_ndarray_creation_inputs(cls, iterable=None, shape=None, **kwargs):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray

        if shape is None:
            if iterable is None:
                shape = ()
                iterable = ()
            # Construction of a sparse array from a sparse array
            elif isinstance(iterable, SparseNDimArray):
                return iterable._shape, iterable._sparse_array

            # Construct N-dim array from another N-dim array:
            elif isinstance(iterable, NDimArray):
                shape = iterable.shape

            # Construct N-dim array from an iterable (numpy arrays included):
            elif isinstance(iterable, Iterable):
                iterable, shape = cls._scan_iterable_shape(iterable)

            # Construct N-dim array from a Matrix:
            elif isinstance(iterable, MatrixBase):
                shape = iterable.shape

            else:
                shape = ()
                iterable = (iterable,)

        if isinstance(iterable, (Dict, dict)) and shape is not None:
            new_dict = iterable.copy()
            for k in new_dict:
                if isinstance(k, (tuple, Tuple)):
                    new_key = 0
                    for i, idx in enumerate(k):
                        new_key = new_key * shape[i] + idx
                    iterable[new_key] = iterable[k]
                    del iterable[k]

        if isinstance(shape, (SYMPY_INTS, Integer)):
            shape = (shape,)

        if not all(isinstance(dim, (SYMPY_INTS, Integer)) for dim in shape):
            raise TypeError("Shape should contain integers only.")

        return tuple(shape), iterable

    def __len__(self):
        """Overload common function len(). Returns number of elements in array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        >>> len(a)
        9

        """
        return self._loop_size

    @property
    def shape(self):
        """
        Returns array shape (dimension).

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a.shape
        (3, 3)

        """
        return self._shape

    def rank(self):
        """
        Returns rank of array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3,4,5,6,3)
        >>> a.rank()
        5

        """
        return self._rank

    def diff(self, *args, **kwargs):
        """
        Calculate the derivative of each element in the array.

        Examples
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> from sympy.abc import x, y
        >>> M = ImmutableDenseNDimArray([[x, y], [1, x*y]])
        >>> M.diff(x)
        [[1, 0], [0, y]]

        """
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        kwargs.setdefault('evaluate', True)
        return ArrayDerivative(self.as_immutable(), *args, **kwargs)

    def _eval_derivative(self, base):
        # Types are (base: scalar, self: array)
        return self.applyfunc(lambda x: base.diff(x))

    def _eval_derivative_n_times(self, s, n):
        return Basic._eval_derivative_n_times(self, s, n)

    def applyfunc(self, f):
        """Apply a function to each element of the N-dim array.

        Examples
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> m = ImmutableDenseNDimArray([i*2+j for i in range(2) for j in range(2)], (2, 2))
        >>> m
        [[0, 1], [2, 3]]
        >>> m.applyfunc(lambda i: 2*i)
        [[0, 2], [4, 6]]
        """
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(self, SparseNDimArray) and f(S.Zero) == 0:
            return type(self)({k: f(v) for k, v in self._sparse_array.items() if f(v) != 0}, self.shape)

        return type(self)(map(f, Flatten(self)), self.shape)

    def _sympystr(self, printer):
        def f(sh, shape_left, i, j):
            if len(shape_left) == 1:
                return "["+", ".join([printer._print(self[self._get_tuple_index(e)]) for e in range(i, j)])+"]"

            sh //= shape_left[0]
            return "[" + ", ".join([f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh) for e in range(shape_left[0])]) + "]" # + "\n"*len(shape_left)

        if self.rank() == 0:
            return printer._print(self[()])

        return f(self._loop_size, self.shape, 0, self._loop_size)

    def tolist(self):
        """
        Converting MutableDenseNDimArray to one-dim list

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))
        >>> a
        [[1, 2], [3, 4]]
        >>> b = a.tolist()
        >>> b
        [[1, 2], [3, 4]]
        """

        def f(sh, shape_left, i, j):
            if len(shape_left) == 1:
                return [self[self._get_tuple_index(e)] for e in range(i, j)]
            result = []
            sh //= shape_left[0]
            for e in range(shape_left[0]):
                result.append(f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh))
            return result

        return f(self._loop_size, self.shape, 0, self._loop_size)

    def __add__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        if not isinstance(other, NDimArray):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i+j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __sub__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        if not isinstance(other, NDimArray):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i-j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __mul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [i*other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rmul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [other*i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __truediv__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected")

        other = sympify(other)
        if isinstance(self, SparseNDimArray) and other != S.Zero:
            return type(self)({k: v/other for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [i/other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rtruediv__(self, other):
        raise NotImplementedError('unsupported operation on NDimArray')

    def __neg__(self):
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(self, SparseNDimArray):
            return type(self)({k: -v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [-i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __iter__(self):
        def iterator():
            if self._shape:
                for i in range(self._shape[0]):
                    yield self[i]
            else:
                yield self[()]

        return iterator()

    def __eq__(self, other):
        """
        NDimArray instances can be compared to each other.
        Instances equal if they have same shape and data.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2, 3)
        >>> b = MutableDenseNDimArray.zeros(2, 3)
        >>> a == b
        True
        >>> c = a.reshape(3, 2)
        >>> c == b
        False
        >>> a[0,0] = 1
        >>> b[0,0] = 2
        >>> a == b
        False
        """
        from sympy.tensor.array import SparseNDimArray
        if not isinstance(other, NDimArray):
            return False

        if not self.shape == other.shape:
            return False

        if isinstance(self, SparseNDimArray) and isinstance(other, SparseNDimArray):
            return dict(self._sparse_array) == dict(other._sparse_array)

        return list(self) == list(other)

    def __ne__(self, other):
        return not self == other

    def _eval_transpose(self):
        if self.rank() != 2:
            raise ValueError("array rank not 2")
        from .arrayop import permutedims
        return permutedims(self, (1, 0))

    def transpose(self):
        return self._eval_transpose()

    def _eval_conjugate(self):
        from sympy.tensor.array.arrayop import Flatten

        return self.func([i.conjugate() for i in Flatten(self)], self.shape)

    def conjugate(self):
        return self._eval_conjugate()

    def _eval_adjoint(self):
        return self.transpose().conjugate()

    def adjoint(self):
        return self._eval_adjoint()

    def _slice_expand(self, s, dim):
        if not isinstance(s, slice):
                return (s,)
        start, stop, step = s.indices(dim)
        return [start + i*step for i in range((stop-start)//step)]

    def _get_slice_data_for_array_access(self, index):
        sl_factors = [self._slice_expand(i, dim) for (i, dim) in zip(index, self.shape)]
        eindices = itertools.product(*sl_factors)
        return sl_factors, eindices

    def _get_slice_data_for_array_assignment(self, index, value):
        if not isinstance(value, NDimArray):
            value = type(self)(value)
        sl_factors, eindices = self._get_slice_data_for_array_access(index)
        slice_offsets = [min(i) if isinstance(i, list) else None for i in sl_factors]
        # TODO: add checks for dimensions for `value`?
        return value, eindices, slice_offsets

    @classmethod
    def _check_special_bounds(cls, flat_list, shape):
        if shape == () and len(flat_list) != 1:
            raise ValueError("arrays without shape need one scalar value")
        if shape == (0,) and len(flat_list) > 0:
            raise ValueError("if array shape is (0,) there cannot be elements")

    def _check_index_for_getitem(self, index):
        if isinstance(index, (SYMPY_INTS, Integer, slice)):
            index = (index,)

        if len(index) < self.rank():
            index = tuple(index) + \
                          tuple(slice(None) for i in range(len(index), self.rank()))

        if len(index) > self.rank():
            raise ValueError('Dimension of index greater than rank of array')

        return index


class ImmutableNDimArray(NDimArray, Basic):
    _op_priority = 11.0

    def __hash__(self):
        return Basic.__hash__(self)

    def as_immutable(self):
        return self

    def as_mutable(self):
        raise NotImplementedError("abstract method")
