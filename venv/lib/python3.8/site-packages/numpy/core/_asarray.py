"""
Functions in the ``as*array`` family that promote array-likes into arrays.

`require` fits this category despite its name not matching this pattern.
"""
from .overrides import set_module
from .multiarray import array


__all__ = [
    "asarray", "asanyarray", "ascontiguousarray", "asfortranarray", "require",
]

@set_module('numpy')
def asarray(a, dtype=None, order=None):
    """Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or
        column-major (Fortran-style) memory representation.
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.

    See Also
    --------
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfarray : Convert input to a floating point ndarray.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.asarray(a, dtype=np.float32) is a
    True
    >>> np.asarray(a, dtype=np.float64) is a
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.recarray, np.ndarray)
    True
    >>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asarray(a) is a
    False
    >>> np.asanyarray(a) is a
    True

    """
    return array(a, dtype, copy=False, order=order)


@set_module('numpy')
def asanyarray(a, dtype=None, order=None):
    """Convert the input to an ndarray, but pass ndarray subclasses through.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or column-major
        (Fortran-style) memory representation.  Defaults to 'C'.

    Returns
    -------
    out : ndarray or an ndarray subclass
        Array interpretation of `a`.  If `a` is an ndarray or a subclass
        of ndarray, it is returned as-is and no copy is performed.

    See Also
    --------
    asarray : Similar function which always returns ndarrays.
    ascontiguousarray : Convert input to a contiguous array.
    asfarray : Convert input to a floating point ndarray.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and
                        Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asanyarray(a)
    array([1, 2])

    Instances of `ndarray` subclasses are passed through as-is:

    >>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asanyarray(a) is a
    True

    """
    return array(a, dtype, copy=False, order=order, subok=True)


@set_module('numpy')
def ascontiguousarray(a, dtype=None):
    """
    Return a contiguous array (ndim >= 1) in memory (C order).

    Parameters
    ----------
    a : array_like
        Input array.
    dtype : str or dtype object, optional
        Data-type of returned array.

    Returns
    -------
    out : ndarray
        Contiguous array of same shape and content as `a`, with type `dtype`
        if specified.

    See Also
    --------
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    require : Return an ndarray that satisfies requirements.
    ndarray.flags : Information about the memory layout of the array.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> np.ascontiguousarray(x, dtype=np.float32)
    array([[0., 1., 2.],
           [3., 4., 5.]], dtype=float32)
    >>> x.flags['C_CONTIGUOUS']
    True

    Note: This function returns an array with at least one-dimension (1-d) 
    so it will not preserve 0-d arrays.  

    """
    return array(a, dtype, copy=False, order='C', ndmin=1)


@set_module('numpy')
def asfortranarray(a, dtype=None):
    """
    Return an array (ndim >= 1) laid out in Fortran order in memory.

    Parameters
    ----------
    a : array_like
        Input array.
    dtype : str or dtype object, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : ndarray
        The input `a` in Fortran, or column-major, order.

    See Also
    --------
    ascontiguousarray : Convert input to a contiguous (C order) array.
    asanyarray : Convert input to an ndarray with either row or
        column-major memory order.
    require : Return an ndarray that satisfies requirements.
    ndarray.flags : Information about the memory layout of the array.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> y = np.asfortranarray(x)
    >>> x.flags['F_CONTIGUOUS']
    False
    >>> y.flags['F_CONTIGUOUS']
    True

    Note: This function returns an array with at least one-dimension (1-d) 
    so it will not preserve 0-d arrays.  

    """
    return array(a, dtype, copy=False, order='F', ndmin=1)


@set_module('numpy')
def require(a, dtype=None, requirements=None):
    """
    Return an ndarray of the provided type that satisfies requirements.

    This function is useful to be sure that an array with the correct flags
    is returned for passing to compiled code (perhaps through ctypes).

    Parameters
    ----------
    a : array_like
       The object to be converted to a type-and-requirement-satisfying array.
    dtype : data-type
       The required data-type. If None preserve the current dtype. If your
       application requires the data to be in native byteorder, include
       a byteorder specification as a part of the dtype specification.
    requirements : str or list of str
       The requirements list can be any of the following

       * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array
       * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array
       * 'ALIGNED' ('A')      - ensure a data-type aligned array
       * 'WRITEABLE' ('W')    - ensure a writable array
       * 'OWNDATA' ('O')      - ensure an array that owns its own data
       * 'ENSUREARRAY', ('E') - ensure a base array, instead of a subclass

    Returns
    -------
    out : ndarray
        Array with specified requirements and type if given.

    See Also
    --------
    asarray : Convert input to an ndarray.
    asanyarray : Convert to an ndarray, but pass through ndarray subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    ndarray.flags : Information about the memory layout of the array.

    Notes
    -----
    The returned array will be guaranteed to have the listed requirements
    by making a copy if needed.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> x.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : False
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False
      UPDATEIFCOPY : False

    >>> y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])
    >>> y.flags
      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False
      UPDATEIFCOPY : False

    """
    possible_flags = {'C': 'C', 'C_CONTIGUOUS': 'C', 'CONTIGUOUS': 'C',
                      'F': 'F', 'F_CONTIGUOUS': 'F', 'FORTRAN': 'F',
                      'A': 'A', 'ALIGNED': 'A',
                      'W': 'W', 'WRITEABLE': 'W',
                      'O': 'O', 'OWNDATA': 'O',
                      'E': 'E', 'ENSUREARRAY': 'E'}
    if not requirements:
        return asanyarray(a, dtype=dtype)
    else:
        requirements = {possible_flags[x.upper()] for x in requirements}

    if 'E' in requirements:
        requirements.remove('E')
        subok = False
    else:
        subok = True

    order = 'A'
    if requirements >= {'C', 'F'}:
        raise ValueError('Cannot specify both "C" and "F" order')
    elif 'F' in requirements:
        order = 'F'
        requirements.remove('F')
    elif 'C' in requirements:
        order = 'C'
        requirements.remove('C')

    arr = array(a, dtype=dtype, order=order, copy=False, subok=subok)

    for prop in requirements:
        if not arr.flags[prop]:
            arr = arr.copy(order)
            break
    return arr
