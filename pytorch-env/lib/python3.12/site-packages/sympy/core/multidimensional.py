"""
Provides functionality for multidimensional usage of scalar-functions.

Read the vectorize docstring for more details.
"""

from functools import wraps


def apply_on_element(f, args, kwargs, n):
    """
    Returns a structure with the same dimension as the specified argument,
    where each basic element is replaced by the function f applied on it. All
    other arguments stay the same.
    """
    # Get the specified argument.
    if isinstance(n, int):
        structure = args[n]
        is_arg = True
    elif isinstance(n, str):
        structure = kwargs[n]
        is_arg = False

    # Define reduced function that is only dependent on the specified argument.
    def f_reduced(x):
        if hasattr(x, "__iter__"):
            return list(map(f_reduced, x))
        else:
            if is_arg:
                args[n] = x
            else:
                kwargs[n] = x
            return f(*args, **kwargs)

    # f_reduced will call itself recursively so that in the end f is applied to
    # all basic elements.
    return list(map(f_reduced, structure))


def iter_copy(structure):
    """
    Returns a copy of an iterable object (also copying all embedded iterables).
    """
    return [iter_copy(i) if hasattr(i, "__iter__") else i for i in structure]


def structure_copy(structure):
    """
    Returns a copy of the given structure (numpy-array, list, iterable, ..).
    """
    if hasattr(structure, "copy"):
        return structure.copy()
    return iter_copy(structure)


class vectorize:
    """
    Generalizes a function taking scalars to accept multidimensional arguments.

    Examples
    ========

    >>> from sympy import vectorize, diff, sin, symbols, Function
    >>> x, y, z = symbols('x y z')
    >>> f, g, h = list(map(Function, 'fgh'))

    >>> @vectorize(0)
    ... def vsin(x):
    ...     return sin(x)

    >>> vsin([1, x, y])
    [sin(1), sin(x), sin(y)]

    >>> @vectorize(0, 1)
    ... def vdiff(f, y):
    ...     return diff(f, y)

    >>> vdiff([f(x, y, z), g(x, y, z), h(x, y, z)], [x, y, z])
    [[Derivative(f(x, y, z), x), Derivative(f(x, y, z), y), Derivative(f(x, y, z), z)], [Derivative(g(x, y, z), x), Derivative(g(x, y, z), y), Derivative(g(x, y, z), z)], [Derivative(h(x, y, z), x), Derivative(h(x, y, z), y), Derivative(h(x, y, z), z)]]
    """
    def __init__(self, *mdargs):
        """
        The given numbers and strings characterize the arguments that will be
        treated as data structures, where the decorated function will be applied
        to every single element.
        If no argument is given, everything is treated multidimensional.
        """
        for a in mdargs:
            if not isinstance(a, (int, str)):
                raise TypeError("a is of invalid type")
        self.mdargs = mdargs

    def __call__(self, f):
        """
        Returns a wrapper for the one-dimensional function that can handle
        multidimensional arguments.
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get arguments that should be treated multidimensional
            if self.mdargs:
                mdargs = self.mdargs
            else:
                mdargs = range(len(args)) + kwargs.keys()

            arglength = len(args)

            for n in mdargs:
                if isinstance(n, int):
                    if n >= arglength:
                        continue
                    entry = args[n]
                    is_arg = True
                elif isinstance(n, str):
                    try:
                        entry = kwargs[n]
                    except KeyError:
                        continue
                    is_arg = False
                if hasattr(entry, "__iter__"):
                    # Create now a copy of the given array and manipulate then
                    # the entries directly.
                    if is_arg:
                        args = list(args)
                        args[n] = structure_copy(entry)
                    else:
                        kwargs[n] = structure_copy(entry)
                    result = apply_on_element(wrapper, args, kwargs, n)
                    return result
            return f(*args, **kwargs)
        return wrapper
