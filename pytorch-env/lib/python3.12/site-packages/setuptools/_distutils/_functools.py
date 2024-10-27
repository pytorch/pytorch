import collections.abc
import functools


# from jaraco.functools 3.5
def pass_none(func):
    """
    Wrap func so it's not called if its first param is None

    >>> print_text = pass_none(print)
    >>> print_text('text')
    text
    >>> print_text(None)
    """

    @functools.wraps(func)
    def wrapper(param, *args, **kwargs):
        if param is not None:
            return func(param, *args, **kwargs)

    return wrapper


# from jaraco.functools 4.0
@functools.singledispatch
def _splat_inner(args, func):
    """Splat args to func."""
    return func(*args)


@_splat_inner.register
def _(args: collections.abc.Mapping, func):
    """Splat kargs to func as kwargs."""
    return func(**args)


def splat(func):
    """
    Wrap func to expect its parameters to be passed positionally in a tuple.

    Has a similar effect to that of ``itertools.starmap`` over
    simple ``map``.

    >>> import itertools, operator
    >>> pairs = [(-1, 1), (0, 2)]
    >>> _ = tuple(itertools.starmap(print, pairs))
    -1 1
    0 2
    >>> _ = tuple(map(splat(print), pairs))
    -1 1
    0 2

    The approach generalizes to other iterators that don't have a "star"
    equivalent, such as a "starfilter".

    >>> list(filter(splat(operator.add), pairs))
    [(0, 2)]

    Splat also accepts a mapping argument.

    >>> def is_nice(msg, code):
    ...     return "smile" in msg or code == 0
    >>> msgs = [
    ...     dict(msg='smile!', code=20),
    ...     dict(msg='error :(', code=1),
    ...     dict(msg='unknown', code=0),
    ... ]
    >>> for msg in filter(splat(is_nice), msgs):
    ...     print(msg)
    {'msg': 'smile!', 'code': 20}
    {'msg': 'unknown', 'code': 0}
    """
    return functools.wraps(func)(functools.partial(_splat_inner, func=func))
