"""Miscellaneous stuff that does not really fit anywhere else."""

from __future__ import annotations

import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent


class Undecidable(ValueError):
    # an error to be raised when a decision cannot be made definitively
    # where a definitive answer is needed
    pass


def filldedent(s, w=70, **kwargs):
    """
    Strips leading and trailing empty lines from a copy of ``s``, then dedents,
    fills and returns it.

    Empty line stripping serves to deal with docstrings like this one that
    start with a newline after the initial triple quote, inserting an empty
    line at the beginning of the string.

    Additional keyword arguments will be passed to ``textwrap.fill()``.

    See Also
    ========
    strlines, rawlines

    """
    return '\n' + fill(dedent(str(s)).strip('\n'), width=w, **kwargs)


def strlines(s, c=64, short=False):
    """Return a cut-and-pastable string that, when printed, is
    equivalent to the input.  The lines will be surrounded by
    parentheses and no line will be longer than c (default 64)
    characters. If the line contains newlines characters, the
    `rawlines` result will be returned.  If ``short`` is True
    (default is False) then if there is one line it will be
    returned without bounding parentheses.

    Examples
    ========

    >>> from sympy.utilities.misc import strlines
    >>> q = 'this is a long string that should be broken into shorter lines'
    >>> print(strlines(q, 40))
    (
    'this is a long string that should be b'
    'roken into shorter lines'
    )
    >>> q == (
    ... 'this is a long string that should be b'
    ... 'roken into shorter lines'
    ... )
    True

    See Also
    ========
    filldedent, rawlines
    """
    if not isinstance(s, str):
        raise ValueError('expecting string input')
    if '\n' in s:
        return rawlines(s)
    q = '"' if repr(s).startswith('"') else "'"
    q = (q,)*2
    if '\\' in s:  # use r-string
        m = '(\nr%s%%s%s\n)' % q
        j = '%s\nr%s' % q
        c -= 3
    else:
        m = '(\n%s%%s%s\n)' % q
        j = '%s\n%s' % q
        c -= 2
    out = []
    while s:
        out.append(s[:c])
        s=s[c:]
    if short and len(out) == 1:
        return (m % out[0]).splitlines()[1]  # strip bounding (\n...\n)
    return m % j.join(out)


def rawlines(s):
    """Return a cut-and-pastable string that, when printed, is equivalent
    to the input. Use this when there is more than one line in the
    string. The string returned is formatted so it can be indented
    nicely within tests; in some cases it is wrapped in the dedent
    function which has to be imported from textwrap.

    Examples
    ========

    Note: because there are characters in the examples below that need
    to be escaped because they are themselves within a triple quoted
    docstring, expressions below look more complicated than they would
    be if they were printed in an interpreter window.

    >>> from sympy.utilities.misc import rawlines
    >>> from sympy import TableForm
    >>> s = str(TableForm([[1, 10]], headings=(None, ['a', 'bee'])))
    >>> print(rawlines(s))
    (
        'a bee\\n'
        '-----\\n'
        '1 10 '
    )
    >>> print(rawlines('''this
    ... that'''))
    dedent('''\\
        this
        that''')

    >>> print(rawlines('''this
    ... that
    ... '''))
    dedent('''\\
        this
        that
        ''')

    >>> s = \"\"\"this
    ... is a triple '''
    ... \"\"\"
    >>> print(rawlines(s))
    dedent(\"\"\"\\
        this
        is a triple '''
        \"\"\")

    >>> print(rawlines('''this
    ... that
    ...     '''))
    (
        'this\\n'
        'that\\n'
        '    '
    )

    See Also
    ========
    filldedent, strlines
    """
    lines = s.split('\n')
    if len(lines) == 1:
        return repr(lines[0])
    triple = ["'''" in s, '"""' in s]
    if any(li.endswith(' ') for li in lines) or '\\' in s or all(triple):
        rv = []
        # add on the newlines
        trailing = s.endswith('\n')
        last = len(lines) - 1
        for i, li in enumerate(lines):
            if i != last or trailing:
                rv.append(repr(li + '\n'))
            else:
                rv.append(repr(li))
        return '(\n    %s\n)' % '\n    '.join(rv)
    else:
        rv = '\n    '.join(lines)
        if triple[0]:
            return 'dedent("""\\\n    %s""")' % rv
        else:
            return "dedent('''\\\n    %s''')" % rv

ARCH = str(struct.calcsize('P') * 8) + "-bit"


# XXX: PyPy does not support hash randomization
HASH_RANDOMIZATION = getattr(sys.flags, 'hash_randomization', False)

_debug_tmp: list[str] = []
_debug_iter = 0

def debug_decorator(func):
    """If SYMPY_DEBUG is True, it will print a nice execution tree with
    arguments and results of all decorated functions, else do nothing.
    """
    from sympy import SYMPY_DEBUG

    if not SYMPY_DEBUG:
        return func

    def maketree(f, *args, **kw):
        global _debug_tmp
        global _debug_iter
        oldtmp = _debug_tmp
        _debug_tmp = []
        _debug_iter += 1

        def tree(subtrees):
            def indent(s, variant=1):
                x = s.split("\n")
                r = "+-%s\n" % x[0]
                for a in x[1:]:
                    if a == "":
                        continue
                    if variant == 1:
                        r += "| %s\n" % a
                    else:
                        r += "  %s\n" % a
                return r
            if len(subtrees) == 0:
                return ""
            f = []
            for a in subtrees[:-1]:
                f.append(indent(a))
            f.append(indent(subtrees[-1], 2))
            return ''.join(f)

        # If there is a bug and the algorithm enters an infinite loop, enable the
        # following lines. It will print the names and parameters of all major functions
        # that are called, *before* they are called
        #from functools import reduce
        #print("%s%s %s%s" % (_debug_iter, reduce(lambda x, y: x + y, \
        #    map(lambda x: '-', range(1, 2 + _debug_iter))), f.__name__, args))

        r = f(*args, **kw)

        _debug_iter -= 1
        s = "%s%s = %s\n" % (f.__name__, args, r)
        if _debug_tmp != []:
            s += tree(_debug_tmp)
        _debug_tmp = oldtmp
        _debug_tmp.append(s)
        if _debug_iter == 0:
            print(_debug_tmp[0])
            _debug_tmp = []
        return r

    def decorated(*args, **kwargs):
        return maketree(func, *args, **kwargs)

    return decorated


def debug(*args):
    """
    Print ``*args`` if SYMPY_DEBUG is True, else do nothing.
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(*args, file=sys.stderr)


def debugf(string, args):
    """
    Print ``string%args`` if SYMPY_DEBUG is True, else do nothing. This is
    intended for debug messages using formatted strings.
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(string%args, file=sys.stderr)


def find_executable(executable, path=None):
    """Try to find 'executable' in the directories listed in 'path' (a
    string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH']).  Returns the complete filename or None if not
    found
    """
    from .exceptions import sympy_deprecation_warning
    sympy_deprecation_warning(
        """
        sympy.utilities.misc.find_executable() is deprecated. Use the standard
        library shutil.which() function instead.
        """,
        deprecated_since_version="1.7",
        active_deprecations_target="deprecated-find-executable",
    )
    if path is None:
        path = os.environ['PATH']
    paths = path.split(os.pathsep)
    extlist = ['']
    if os.name == 'os2':
        (base, ext) = os.path.splitext(executable)
        # executable files on OS/2 can have an arbitrary extension, but
        # .exe is automatically appended if no dot is present in the name
        if not ext:
            executable = executable + ".exe"
    elif sys.platform == 'win32':
        pathext = os.environ['PATHEXT'].lower().split(os.pathsep)
        (base, ext) = os.path.splitext(executable)
        if ext.lower() not in pathext:
            extlist = pathext
    for ext in extlist:
        execname = executable + ext
        if os.path.isfile(execname):
            return execname
        else:
            for p in paths:
                f = os.path.join(p, execname)
                if os.path.isfile(f):
                    return f

    return None


def func_name(x, short=False):
    """Return function name of `x` (if defined) else the `type(x)`.
    If short is True and there is a shorter alias for the result,
    return the alias.

    Examples
    ========

    >>> from sympy.utilities.misc import func_name
    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> func_name(Matrix.eye(3))
    'MutableDenseMatrix'
    >>> func_name(x < 1)
    'StrictLessThan'
    >>> func_name(x < 1, short=True)
    'Lt'
    """
    alias = {
    'GreaterThan': 'Ge',
    'StrictGreaterThan': 'Gt',
    'LessThan': 'Le',
    'StrictLessThan': 'Lt',
    'Equality': 'Eq',
    'Unequality': 'Ne',
    }
    typ = type(x)
    if str(typ).startswith("<type '"):
        typ = str(typ).split("'")[1].split("'")[0]
    elif str(typ).startswith("<class '"):
        typ = str(typ).split("'")[1].split("'")[0]
    rv = getattr(getattr(x, 'func', x), '__name__', typ)
    if '.' in rv:
        rv = rv.split('.')[-1]
    if short:
        rv = alias.get(rv, rv)
    return rv


def _replace(reps):
    """Return a function that can make the replacements, given in
    ``reps``, on a string. The replacements should be given as mapping.

    Examples
    ========

    >>> from sympy.utilities.misc import _replace
    >>> f = _replace(dict(foo='bar', d='t'))
    >>> f('food')
    'bart'
    >>> f = _replace({})
    >>> f('food')
    'food'
    """
    if not reps:
        return lambda x: x
    D = lambda match: reps[match.group(0)]
    pattern = _re.compile("|".join(
        [_re.escape(k) for k, v in reps.items()]), _re.M)
    return lambda string: pattern.sub(D, string)


def replace(string, *reps):
    """Return ``string`` with all keys in ``reps`` replaced with
    their corresponding values, longer strings first, irrespective
    of the order they are given.  ``reps`` may be passed as tuples
    or a single mapping.

    Examples
    ========

    >>> from sympy.utilities.misc import replace
    >>> replace('foo', {'oo': 'ar', 'f': 'b'})
    'bar'
    >>> replace("spamham sha", ("spam", "eggs"), ("sha","md5"))
    'eggsham md5'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap (i.e. are the same
    length and have some identical sequence at the
    beginning/end):

    >>> reps = [
    ...     ('ab', 'x'),
    ...     ('bc', 'y')]
    >>> replace('abc', *reps) in ('xc', 'ay')
    True

    References
    ==========

    .. [1] https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
    """
    if len(reps) == 1:
        kv = reps[0]
        if isinstance(kv, dict):
            reps = kv
        else:
            return string.replace(*kv)
    else:
        reps = dict(reps)
    return _replace(reps)(string)


def translate(s, a, b=None, c=None):
    """Return ``s`` where characters have been replaced or deleted.

    SYNTAX
    ======

    translate(s, None, deletechars):
        all characters in ``deletechars`` are deleted
    translate(s, map [,deletechars]):
        all characters in ``deletechars`` (if provided) are deleted
        then the replacements defined by map are made; if the keys
        of map are strings then the longer ones are handled first.
        Multicharacter deletions should have a value of ''.
    translate(s, oldchars, newchars, deletechars)
        all characters in ``deletechars`` are deleted
        then each character in ``oldchars`` is replaced with the
        corresponding character in ``newchars``

    Examples
    ========

    >>> from sympy.utilities.misc import translate
    >>> abc = 'abc'
    >>> translate(abc, None, 'a')
    'bc'
    >>> translate(abc, {'a': 'x'}, 'c')
    'xb'
    >>> translate(abc, {'abc': 'x', 'a': 'y'})
    'x'

    >>> translate('abcd', 'ac', 'AC', 'd')
    'AbC'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap are the same
    length and have some identical sequences at the
    beginning/end:

    >>> translate(abc, {'ab': 'x', 'bc': 'y'}) in ('xc', 'ay')
    True
    """

    mr = {}
    if a is None:
        if c is not None:
            raise ValueError('c should be None when a=None is passed, instead got %s' % c)
        if b is None:
            return s
        c = b
        a = b = ''
    else:
        if isinstance(a, dict):
            short = {}
            for k in list(a.keys()):
                if len(k) == 1 and len(a[k]) == 1:
                    short[k] = a.pop(k)
            mr = a
            c = b
            if short:
                a, b = [''.join(i) for i in list(zip(*short.items()))]
            else:
                a = b = ''
        elif len(a) != len(b):
            raise ValueError('oldchars and newchars have different lengths')

    if c:
        val = str.maketrans('', '', c)
        s = s.translate(val)
    s = replace(s, mr)
    n = str.maketrans(a, b)
    return s.translate(n)


def ordinal(num):
    """Return ordinal number string of num, e.g. 1 becomes 1st.
    """
    # modified from https://codereview.stackexchange.com/questions/41298/producing-ordinal-numbers
    n = as_int(num)
    k = abs(n) % 100
    if 11 <= k <= 13:
        suffix = 'th'
    elif k % 10 == 1:
        suffix = 'st'
    elif k % 10 == 2:
        suffix = 'nd'
    elif k % 10 == 3:
        suffix = 'rd'
    else:
        suffix = 'th'
    return str(n) + suffix


def as_int(n, strict=True):
    """
    Convert the argument to a builtin integer.

    The return value is guaranteed to be equal to the input. ValueError is
    raised if the input has a non-integral value. When ``strict`` is True, this
    uses `__index__ <https://docs.python.org/3/reference/datamodel.html#object.__index__>`_
    and when it is False it uses ``int``.


    Examples
    ========

    >>> from sympy.utilities.misc import as_int
    >>> from sympy import sqrt, S

    The function is primarily concerned with sanitizing input for
    functions that need to work with builtin integers, so anything that
    is unambiguously an integer should be returned as an int:

    >>> as_int(S(3))
    3

    Floats, being of limited precision, are not assumed to be exact and
    will raise an error unless the ``strict`` flag is False. This
    precision issue becomes apparent for large floating point numbers:

    >>> big = 1e23
    >>> type(big) is float
    True
    >>> big == int(big)
    True
    >>> as_int(big)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    >>> as_int(big, strict=False)
    99999999999999991611392

    Input that might be a complex representation of an integer value is
    also rejected by default:

    >>> one = sqrt(3 + 2*sqrt(2)) - sqrt(2)
    >>> int(one) == 1
    True
    >>> as_int(one)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    """
    if strict:
        try:
            if isinstance(n, bool):
                raise TypeError
            return operator.index(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
    else:
        try:
            result = int(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
        if n - result:
            raise ValueError('%s is not an integer' % (n,))
        return result
