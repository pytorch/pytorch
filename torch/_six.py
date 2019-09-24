# Copyright (c) 2010-2017 Benjamin Peterson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
import sys
import builtins
import types


PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
PY37 = sys.version_info[0] == 3 and sys.version_info[1] == 7

if PY2:
    inf = float('inf')
    nan = float('nan')
else:
    import math
    inf = math.inf
    nan = math.nan

if PY2:
    string_classes = basestring
else:
    string_classes = (str, bytes)


if PY2:
    int_classes = (int, long)
else:
    int_classes = int


if PY2:
    FileNotFoundError = IOError
else:
    FileNotFoundError = builtins.FileNotFoundError


if PY2:
    import Queue as queue  # noqa: F401
else:
    import queue  # noqa: F401


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})


# A portable way of referring to the generator version of map
# in both Python 2 and Python 3.
if hasattr(itertools, 'imap'):
    imap = itertools.imap  # type: ignore
else:
    imap = map  # type: ignore


if PY3:
    import builtins
    # See https://github.com/PyCQA/flake8-bugbear/issues/64
    exec_ = getattr(builtins, "exec")  # noqa: B009
else:
    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")


if sys.version_info[:2] == (3, 2):
    exec_("""def raise_from(value, from_value):
    try:
        if from_value is None:
            raise value
        raise value from from_value
    finally:
        value = None
""")
elif sys.version_info[:2] > (3, 2):
    exec_("""def raise_from(value, from_value):
    try:
        raise value from from_value
    finally:
        value = None
""")
else:
    def raise_from(value, from_value):
        raise value

if PY2:
    import collections
    container_abcs = collections
elif PY3:
    import collections.abc
    container_abcs = collections.abc

# Gets a function from the name of a method on a type
if PY2:
    def get_function_from_type(cls, name):
        method = getattr(cls, name, None)
        return getattr(method, "__func__", None)
elif PY3:
    def get_function_from_type(cls, name):
        return getattr(cls, name, None)

if PY2:
    import __builtin__ as builtins
elif PY3:
    import builtins

if PY2:
    import StringIO
    StringIO = StringIO.StringIO
elif PY3:
    import io
    StringIO = io.StringIO


# The codes below is not copied from the six package, so the copyright
# declaration at the beginning does not apply.
#
# Copyright(c) PyTorch contributors
#

def istuple(obj):
    # Usually instances of PyStructSequence is also an instance of tuple
    # but in some py2 environment it is not, so we have to manually check
    # the name of the type to determine if it is a namedtupled returned
    # by a pytorch operator.
    t = type(obj)
    return isinstance(obj, tuple) or t.__module__ == 'torch.return_types'

def bind_method(fn, obj, obj_type):
    if PY2:
        fn = fn.__func__
        return types.MethodType(fn, obj, obj_type)
    else:
        return types.MethodType(fn, obj)
