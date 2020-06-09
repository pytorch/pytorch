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

import sys
import types

PY37 = sys.version_info[0] == 3 and sys.version_info[1] == 7


import math
inf = math.inf
nan = math.nan

string_classes = (str, bytes)


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})


import builtins
# See https://github.com/PyCQA/flake8-bugbear/issues/64
exec_ = getattr(builtins, "exec")  # noqa: B009


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


import collections.abc
container_abcs = collections.abc


# Gets a function from the name of a method on a type
def get_function_from_type(cls, name):
    return getattr(cls, name, None)


# The codes below is not copied from the six package, so the copyright
# declaration at the beginning does not apply.
#
# Copyright(c) PyTorch contributors
#


def bind_method(fn, obj, obj_type):
    return types.MethodType(fn, obj)
