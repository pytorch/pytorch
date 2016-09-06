from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib

from caffe2.proto import caffe2_pb2

# Python 2 and 3 compatibility: test if basestring exists
try:
    basestring  # NOQA
except NameError:
    # This is python3 so we define basestring.
    basestring = str

# The name scope and device scope when creating a new operator.
NAMESCOPE = ''
DEVICESCOPE = None

_NAMESCOPE_SEPARATOR = '/'


@contextlib.contextmanager
def NameScope(prefix, reset=False):
    global NAMESCOPE
    assert isinstance(prefix, basestring), \
        "NameScope takes in a string as its argument."
    old_scope = NAMESCOPE
    prefix = prefix + _NAMESCOPE_SEPARATOR if prefix is not '' else ''
    if reset:
        NAMESCOPE = prefix
    else:
        NAMESCOPE = NAMESCOPE + prefix
    yield
    assert NAMESCOPE.endswith(prefix), \
        "The namescope variable is changed from outside NameScope() calls."
    NAMESCOPE = old_scope


@contextlib.contextmanager
def DeviceScope(scope):
    assert isinstance(scope, caffe2_pb2.DeviceOption), \
        "DeviceScope takes in a caffe2_pb2.DeviceOption as its argument."
    global DEVICESCOPE
    old_scope = DEVICESCOPE
    DEVICESCOPE = scope
    yield
    assert DEVICESCOPE == scope, \
        "The device scope is changed from outside DeviceScope() calls."
    DEVICESCOPE = old_scope
