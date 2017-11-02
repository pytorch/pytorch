# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package scope
# Module caffe2.python.scope
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib
import threading
from past.builtins import basestring

from caffe2.proto import caffe2_pb2


# The name scope and device scope when creating a new operator.
_NAMESCOPE_SEPARATOR = '/'

_threadlocal_scope = threading.local()


def CurrentNameScope():
    global _threadlocal_scope
    if not hasattr(_threadlocal_scope, "namescope"):
        _threadlocal_scope.namescope = ''
    return _threadlocal_scope.namescope


def CurrentDeviceScope():
    global _threadlocal_scope
    if not hasattr(_threadlocal_scope, "devicescope"):
        _threadlocal_scope.devicescope = None
    return _threadlocal_scope.devicescope


@contextlib.contextmanager
def NameScope(prefix, reset=False):
    global _threadlocal_scope
    assert isinstance(prefix, basestring), \
        "NameScope takes in a string as its argument."
    old_scope = CurrentNameScope()
    prefix = prefix + _NAMESCOPE_SEPARATOR if prefix is not '' else ''
    if reset:
        _threadlocal_scope.namescope = prefix
    else:
        _threadlocal_scope.namescope = _threadlocal_scope.namescope + prefix

    try:
        yield
    finally:
        assert _threadlocal_scope.namescope.endswith(prefix), \
            "The namescope variable is changed from outside NameScope() calls."
        _threadlocal_scope.namescope = old_scope


@contextlib.contextmanager
def DeviceScope(scope, node_name=None):
    new_scope = caffe2_pb2.DeviceOption()
    if scope:
        assert isinstance(scope, caffe2_pb2.DeviceOption), \
            "DeviceScope takes in a caffe2_pb2.DeviceOption as its argument."
        new_scope.CopyFrom(scope)
    else:
        assert node_name, "At least one argument should be non-null in DeviceScope"

    # rewrite node_name if it is explicitly given
    if node_name:
        new_scope.node_name = node_name
    global _threadlocal_scope
    old_scope = CurrentDeviceScope()
    # nested scope should inherit the node_name if it is not explicitly set
    if old_scope and old_scope.HasField('node_name') and \
            not new_scope.HasField('node_name'):
        new_scope.node_name = old_scope.node_name
    _threadlocal_scope.devicescope = new_scope
    try:
        yield
    finally:
        assert _threadlocal_scope.devicescope == new_scope, \
            "The device scope is changed from outside DeviceScope() calls."
        _threadlocal_scope.devicescope = old_scope


@contextlib.contextmanager
def EmptyDeviceScope():
    """
    Allow users to 'disable' the device scope behaviour (so it can be
    controlled at a NetDef::DeviceOption level, not overridden at
    OperatorDef::DeviceOption level).

    This sets the CurrentDeviceScope() to None, so that the field is
    not set in CreateOperator(...), etc.
    """
    old_scope = CurrentDeviceScope()
    try:
        _threadlocal_scope.devicescope = None
        yield
    finally:
        _threadlocal_scope.devicescope = old_scope
        return
