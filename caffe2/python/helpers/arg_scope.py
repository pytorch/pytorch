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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import copy
import threading

_threadlocal_scope = threading.local()


@contextlib.contextmanager
def arg_scope(single_helper_or_list, **kwargs):
    global _threadlocal_scope
    if not isinstance(single_helper_or_list, list):
        assert callable(single_helper_or_list), \
            "arg_scope is only supporting single or a list of helper functions."
        single_helper_or_list = [single_helper_or_list]
    old_scope = copy.deepcopy(get_current_scope())
    for helper in single_helper_or_list:
        assert callable(helper), \
            "arg_scope is only supporting a list of callable helper functions."
        helper_key = helper.__name__
        if helper_key not in old_scope:
            _threadlocal_scope.current_scope[helper_key] = {}
        _threadlocal_scope.current_scope[helper_key].update(kwargs)

    yield
    _threadlocal_scope.current_scope = old_scope


def get_current_scope():
    global _threadlocal_scope
    if not hasattr(_threadlocal_scope, "current_scope"):
        _threadlocal_scope.current_scope = {}
    return _threadlocal_scope.current_scope
