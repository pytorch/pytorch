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

## @package control_ops
# Module caffe2.python.helpers.control_ops
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.control_ops_util import add_if_op, add_while_op


def cond(model, cond_blob, external_blobs, then_model, else_model=None):
    """Condition"""
    add_if_op(
        model.net,
        cond_blob,
        external_blobs,
        then_model.net,
        else_model.net if else_model else None)


def loop(model, cond_blob, external_blobs, loop_model, cond_model=None):
    """Loop"""
    add_while_op(
        model.net,
        cond_blob,
        external_blobs,
        loop_model.net,
        cond_model.net if cond_model else None)
