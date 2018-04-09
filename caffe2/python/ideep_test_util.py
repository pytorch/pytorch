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

## @package ideep_test_util
# Module caffe2.python.ideep_test_util
"""
The IDEEP test utils is a small addition on top of the hypothesis test utils
under caffe2/python, which allows one to more easily test IDEEP related
operators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
from caffe2.python import hypothesis_test_util as hu

cpu_do = hu.cpu_do
ideep_do = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.IDEEP)
device_options = hu.device_options + ([ideep_do])


def device_checker_device_options():
    return st.just(device_options)


def gradient_checker_device_option():
    return st.sampled_from(device_options)


gcs = dict(
    gc=gradient_checker_device_option(),
    dc=device_checker_device_options()
)

gcs_cpu_only = dict(gc=st.sampled_from([cpu_do]), dc=st.just([cpu_do]))
gcs_ideep_only = dict(gc=st.sampled_from([ideep_do]), dc=st.just([ideep_do]))
gcs_cpu_ideep = dict(gc=st.sampled_from([cpu_do, ideep_do]), dc=st.just([cpu_do, ideep_do]))
