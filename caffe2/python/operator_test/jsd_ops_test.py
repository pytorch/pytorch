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
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


def entropy(p):
    q = 1. - p
    return -p * np.log(p) - q * np.log(q)


def jsd(p, q):
    return [entropy(p / 2. + q / 2.) - entropy(p) / 2. - entropy(q) / 2.]


def jsd_grad(go, o, pq_list):
    p, q = pq_list
    m = (p + q) / 2.
    return [np.log(p * (1 - m) / (1 - p) / m) / 2. * go, None]


class TestJSDOps(hu.HypothesisTestCase):
    @given(n=st.integers(10, 100), **hu.gcs_cpu_only)
    def test_bernoulli_jsd(self, n, gc, dc):
        p = np.random.rand(n).astype(np.float32)
        q = np.random.rand(n).astype(np.float32)
        op = core.CreateOperator("BernoulliJSD", ["p", "q"], ["l"])
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[p, q],
            reference=jsd,
            output_to_grad='l',
            grad_reference=jsd_grad,
        )
