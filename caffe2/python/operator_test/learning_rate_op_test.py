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
import numpy as np


class TestLearningRate(hu.HypothesisTestCase):
    @given(
        **hu.gcs_cpu_only
    )
    def test_learningrate(self, gc, dc):
        data = np.random.randint(low=1, high=1e5, size=1)
        active_period = int(np.random.randint(low=1, high=1e3, size=1))
        inactive_period = int(np.random.randint(low=1, high=1e3, size=1))
        base_lr = float(np.random.random(1))

        def ref(data):
            iter_num = float(data)
            reminder = iter_num % (active_period + inactive_period)
            if reminder < active_period:
                return (np.array(base_lr), )
            else:
                return (np.array(0.), )

        op = core.CreateOperator(
            'LearningRate',
            'data',
            'out',
            policy="alter",
            active_first=True,
            base_lr=base_lr,
            active_period=active_period,
            inactive_period=inactive_period
        )

        self.assertReferenceChecks(gc, op, [data], ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
