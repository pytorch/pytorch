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

import numpy as np

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestDuplicateOperands(TestCase):
    def test_duplicate_operands(self):
        net = core.Net('net')
        shape = (2, 4)
        x_in = np.random.uniform(size=shape)
        x = net.GivenTensorFill([], 'X', shape=shape,
                                values=x_in.flatten().tolist())
        xsq = net.Mul([x, x])
        y = net.DotProduct([xsq, xsq])
        net.AddGradientOperators([y])
        workspace.RunNetOnce(net)
        self.assertTrue(np.allclose(workspace.FetchBlob('X_grad'),
                                    4 * x_in**3))

if __name__ == "__main__":
    import unittest
    unittest.main()
