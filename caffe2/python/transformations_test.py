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

from caffe2.python import core, workspace, test_util

from caffe2.python.transformations import addNNPACK


class TestTransformations(test_util.TestCase):
    def test_addNNPACK(self):
        net = core.Net("net")
        net.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net.Relu(["Y"], ["Y2"])
        addNNPACK(net)
        assert (net.Proto().op[0].engine == "NNPACK")
