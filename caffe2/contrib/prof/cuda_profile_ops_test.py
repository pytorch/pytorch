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

import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import core, dyndep, workspace

dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/prof:cuda_profile_ops")


class CudaProfileOpsTest(unittest.TestCase):
    @unittest.skipIf(workspace.NumCudaDevices() < 1, "Need at least 1 GPU")
    def test_run(self):
        net = core.Net("net")
        net.CudaProfileInitialize([], [], output="/tmp/cuda_profile_test")
        net.CudaProfileStart([], [])
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            net.ConstantFill([], ["out"], shape=[1, 3, 244, 244])
        net.CudaProfileStop([], [])

        workspace.CreateNet(net)
        workspace.RunNet(net)
