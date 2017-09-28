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

import unittest
from caffe2.python import convnet_benchmarks as cb
from caffe2.python import test_util, workspace


@unittest.skipIf(not workspace.has_gpu_support, "no gpu")
class TestConvnetBenchmarks(test_util.TestCase):
    def testConvnetBenchmarks(self):
        all_args = [
            '--batch_size 16 --order NCHW --iterations 1 '
            '--warmup_iterations 1',
            '--batch_size 16 --order NCHW --iterations 1 '
            '--warmup_iterations 1 --forward_only',
        ]
        for model in [cb.AlexNet, cb.OverFeat, cb.VGGA, cb.Inception]:
            for arg_str in all_args:
                args = cb.GetArgumentParser().parse_args(arg_str.split(' '))
                cb.Benchmark(model, args)


if __name__ == '__main__':
    unittest.main()
