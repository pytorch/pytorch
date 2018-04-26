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

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, gradient_checker

import numpy as np
import unittest


class TestGroupNormOp(unittest.TestCase):
    @unittest.skipIf(not workspace.has_gpu_support, "no gpu support")
    def test_forward_ref(self):
        np.random.seed(1)

        N, C, H, W = 2, 6, 5, 7
        G = 3

        scale = np.random.randn(C, )
        bias = np.random.randn(C, )
        X = np.random.randn(N, C, H, W) + 2

        net = core.Net('test')
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            workspace.FeedBlob('X', X.astype(np.float32))
            workspace.FeedBlob('scale', scale.astype(np.float32))
            workspace.FeedBlob('bias', bias.astype(np.float32))
            net.GroupNorm(
                ['X', 'scale', 'bias'],
                ['Y', 'mu', 'sig'],
                num_groups=G)
            net.AddGradientOperators(['Y'])

        workspace.CreateNet(net)
        workspace.RunNet(net.Proto().name)

        Y = workspace.FetchBlob('Y')
        mu = workspace.FetchBlob('mu')
        sig = workspace.FetchBlob('sig')

        np.set_printoptions(precision=3, suppress=True)

        mu_ref = np.mean(X.reshape(N * G, -1), axis=1)
        sig_ref = np.mean(
            ((X.reshape(N * G, -1).transpose() - mu).transpose())**2,
            axis=1) ** .5
        Y_ref = (
            ((X.reshape(N * G, -1).transpose() - mu) / sig)
        ).transpose()
        Y_ref = Y_ref.reshape(Y.shape)
        Y_ref = (
            Y_ref.transpose((0, 2, 3, 1)) * scale + bias
        ).transpose((0, 3, 1, 2))

        self.assertTrue(
            np.allclose(mu_ref, mu),
            'mu not close: max abs diff {}'.format(
                np.absolute(mu_ref - mu).max())
        )
        self.assertTrue(
            np.allclose(sig_ref, sig),
            'sig not close: max abs diff {}'.format(
                np.absolute(sig_ref - sig).max())
        )
        self.assertTrue(
            np.allclose(Y_ref, Y, atol=1e-06),
            'Y not close: max abs diff {}'.format(
                np.absolute(Y_ref - Y).max())
        )

    @unittest.skipIf(not workspace.has_gpu_support, "no gpu support")
    def test_gradient_ref(self):
        N, C, H, W = 2, 6, 5, 7
        G = 3

        scale = np.random.randn(C, ).astype(np.float32)
        bias = np.random.randn(C, ).astype(np.float32)
        X = np.random.randn(N, C, H, W).astype(np.float32) + 2

        op = core.CreateOperator(
            'GroupNorm',
            ['X', 'scale', 'bias'],
            ['Y', 'mu', 'sig'],
            num_groups=G)

        gc = gradient_checker.GradientChecker(
            stepsize=0.05,
            threshold=0.005,
            device_option=core.DeviceOption(caffe2_pb2.CUDA, 0))

        for i in range(3):  # 3 inputs
            res, grad, grad_estimated = gc.CheckSimple(
                op, [X, scale, bias],
                i, [0])

            self.assertTrue(
                grad.shape == grad_estimated.shape,
                'grad.shape not matching.')
            self.assertTrue(
                res,
                'gradient check fail for input[{}]'.format(i))
