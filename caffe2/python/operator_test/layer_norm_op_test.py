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

from caffe2.python import brew, core, workspace
from functools import reduce
from hypothesis import given
from operator import mul
import caffe2.python.hypothesis_test_util as hu
import numpy as np

from caffe2.python.model_helper import ModelHelper

class TestLayerNormOp(hu.HypothesisTestCase):
    @given(X=hu.tensors(n=1), **hu.gcs)
    def test_layer_norm_grad_op(self, X, gc, dc):
        X = X[0]
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        axis = np.random.randint(0, len(X.shape))
        epsilon = 1e-4
        op = core.CreateOperator(
            "LayerNormGradient",
            ["gout", "out", "mean", "stdev", "in"],
            ["gin"],
            axis=axis,
            epsilon=epsilon,
        )

        def layer_norm_ref(X):
            left = reduce(mul, X.shape[:axis], 1)
            reshaped = np.reshape(X, [left, -1])
            mean = np.mean(reshaped, axis=1).reshape([left, 1])
            stdev = np.sqrt(
                np.mean(np.power(reshaped, 2), axis=1).reshape([left, 1]) -
                np.power(mean, 2) + epsilon
            )
            norm = (reshaped - mean) / (stdev)
            norm = np.reshape(norm, X.shape)
            mean = np.reshape(mean, X.shape[:axis] + (1,))
            stdev = np.reshape(stdev, X.shape[:axis] + (1,))
            return [norm, mean, stdev]

        norm, mean, stdev = layer_norm_ref(X)
        gout = norm

        def layer_norm_grad_ref(gout_full, norm, mean_full, stdev_full, X_full):
            left = reduce(mul, X_full.shape[:axis], 1)
            right = reduce(mul, X_full.shape[axis:], 1)
            X = np.reshape(X_full, [left, right])
            stdev = np.reshape(stdev_full, [left, 1])
            mean = np.reshape(mean_full, [left, 1])
            gout = np.reshape(gout_full, [left, right])
            dstdev_end = (-1.0) / np.power(stdev, 2.0) \
                    * np.sum((X - mean) * gout, axis=1).reshape([left, 1])
            dmean_end = np.sum(-1.0 / stdev * gout, axis=1).reshape([left, 1])
            dx_end = 1.0 / stdev * gout

            # stdev block
            dmean_stdev = -1.0 * mean / stdev * dstdev_end
            dx_stdev = X / (right * stdev) * dstdev_end

            # mean block
            dmean = dmean_end + dmean_stdev
            dxmean = (1.0 / right) * dmean

            # final outputs
            dx = dx_end + dx_stdev + dxmean
            dx = dx.reshape(X_full.shape)

            return [dx]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[gout, norm, mean, stdev, X],
            reference=layer_norm_grad_ref
        )
        self.assertDeviceChecks(
            device_options=dc,
            op=op,
            inputs=[gout, norm, mean, stdev, X],
            outputs_to_check=[0],
        )

    @given(X=hu.tensors(n=1), **hu.gcs)
    def test_layer_norm_op(self, X, gc, dc):
        X = X[0]
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        axis = np.random.randint(0, len(X.shape))
        epsilon = 1e-4
        op = core.CreateOperator(
            "LayerNorm",
            ["input"],
            ["output", "mean", "stdev"],
            axis=axis,
            epsilon=epsilon,
        )

        def layer_norm_ref(X):
            left = reduce(mul, X.shape[:axis], 1)
            reshaped = np.reshape(X, [left, -1])
            mean = np.mean(reshaped, axis=1).reshape([left, 1])
            stdev = np.sqrt(
                np.mean(np.power(reshaped, 2), axis=1).reshape([left, 1]) -
                np.power(mean, 2) + epsilon
            )
            norm = (reshaped - mean) / (stdev)
            norm = np.reshape(norm, X.shape)
            mean = np.reshape(mean, X.shape[:axis] + (1,))
            stdev = np.reshape(stdev, X.shape[:axis] + (1,))
            return [norm, mean, stdev]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=layer_norm_ref
        )
        self.assertDeviceChecks(
            device_options=dc,
            op=op,
            inputs=[X],
            outputs_to_check=[0, 1, 2],
        )

    @given(X=hu.tensors(n=1), **hu.gcs)
    def test_layer_norm_brew_wrapper(self, X, gc, dc):
        X = X[0]
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        axis = np.random.randint(0, len(X.shape))
        epsilon = 1e-4

        workspace.FeedBlob('X', X)

        model = ModelHelper(name='test_layer_norm_brew_wrapper')
        brew.layer_norm(
            model,
            'X',
            'Y',
            axis=axis,
            epsilon=epsilon,
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
