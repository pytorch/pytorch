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
from caffe2.python import workspace, brew, model_helper
from caffe2.python.modeling.gradient_clipping import GradientClipping

import numpy as np


class GradientClippingTest(unittest.TestCase):
    def test_gradient_clipping(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        fc2 = brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')

        grad_map = model.AddGradientOperators([loss])

        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}

        net_modifier = GradientClipping(
            grad_clip_method='by_norm',
            clip_norm_type='l2_norm',
            clip_threshold=0.1,
        )

        net_modifier(model.net, grad_map=grad_map_for_param)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        # 5 forward ops + 6 backward ops + 2 * (3 gradient clipping ops)
        self.assertEqual(len(model.net.Proto().op), 17)

    def test_gradient_clipping_l1_norm(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        fc2 = brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')

        grad_map = model.AddGradientOperators([loss])

        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}

        net_modifier = GradientClipping(
            grad_clip_method='by_norm',
            clip_norm_type='l1_norm',
            clip_threshold=0.1,
        )

        net_modifier(model.net, grad_map=grad_map_for_param)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        # 5 forward ops + 6 backward ops + 2 * (2 gradient clipping ops)
        self.assertEqual(len(model.net.Proto().op), 15)

    def test_gradient_clipping_using_param_norm(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        fc2 = brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')

        grad_map = model.AddGradientOperators([loss])

        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}

        net_modifier = GradientClipping(
            grad_clip_method='by_norm',
            clip_norm_type='l2_norm',
            clip_threshold=0.1,
            use_parameter_norm=True,
        )

        net_modifier(model.net, grad_map=grad_map_for_param)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        # 5 forward ops + 6 backward ops + 2 * (5 gradient clipping ops)
        self.assertEqual(len(model.net.Proto().op), 21)

    def test_gradient_clipping_compute_norm_ratio(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        fc2 = brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')

        grad_map = model.AddGradientOperators([loss])

        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}

        net_modifier = GradientClipping(
            grad_clip_method='by_norm',
            clip_norm_type='l2_norm',
            clip_threshold=0.1,
            use_parameter_norm=True,
            compute_norm_ratio=True,
        )

        net_modifier(model.net, grad_map=grad_map_for_param)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        # 5 forward ops + 6 backward ops + 2 * (6 gradient clipping ops)
        self.assertEqual(len(model.net.Proto().op), 23)
