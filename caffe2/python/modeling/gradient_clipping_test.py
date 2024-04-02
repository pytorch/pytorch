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
from caffe2.python import workspace, brew, model_helper
from caffe2.python.modeling.gradient_clipping import GradientClipping

import numpy as np


class GradientClippingTest(unittest.TestCase):
    def test_gradient_clipping_by_norm(self):
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

    def test_gradient_clipping_by_norm_l1_norm(self):
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

    def test_gradient_clipping_by_norm_using_param_norm(self):
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

    def test_gradient_clipping_by_norm_compute_norm_ratio(self):
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

    def test_gradient_clipping_by_value(self):
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

        clip_max = 1e-8
        clip_min = 0
        net_modifier = GradientClipping(
            grad_clip_method='by_value',
            clip_max=clip_max,
            clip_min=clip_min,
        )

        net_modifier(model.net, grad_map=grad_map_for_param)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        # 5 forward ops + 6 backward ops + 2 * (1 gradient clipping ops)
        self.assertEqual(len(model.net.Proto().op), 13)

        fc1_w_grad = workspace.FetchBlob('fc1_w_grad')
        self.assertLessEqual(np.amax(fc1_w_grad), clip_max)
        self.assertGreaterEqual(np.amin(fc1_w_grad), clip_min)

    def test_gradient_clipping_by_norm_including_blobs(self):
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
            blobs_to_include=['fc1_w'],
            blobs_to_exclude=None
        )

        net_modifier(model.net, grad_map=grad_map_for_param)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        # 5 forward ops + 6 backward ops + 1 * (3 gradient clipping ops)
        self.assertEqual(len(model.net.Proto().op), 14)

    def test_gradient_clipping_by_norm_excluding_blobs(self):
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
            blobs_to_include=None,
            blobs_to_exclude=['fc1_w', 'fc2_w']
        )

        net_modifier(model.net, grad_map=grad_map_for_param)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        # 5 forward ops + 6 backward ops + 0 * (3 gradient clipping ops)
        self.assertEqual(len(model.net.Proto().op), 11)
