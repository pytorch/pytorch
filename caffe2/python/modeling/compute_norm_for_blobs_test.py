from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from caffe2.python import workspace, brew, model_helper
from caffe2.python.modeling.compute_norm_for_blobs import ComputeNormForBlobs

import numpy as np


class ComputeNormForBlobsTest(unittest.TestCase):
    def test_compute_norm_for_blobs(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        net_modifier = ComputeNormForBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=10,
        )

        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_l2_norm = workspace.FetchBlob('fc1_w_l2_norm')

        self.assertEqual(fc1_w_l2_norm.size, 1)
        self.assertAlmostEqual(fc1_w_l2_norm[0],
                               np.linalg.norm(fc1_w)**2,
                               delta=1e-5)

        self.assertEqual(len(model.net.Proto().op), 6)

        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()

    def test_compute_averaged_norm_for_blobs(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        net_modifier = ComputeNormForBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=10,
            compute_averaged_norm=True,
        )

        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_l2_averaged_norm = workspace.FetchBlob('fc1_w_averaged_l2_norm')

        self.assertEqual(fc1_w_l2_averaged_norm.size, 1)
        self.assertAlmostEqual(fc1_w_l2_averaged_norm[0],
                               np.linalg.norm(fc1_w)**2 / fc1_w.size,
                               delta=1e-5)

        self.assertEqual(len(model.net.Proto().op), 6)

        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()

    def test_compute_norm_for_blobs_no_print(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        net_modifier = ComputeNormForBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=-1,
        )

        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_l2_norm = workspace.FetchBlob('fc1_w_l2_norm')

        self.assertEqual(fc1_w_l2_norm.size, 1)
        self.assertAlmostEqual(fc1_w_l2_norm[0],
                               np.linalg.norm(fc1_w)**2,
                               delta=1e-5)

        self.assertEqual(len(model.net.Proto().op), 4)

        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()

    def test_compute_l1_norm_for_blobs(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        net_modifier = ComputeNormForBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=10,
            p=1,
        )

        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_l1_norm = workspace.FetchBlob('fc1_w_l1_norm')

        self.assertEqual(fc1_w_l1_norm.size, 1)
        self.assertAlmostEqual(fc1_w_l1_norm[0],
                               np.sum(np.abs(fc1_w)),
                               delta=1e-5)

        self.assertEqual(len(model.net.Proto().op), 6)

        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()

    def test_compute_l1_averaged_norm_for_blobs(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        net_modifier = ComputeNormForBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=10,
            p=1,
            compute_averaged_norm=True,
        )

        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_averaged_l1_norm = workspace.FetchBlob('fc1_w_averaged_l1_norm')

        self.assertEqual(fc1_w_averaged_l1_norm.size, 1)
        self.assertAlmostEqual(fc1_w_averaged_l1_norm[0],
                               np.sum(np.abs(fc1_w)) / fc1_w.size,
                               delta=1e-5)

        self.assertEqual(len(model.net.Proto().op), 6)

        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
