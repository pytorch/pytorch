from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from caffe2.python import workspace, brew, model_helper
from caffe2.python.modeling.compute_statistics_for_blobs import (
    ComputeStatisticsForBlobs
)

import numpy as np


class ComputeStatisticsForBlobsTest(unittest.TestCase):
    def test_compute_statistics_for_blobs(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        net_modifier = ComputeStatisticsForBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=10,
        )

        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_summary = workspace.FetchBlob('fc1_w_summary')

        # std is unbiased here
        stats_ref = np.array([fc1_w.flatten().min(), fc1_w.flatten().max(),
                     fc1_w.flatten().mean(), fc1_w.flatten().std(ddof=1)])

        self.assertAlmostEqual(np.linalg.norm(stats_ref - fc1_w_summary), 0,
                               delta=1e-5)
        self.assertEqual(fc1_w_summary.size, 4)

        self.assertEqual(len(model.net.Proto().op), 8)
        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs()
