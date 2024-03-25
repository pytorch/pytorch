




import unittest
from caffe2.python import workspace, brew, model_helper
from caffe2.python.modeling.compute_histogram_for_blobs import (
    ComputeHistogramForBlobs
)

import numpy as np


class ComputeHistogramForBlobsTest(unittest.TestCase):

    def histogram(self, X, lower_bound=0.0, upper_bound=1.0, num_buckets=20):
        assert X.ndim == 2, ('this test assume 2d array,  but X.ndim is {0}'.
            format(X.ndim))
        N, M = X.shape
        hist = np.zeros((num_buckets + 2, ), dtype=np.int32)
        segment = (upper_bound - lower_bound) / num_buckets
        Y = np.zeros((N, M), dtype=np.int32)
        Y[X < lower_bound] = 0
        Y[X >= upper_bound] = num_buckets + 1
        Y[(X >= lower_bound) & (X < upper_bound)] = \
            ((X[(X >= lower_bound) & (X < upper_bound)] - lower_bound) /
                    segment + 1).astype(np.int32)

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                hist[Y[i][j]] += 1

        cur_hist = hist.astype(np.float32) / (N * M)
        acc_hist = cur_hist
        return [cur_hist, acc_hist]

    def test_compute_histogram_for_blobs(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        num_buckets = 20
        lower_bound = 0.2
        upper_bound = 0.8
        accumulate = False
        net_modifier = ComputeHistogramForBlobs(blobs=['fc1_w', 'fc2_w'],
                                                logging_frequency=10,
                                                num_buckets=num_buckets,
                                                lower_bound=lower_bound,
                                                upper_bound=upper_bound,
                                                accumulate=accumulate)
        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_curr_normalized_hist = workspace.FetchBlob('fc1_w_curr_normalized_hist')
        cur_hist, acc_hist = self.histogram(fc1_w,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            num_buckets=num_buckets)

        self.assertEqual(fc1_w_curr_normalized_hist.size, num_buckets + 2)
        self.assertAlmostEqual(np.linalg.norm(
            fc1_w_curr_normalized_hist - cur_hist), 0.0, delta=1e-5)
        self.assertEqual(len(model.net.Proto().op), 12)

        assert model.net.output_record() is None

    def test_compute_histogram_for_blobs_modify_output_record(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=2)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=2, dim_out=1)

        num_buckets = 20
        lower_bound = 0.2
        upper_bound = 0.8
        accumulate = False
        net_modifier = ComputeHistogramForBlobs(blobs=['fc1_w', 'fc2_w'],
                                                logging_frequency=10,
                                                num_buckets=num_buckets,
                                                lower_bound=lower_bound,
                                                upper_bound=upper_bound,
                                                accumulate=accumulate)
        net_modifier(model.net, modify_output_record=True)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_curr_normalized_hist = workspace.FetchBlob('fc1_w_curr_normalized_hist')
        cur_hist, acc_hist = self.histogram(fc1_w,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            num_buckets=num_buckets)

        self.assertEqual(fc1_w_curr_normalized_hist.size, num_buckets + 2)
        self.assertAlmostEqual(np.linalg.norm(
            fc1_w_curr_normalized_hist - cur_hist), 0.0, delta=1e-5)
        self.assertEqual(len(model.net.Proto().op), 12)

        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
