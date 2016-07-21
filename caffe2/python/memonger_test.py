from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import workspace, cnn, memonger
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from hypothesis import given


class MemongerTest(hu.HypothesisTestCase):
    @given(input_dim=st.integers(min_value=1, max_value=10),
           output_dim=st.integers(min_value=1, max_value=10),
           batch_size=st.integers(min_value=1, max_value=10),
           do=st.sampled_from(hu.device_options))
    def test_simple_memonger(self, input_dim, output_dim, batch_size, do):
        m = cnn.CNNModelHelper()
        fc1 = m.FC("data", "fc1", dim_in=input_dim, dim_out=output_dim)
        fc2 = m.FC(fc1, "fc2", dim_in=output_dim, dim_out=output_dim)
        fc3 = m.FC(fc2, "fc3", dim_in=output_dim, dim_out=output_dim)

        fc3.Relu([], fc3)\
           .Softmax([], "pred") \
           .LabelCrossEntropy(["label"], ["xent"]) \
           .AveragedLoss([], "loss")
        input_to_grad = m.AddGradientOperators(["loss"])
        m.net.Proto().device_option.CopyFrom(do)
        m.param_init_net.Proto().device_option.CopyFrom(do)
        static_blobs = \
            [o for op in m.param_init_net.Proto().op for o in op.output] + \
            ["data", "label", "loss", input_to_grad["fc1_w"]]

        optimization = memonger.optimize_interference(m.Proto(), static_blobs)
        data = np.random.randn(batch_size, input_dim).astype(np.float32)
        label = np.random.randint(
            low=0, high=output_dim, size=(batch_size,)).astype(np.int32)
        workspace.RunNetOnce(m.param_init_net)
        workspace.FeedBlob("data", data, device_option=do)
        workspace.FeedBlob("label", label, device_option=do)
        workspace.RunNetOnce(m.net)
        loss = workspace.FetchBlob("loss")
        grad = workspace.FetchBlob(str(input_to_grad["fc1_w"]))
        workspace.RunNetOnce(optimization.net)
        optimized_loss = workspace.FetchBlob("loss")
        optimized_grad = workspace.FetchBlob(str(input_to_grad["fc1_w"]))
        np.testing.assert_almost_equal(loss, optimized_loss)
        np.testing.assert_almost_equal(grad, optimized_grad)
        stats = memonger.compute_statistics(optimization.assignments)
        self.assertLess(stats.optimized_nbytes, stats.baseline_nbytes)
