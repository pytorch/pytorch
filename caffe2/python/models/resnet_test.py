from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import workspace, cnn, memonger, core
import caffe2.python.models.resnet as resnet
import hypothesis.strategies as st
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu


def has_blob(proto, needle):
    for op in proto.op:
        for inp in op.input:
            if inp == needle:
                return True
        for outp in op.output:
            if outp == needle:
                return True
    return False


def count_blobs(proto):
    blobs = set()
    for op in proto.op:
        blobs = blobs.union(set(op.input)).union(set(op.output))
    return len(blobs)


def count_shared_blobs(proto):
    blobs = set()
    for op in proto.op:
        blobs = blobs.union(set(op.input)).union(set(op.output))
    return len([b for b in blobs if "_shared" in b])


class ResnetMemongerTest(hu.HypothesisTestCase):

    @given(with_shapes=st.booleans(), **hu.gcs_cpu_only)
    @settings(max_examples=2, timeout=120)
    def test_resnet_shared_grads(self, with_shapes, gc, dc):
        model = cnn.CNNModelHelper(
            order="NCHW",
            name="test",
            cudnn_exhaustive_search=True,
        )
        with core.NameScope("gpu_0"):
            data = model.net.AddExternalInput("gpu_0/data")
            label = model.net.AddExternalInput("gpu_0/label")
            (_softmax, loss) = resnet.create_resnet50(
                model,
                data,
                num_input_channels=3,
                num_labels=1000,
                label=label,
                is_test=False,
            )

        param_to_grad = model.AddGradientOperators([loss])

        (shapes, types) = workspace.InferShapesAndTypes(
            [model.param_init_net, model.net],
            {'gpu_0/data': [4, 3, 227, 227],
                         'gpu_0/label': [4]},
        )

        count_before = count_blobs(model.net.Proto())
        optim_proto = memonger.share_grad_blobs(
            model.net,
            ["gpu_0/loss"],
            set(model.param_to_grad.values()),
            "gpu_0/",
            share_activations=True,
            dont_share_blobs=set([str(param_to_grad["gpu_0/conv1_w"])]),
            blob_shapes=shapes if with_shapes else None,
        )
        self.assertTrue(memonger.verify_graph_equality(model.net.Proto(), optim_proto))
        count_after = count_blobs(optim_proto)
        self.assertTrue(count_after < count_before)

        # Run model and compare results. We check that the loss is same
        # and also that the final gradient (conv1_w_grad is same)
        workspace.RunNetOnce(model.param_init_net)
        data = np.random.rand(4, 3, 227, 227).astype(np.float32)
        label = (np.random.rand(4) * 1000).astype(np.int32)

        workspace.FeedBlob("gpu_0/data", data)
        workspace.FeedBlob("gpu_0/label", label)

        workspace.RunNetOnce(model.net)
        model.net.Proto().type = 'dag'
        model.net.Proto().num_workers = 4
        loss1 = workspace.FetchBlob("gpu_0/last_out_L1000")
        conv1_w_grad = workspace.FetchBlob(param_to_grad["gpu_0/conv1_w"])
        workspace.FeedBlob(param_to_grad["gpu_0/conv1_w"], np.array([0.0]))

        workspace.RunNetOnce(optim_proto)
        optimized_loss1 = workspace.FetchBlob("gpu_0/last_out_L1000")
        optim_conv1_w_grad = workspace.FetchBlob(param_to_grad["gpu_0/conv1_w"])

        print("before: {} after: {}".format(count_before, count_after))

        np.testing.assert_almost_equal(loss1, optimized_loss1)
        np.testing.assert_almost_equal(conv1_w_grad, optim_conv1_w_grad)

    def test_resnet_forward_only(self):
        model = cnn.CNNModelHelper(
            order="NCHW",
            name="test",
            cudnn_exhaustive_search=True,
        )
        with core.NameScope("gpu_0"):
                data = model.net.AddExternalInput("gpu_0/data")
                resnet.create_resnet50(
                    model,
                    data,
                    num_input_channels=3,
                    num_labels=1000,
                    is_test=True
                )

        count_before = count_blobs(model.net.Proto())
        optim_proto = memonger.optimize_inference_for_dag(
            model.net, ["gpu_0/data"], "gpu_0/"
        )
        count_after = count_blobs(optim_proto)
        num_shared_blobs = count_shared_blobs(optim_proto)

        # Run model and compare results

        workspace.RunNetOnce(model.param_init_net)
        data = np.random.rand(4, 3, 227, 227).astype(np.float32)

        workspace.FeedBlob("gpu_0/data", data)
        workspace.RunNetOnce(model.net)
        model.net.Proto().type = 'dag'
        model.net.Proto().num_workers = 4
        loss1 = workspace.FetchBlob("gpu_0/last_out_L1000")
        self.assertTrue(memonger.verify_graph_equality(
            model.net.Proto(), optim_proto))

        workspace.RunNetOnce(optim_proto)
        optimized_loss1 = workspace.FetchBlob("gpu_0/last_out_L1000")
        self.assertTrue(count_after < count_before)
        self.assertTrue(num_shared_blobs < 7)
        np.testing.assert_almost_equal(loss1, optimized_loss1)


if __name__ == "__main__":
    import unittest
    import random
    random.seed(2603)
    workspace.GlobalInit([
        'caffe2',
        '--caffe2_log_level=0',
        '--caffe2_print_blob_sizes_at_exit=0',
        '--caffe2_gpu_memory_tracking=1'])
    unittest.main()
