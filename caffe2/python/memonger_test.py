from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import workspace, cnn, memonger, core
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

    @given(input_dim=st.integers(min_value=1, max_value=4),
           output_dim=st.integers(min_value=1, max_value=4),
           batch_size=st.integers(min_value=1, max_value=4))
    def test_gradient_optim(self, input_dim, output_dim, batch_size):
        m = cnn.CNNModelHelper()
        with core.NameScope("name_x"):
            fc1 = m.FC("data", "fc1", dim_in=input_dim, dim_out=output_dim)
            fc2 = m.FC(fc1, "fc2", dim_in=output_dim, dim_out=output_dim)
            fc3 = m.FC(fc2, "fc3", dim_in=output_dim, dim_out=output_dim)
            fc4 = m.FC(fc3, "fc4", dim_in=output_dim, dim_out=output_dim)
            fc5 = m.FC(fc4, "fc5", dim_in=output_dim, dim_out=output_dim)
            fc5.Relu([], fc5)\
               .Softmax([], "pred") \
               .LabelCrossEntropy(["label"], ["xent"]) \
               .AveragedLoss([], "loss")
        input_to_grad = m.AddGradientOperators(["name_x/loss"])

        def count_blobs(proto):
            blob_set = set()
            for op in proto.op:
                for inp in op.input:
                    blob_set.add(inp)
                for outp in op.output:
                    blob_set.add(outp)
            return len(blob_set)

        blobs_before = count_blobs(m.net.Proto())
        optim_proto = memonger.share_grad_blobs(
            m.net,
            ["name_x/loss"],
            set(m.param_to_grad.values()),
            "name_x/",
        )
        blobs_after = count_blobs(optim_proto)
        self.assertLess(blobs_after, blobs_before)

        # Test networks produce exactly same gradients
        data = np.random.randn(batch_size, input_dim).astype(np.float32)
        label = np.random.randint(
            low=0, high=output_dim, size=(batch_size,)).astype(np.int32)
        workspace.RunNetOnce(m.param_init_net)
        workspace.FeedBlob("name_x/data", data)
        workspace.FeedBlob("name_x/label", label)
        workspace.RunNetOnce(m.net)
        loss = workspace.FetchBlob("name_x/loss")
        grad = workspace.FetchBlob(str(input_to_grad["name_x/fc1_w"]))
        workspace.RunNetOnce(optim_proto)
        optimized_loss = workspace.FetchBlob("name_x/loss")
        optimized_grad = workspace.FetchBlob(str(input_to_grad["name_x/fc1_w"]))
        np.testing.assert_almost_equal(loss, optimized_loss)
        np.testing.assert_almost_equal(grad, optimized_grad)

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_gradient_optim_tree(self, input_dim, output_dim, batch_size):
        m = cnn.CNNModelHelper()
        with core.NameScope("name_x"):
            fc1 = m.FC("data", "fc1", dim_in=input_dim, dim_out=output_dim)
            fc2 = m.FC(fc1, "fc2", dim_in=output_dim, dim_out=output_dim)
            fc3 = m.FC(fc2, "fc3", dim_in=output_dim, dim_out=output_dim)
            fc4 = m.FC(fc3, "fc4", dim_in=output_dim, dim_out=output_dim)
            fc5 = m.FC(fc4, "fc5", dim_in=output_dim, dim_out=output_dim)
            fc5.Relu([], fc5) \
               .Softmax([], "pred1") \
               .LabelCrossEntropy(["label"], ["xent1"]) \
               .AveragedLoss([], "loss1")
            fc6 = m.FC(fc5, "fc6", dim_in=output_dim, dim_out=output_dim)
            fc6.Relu([], fc6) \
               .Softmax([], "pred2") \
               .LabelCrossEntropy(["label"], ["xent2"]) \
               .AveragedLoss([], "loss2")
        input_to_grad = m.AddGradientOperators(["name_x/loss1", "name_x/loss2"])

        def count_blobs(proto):
            blob_set = set()
            for op in proto.op:
                for inp in op.input:
                    blob_set.add(inp)
                for outp in op.output:
                    blob_set.add(outp)
            return len(blob_set)

        blobs_before = count_blobs(m.net.Proto())
        optim_proto = memonger.share_grad_blobs(
            m.net,
            ["name_x/loss1", "name_x/loss2"],
            set(m.param_to_grad.values()),
            "name_x",  # "name_x//shared_gradinp_0_shared" if using "name_x/"
        )
        blobs_after = count_blobs(optim_proto)
        self.assertLess(blobs_after, blobs_before)

        print(str(optim_proto))

        # Test networks produce exactly same gradients
        data = np.random.randn(batch_size, input_dim).astype(np.float32)
        label = np.random.randint(
            low=0, high=output_dim, size=(batch_size,)).astype(np.int32)
        workspace.RunNetOnce(m.param_init_net)
        workspace.FeedBlob("name_x/data", data)
        workspace.FeedBlob("name_x/label", label)
        workspace.RunNetOnce(m.net)
        loss1 = workspace.FetchBlob("name_x/loss1")
        loss2 = workspace.FetchBlob("name_x/loss2")
        grad = workspace.FetchBlob(str(input_to_grad["name_x/fc1_w"]))
        workspace.RunNetOnce(optim_proto)
        optimized_loss1 = workspace.FetchBlob("name_x/loss1")
        optimized_loss2 = workspace.FetchBlob("name_x/loss2")
        optimized_grad = workspace.FetchBlob(str(input_to_grad["name_x/fc1_w"]))
        np.testing.assert_almost_equal(loss1, optimized_loss1)
        np.testing.assert_almost_equal(loss2, optimized_loss2)
        np.testing.assert_almost_equal(grad, optimized_grad)

    def test_topological_sort_longest_path(self):
        m = cnn.CNNModelHelper()
        # 0
        m.Copy("conv0_w_comp", "conv0_w")
        # 1
        conv0 = m.Conv("data", "conv0", 32, 32, 4)
        # 2
        m.Copy("conv2_w", "conv2_w")
        # 3
        m.Conv(conv0, "conv2", 16, 32, 4)

        g = memonger.compute_interference_graph(m.net.Proto().op)

        orders_org = memonger.topological_sort_traversal(g)
        orders_gt_org = [2, 0, 1, 3]
        self.assertEqual(orders_gt_org, orders_org)

        orders = memonger.topological_sort_traversal_longest_path(g)
        # longer path is in front of the shorter one
        orders_gt = [0, 1, 2, 3]
        self.assertEqual(orders_gt, orders)

    def test_topological_sort_longest_path_multi_target(self):
        # two outputs: conv2 and data4
        m = cnn.CNNModelHelper()
        # 0
        m.Copy("conv0_w_comp", "conv0_w")
        # 1
        conv0 = m.Conv("data", "conv0", 32, 32, 4)
        # 2
        m.Copy("conv2_w", "conv2_w")
        # 3
        m.Conv(conv0, "conv2", 16, 32, 4)
        # 4
        m.Copy("data1", "data2")
        # 5
        m.Copy("data2", "data3")

        g = memonger.compute_interference_graph(m.net.Proto().op)

        orders_org = memonger.topological_sort_traversal(g)
        orders_gt_org = [4, 5, 2, 0, 1, 3]
        self.assertEqual(orders_gt_org, orders_org)

        orders = memonger.topological_sort_traversal_longest_path(g)
        # longer path is in front of the shorter one
        orders_gt = [0, 1, 2, 3, 4, 5]
        self.assertEqual(orders_gt, orders)

    def test_topological_sort_longest_path_single_node(self):
        # single node
        m = cnn.CNNModelHelper()
        # 0
        m.Copy("conv0_w_comp", "conv0_w")

        g = memonger.compute_interference_graph(m.net.Proto().op)

        orders_org = memonger.topological_sort_traversal(g)
        orders_gt_org = [0]
        self.assertEqual(orders_gt_org, orders_org)

        orders = memonger.topological_sort_traversal_longest_path(g)
        # longer path is in front of the shorter one
        orders_gt = [0]
        self.assertEqual(orders_gt, orders)
