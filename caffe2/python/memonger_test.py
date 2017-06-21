from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import workspace, memonger, core, model_helper, brew
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from hypothesis import given, settings


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


class MemongerTest(hu.HypothesisTestCase):
    @given(input_dim=st.integers(min_value=1, max_value=10),
           output_dim=st.integers(min_value=1, max_value=10),
           batch_size=st.integers(min_value=1, max_value=10),
           do=st.sampled_from(hu.device_options),
           algo=st.sampled_from(memonger.AssignmentAlgorithm))
    @settings(max_examples=5, timeout=120)
    def test_simple_memonger(self, input_dim, output_dim, batch_size, do, algo):
        m = model_helper.ModelHelper()
        fc1 = brew.fc(m, "data", "fc1", dim_in=input_dim, dim_out=output_dim)
        fc2 = brew.fc(m, fc1, "fc2", dim_in=output_dim, dim_out=output_dim)
        fc3 = brew.fc(m, fc2, "fc3", dim_in=output_dim, dim_out=output_dim)

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

        optimization = memonger.optimize_interference(
            m.Proto(), static_blobs, algo=algo)
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

        # run with blob sizes
        blob_sizes = memonger.collect_blob_sizes(m.Proto())
        optimization1 = memonger.optimize_interference(
            m.Proto(), static_blobs, blob_sizes=blob_sizes, algo=algo)
        workspace.RunNetOnce(optimization1.net)
        optimized_loss = workspace.FetchBlob("loss")
        optimized_grad = workspace.FetchBlob(str(input_to_grad["fc1_w"]))
        np.testing.assert_almost_equal(loss, optimized_loss)
        np.testing.assert_almost_equal(grad, optimized_grad)
        stats = memonger.compute_statistics(optimization1.assignments)
        self.assertLessEqual(stats.optimized_nbytes, stats.baseline_nbytes)

    @given(input_dim=st.integers(min_value=1, max_value=4),
           output_dim=st.integers(min_value=1, max_value=4),
           batch_size=st.integers(min_value=1, max_value=4))
    def test_gradient_optim(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "fc1", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m, fc1, "fc2", dim_in=output_dim, dim_out=output_dim)
            fc3 = brew.fc(m, fc2, "fc3", dim_in=output_dim, dim_out=output_dim)
            fc4 = brew.fc(m, fc3, "fc4", dim_in=output_dim, dim_out=output_dim)
            fc5 = brew.fc(m, fc4, "fc5", dim_in=output_dim, dim_out=output_dim)
            fc5.Relu([], fc5)\
               .Softmax([], "pred") \
               .LabelCrossEntropy(["label"], ["xent"]) \
               .AveragedLoss([], "loss")
        input_to_grad = m.AddGradientOperators(["name_x/loss"])

        blobs_before = count_blobs(m.net.Proto())
        optim_proto = memonger.share_grad_blobs(
            m.net,
            ["name_x/loss"],
            set(m.param_to_grad.values()),
            "name_x/",
            share_activations=False,
        )
        self.assertTrue(memonger.verify_graph_equality(m.Proto(), optim_proto))
        blobs_after = count_blobs(optim_proto)
        self.assertLess(blobs_after, blobs_before)

        optim_proto_wacts = memonger.share_grad_blobs(
            m.net,
            ["name_x/loss"],
            set(m.param_to_grad.values()),
            "name_x/",
            share_activations=True,
            dont_share_blobs=set([str(input_to_grad["name_x/fc1_w"])]),
        )
        self.assertTrue(memonger.verify_graph_equality(m.Proto(), optim_proto_wacts))
        blobs_wact_optim = count_blobs(optim_proto_wacts)
        self.assertLessEqual(blobs_wact_optim, blobs_after)

        # Check that the last activations are not shared
        self.assertTrue(has_blob(optim_proto, "name_x/fc5"))
        self.assertTrue(
            has_blob(optim_proto_wacts, "name_x/fc5"),
            "Dont remap final activation",
        )

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

        workspace.FeedBlob(str(input_to_grad["name_x/fc1_w"]), np.array([0.0]))

        # Run with the forward optimization
        workspace.RunNetOnce(optim_proto_wacts)
        optimized_loss = workspace.FetchBlob("name_x/loss")
        optimized_grad = workspace.FetchBlob(str(input_to_grad["name_x/fc1_w"]))
        np.testing.assert_almost_equal(loss, optimized_loss)
        np.testing.assert_almost_equal(grad, optimized_grad)

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_gradient_optim_tree(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "fc1", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m, fc1, "fc2", dim_in=output_dim, dim_out=output_dim)
            fc3 = brew.fc(m, fc2, "fc3", dim_in=output_dim, dim_out=output_dim)
            fc4 = brew.fc(m, fc3, "fc4", dim_in=output_dim, dim_out=output_dim)
            fc5 = brew.fc(m, fc4, "fc5", dim_in=output_dim, dim_out=output_dim)
            fc5.Relu([], fc5) \
               .Softmax([], "pred1") \
               .LabelCrossEntropy(["label"], ["xent1"]) \
               .AveragedLoss([], "loss1")
            fc6 = brew.fc(m, fc5, "fc6", dim_in=output_dim, dim_out=output_dim)
            fc6.Relu([], fc6) \
               .Softmax([], "pred2") \
               .LabelCrossEntropy(["label"], ["xent2"]) \
               .AveragedLoss([], "loss2")
        input_to_grad = m.AddGradientOperators(["name_x/loss1", "name_x/loss2"])

        blobs_before = count_blobs(m.net.Proto())
        optim_proto = memonger.share_grad_blobs(
            m.net,
            ["name_x/loss1", "name_x/loss2"],
            set(m.param_to_grad.values()),
            "name_x",  # "name_x//shared_gradinp_0_shared" if using "name_x/"
            share_activations=True,
            dont_share_blobs=set(['name_x/fc6', 'name_x/fc5',
                                   str(input_to_grad["name_x/fc1_w"])]),
        )
        self.assertTrue(memonger.verify_graph_equality(m.net.Proto(), optim_proto))
        blobs_after = count_blobs(optim_proto)
        self.assertLess(blobs_after, blobs_before)
        self.assertTrue(has_blob(optim_proto, "name_x/fc6"))

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
        workspace.FeedBlob(str(input_to_grad["name_x/fc1_w"]), np.array([0.0]))

        workspace.RunNetOnce(optim_proto)
        optimized_loss1 = workspace.FetchBlob("name_x/loss1")
        optimized_loss2 = workspace.FetchBlob("name_x/loss2")
        optimized_grad = workspace.FetchBlob(str(input_to_grad["name_x/fc1_w"]))
        np.testing.assert_almost_equal(loss1, optimized_loss1)
        np.testing.assert_almost_equal(loss2, optimized_loss2)
        np.testing.assert_almost_equal(grad, optimized_grad)

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_forward_optim_tree_daggy(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        m.Proto().type = "dag"
        m.Proto().num_workers = 4

        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "fc1", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m, fc1, "fc2", dim_in=output_dim, dim_out=output_dim)

            fc3 = brew.fc(m, fc2, "fc3", dim_in=output_dim, dim_out=output_dim)
            fc4 = brew.fc(m, fc3, "fc4", dim_in=output_dim, dim_out=output_dim)
            fc5 = brew.fc(m, fc4, "fc5", dim_in=output_dim, dim_out=output_dim)

            # Branch
            fc3b = brew.fc(m, fc2, "fc3b", dim_in=output_dim, dim_out=output_dim)
            fc4b = brew.fc(m, fc3b, "fc4b", dim_in=output_dim, dim_out=output_dim)
            fc5b = brew.fc(m, fc4b, "fc5b", dim_in=output_dim, dim_out=output_dim)

            fc5sum = brew.sum(m, [fc5, fc5b], "fc5sum")

            fc5.Relu([], fc5sum) \
               .Softmax([], "pred1") \
               .LabelCrossEntropy(["label"], ["xent1"]) \
               .AveragedLoss([], "loss1")
            fc6 = brew.fc(m, fc5, "fc6", dim_in=output_dim, dim_out=output_dim)
            fc6.Relu([], fc6) \
               .Softmax([], "pred2") \
               .LabelCrossEntropy(["label"], ["xent2"]) \
               .AveragedLoss([], "loss2")

        blobs_before = count_blobs(m.net.Proto())
        optim_proto = memonger.optimize_inference_for_dag(
            m.net, ["name_x/data"], "name_x"
        )
        self.assertTrue(memonger.verify_graph_equality(m.net.Proto(), optim_proto))
        blobs_after = count_blobs(optim_proto)
        self.assertLess(blobs_after, blobs_before)

        # Test networks produce exactly same results
        data = np.random.randn(batch_size, input_dim).astype(np.float32)
        label = np.random.randint(
            low=0, high=output_dim, size=(batch_size,)).astype(np.int32)
        workspace.RunNetOnce(m.param_init_net)
        workspace.FeedBlob("name_x/data", data)
        workspace.FeedBlob("name_x/label", label)
        workspace.RunNetOnce(m.net)
        loss1 = workspace.FetchBlob("name_x/loss1")
        loss2 = workspace.FetchBlob("name_x/loss2")
        workspace.RunNetOnce(optim_proto)
        optimized_loss1 = workspace.FetchBlob("name_x/loss1")
        optimized_loss2 = workspace.FetchBlob("name_x/loss2")
        np.testing.assert_almost_equal(loss1, optimized_loss1)
        np.testing.assert_almost_equal(loss2, optimized_loss2)

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_forward_optim_tree_harder(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        m.net.Proto().type = "dag"
        m.net.Proto().num_workers = 4
        m.net.AddExternalInput("label")
        m.net.AddExternalInput("data")

        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "fc1", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m, fc1, "fc2", dim_in=output_dim, dim_out=output_dim)

            fc3 = brew.fc(m, fc2, "fc3", dim_in=output_dim, dim_out=output_dim)
            fc4 = brew.fc(m, fc3, "fc4", dim_in=output_dim, dim_out=output_dim)
            fc5 = brew.fc(m, fc4, "fc5", dim_in=output_dim, dim_out=output_dim)

            # Branch
            fc3b = brew.fc(m, fc2, "fc3b", dim_in=output_dim, dim_out=output_dim)
            fc4b = brew.fc(m, fc3b, "fc4b", dim_in=output_dim, dim_out=output_dim)
            fc5b = brew.fc(m, fc4b, "fc5b", dim_in=output_dim, dim_out=output_dim)

            fc5sum = brew.sum(m, [fc5, fc5b], "fc5sum")
            fc5sum.Relu([], "relu1") \
               .Softmax([], "pred1") \
               .LabelCrossEntropy(["label"], ["xent1"]) \
               .AveragedLoss([], "loss1")
            fc6 = brew.fc(m, fc5, "fc6", dim_in=output_dim, dim_out=output_dim)
            fc6.Relu([], fc6) \
               .Softmax([], "pred2") \
               .LabelCrossEntropy(["label"], ["xent2"]) \
               .AveragedLoss([], "loss2")

        blobs_before = count_blobs(m.net.Proto())
        optim_proto = memonger.optimize_inference_for_dag(
            m.net, ["name_x/data"], "name_x/"
        )
        self.assertTrue(
            memonger.verify_graph_equality(m.net.Proto(), optim_proto))
        blobs_after = count_blobs(optim_proto)
        print(str(optim_proto))
        self.assertLess(blobs_after, blobs_before)

        # Test networks produce exactly same results
        data = np.random.randn(batch_size, input_dim).astype(np.float32)
        label = np.random.randint(
            low=0, high=output_dim, size=(batch_size,)).astype(np.int32)
        workspace.RunNetOnce(m.param_init_net)
        workspace.FeedBlob("name_x/data", data)
        workspace.FeedBlob("name_x/label", label)
        workspace.RunNetOnce(m.net)
        loss1 = workspace.FetchBlob("name_x/loss1")
        loss2 = workspace.FetchBlob("name_x/loss2")
        workspace.RunNetOnce(optim_proto)
        optimized_loss1 = workspace.FetchBlob("name_x/loss1")
        optimized_loss2 = workspace.FetchBlob("name_x/loss2")
        np.testing.assert_almost_equal(loss1, optimized_loss1)
        np.testing.assert_almost_equal(loss2, optimized_loss2)

    def test_compute_interference_graph_inplace_ops(self):
        m = model_helper.ModelHelper()
        m.Copy("b1", "b1")
        m.Copy("b1", "b1")
        m.Copy("b1", "b1")
        g = memonger.compute_interference_graph(m.net.Proto().op)
        self.assertEqual(g.edges(), [(0, 1), (0, 2), (1, 2)])

    def test_topological_sort_longest_path(self):
        m = model_helper.ModelHelper()
        # 0
        m.Copy("conv0_w_comp", "conv0_w")
        # 1
        conv0 = brew.conv(m, "data", "conv0", 32, 32, 4)
        # 2
        m.Copy("conv2_w", "conv2_w")
        # 3
        brew.conv(m, conv0, "conv2", 16, 32, 4)

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
        m = model_helper.ModelHelper()
        # 0
        m.Copy("conv0_w_comp", "conv0_w")
        # 1
        conv0 = brew.conv(m, "data", "conv0", 32, 32, 4)
        # 2
        m.Copy("conv2_w", "conv2_w")
        # 3
        brew.conv(m, conv0, "conv2", 16, 32, 4)
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
        m = model_helper.ModelHelper()
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

    def test_compute_assignments_greedy(self):
        LiveRange = memonger.LiveRange
        ranges_sorted = [
            ('b1', LiveRange(1, 3, 10)),
            ('b2', LiveRange(3, 4, 1)),
            ('b3', LiveRange(5, 6, 1)),
            ('b4', LiveRange(5, 7, 10)),
        ]
        assignment_gt = [
            [ranges_sorted[0], ranges_sorted[3]],
            [ranges_sorted[1], ranges_sorted[2]],
        ]

        best = memonger.compute_assignments_greedy(ranges_sorted, None)
        self.assertEqual(memonger.get_memory_usage(best), 11)
        self.assertEqual(best, assignment_gt)

    def test_compute_assignments_dp(self):
        LiveRange = memonger.LiveRange
        ranges_sorted = [
            ('b1', LiveRange(1, 3, 10)),
            ('b2', LiveRange(3, 4, 1)),
            ('b3', LiveRange(5, 6, 1)),
            ('b4', LiveRange(5, 7, 10)),
        ]

        best = memonger.compute_assignments_dp(ranges_sorted, None)
        self.assertEqual(memonger.get_memory_usage(best), 11)

    def test_compute_assignments_dp1(self):
        LiveRange = memonger.LiveRange
        ranges_sorted = [
            ('b1', LiveRange(1, 2, 10)),
            ('b2', LiveRange(4, 6, 1)),
            ('b3', LiveRange(5, 6, 10)),
        ]

        best = memonger.compute_assignments_dp(ranges_sorted, [])
        self.assertEqual(memonger.get_memory_usage(best), 11)

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_verify_graph_equality(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        m.Proto().type = "dag"
        m.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "x", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc3 = brew.fc(m, fc1, "z", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m, [fc2, fc3], "out")

        m2 = model_helper.ModelHelper()
        m2.Proto().type = "dag"
        m2.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m2, "data", "other_x", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m2, fc1, "other_y", dim_in=output_dim, dim_out=output_dim)
            fc3 = brew.fc(m2, fc1, "other_z", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m2, [fc2, fc3], "out")

        self.assertTrue(memonger.verify_graph_equality(m.net.Proto(), m2.net.Proto()))

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_verify_graph_equality_harder(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        m.Proto().type = "dag"
        m.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "x", dim_in=input_dim, dim_out=output_dim)
            fc2a = brew.fc(m, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc2b = brew.fc(m, fc1, "z", dim_in=output_dim, dim_out=output_dim)
            fc3a = brew.fc(m, fc2a, "u", dim_in=output_dim, dim_out=output_dim)
            fc3b = brew.fc(m, fc2b, "v", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m, [fc3a, fc3b], "out")

        m2 = model_helper.ModelHelper()
        m2.Proto().type = "dag"
        m2.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m2, "data", "x", dim_in=input_dim, dim_out=output_dim)
            fc2a = brew.fc(m2, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc2b = brew.fc(m2, fc1, "z", dim_in=output_dim, dim_out=output_dim)
            fc3a = brew.fc(m2, fc2a, "y", dim_in=output_dim, dim_out=output_dim)
            fc3b = brew.fc(m2, fc2b, "z", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m2, [fc3a, fc3b], "out")

        self.assertTrue(memonger.verify_graph_equality(m.net.Proto(), m2.net.Proto()))

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_verify_graph_inequality(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        m.Proto().type = "dag"
        m.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "x", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc3 = brew.fc(m, fc1, "z", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m, [fc2, fc3], "out")

        m2 = model_helper.ModelHelper()
        m2.Proto().type = "dag"
        m2.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m2, "data", "x", dim_in=input_dim, dim_out=output_dim)
            fc2 = brew.fc(m2, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc3 = brew.fc(m2, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m2, [fc2, fc3], "out")

        self.assertFalse(memonger.verify_graph_equality(m.net.Proto(), m2.net.Proto()))

    @given(input_dim=st.integers(min_value=4, max_value=4),
           output_dim=st.integers(min_value=4, max_value=4),
           batch_size=st.integers(min_value=4, max_value=4))
    def test_verify_graph_inequality_harder(self, input_dim, output_dim, batch_size):
        m = model_helper.ModelHelper()
        m.Proto().type = "dag"
        m.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m, "data", "x", dim_in=input_dim, dim_out=output_dim)
            fc2a = brew.fc(m, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc2b = brew.fc(m, fc1, "z", dim_in=output_dim, dim_out=output_dim)
            fc3a = brew.fc(m, fc2a, "u", dim_in=output_dim, dim_out=output_dim)
            fc3b = brew.fc(m, fc2b, "v", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m, [fc3a, fc3b], "out")

        m2 = model_helper.ModelHelper()
        m2.Proto().type = "dag"
        m2.Proto().num_workers = 4
        with core.NameScope("name_x"):
            fc1 = brew.fc(m2, "data", "x", dim_in=input_dim, dim_out=output_dim)
            fc2a = brew.fc(m2, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc2b = brew.fc(m2, fc1, "y", dim_in=output_dim, dim_out=output_dim)
            fc3a = brew.fc(m2, fc2a, "u", dim_in=output_dim, dim_out=output_dim)
            fc3b = brew.fc(m2, fc2b, "v", dim_in=output_dim, dim_out=output_dim)
            brew.sum(m2, [fc3a, fc3b], "out")

        self.assertFalse(memonger.verify_graph_equality(m.net.Proto(), m2.net.Proto()))
