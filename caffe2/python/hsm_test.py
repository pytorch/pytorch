from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from hypothesis import given
import numpy as np
import unittest

from caffe2.proto import caffe2_pb2, hsm_pb2
from caffe2.python import workspace, core, gradient_checker
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.hsm_util as hsmu

# User inputs tree using protobuf file or, in this case, python utils
# The hierarchy in this test looks as shown below. Note that the final subtrees
# (with word_ids as leaves) have been collapsed for visualization
#           *
#         /  \
#        *    5,6,7,8
#       / \
#  0,1,2   3,4
tree = hsm_pb2.TreeProto()
words = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
node1 = hsmu.create_node_with_words(words[0])
node2 = hsmu.create_node_with_words(words[1])
node3 = hsmu.create_node_with_words(words[2])
node4 = hsmu.create_node_with_nodes([node1, node2])
node = hsmu.create_node_with_nodes([node4, node3])
tree.root_node.MergeFrom(node)

# Internal util to translate input tree to list of (word_id,path). serialized
# hierarchy is passed into the operator_def as a string argument,
hierarchy_proto = hsmu.create_hierarchy(tree)
arg = caffe2_pb2.Argument()
arg.name = "hierarchy"
arg.s = hierarchy_proto.SerializeToString()


class TestHsm(hu.HypothesisTestCase):
    def test_hsm_run_once(self):
        workspace.GlobalInit(['caffe2'])
        workspace.FeedBlob("data",
                           np.random.randn(1000, 100).astype(np.float32))
        workspace.FeedBlob("weights",
                           np.random.randn(1000, 100).astype(np.float32))
        workspace.FeedBlob("bias", np.random.randn(1000).astype(np.float32))
        workspace.FeedBlob("labels", np.random.randn(1000).astype(np.int32))
        op = core.CreateOperator(
            'HSoftmax',
            ['data', 'weights', 'bias', 'labels'],
            ['output', 'intermediate_output'],
            'HSoftmax',
            arg=[arg])
        self.assertTrue(workspace.RunOperatorOnce(op))

    # Test to check value of sum of squared losses in forward pass for given
    # input
    def test_hsm_forward(self):
        cpu_device_option = caffe2_pb2.DeviceOption()
        grad_checker = gradient_checker.GradientChecker(
            0.01, 0.05, cpu_device_option, "default")
        samples = 10
        dim_in = 5
        X = np.zeros((samples, dim_in)).astype(np.float32) + 1
        w = np.zeros((hierarchy_proto.size, dim_in)).astype(np.float32) + 1
        b = np.array([i for i in range(hierarchy_proto.size)])\
            .astype(np.float32)
        labels = np.array([i for i in range(samples)]).astype(np.int32)

        workspace.GlobalInit(['caffe2'])
        workspace.FeedBlob("data", X)
        workspace.FeedBlob("weights", w)
        workspace.FeedBlob("bias", b)
        workspace.FeedBlob("labels", labels)

        op = core.CreateOperator(
            'HSoftmax',
            ['data', 'weights', 'bias', 'labels'],
            ['output', 'intermediate_output'],
            'HSoftmax',
            arg=[arg])
        grad_ops, g_input = core.GradientRegistry.GetGradientForOp(
            op, [s + '_grad' for s in op.output])

        loss, _ = grad_checker.GetLossAndGrad(
            op, grad_ops, X, op.input[0], g_input[0], [0]
        )
        self.assertAlmostEqual(loss, 44.269, delta=0.001)

    # Test to compare gradient calculated using the gradient operator and the
    # symmetric derivative calculated using Euler Method
    # TODO: convert to both cpu and gpu test when ready.
    @given(**hu.gcs_cpu_only)
    def test_hsm_gradient(self, gc, dc):
        samples = 10
        dim_in = 5
        X = np.random.rand(samples, dim_in).astype(np.float32) - 0.5
        w = np.random.rand(hierarchy_proto.size, dim_in) \
            .astype(np.float32) - 0.5
        b = np.random.rand(hierarchy_proto.size).astype(np.float32) - 0.5
        labels = np.array([np.random.randint(0, 8) for i in range(samples)]) \
            .astype(np.int32)

        workspace.GlobalInit(['caffe2'])
        workspace.FeedBlob("data", X)
        workspace.FeedBlob("weights", w)
        workspace.FeedBlob("bias", b)
        workspace.FeedBlob("labels", labels)

        op = core.CreateOperator(
            'HSoftmax',
            ['data', 'weights', 'bias', 'labels'],
            ['output', 'intermediate_output'],
            'HSoftmax',
            arg=[arg])

        self.assertDeviceChecks(dc, op, [X, w, b, labels], [0])

        for i in range(3):
            self.assertGradientChecks(gc, op, [X, w, b, labels], i, [0])


if __name__ == '__main__':
    unittest.main()
