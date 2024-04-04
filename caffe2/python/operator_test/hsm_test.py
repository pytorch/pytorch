



from hypothesis import given, settings
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
node1 = hsmu.create_node_with_words(words[0], "node1")
node2 = hsmu.create_node_with_words(words[1], "node2")
node3 = hsmu.create_node_with_words(words[2], "node3")
node4 = hsmu.create_node_with_nodes([node1, node2], "node4")
node = hsmu.create_node_with_nodes([node4, node3], "node5")
tree.root_node.MergeFrom(node)

# structure:
# node5: [0, 2, ["node4", "node3"]] # offset, length, "node4, node3"
# node4: [2, 2, ["node1", "node2"]]
# node1: [4, 3, [0, 1 ,2]]
# node2: [7, 2, [3, 4]
# node3: [9, 4, [5, 6, 7, 8]
struct = [[0, 2, ["node4", "node3"], "node5"],
            [2, 2, ["node1", "node2"], "node4"],
            [4, 3, [0, 1, 2], "node1"],
            [7, 2, [3, 4], "node2"],
            [9, 4, [5, 6, 7, 8], "node3"]]

# Internal util to translate input tree to list of (word_id,path). serialized
# hierarchy is passed into the operator_def as a string argument,
hierarchy_proto = hsmu.create_hierarchy(tree)
arg = caffe2_pb2.Argument()
arg.name = "hierarchy"
arg.s = hierarchy_proto.SerializeToString()

beam = 5
args_search = []
arg_search = caffe2_pb2.Argument()
arg_search.name = "tree"
arg_search.s = tree.SerializeToString()
args_search.append(arg_search)
arg_search = caffe2_pb2.Argument()
arg_search.name = "beam"
arg_search.f = beam
args_search.append(arg_search)


class TestHsm(hu.HypothesisTestCase):
    def test_hsm_search(self):
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
            'HSoftmaxSearch',
            ['data', 'weights', 'bias'],
            ['names', 'scores'],
            'HSoftmaxSearch',
            arg=args_search)
        workspace.RunOperatorOnce(op)
        names = workspace.FetchBlob('names')
        scores = workspace.FetchBlob('scores')

        def simulation_hsm_search():
            names = []
            scores = []
            for line in struct:
                s, e = line[0], line[0] + line[1]
                score = np.dot(X, w[s:e].transpose()) + b[s:e]
                score = np.exp(score - np.max(score, axis=1, keepdims=True))
                score /= score.sum(axis=1, keepdims=True)
                score = -np.log(score)

                score = score.transpose()
                idx = -1
                for j, n in enumerate(names):
                    if n == line[3]:
                        idx = j
                        score += scores[j]
                if idx == -1:
                    score[score > beam] = np.inf
                else:
                    score[score - scores[idx] > beam] = np.inf

                for i, name in enumerate(line[2]):
                    scores.append(score[i])
                    names.append(name)
            scores = np.vstack(scores)
            return names, scores.transpose()

        p_names, p_scores = simulation_hsm_search()
        idx = np.argsort(p_scores, axis=1)
        p_scores = np.sort(p_scores, axis=1)
        p_names = np.array(p_names)[idx]
        for i in range(names.shape[0]):
            for j in range(names.shape[1]):
                if names[i][j]:
                    self.assertEqual(
                        names[i][j], p_names[i][j].item().encode('utf-8'))
                    self.assertAlmostEqual(
                        scores[i][j], p_scores[i][j], delta=0.001)

    def test_hsm_run_once(self):
        workspace.GlobalInit(['caffe2'])
        workspace.FeedBlob("data",
                           np.random.randn(1000, 100).astype(np.float32))
        workspace.FeedBlob("weights",
                           np.random.randn(1000, 100).astype(np.float32))
        workspace.FeedBlob("bias", np.random.randn(1000).astype(np.float32))
        workspace.FeedBlob("labels", np.random.rand(1000).astype(np.int32) * 9)
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
        samples = 9
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
            op, grad_ops, [X, w, b, labels], op.input, 0, g_input[0], [0]
        )
        self.assertAlmostEqual(loss, 44.269, delta=0.001)

    # Test to compare gradient calculated using the gradient operator and the
    # symmetric derivative calculated using Euler Method
    # TODO : convert to both cpu and gpu test when ready.
    @given(**hu.gcs_cpu_only)
    @settings(deadline=10000)
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

    def test_huffman_tree_hierarchy(self):
        workspace.GlobalInit(['caffe2'])
        labelSet = list(range(0, 6))
        counts = [1, 2, 3, 4, 5, 6]
        labels = sum([[l] * c for (l, c) in zip(labelSet, counts)], [])
        Y = np.array(labels).astype(np.int64)
        workspace.FeedBlob("labels", Y)
        arg = caffe2_pb2.Argument()
        arg.name = 'num_classes'
        arg.i = 6
        op = core.CreateOperator(
            'HuffmanTreeHierarchy',
            ['labels'],
            ['huffman_tree'],
            'HuffmanTreeHierarchy',
            arg=[arg])
        workspace.RunOperatorOnce(op)
        huffmanTreeOutput = workspace.FetchBlob('huffman_tree')
        treeOutput = hsm_pb2.TreeProto()
        treeOutput.ParseFromString(huffmanTreeOutput[0])
        treePathOutput = hsmu.create_hierarchy(treeOutput)

        label_to_path = {}
        for path in treePathOutput.paths:
            label_to_path[path.word_id] = path

        def checkPath(label, indices, code):
            path = label_to_path[label]
            self.assertEqual(len(path.path_nodes), len(code))
            self.assertEqual(len(path.path_nodes), len(code))
            for path_node, index, target in \
                    zip(path.path_nodes, indices, code):
                self.assertEqual(path_node.index, index)
                self.assertEqual(path_node.target, target)
        checkPath(0, [0, 4, 6, 8], [1, 0, 0, 0])
        checkPath(1, [0, 4, 6, 8], [1, 0, 0, 1])
        checkPath(2, [0, 4, 6], [1, 0, 1])
        checkPath(3, [0, 2], [0, 0])
        checkPath(4, [0, 2], [0, 1])
        checkPath(5, [0, 4], [1, 1])

if __name__ == '__main__':
    unittest.main()
