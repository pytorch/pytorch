import numpy as np
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestSplitOpCost(TestCase):
    def _verify_cost(self, workspace, split_op):
        flops, bytes_written, bytes_read = workspace.GetOperatorCost(
            split_op, split_op.input
        )
        self.assertEqual(flops, 0)
        self.assertEqual(
            bytes_read,
            sum(workspace.FetchBlob(b).nbytes for b in split_op.input),
        )
        self.assertEqual(
            bytes_written,
            sum(workspace.FetchBlob(b).nbytes for b in split_op.output),
        )

    def test_columnwise_equal_outputSplit(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob("input", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2", "output_3"],
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (2, 1))
        np.testing.assert_array_equal(output_1, [[1], [4]])

        output_2 = workspace.FetchBlob("output_2")
        np.testing.assert_array_equal(output_2, [[2], [5]])

        output_3 = workspace.FetchBlob("output_3")
        np.testing.assert_array_equal(output_3, [[3], [6]])

        self._verify_cost(workspace, split_op)

    def test_rowwise_equal_outputSplit(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob("input", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2"],
            axis=0,
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (1, 3))
        np.testing.assert_array_equal(output_1, [[1, 2, 3]])

        output_2 = workspace.FetchBlob("output_2")
        np.testing.assert_array_equal(output_2, [[4, 5, 6]])

        self._verify_cost(workspace, split_op)

    def test_columnwise_equal_outputSplit_columnRemoved(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob("input", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
        # To be able to use 'add_axis' (which should have been called 'remove_axis') on 'axis',
        # the dimensions of split tensors must match on 'axis'
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2", "output_3"],
            axis=1,
            add_axis=1,
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (2,))
        np.testing.assert_array_equal(output_1, [1, 4])

        output_2 = workspace.FetchBlob("output_2")
        np.testing.assert_array_equal(output_2, [2, 5])

        output_3 = workspace.FetchBlob("output_3")
        np.testing.assert_array_equal(output_3, [3, 6])

        self._verify_cost(workspace, split_op)

    def test_rowwise_equal_outputSplit_rowRemoved(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob("input", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2"],
            axis=0,
            add_axis=1,
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (3,))
        np.testing.assert_array_equal(output_1, [1, 2, 3])

        output_2 = workspace.FetchBlob("output_2")
        np.testing.assert_array_equal(output_2, [4, 5, 6])

        self._verify_cost(workspace, split_op)

    def test_rowwise_unequal_argSplit(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob(
            "input", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        )
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2"],
            axis=0,
            split=[1, 2],
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (1, 3))
        np.testing.assert_array_equal(output_1, [[1, 2, 3]])

        output_2 = workspace.FetchBlob("output_2")
        self.assertTupleEqual(output_2.shape, (2, 3))
        np.testing.assert_array_equal(output_2, [[4, 5, 6], [7, 8, 9]])

        self._verify_cost(workspace, split_op)

    def test_rowwise_unequal_argSplit_rowRemoved(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob(
            "input", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        )
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2", "output_3"],
            axis=0,
            split=[1, 1, 1],
            add_axis=1,
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (3,))
        np.testing.assert_array_equal(output_1, [1, 2, 3])

        output_2 = workspace.FetchBlob("output_2")
        np.testing.assert_array_equal(output_2, [4, 5, 6])

        output_3 = workspace.FetchBlob("output_3")
        np.testing.assert_array_equal(output_3, [7, 8, 9])

        self._verify_cost(workspace, split_op)

    def test_rowwise_unequal_blobSplit(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob(
            "input", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        )
        workspace.FeedBlob("split", np.array([1, 2], dtype=np.int32))
        split_op = core.CreateOperator(
            "Split",
            ["input", "split"],
            ["output_1", "output_2"],
            axis=0,
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (1, 3))
        np.testing.assert_array_equal(output_1, [[1, 2, 3]])

        output_2 = workspace.FetchBlob("output_2")
        self.assertTupleEqual(output_2.shape, (2, 3))
        np.testing.assert_array_equal(output_2, [[4, 5, 6], [7, 8, 9]])

        self._verify_cost(workspace, split_op)

    def test_columnwise_unequal_argSplit(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob("input", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2"],
            axis=1,
            split=[1, 2],
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (2, 1))
        np.testing.assert_array_equal(output_1, [[1], [4]])

        output_2 = workspace.FetchBlob("output_2")
        self.assertTupleEqual(output_2.shape, (2, 2))
        np.testing.assert_array_equal(output_2, [[2, 3], [5, 6]])

        self._verify_cost(workspace, split_op)

    def test_columnWise_unequal_blobSplit_columnRemoved(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob("input", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
        workspace.FeedBlob("split", np.array([1, 1, 1], dtype=np.int32))
        split_op = core.CreateOperator(
            "Split",
            ["input", "split"],
            ["output_1", "output_2", "output_3"],
            axis=1,
            add_axis=1,
        )
        workspace.RunOperatorOnce(split_op)

        output_1 = workspace.FetchBlob("output_1")
        self.assertTupleEqual(output_1.shape, (2,))
        np.testing.assert_array_equal(output_1, [1, 4])

        output_2 = workspace.FetchBlob("output_2")
        np.testing.assert_array_equal(output_2, [2, 5])

        output_3 = workspace.FetchBlob("output_3")
        np.testing.assert_array_equal(output_3, [3, 6])

        self._verify_cost(workspace, split_op)

    def test_equal_outputSplit_NHWC(self):
        workspace.ResetWorkspace()
        workspace.FeedBlob("input", np.random.rand(2, 5, 7, 9).astype(np.int32))
        split_op = core.CreateOperator(
            "Split",
            ["input"],
            ["output_1", "output_2", "output_3"],
            order="NHWC",
        )
        workspace.RunOperatorOnce(split_op)

        for b in split_op.output:
            self.assertTupleEqual(workspace.FetchBlob(b).shape, (2, 5, 7, 3))

        self._verify_cost(workspace, split_op)
