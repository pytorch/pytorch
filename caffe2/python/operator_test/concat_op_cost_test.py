from collections import namedtuple

import numpy as np
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestConcatOpCost(TestCase):
    def test_columnwise_concat(self):
        def _test_columnwise_concat_for_type(dtype):
            workspace.ResetWorkspace()
            workspace.FeedBlob("input_1", np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype))
            workspace.FeedBlob("input_2", np.array([[7], [8]], dtype=dtype))
            concat_op = core.CreateOperator(
                "Concat",
                ["input_1", "input_2"],
                ["output", "split_info"],
            )
            workspace.RunOperatorOnce(concat_op)

            output = workspace.FetchBlob("output")
            self.assertTupleEqual(output.shape, (2, 4))
            np.testing.assert_array_equal(output, [[1, 2, 3, 7], [4, 5, 6, 8]])

            flops, bytes_written, bytes_read = workspace.GetOperatorCost(
                concat_op, concat_op.input
            )

            self.assertEqual(flops, 0)
            self.assertEqual(
                bytes_read,
                sum(workspace.FetchBlob(b).nbytes for b in concat_op.input),
            )
            self.assertEqual(
                bytes_written,
                sum(workspace.FetchBlob(b).nbytes for b in concat_op.output),
            )

        [
            _test_columnwise_concat_for_type(t)
            for t in [np.int64, np.float64, np.half, np.int8]
        ]

    def test_split_then_concat(self):
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

        concat_op = core.CreateOperator(
            "Concat",
            ["output_1", "output_2", "output_3"],
            ["output", "split_info"],
            axis=1,
            add_axis=1,
        )
        workspace.RunOperatorOnce(concat_op)

        np.testing.assert_array_equal(
            workspace.FetchBlob("input"), workspace.FetchBlob("output")
        )

        split_cost = workspace.GetOperatorCost(split_op, split_op.input)
        self.assertTupleEqual(
            split_cost,
            namedtuple("expected_cost", ["flops", "bytes_written", "bytes_read"])(
                0, 24, 36
            ),
        )

        concat_cost = workspace.GetOperatorCost(concat_op, concat_op.input)
        self.assertTupleEqual(
            concat_cost,
            namedtuple("expected_cost", ["flops", "bytes_written", "bytes_read"])(
                0, 36, 24
            ),
        )
