from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import workspace
from caffe2.python.functional import Functional
import numpy as np


@st.composite
def _tensor_splits(draw, add_axis=False):
    """Generates (axis, split_info, tensor_splits) tuples."""
    tensor = draw(hu.tensor(min_value=4))  # Each dim has at least 4 elements.
    axis = draw(st.integers(0, len(tensor.shape) - 1))
    if add_axis:
        # Simple case: get individual slices along one axis, where each of them
        # is (N-1)-dimensional. The axis will be added back upon concatenation.
        return (
            axis, np.ones(tensor.shape[axis], dtype=np.int32), [
                np.array(tensor.take(i, axis=axis))
                for i in range(tensor.shape[axis])
            ]
        )
    else:
        # General case: pick some (possibly consecutive, even non-unique)
        # indices at which we will split the tensor, along the given axis.
        splits = sorted(
            draw(
                st.
                lists(elements=st.integers(0, tensor.shape[axis]), max_size=4)
            ) + [0, tensor.shape[axis]]
        )
        return (
            axis, np.array(np.diff(splits), dtype=np.int32), [
                tensor.take(range(splits[i], splits[i + 1]), axis=axis)
                for i in range(len(splits) - 1)
            ],
        )


class TestFunctional(hu.HypothesisTestCase):
    @given(X=hu.tensor(), engine=st.sampled_from(["", "CUDNN"]), **hu.gcs)
    def test_relu(self, X, engine, gc, dc):
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        output = Functional.Relu(X, device_option=gc)
        Y_l = output[0]
        Y_d = output["output_0"]

        with workspace.WorkspaceGuard("tmp_workspace"):
            op = core.CreateOperator("Relu", ["X"], ["Y"], engine=engine)
            workspace.FeedBlob("X", X)
            workspace.RunOperatorOnce(op)
            Y_ref = workspace.FetchBlob("Y")

        np.testing.assert_array_equal(
            Y_l, Y_ref, err_msg='Functional Relu result mismatch'
        )

        np.testing.assert_array_equal(
            Y_d, Y_ref, err_msg='Functional Relu result mismatch'
        )

    @given(tensor_splits=_tensor_splits(), **hu.gcs)
    def test_concat(self, tensor_splits, gc, dc):
        # Input Size: 1 -> inf
        axis, _, splits = tensor_splits
        concat_result, split_info = Functional.Concat(*splits, axis=axis, device_option=gc)

        concat_result_ref = np.concatenate(splits, axis=axis)
        split_info_ref = np.array([a.shape[axis] for a in splits])

        np.testing.assert_array_equal(
            concat_result,
            concat_result_ref,
            err_msg='Functional Concat result mismatch'
        )

        np.testing.assert_array_equal(
            split_info,
            split_info_ref,
            err_msg='Functional Concat split info mismatch'
        )

    @given(tensor_splits=_tensor_splits(), split_as_arg=st.booleans(), **hu.gcs)
    def test_split(self, tensor_splits, split_as_arg, gc, dc):
        # Output Size: 1 - inf
        axis, split_info, splits = tensor_splits

        split_as_arg = True

        if split_as_arg:
            input_tensors = [np.concatenate(splits, axis=axis)]
            kwargs = dict(axis=axis, split=split_info, num_output=len(splits))
        else:
            input_tensors = [np.concatenate(splits, axis=axis), split_info]
            kwargs = dict(axis=axis, num_output=len(splits))
        result = Functional.Split(*input_tensors, device_option=gc, **kwargs)

        def split_ref(input, split=split_info):
            s = np.cumsum([0] + list(split))
            return [
                np.array(input.take(np.arange(s[i], s[i + 1]), axis=axis))
                for i in range(len(split))
            ]

        result_ref = split_ref(*input_tensors)
        for i, ref in enumerate(result_ref):
            np.testing.assert_array_equal(
                result[i], ref, err_msg='Functional Relu result mismatch'
            )


if __name__ == '__main__':
    unittest.main()
