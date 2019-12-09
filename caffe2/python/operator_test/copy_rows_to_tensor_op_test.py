from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import caffe2.python.hypothesis_test_util as hu
import numpy as np
from caffe2.python import core
from hypothesis import given, strategies as st


logger = logging.getLogger(__name__)


def get_input_tensors():
    height = np.random.randint(1, 10)
    width = np.random.randint(1, 10)
    dtype = np.float32
    input_tensor = hu.arrays(
        dims=[height, width],
        dtype=dtype,
        elements=st.integers(min_value=0, max_value=100),
    )

    return input_tensor


class TestCopyRowsToTensor(hu.HypothesisTestCase):
    @given(input_tensor=get_input_tensors(), **hu.gcs_cpu_only)
    def test_copy_rows_to_tensor(self, input_tensor, gc, dc):
        dtype = np.random.choice([np.float16, np.float32, np.int32, np.int64], 1)[0]
        input_tensor = np.array(input_tensor).astype(dtype)
        height = np.shape(input_tensor)[0]
        width = np.shape(input_tensor)[1]
        row = np.random.rand(width).astype(dtype)
        indices_lengths = np.random.randint(height)
        all_indices = np.arange(height)
        np.random.shuffle(all_indices)
        indices = all_indices[:indices_lengths]

        def ref(input_tensor, indices, row):
            for idx in indices:
                input_tensor[idx] = row
            return [input_tensor]
        op = core.CreateOperator(
            "CopyRowsToTensor", ["input_tensor", "indices", "row"], ["input_tensor"]
        )
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_tensor, indices, row],
            reference=ref,
        )

    @given(input_tensor=get_input_tensors(), **hu.gcs_cpu_only)
    def test_copy_rows_to_tensor_invalid_input(self, input_tensor, gc, dc):
        input_tensor = np.array(input_tensor).astype(np.float32)
        height = np.shape(input_tensor)[0]
        width = np.shape(input_tensor)[1]
        row = np.random.rand(width + 1).astype(np.float32)
        indices_lengths = np.random.randint(height)
        all_indices = np.arange(height)
        np.random.shuffle(all_indices)
        indices = all_indices[:indices_lengths]

        self.assertRunOpRaises(
            device_option=gc,
            op=core.CreateOperator(
                "CopyRowsToTensor", ["input_tensor", "indices", "row"], ["input_tensor"]
            ),
            inputs=[input_tensor, indices, row],
            exception=RuntimeError,
            regexp="width of input tensor should match lengths of row",
        )
