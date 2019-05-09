from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestLengthsTileOp(serial.SerializedTestCase):

    @serial.given(
        inputs=st.integers(min_value=1, max_value=20).flatmap(
            lambda size: st.tuples(
                hu.arrays([size], dtype=np.float32),
                hu.arrays([size], dtype=np.int32,
                          elements=st.integers(min_value=0, max_value=20)),
            )
        ),
        **hu.gcs)
    def test_lengths_tile(self, inputs, gc, dc):
        data, lengths = inputs

        def lengths_tile_op(data, lengths):
            return [np.concatenate([
                [d] * l for d, l in zip(data, lengths)
            ])]

        op = core.CreateOperator(
            "LengthsTile",
            ["data", "lengths"],
            ["output"],
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[data, lengths],
            reference=lengths_tile_op,
        )

        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=[data, lengths],
            outputs_to_check=0,
            outputs_with_grads=[0]
        )
