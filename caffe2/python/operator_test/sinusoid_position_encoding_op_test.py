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
import math

MAX_TEST_EMBEDDING_SIZE = 20
MAX_TEST_SEQUENCE_LENGTH = 10
MAX_TEST_BATCH_SIZE = 5
MIN_TEST_ALPHA = 5000.0
MAX_TEST_ALPHA = 20000.0
MIN_TEST_AMPLITUDE = 0.1
MAX_TEST_AMPLITUDE = 10.0


class TestSinusoidPositionEncodingOp(serial.SerializedTestCase):
    @serial.given(
        positions_vec=hu.arrays(
            dims=[MAX_TEST_SEQUENCE_LENGTH],
            dtype=np.int32,
            elements=st.integers(1, MAX_TEST_SEQUENCE_LENGTH)
        ),
        embedding_size=st.integers(1, MAX_TEST_EMBEDDING_SIZE),
        batch_size=st.integers(1, MAX_TEST_BATCH_SIZE),
        alpha=st.floats(MIN_TEST_ALPHA, MAX_TEST_ALPHA),
        amplitude=st.floats(MIN_TEST_AMPLITUDE, MAX_TEST_AMPLITUDE),
        **hu.gcs_cpu_only
    )
    def test_sinusoid_embedding(
        self, positions_vec, embedding_size, batch_size, alpha, amplitude, gc, dc
    ):
        positions = np.tile(positions_vec, [batch_size, 1]).transpose()

        op = core.CreateOperator(
            "SinusoidPositionEncoding",
            ["positions"],
            ["output"],
            embedding_size=embedding_size,
            alpha=alpha,
            amplitude=amplitude,
        )

        def sinusoid_encoding(dim, position):
            x = 1. * position / math.pow(alpha, 1. * dim / embedding_size)
            if dim % 2 == 0:
                return amplitude * math.sin(x)
            else:
                return amplitude * math.cos(x)

        def sinusoid_embedding_op(positions):
            output_shape = (len(positions), len(positions[0]), embedding_size)
            ar = np.zeros(output_shape)
            for i, position_vector in enumerate(positions):
                for j, position in enumerate(position_vector):
                    for k in range(embedding_size):
                        ar[i, j, k] = sinusoid_encoding(k, position)
            return [ar]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[positions],
            reference=sinusoid_embedding_op,
        )
