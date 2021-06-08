




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestLengthsPadOp(serial.SerializedTestCase):

    @serial.given(
        inputs=hu.lengths_tensor(
            dtype=np.float32,
            min_value=1,
            max_value=5,
            allow_empty=True,
        ),
        delta_length=st.integers(0, 10),
        padding_value=st.floats(-10.0, 10.0),
        **hu.gcs
    )
    def test_lengths_pad(self, inputs, delta_length, padding_value, gc, dc):
        data, lengths = inputs
        max_length = np.max(lengths) if len(lengths) > 0 else 0
        target_length = max(max_length + delta_length, 1)

        def lengths_pad_op(data, lengths):
            N = len(lengths)
            output = np.ndarray(
                shape=(target_length * N, ) + data.shape[1:], dtype=np.float32)
            output.fill(padding_value)
            ptr1, ptr2 = 0, 0
            for i in range(N):
                output[ptr1:ptr1 + lengths[i]] = data[ptr2:ptr2 + lengths[i]]
                ptr1 += target_length
                ptr2 += lengths[i]

            return [output]

        op = core.CreateOperator(
            "LengthsPad",
            ["data", "lengths"],
            ["data_padded"],
            target_length=target_length,
            padding_value=padding_value,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[data, lengths],
            reference=lengths_pad_op,
        )
