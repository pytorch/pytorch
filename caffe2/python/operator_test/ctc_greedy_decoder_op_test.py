




from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import unittest


class TestCTCGreedyDecoderOp(serial.SerializedTestCase):

    @given(
        batch=st.sampled_from([2, 4, 128, 256]),
        max_time=st.sampled_from([2, 10, 30, 50]),
        num_classes=st.sampled_from([2, 10, 26, 40]),
        merge_repeated=st.sampled_from([True, False]),
        **hu.gcs_cpu_only
    )
    @settings(deadline=10000)
    def test_ctc_greedy_decoder(
        self, batch, max_time,
        num_classes, merge_repeated, gc, dc
    ):

        def input_generater():
            inputs = np.random.rand(max_time, batch, num_classes)\
                .astype(np.float32)
            seq_len = np.random.randint(1, max_time + 1, size=batch)\
                .astype(np.int32)
            return inputs, seq_len

        def ref_ctc_decoder(inputs, seq_len):
            merge = merge_repeated
            output_len = np.array([]).astype(np.int32)
            val = np.array([]).astype(np.int32)
            for i in range(batch):
                prev_id = 0
                t_dec = 0
                len_i = seq_len[i] if seq_len is not None else max_time
                for t in range(len_i):
                    max_id = np.argmax(inputs[t, i, :])
                    if max_id == 0:
                        prev_id = max_id
                        continue
                    if max_id == prev_id and merge:
                        prev_id = max_id
                        continue
                    t_dec += 1
                    val = np.append(val, max_id)
                    prev_id = max_id
                output_len = np.append(output_len, t_dec)

            return [output_len, val]

        def ref_ctc_decoder_max_time(inputs):
            return ref_ctc_decoder(inputs, None)

        inputs, seq_len = input_generater()
        op = core.CreateOperator('CTCGreedyDecoder',
            ['INPUTS', 'SEQ_LEN'],
            ['OUTPUT_LEN', 'VALUES'],
            merge_repeated=merge_repeated)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[inputs, seq_len],
            reference=ref_ctc_decoder,
        )

        op_1 = core.CreateOperator('CTCGreedyDecoder',
            ['INPUTS'],
            ['OUTPUT_LEN', 'VALUES'],
            merge_repeated=merge_repeated)

        self.assertReferenceChecks(
            device_option=gc,
            op=op_1,
            inputs=[inputs],
            reference=ref_ctc_decoder_max_time,
        )

    @given(
        batch=st.sampled_from([2, 4, 128, 256]),
        max_time=st.sampled_from([2, 10, 30, 50]),
        num_classes=st.sampled_from([2, 10, 26, 40]),
        **hu.gcs_cpu_only
    )
    @settings(deadline=10000)
    def test_ctc_greedy_decoder_no_merge_arg(
        self, batch, max_time,
        num_classes, gc, dc
    ):

        def input_generater():
            inputs = np.random.rand(max_time, batch, num_classes)\
                .astype(np.float32)
            seq_len = np.random.randint(1, max_time + 1, size=batch)\
                .astype(np.int32)
            return inputs, seq_len

        def ref_ctc_decoder_no_merge_arg(inputs, seq_len):
            merge = True

            output_len = np.array([]).astype(np.int32)
            val = np.array([]).astype(np.int32)
            for i in range(batch):
                prev_id = 0
                t_dec = 0
                len_i = seq_len[i] if seq_len is not None else max_time
                for t in range(len_i):
                    max_id = np.argmax(inputs[t, i, :])
                    if max_id == 0:
                        prev_id = max_id
                        continue
                    if max_id == prev_id and merge:
                        prev_id = max_id
                        continue
                    t_dec += 1
                    val = np.append(val, max_id)
                    prev_id = max_id
                output_len = np.append(output_len, t_dec)

            return [output_len, val]

        def ref_ctc_decoder_max_time(inputs):
            return ref_ctc_decoder_no_merge_arg(inputs, None)

        inputs, seq_len = input_generater()

        op = core.CreateOperator('CTCGreedyDecoder',
            ['INPUTS'],
            ['OUTPUT_LEN', 'VALUES'])

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[inputs],
            reference=ref_ctc_decoder_max_time,
        )


if __name__ == "__main__":
    import random
    random.seed(2603)
    unittest.main()
