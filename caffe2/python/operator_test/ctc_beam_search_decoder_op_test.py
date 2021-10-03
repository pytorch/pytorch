




from caffe2.python import core
from collections import defaultdict, Counter
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

import unittest

DEFAULT_BEAM_WIDTH = 10
DEFAULT_PRUNE_THRESHOLD = 0.001


class TestCTCBeamSearchDecoderOp(serial.SerializedTestCase):
    @given(
        batch=st.sampled_from([1, 2, 4]),
        max_time=st.sampled_from([1, 8, 64]),
        alphabet_size=st.sampled_from([1, 2, 32, 128, 512]),
        beam_width=st.sampled_from([1, 2, 16, None]),
        num_candidates=st.sampled_from([1, 2]),
        **hu.gcs_cpu_only
    )
    @settings(deadline=None, max_examples=30)
    def test_ctc_beam_search_decoder(
        self, batch, max_time, alphabet_size, beam_width, num_candidates, gc, dc
    ):
        if not beam_width:
            beam_width = DEFAULT_BEAM_WIDTH
            op_seq_len = core.CreateOperator('CTCBeamSearchDecoder',
                ['INPUTS', 'SEQ_LEN'],
                ['OUTPUT_LEN', 'VALUES', 'OUTPUT_PROB'],
                num_candidates=num_candidates)

            op_no_seq_len = core.CreateOperator('CTCBeamSearchDecoder',
                ['INPUTS'],
                ['OUTPUT_LEN', 'VALUES', 'OUTPUT_PROB'],
                num_candidates=num_candidates)
        else:
            num_candidates = min(num_candidates, beam_width)
            op_seq_len = core.CreateOperator('CTCBeamSearchDecoder',
                ['INPUTS', 'SEQ_LEN'],
                ['OUTPUT_LEN', 'VALUES', 'OUTPUT_PROB'],
                beam_width=beam_width,
                num_candidates=num_candidates)

            op_no_seq_len = core.CreateOperator('CTCBeamSearchDecoder',
                ['INPUTS'],
                ['OUTPUT_LEN', 'VALUES', 'OUTPUT_PROB'],
                beam_width=beam_width,
                num_candidates=num_candidates)

        def input_generater():
            inputs = np.random.rand(max_time, batch, alphabet_size)\
                .astype(np.float32)
            seq_len = np.random.randint(1, max_time + 1, size=batch)\
                .astype(np.int32)
            return inputs, seq_len

        def ref_ctc_decoder(inputs, seq_len):
            output_len = np.zeros(batch * num_candidates, dtype=np.int32)
            output_prob = np.zeros(batch * num_candidates, dtype=np.float32)
            val = np.array([]).astype(np.int32)

            for i in range(batch):
                Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
                Pb[0][()] = 1
                Pnb[0][()] = 0
                A_prev = [()]
                ctc = inputs[:, i, :]
                ctc = np.vstack((np.zeros(alphabet_size), ctc))
                len_i = seq_len[i] if seq_len is not None else max_time

                for t in range(1, len_i + 1):
                    pruned_alphabet = np.where(ctc[t] > DEFAULT_PRUNE_THRESHOLD)[0]
                    for l in A_prev:
                        for c in pruned_alphabet:
                            if c == 0:
                                Pb[t][l] += ctc[t][c] * (Pb[t - 1][l] + Pnb[t - 1][l])
                            else:
                                l_plus = l + (c,)
                                if len(l) > 0 and c == l[-1]:
                                    Pnb[t][l_plus] += ctc[t][c] * Pb[t - 1][l]
                                    Pnb[t][l] += ctc[t][c] * Pnb[t - 1][l]
                                else:
                                    Pnb[t][l_plus] += \
                                        ctc[t][c] * (Pb[t - 1][l] + Pnb[t - 1][l])

                                if l_plus not in A_prev:
                                    Pb[t][l_plus] += \
                                        ctc[t][0] * \
                                        (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                                    Pnb[t][l_plus] += ctc[t][c] * Pnb[t - 1][l_plus]

                    A_next = Pb[t] + Pnb[t]
                    A_prev = sorted(A_next, key=A_next.get, reverse=True)
                    A_prev = A_prev[:beam_width]

                candidates = A_prev[:num_candidates]
                index = 0
                for candidate in candidates:
                    val = np.hstack((val, candidate))
                    output_len[i * num_candidates + index] = len(candidate)
                    output_prob[i * num_candidates + index] = Pb[t][candidate] + Pnb[t][candidate]
                    index += 1

            return [output_len, val, output_prob]

        def ref_ctc_decoder_max_time(inputs):
            return ref_ctc_decoder(inputs, None)

        inputs, seq_len = input_generater()

        self.assertReferenceChecks(
            device_option=gc,
            op=op_seq_len,
            inputs=[inputs, seq_len],
            reference=ref_ctc_decoder,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op_no_seq_len,
            inputs=[inputs],
            reference=ref_ctc_decoder_max_time,
        )


if __name__ == "__main__":
    import random
    random.seed(2603)
    unittest.main()
