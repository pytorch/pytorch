from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


def calculate_ap(predictions, labels):
    N, D = predictions.shape
    ap = np.zeros(D)
    num_range = np.arange((N), dtype=np.float32) + 1
    for k in range(D):
        scores = predictions[:N, k]
        label = labels[:N, k]
        sortind = np.argsort(-scores, kind='mergesort')
        truth = label[sortind]
        precision = np.cumsum(truth) / num_range
        ap[k] = precision[truth.astype(np.bool)].sum() / max(1, truth.sum())
    return ap


class TestAPMeterOps(hu.HypothesisTestCase):
    @given(predictions=hu.arrays(dims=[10, 3],
           elements=st.floats(allow_nan=False,
                              allow_infinity=False,
                              min_value=0.1,
                              max_value=1)),
           labels=hu.arrays(dims=[10, 3],
                            dtype=np.int32,
                            elements=st.integers(min_value=0,
                                                 max_value=1)),
           **hu.gcs_cpu_only)
    def test_average_precision(self, predictions, labels, gc, dc):
        op = core.CreateOperator(
            "APMeter",
            ["predictions", "labels"],
            ["AP"],
            buffer_size=10,
        )

        def op_ref(predictions, labels):
            ap = calculate_ap(predictions, labels)
            return (ap, )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[predictions, labels],
            reference=op_ref)

    @given(predictions=hu.arrays(dims=[10, 3],
           elements=st.floats(allow_nan=False,
                              allow_infinity=False,
                              min_value=0.1,
                              max_value=1)),
           labels=hu.arrays(dims=[10, 3],
                            dtype=np.int32,
                            elements=st.integers(min_value=0,
                                                 max_value=1)),
           **hu.gcs_cpu_only)
    def test_average_precision_small_buffer(self, predictions, labels, gc, dc):
        op_small_buffer = core.CreateOperator(
            "APMeter",
            ["predictions", "labels"],
            ["AP"],
            buffer_size=5,
        )

        def op_ref(predictions, labels):
            # We can only hold the last 5 in the buffer
            ap = calculate_ap(predictions[5:], labels[5:])
            return (ap, )

        self.assertReferenceChecks(
            device_option=gc,
            op=op_small_buffer,
            inputs=[predictions, labels],
            reference=op_ref
        )
