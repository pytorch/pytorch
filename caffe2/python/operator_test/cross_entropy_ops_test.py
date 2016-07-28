from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_cross_entropy_with_logits(x, z):
    return np.maximum(x, 0) - x * z + np.log(1 + np.exp(-np.abs(x)))


def sigmoid_cross_entropy_with_logits_grad(x, z):
    return z - sigmoid(x)


class TestCrossEntropyOps(hu.HypothesisTestCase):
    @given(
        inputs=st.lists(
            elements=st.integers(min_value=1, max_value=5),
            min_size=1,
            max_size=2,
            average_size=2,
        ).flatmap(
            lambda shape: st.tuples(
                hu.arrays(
                    dims=shape,
                    elements=st.one_of(
                        st.floats(min_value=-1.0, max_value=-0.1),
                        st.floats(min_value=0.1, max_value=1.0),
                    )),
                hu.arrays(
                    dims=shape,
                    elements=st.sampled_from([0.0, 1.0]),
                ),
            )
        ),
    )
    def test_sigmoid_cross_entropy_with_logits(self, inputs):
        logits, targets = inputs

        def sigmoid_xentr_logit_ref(logits, targets):
            s = sigmoid_cross_entropy_with_logits(logits, targets)
            m = np.mean(s, axis=len(logits.shape) - 1)
            return (m, )

        def sigmoid_xentr_logit_grad_ref(g_out, outputs, fwd_inputs):
            fwd_logits, fwd_targets = fwd_inputs
            inner_size = fwd_logits.shape[-1]
            m = fwd_targets - sigmoid(fwd_logits)
            g_in = -np.expand_dims(g_out, axis=-1) * m / inner_size
            return (g_in, None)

        op = core.CreateOperator(
            'SigmoidCrossEntropyWithLogits',
            ['logits', 'targets'],
            ['xentropy'])
        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [logits, targets],
            sigmoid_xentr_logit_ref,
            output_to_grad='xentropy',
            grad_reference=sigmoid_xentr_logit_grad_ref)
