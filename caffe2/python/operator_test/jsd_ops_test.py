




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


def entropy(p):
    q = 1. - p
    return -p * np.log(p) - q * np.log(q)


def jsd(p, q):
    return [entropy(p / 2. + q / 2.) - entropy(p) / 2. - entropy(q) / 2.]


def jsd_grad(go, o, pq_list):
    p, q = pq_list
    m = (p + q) / 2.
    return [np.log(p * (1 - m) / (1 - p) / m) / 2. * go, None]


class TestJSDOps(serial.SerializedTestCase):
    @serial.given(n=st.integers(10, 100), **hu.gcs_cpu_only)
    def test_bernoulli_jsd(self, n, gc, dc):
        p = np.random.rand(n).astype(np.float32)
        q = np.random.rand(n).astype(np.float32)
        op = core.CreateOperator("BernoulliJSD", ["p", "q"], ["l"])
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[p, q],
            reference=jsd,
            output_to_grad='l',
            grad_reference=jsd_grad,
        )
