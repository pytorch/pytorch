from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


class TestGatherOps(hu.HypothesisTestCase):
    @given(rows_num=st.integers(1, 10000),
           index_num=st.integers(1, 5000),
           **hu.gcs)
    def test_gather_ops(self, rows_num, index_num, gc, dc):
        data = np.random.random((rows_num, 10, 20)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(index_num, 1)).astype('int32')
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'])

        def ref_gather(data, ind):
            output = [r for r in [data[i] for i in ind]]
            return [output]

        self.assertReferenceChecks(gc, op, [data, ind], ref_gather)


@st.composite
def _inputs(draw):
    rows_num = draw(st.integers(1, 100))
    index_num = draw(st.integers(1, 10))
    batch_size = draw(st.integers(2, 10))
    return (
        draw(hnp.arrays(
            np.float32,
            (batch_size, rows_num, 2),
            elements=st.floats(-10.0, 10.0),
        )),
        draw(hnp.arrays(
            np.int32,
            (index_num, 1),
            elements=st.integers(0, rows_num - 1),
        )),
    )


class TestBatchGatherOps(hu.HypothesisTestCase):
    @given(inputs=_inputs(),
           **hu.gcs_cpu_only)
    def test_batch_gather_ops(self, inputs, gc, dc):
        data, ind = inputs
        op = core.CreateOperator(
            'BatchGather',
            ['data', 'ind'],
            ['output'])

        def ref_batch_gather(data, ind):
            output = []
            for b in range(data.shape[0]):
                output.append([r for r in [data[b][i] for i in ind]])
            return [output]

        self.assertReferenceChecks(gc, op, [data, ind], ref_batch_gather)
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
