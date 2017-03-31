from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st

class TestGatherOps(hu.HypothesisTestCase):
    @given(rows_num=st.integers(1, 10000), 
           index_num=st.integers(1, 5000),
           **hu.gcs)
    def test_gather_ops(self, rows_num, index_num, gc, dc):
        data = np.random.random((rows_num, 10, 20)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(index_num,1)).astype('int32')
        op = core.CreateOperator(
            'Gather', 
            ['data', 'ind'], 
            ['output'])
        def ref_gather(data, ind):
            output = [r for r in [data[i] for i in ind]]
            return [output]
        self.assertReferenceChecks(
            gc,
            op=op,
            inputs=[data, ind],
            reference=ref_gather,
        )

if __name__ == "__main__":
    import unittest
    unittest.main()
