



from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestConditionalOp(serial.SerializedTestCase):
    @serial.given(rows_num=st.integers(1, 10000), **hu.gcs_cpu_only)
    def test_conditional(self, rows_num, gc, dc):
        op = core.CreateOperator(
            "Conditional", ["condition", "data_t", "data_f"], "output"
        )
        data_t = np.random.random((rows_num, 10, 20)).astype(np.float32)
        data_f = np.random.random((rows_num, 10, 20)).astype(np.float32)
        condition = np.random.choice(a=[True, False], size=rows_num)

        def ref(condition, data_t, data_f):
            output = [
                data_t[i] if condition[i] else data_f[i]
                for i in range(rows_num)
            ]
            return (output,)

        self.assertReferenceChecks(gc, op, [condition, data_t, data_f], ref)
