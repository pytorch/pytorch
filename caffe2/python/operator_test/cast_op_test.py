





from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given
import numpy as np


class TestCastOp(hu.HypothesisTestCase):

    @given(**hu.gcs)
    def test_cast_int_float(self, gc, dc):
        data = np.random.rand(5, 5).astype(np.int32)
        # from int to float
        op = core.CreateOperator('Cast', 'data', 'data_cast', to=1, from_type=2)
        self.assertDeviceChecks(dc, op, [data], [0])
        # This is actually 0
        self.assertGradientChecks(gc, op, [data], 0, [0])

    @given(**hu.gcs)
    def test_cast_int_float_empty(self, gc, dc):
        data = np.random.rand(0).astype(np.int32)
        # from int to float
        op = core.CreateOperator('Cast', 'data', 'data_cast', to=1, from_type=2)
        self.assertDeviceChecks(dc, op, [data], [0])
        # This is actually 0
        self.assertGradientChecks(gc, op, [data], 0, [0])

    @given(data=hu.tensor(dtype=np.int32), **hu.gcs_cpu_only)
    def test_cast_int_to_string(self, data, gc, dc):
        op = core.CreateOperator(
            'Cast', 'data', 'data_cast', to=core.DataType.STRING)

        def ref(data):
            ret = data.astype(dtype=np.str)
            # the string blob will be fetched as object, we feed and re-fetch
            # to mimic this.
            with hu.temp_workspace('tmp_ref_int_to_string'):
                workspace.FeedBlob('tmp_blob', ret)
                fetched_ret = workspace.FetchBlob('tmp_blob')
            return (fetched_ret, )

        self.assertReferenceChecks(gc, op, inputs=[data], reference=ref)
