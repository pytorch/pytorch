



import numpy as np

from caffe2.python import core, workspace
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

# Basic implementation of gather for axis == 0, shich is lookup of indices
# in the outer dimension. Keeping it for reference here, although is similar
# to more general function below.
def ref_gather_axis0():
    def inner(data, ind):
        if ind.size == 0 or data.shape[0] == 0:
            return [np.zeros((0, 10, 20)).astype(np.float32)]
        output = [data[i] for i in ind]
        return [output]
    return inner

# Returns axis-based lookup. We just use numpy take() which handles different
# axis values as we want.
def ref_gather(axis):
    def inner(data, ind):
        if ind.size == 0 or data.shape[axis] == 0:
            shape = list(data.shape)
            shape[0] = 0
            return [np.zeros(tuple(shape)).astype(np.float32)]
        # np.take() does axis lookup same as gather
        output = data.take(ind, axis).astype(np.float32)
        return [output]
    return inner

# Gather(..., match_outer==True)
def ref_gather_match_outer(axis=1):
    def inner(data, ind):
        if ind.size == 0 or data.shape[axis] == 0:
            shape = list(data.shape)
            shape[0] = 0
            return [np.zeros(tuple(shape)).astype(np.float32)]
        input_shape = list(data.shape)
        output_shape = input_shape[:axis] + list(ind.shape[axis:]) + input_shape[axis + 1:]
        output = np.zeros(tuple(output_shape)).astype(np.float32)
        if axis == 1:
            for i in range(data.shape[0]):
                output[i] = data[i, ind[i], ]
        elif axis == 2:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    output[i, j] = data[i, j, ind[i, j], ]
        else:
            raise NotImplementedError
        return [output]
    return inner

class TestGatherOps(serial.SerializedTestCase):
    @given(rows_num=st.integers(0, 10000),
           index_num=st.integers(0, 5000),
           **hu.gcs)
    @settings(deadline=10000)
    def test_gather_ops(self, rows_num, index_num, gc, dc):
        data = np.random.random((rows_num, 10, 20)).astype(np.float32)

        if rows_num > 0:
            ind = np.random.randint(rows_num, size=(index_num, )).astype('int32')
        else:
            ind = np.random.randint(10, size=(index_num, )).astype('int32')
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'])

        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_axis0())
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        return

    # Test axis == 2, this keeps outer dimension but will replace data
    # within axis by lookup of index array (repeated for each outer entry)
    @given(batch_num=st.integers(1, 4000),
           rows_num=st.integers(1, 6),
           index_num=st.integers(1, 20),
           **hu.gcs)
    def test_gather_ops_axis2(self, batch_num, rows_num, index_num, gc, dc):
        data = np.random.random((batch_num, rows_num, 5)).astype(np.float32)
        ind = np.random.randint(5, size=(index_num, )).astype('int32')
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'],
            axis=2)

        self.assertReferenceChecks(gc, op, [data, ind], ref_gather(axis=2))
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        return

    # Test match_outer == true, the indices has the same outer dimensions as data
    @given(batch_num=st.integers(1, 40),
           rows_num=st.integers(1, 6),
           index_num=st.integers(1, 20),
           **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_gather_ops_match_outer(self, batch_num, rows_num, index_num, gc, dc):
        data = np.random.random((batch_num, rows_num, 5)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(batch_num, index_num)).astype('int32')
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'],
            axis=1,
            match_outer=True)

        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_match_outer())
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])
        return

    # Test BatchGather with match_outer == true, the indices has the same outer dimensions as data
    # Note BatchGather is equivalent to Gather(..., axis=1)
    @given(batch_num=st.integers(1, 40),
           rows_num=st.integers(1, 6),
           index_num=st.integers(1, 20),
           **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_batch_gather_op_match_outer(self, batch_num, rows_num, index_num, gc, dc):
        data = np.random.random((batch_num, rows_num, 5)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(batch_num, index_num)).astype('int32')
        op = core.CreateOperator(
            'BatchGather',
            ['data', 'ind'],
            ['output'],
            match_outer=True)

        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_match_outer())
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])
        return

    # when the data is larger,
    # this test sometimes passes, sometimes fails,
    # test log here: https://fb.quip.com/SeiyAVWQXvsN (second run failed)
    # after some digging, this turns out to be numerical error,
    # the failed run has max|grad - estimated_grad| = 0.009
    # so here we changed the gradient checking threshold to 0.02 for this test to pass
    @given(batch_num=st.integers(1, 30),
           rows_num=st.integers(1, 6),
           index_num=st.integers(1, 10),
           index_num2=st.integers(1, 10),
           axis2_num=st.integers(1, 10),
           **hu.gcs_cpu_only)
    @settings(deadline=None, max_examples=50)
    def test_gather_op_match_outer_axis2_data4D_ind4D(
        self, batch_num, rows_num, axis2_num, index_num, index_num2, gc, dc
    ):
        data = np.random.random((batch_num, rows_num, axis2_num, 5)).astype(np.float32)
        ind = np.random.randint(axis2_num, size=(batch_num, rows_num, index_num, index_num2)).astype('int32')
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'],
            axis=2,
            match_outer=True)

        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_match_outer(axis=2))
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        self.assertGradientChecks(gc, op, [data, ind], 0, [0], threshold=0.02)
        return


# Generates data arrays of max dims 10x100x2 and indexing array up to rows_num
@st.composite
def _inputs(draw):
    batch_size = draw(st.integers(2, 10))
    rows_num = draw(st.integers(1, 100))
    block_size = draw(st.integers(1, 2))
    index_num = draw(st.integers(1, 10))
    return (
        draw(hnp.arrays(
            np.float32,
            (batch_size, rows_num, block_size),
            elements=hu.floats(-10.0, 10.0),
        )),
        draw(hnp.arrays(
            np.int32,
            (index_num, 1),
            elements=st.integers(0, rows_num - 1),
        )),
    )

class TestBatchGatherOps(hu.HypothesisTestCase):
    @given(inputs=_inputs(),
           **hu.gcs)
    @settings(deadline=10000)
    def test_batch_gather_ops(self, inputs, gc, dc):
        data, ind = inputs
        op = core.CreateOperator(
            'BatchGather',
            ['data', 'ind'],
            ['output'])
        self.assertReferenceChecks(gc, op, [data, ind], ref_gather(axis=1))
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])


class TestGatherFused8BitRowwise(hu.HypothesisTestCase):
    @given(rows_num=st.integers(1, 10000),
           cols_num=st.integers(1, 128),
           index_num=st.integers(0, 5000),
           **hu.gcs)
    @settings(deadline=10000)
    def test_batch_gather_ops(self, rows_num, cols_num, index_num, gc, dc):
        data = np.random.random((rows_num, cols_num)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(index_num, )).astype('int32')

        net = core.Net("bench")

        quantized_data = net.FloatToFused8BitRowwiseQuantized(
            'data', 'quantized_data')
        dequantized_data = net.Fused8BitRowwiseQuantizedToFloat(
            quantized_data, 'dequantized_data')

        net.Gather(
            [dequantized_data, 'ind'], 'gather_reference')
        net.GatherFused8BitRowwise(
            [quantized_data, 'ind'], 'gather_quantized')

        workspace.FeedBlob('data', data)
        workspace.FeedBlob('ind', ind)
        workspace.CreateNet(net)
        workspace.RunNetOnce(net)

        gather_reference = workspace.FetchBlob('gather_reference')
        gather_quantized = workspace.FetchBlob('gather_quantized')
        np.testing.assert_array_almost_equal(gather_reference, gather_quantized)



if __name__ == "__main__":
    import unittest
    unittest.main()
