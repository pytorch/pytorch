from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


class GatherReference:
    @staticmethod
    def build_ref_gather(axis):
        def inner(data, ind):
            if ind.size == 0:
                shape = list(data.shape)
                shape[0] = 0
                return [np.zeros(tuple(shape)).astype(np.float32)]

            output = [np.array(x, dtype=np.float32) for x in data.take(ind, axis).astype(np.float32)]
            return [output]
        return inner


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


class TestGatherOps(hu.HypothesisTestCase):
    @given(rows_num=st.integers(1, 10000),
           index_num=st.integers(0, 5000),
           **hu.gcs)
    def test_gather_ops_axis_0(self, rows_num, index_num, gc, dc):
        data = np.random.random((rows_num, 10, 20)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(index_num, )).astype('int32')
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'])

        self.assertReferenceChecks(gc, op, [data, ind], GatherReference.build_ref_gather(0))


    @given(inputs=_inputs(),
           **hu.gcs)
    def test_gather_ops_axis_1(self, inputs, gc, dc):
        return
        data, ind = inputs
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'],
            axis=1)

        self.assertReferenceChecks(gc, op, [data, ind], GatherReference.build_ref_gather(1))
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])


class TestBatchGatherOps(hu.HypothesisTestCase):
    @given(inputs=_inputs(),
           **hu.gcs)
    def test_batch_gather_ops(self, inputs, gc, dc):
        data, ind = inputs
        op = core.CreateOperator(
            'BatchGather',
            ['data', 'ind'],
            ['output'])

        self.assertReferenceChecks(gc, op, [data, ind], GatherReference.build_ref_gather(1))
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])


class TestGatherFused8BitRowwise(hu.HypothesisTestCase):
    @given(rows_num=st.integers(1, 10000),
           cols_num=st.integers(1, 128),
           index_num=st.integers(0, 5000),
           **hu.gcs)
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
