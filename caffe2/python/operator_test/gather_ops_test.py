from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


class TestGatherOps(hu.HypothesisTestCase):
    @given(rows_num=st.integers(1, 10000),
           index_num=st.integers(0, 5000),
           **hu.gcs)
    def test_gather_ops(self, rows_num, index_num, gc, dc):
        data = np.random.random((rows_num, 10, 20)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(index_num, )).astype('int32')
        op = core.CreateOperator(
            'Gather',
            ['data', 'ind'],
            ['output'])

        def ref_gather(data, ind):
            if ind.size == 0:
                return [np.zeros((0, 10, 20)).astype(np.float32)]

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
           **hu.gcs)
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


class TestNdimGatherOps(hu.HypothesisTestCase):
    @given(num=st.integers(3, 20),
           index_num=st.integers(0, 10),
           **hu.gcs)
    def test_ndim_gather_ops(self, num, index_num, gc, dc):
        # case: axis = 0
        def ref_ndim_gather_0(data, ind):
            if ind.size == 0:
                return [np.zeros(
                    (0, data.shape[1], data.shape[2])).astype(np.float32)]

            output = [r for r in [data[i] for i in ind]]
            return [output]

        # case: axis = 1
        def ref_ndim_gather_1(data, ind):
            if ind.size == 0:
                return [np.zeros(
                    (data.shape[0], 0, data.shape[2])).astype(np.float32)]

            output = []
            for b in range(data.shape[0]):
                output.append([r for r in [data[b][i] for i in ind]])
            return [output]

        # case: axis = 2
        def ref_ndim_gather_2(data, ind):
            if ind.size == 0:
                return [np.zeros(
                    (data.shape[0], data.shape[1], 0)).astype(np.float32)]

            output = []
            for b in range(data.shape[0]):
                matrix = []
                for c in range(data.shape[1]):
                    matrix.append([data[b][c][i] for i in ind])
                output.append(matrix)
            return [output]

        ref_ndim_gathers = [
            ref_ndim_gather_0,
            ref_ndim_gather_1,
            ref_ndim_gather_2,
        ]

        data = np.random.random((
            np.random.randint(2, num),
            np.random.randint(2, num),
            np.random.randint(2, num)
        )).astype(np.float32)
        for _axis in range(3):
            ind = np.random.randint(
                data.shape[_axis], size=(index_num, )).astype('int32')
            op = core.CreateOperator(
                'NdimGather',
                ['data', 'ind'],
                ['output'],
                axis=_axis,
            )
            self.assertReferenceChecks(
                gc, op, [data, ind], ref_ndim_gathers[_axis])
            self.assertGradientChecks(gc, op, [data, ind], 0, [0])

        # test with gather and batch_gather operators
        self.ws.create_blob("data").feed(data)
        gather_ops = ['Gather', 'BatchGather']
        for _axis in range(len(gather_ops)):
            ind = np.random.randint(
                data.shape[_axis], size=(index_num, )).astype('int32')
            self.ws.create_blob("ind").feed(ind)
            self.ws.run(core.CreateOperator(
                'NdimGather',
                ['data', 'ind'],
                ['output'],
                axis=_axis,
            ))
            self.ws.run(core.CreateOperator(
                gather_ops[_axis],
                ['data', 'ind'],
                ['gather_output'],
            ))
            np.testing.assert_allclose(
                self.ws.blobs[("output")].fetch(),
                self.ws.blobs[("gather_output")].fetch(),
                rtol=1e-4,
                atol=1e-4
            )


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
