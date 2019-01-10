from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import time


class TestTensorPackOps(hu.HypothesisTestCase):

    def pack_segments_ref(self, return_presence_mask=False):
        def pack_segments_ref(lengths, data):
            arr = []
            constant_values = 0
            if data.dtype.char == 'S':
                constant_values = ''
            for idx in range(np.size(lengths)):
                chunk = data[np.sum(lengths[:idx]):np.sum(lengths[:idx + 1])]
                pad_length = np.max(lengths) - lengths[idx]

                # ((0, pad_length), (0, 0)) says add pad_length rows of padding
                # below chunk and 0 rows of padding elsewhere
                arr.append(
                    np.pad(
                        chunk, ((0, pad_length), (0, 0)),
                        mode=str("constant"),
                        constant_values=constant_values
                    )
                )
            result = [arr]
            if return_presence_mask:
                presence_arr = []
                for length in lengths:
                    pad_length = np.max(lengths) - length
                    presence_arr.append(
                        np.pad(
                            np.ones((length), dtype=np.bool), ((0, pad_length)),
                            mode=str("constant")
                        )
                    )
                result.append(presence_arr)
            return result

        return pack_segments_ref

    @given(
        num_seq=st.integers(10, 500),
        cell_size=st.integers(1, 10),
        **hu.gcs
    )
    def test_pack_ops(self, num_seq, cell_size, gc, dc):
        # create data
        lengths = np.arange(num_seq, dtype=np.int32) + 1
        num_cell = np.sum(lengths)
        data = np.zeros(num_cell * cell_size, dtype=np.float32)
        left = np.cumsum(np.arange(num_seq) * cell_size)
        right = np.cumsum(lengths * cell_size)
        for i in range(num_seq):
            data[left[i]:right[i]] = i + 1.0
        data.resize(num_cell, cell_size)
        print("\nnum seq:{},    num cell: {},   cell size:{}\n".format(
            num_seq, num_cell, cell_size)
            + "=" * 60
        )
        # run test
        op = core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t'])
        workspace.FeedBlob('l', lengths)
        workspace.FeedBlob('d', data)

        start = time.time()
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[lengths, data],
            reference=self.pack_segments_ref(),
        )
        end = time.time()
        print("{} used time: {}".format(gc, end - start).replace('\n', ' '))

        with core.DeviceScope(gc):
            workspace.FeedBlob('l', lengths)
            workspace.FeedBlob('d', data)
        workspace.RunOperatorOnce(core.CreateOperator(
            'PackSegments',
            ['l', 'd'],
            ['t'],
            device_option=gc))
        workspace.RunOperatorOnce(core.CreateOperator(
            'UnpackSegments',
            ['l', 't'],
            ['newd'],
            device_option=gc))
        assert((workspace.FetchBlob('newd') == workspace.FetchBlob('d')).all())

    @given(
        **hu.gcs_cpu_only
    )
    def test_pack_ops_str(self, gc, dc):
        # GPU does not support string. Test CPU implementation only.
        workspace.FeedBlob('l', np.array([1, 2, 3], dtype=np.int64))
        strs = np.array([
            ["a", "a"],
            ["b", "b"],
            ["bb", "bb"],
            ["c", "c"],
            ["cc", "cc"],
            ["ccc", "ccc"]],
            dtype='|S')
        workspace.FeedBlob('d', strs)
        workspace.RunOperatorOnce(core.CreateOperator(
            'PackSegments',
            ['l', 'd'],
            ['t'],
            device_option=gc))
        workspace.RunOperatorOnce(core.CreateOperator(
            'UnpackSegments',
            ['l', 't'],
            ['newd'],
            device_option=gc))
        assert((workspace.FetchBlob('newd') == workspace.FetchBlob('d')).all())

    def test_pad_minf(self):
        workspace.FeedBlob('l', np.array([1, 2, 3], dtype=np.int32))
        workspace.FeedBlob(
            'd',
            np.array([
                [1.0, 1.1],
                [2.0, 2.1],
                [2.2, 2.2],
                [3.0, 3.1],
                [3.2, 3.3],
                [3.4, 3.5]],
                dtype=np.float32))
        workspace.RunOperatorOnce(core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t'], pad_minf=True))
        workspace.RunOperatorOnce(core.CreateOperator(
            'Exp', ['t'], ['r']
        ))
        result = workspace.FetchBlob('t')
        assert(result[0, -1, 0] < -1000.0)

        # The whole point of padding with -inf is that when we exponentiate it
        # then it should be zero.
        exponentiated = workspace.FetchBlob('r')
        assert(exponentiated[0, -1, 0] == 0.0)

    @given(**hu.gcs_cpu_only)
    def test_presence_mask(self, gc, dc):
        lengths = np.array([1, 2, 3], dtype=np.int32)
        data = np.array(
            [
                [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0],
                [3.0, 3.0]
            ],
            dtype=np.float32
        )

        op = core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t', 'p'], return_presence_mask=True
        )
        workspace.FeedBlob('l', lengths)
        workspace.FeedBlob('d', data)
        inputs = [lengths, data]
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=self.pack_segments_ref(return_presence_mask=True),
        )

        op = core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t', 'p'], return_presence_mask=True
        )
        workspace.RunOperatorOnce(op)

        output = workspace.FetchBlob('t')
        expected_output_shape = (3, 3, 2)
        self.assertEquals(output.shape, expected_output_shape)

        presence_mask = workspace.FetchBlob('p')
        expected_presence_mask = np.array(
            [[True, False, False], [True, True, False], [True, True, True]],
            dtype=np.bool
        )
        self.assertEqual(presence_mask.shape, expected_presence_mask.shape)
        np.testing.assert_array_equal(presence_mask, expected_presence_mask)

    def test_presence_mask_empty(self):
        lengths = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.float32)

        op = core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t', 'p'], return_presence_mask=True
        )
        workspace.FeedBlob('l', lengths)
        workspace.FeedBlob('d', data)
        workspace.RunOperatorOnce(op)

        output = workspace.FetchBlob('p')
        expected_output_shape = (0, 0)
        self.assertEquals(output.shape, expected_output_shape)

    @given(**hu.gcs_cpu_only)
    def test_out_of_bounds(self, gc, dc):
        # Copy pasted from test_pack_ops but with 3 changed to 4
        lengths = np.array([1, 2, 4], dtype=np.int32)
        data = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [3.0, 3.0],
            [3.0, 3.0]], dtype=np.float32)
        op = core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t'])

        inputs = [lengths, data]
        self.assertRunOpRaises(
            device_option=gc,
            op=op,
            inputs=inputs,
            exception=RuntimeError
        )

    @given(**hu.gcs_cpu_only)
    def test_under_bounds(self, gc, dc):
        # Copy pasted from test_pack_ops but with 3 changed to 2
        lengths = np.array([1, 2, 2], dtype=np.int32)
        data = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [3.0, 3.0],
            [3.0, 3.0]], dtype=np.float32)
        op = core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t'])

        inputs = [lengths, data]
        self.assertRunOpRaises(
            device_option=gc,
            op=op,
            inputs=inputs,
            exception=RuntimeError
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
