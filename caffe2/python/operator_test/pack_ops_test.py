from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import time


class TestTensorPackOps(serial.SerializedTestCase):

    def pack_segments_ref(self, return_presence_mask=False, max_length=None):
        def pack_segments_ref(lengths, data, max_length=max_length):
            arr = []
            constant_values = 0
            if data.dtype.char == 'S':
                constant_values = ''
            if max_length is None:
                max_length = np.max(lengths)
            start = 0
            for idx in range(np.size(lengths)):
                len = lengths[idx] if max_length >= lengths[idx] else max_length
                chunk = data[start : start + len]
                pad_length = max_length - len

                # ((0, pad_length), (0, 0)) says add pad_length rows of padding
                # below chunk and 0 rows of padding elsewhere
                arr.append(
                    np.pad(
                        chunk, ((0, pad_length), (0, 0)),
                        mode=str("constant"),
                        constant_values=constant_values
                    )
                )
                start += lengths[idx]
            result = [arr]
            if return_presence_mask:
                presence_arr = []
                for length in lengths:
                    length = length if max_length >= length else max_length
                    pad_length = max_length - length
                    presence_arr.append(
                        np.pad(
                            np.ones((length), dtype=np.bool), ((0, pad_length)),
                            mode=str("constant")
                        )
                    )
                result.append(presence_arr)
            return result

        return pack_segments_ref

    @serial.given(
        num_seq=st.integers(10, 100),
        cell_size=st.integers(1, 10),
        max_length_buffer=st.integers(-5, 5),
        **hu.gcs
    )
    def test_pack_with_max_length_ops(
        self, num_seq, cell_size, max_length_buffer, gc, dc
    ):
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
        max_length = num_seq + max_length_buffer
        op = core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t'], max_length=max_length)
        workspace.FeedBlob('l', lengths)
        workspace.FeedBlob('d', data)
        start = time.time()
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[lengths, data, max_length],
            reference=self.pack_segments_ref(max_length=max_length),
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
            max_length=max_length,
            device_option=gc))
        workspace.RunOperatorOnce(core.CreateOperator(
            'UnpackSegments',
            ['l', 't'],
            ['newd'],
            max_length=max_length,
            device_option=gc))
        assert(workspace.FetchBlob('t').shape[1] == max_length)

        def _cal_unpacked_data(data):
            if max_length >= num_seq:
                return data
            output = None
            start = 0
            for i, length in enumerate(lengths):
                new_len = max_length if length > max_length else length
                chunk = data[start: start + new_len]
                if output is None:
                    output = chunk
                else:
                    output = np.concatenate((output, chunk), axis=0)
                start += length
            return output

        true_newd = _cal_unpacked_data(workspace.FetchBlob('d'))
        assert((workspace.FetchBlob('newd') == true_newd).all())

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

    def test_pad_no_minf(self):
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
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'PackSegments', ['l', 'd'], ['t'], pad_minf=False),
        )
        result = workspace.FetchBlob('t')
        assert(result[0, -1, 0] == 0.0)

        workspace.FeedBlob(
            'i',
            np.array([
                [1, 1],
                [2, 2],
                [2, 2],
                [3, 3],
                [3, 3],
                [3, 3]],
                dtype=np.int32))
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'PackSegments', ['l', 'i'], ['t2'], pad_minf=False),
        )
        result = workspace.FetchBlob('t2')
        assert(result[0, -1, 0] == 0)

    @given(**hu.gcs)
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
