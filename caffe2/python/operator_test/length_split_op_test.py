from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestLengthSplitOperator(hu.HypothesisTestCase):

    def _length_split_op_ref(self, input_lengths, n_splits):
        output = []
        n_splits = n_splits[0]
        for i in range(input_lengths.size):
            x = input_lengths[i]
            mod = x % n_splits
            val = x // n_splits + 1
            for _ in range(n_splits):
                if mod > 0:
                    output.append(val)
                    mod -= 1
                else:
                    output.append(val - 1)
        return [np.array(output).astype(np.int32)]

    @given(**hu.gcs_cpu_only)
    def test_length_split_edge(self, gc, dc):
        input_lengths = np.array([3, 4, 5]).astype(np.int32)
        n_splits_ = np.array([5]).astype(np.int32)

        op = core.CreateOperator(
            'LengthsSplit',
            ['input_lengths',
             'n_splits'],
            ['Y'],
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_lengths,
                    n_splits_],
            reference=self._length_split_op_ref,
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [input_lengths, n_splits_], [0])

    @given(**hu.gcs_cpu_only)
    def test_length_split_arg(self, gc, dc):
        input_lengths = np.array([9, 4, 5]).astype(np.int32)
        n_splits = 3

        op = core.CreateOperator(
            'LengthsSplit',
            ['input_lengths'],
            ['Y'], n_splits=n_splits
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_lengths],
            reference=lambda x : self._length_split_op_ref(x, [n_splits]),
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [input_lengths], [0])

    @given(**hu.gcs_cpu_only)
    def test_length_split_override_arg(self, gc, dc):
        input_lengths = np.array([9, 4, 5]).astype(np.int32)
        n_splits = 2
        n_splits_ = np.array([3]).astype(np.int32)

        op = core.CreateOperator(
            'LengthsSplit',
            ['input_lengths',
             'n_splits'],
            ['Y'], n_splits=n_splits
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_lengths,
                    n_splits_],
            reference=self._length_split_op_ref,
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [input_lengths, n_splits_], [0])

    @given(m=st.integers(1, 100), n_splits=st.integers(1, 20),
           **hu.gcs_cpu_only)
    def test_length_split_even_divide(self, m, n_splits, gc, dc):
        # multiples of n_splits
        input_lengths = np.random.randint(100, size=m).astype(np.int32) * n_splits
        n_splits_ = np.array([n_splits]).astype(np.int32)

        op = core.CreateOperator(
            'LengthsSplit',
            ['input_lengths',
             'n_splits'],
            ['Y'],
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_lengths,
                    n_splits_],
            reference=self._length_split_op_ref,
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [input_lengths, n_splits_], [0])

    @given(m=st.integers(1, 100), n_splits=st.integers(1, 20),
           **hu.gcs_cpu_only)
    def test_length_split_random(self, m, n_splits, gc, dc):
        input_lengths = np.random.randint(100, size=m).astype(np.int32)
        n_splits_ = np.array([n_splits]).astype(np.int32)

        op = core.CreateOperator(
            'LengthsSplit',
            ['input_lengths',
             'n_splits'],
            ['Y'],
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_lengths,
                    n_splits_],
            reference=self._length_split_op_ref,
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [input_lengths, n_splits_], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
