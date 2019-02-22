from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

from hypothesis import given
import hypothesis.strategies as st
import unittest

from caffe2.python import core, dyndep, workspace
import caffe2.python.hypothesis_test_util as hu


class SparseLabelSplitOps(hu.HypothesisTestCase):
    def sparse_label_split_test_core(self, num_labels, N_len, return_offset_map,
                                     gc, dc):
        max_val = 100
        len = np.random.randint(low=0, high=num_labels + 1, size=N_len)\
            .astype(np.int32)
        N_ind = sum(len)

        key = np.random.randint(low=0, high=num_labels, size=N_ind)\
            .astype(np.int64)
        label = np.random.randint(low=0, high=max_val, size=N_ind)\
            .astype(np.float32)
        eid = np.array(
            sum([[i] * len[i] for i in range(N_len)], []), dtype=np.int32
        )

        label_str = ["label{i}".format(i=i) for i in range(num_labels)]
        eid_str = ["eid{i}".format(i=i) for i in range(num_labels)]
        outputs = label_str + eid_str
        if return_offset_map:
            outputs.append('offset_map')
        op = core.CreateOperator(
            "SparseLabelSplit", ["len", "key", "label"],
            outputs,
            num_labels=num_labels
        )

        def ref(len, key, label):
            ret_labels = []
            ret_eids = []
            for i in range(num_labels):
                ret_labels.append(label[key == i])
                ret_eids.append(eid[key == i])

            ret = ret_labels + ret_eids

            if return_offset_map:
                ret_offset = [0] * N_ind
                offset_vec = [0] * num_labels
                for i in range(N_ind):
                    ret_offset[i] = offset_vec[key[i]]
                    offset_vec[key[i]] += 1

                ret.append(ret_offset)

            return ret

        self.assertDeviceChecks(dc, op, [len, key, label], [0])
        self.assertReferenceChecks(gc, op, [len, key, label], ref)

        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=[len, key, label],
            outputs_to_check=2,
            outputs_with_grads=list(range(num_labels)),
        )

    @given(
        num_labels=st.integers(min_value=1, max_value=10),
        N_len=st.integers(min_value=0, max_value=10),
        return_offset_map=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_sparse_label_split(self, num_labels, N_len, return_offset_map, gc, dc):
        self.sparse_label_split_test_core(num_labels, N_len, return_offset_map, gc, dc)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
    @given(
        num_labels=st.integers(min_value=1, max_value=10),
        N_len=st.integers(min_value=0, max_value=10),
        **hu.gcs_gpu_only
    )
    def test_sparse_label_split_gpu(self, num_labels, N_len, gc, dc):
        self.sparse_label_split_test_core(num_labels, N_len, True, gc, dc)
