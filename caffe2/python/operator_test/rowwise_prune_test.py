from __future__ import absolute_import, division, print_function

import unittest

import caffe2.python.hypothesis_test_util as hu
import numpy as np
from caffe2.python import core, workspace


def rowwise_prune_ref(x, indicator, thd, abs):
    selected_idx = (np.abs(indicator) if abs else indicator) > thd
    x_out = x[selected_idx, :]
    return x_out


def rowwise_select_ref(x, compressed_idx):
    x_out = x.copy()
    num_cols = x.shape[1]
    max_new_i = -1
    for old_i, new_i in enumerate(compressed_idx):
        max_new_i = max(new_i, max_new_i)
        if new_i >= 0:
            x_out[new_i, :] = x[old_i, :]

    x_out.resize(max_new_i + 1, num_cols)
    return x_out


class TestQuantile(hu.HypothesisTestCase):
    def _test_rowwise_prune_float_uint8(self, n_rows=1000, n_cols=64, abs=1):
        x = np.random.randint(100, size=(n_rows, n_cols), dtype=np.uint8)
        indicator = np.random.randn(n_rows).astype(dtype=np.float32)
        thd = np.random.rand(1).astype(dtype=np.float32)
        net = core.Net("test_net")
        net.Proto().type = "dag"
        workspace.FeedBlob("X", x)
        workspace.FeedBlob("indicator", indicator)
        workspace.FeedBlob("thd", thd)

        net.RowwisePruneFloatUInt8(
            ["X", "indicator", "thd"], ["X", "compressed_idx"], abs=abs
        )

        workspace.RunNetOnce(net)
        x_out = workspace.FetchBlob("X")
        compressed_idx_out = workspace.FetchBlob("compressed_idx")

        x_expected = rowwise_prune_ref(x=x, indicator=indicator, thd=thd, abs=abs)
        np.testing.assert_equal(x_out, x_expected)

        x_transformed_out = rowwise_select_ref(x=x, compressed_idx=compressed_idx_out)
        np.testing.assert_equal(x_transformed_out, x_expected)

    def test_test_rowwise_prune_float_uint8_1(self):
        self._test_rowwise_prune_float_uint8(abs=1)

    def test_test_rowwise_prune_float_uint8_2(self):
        self._test_rowwise_prune_float_uint8(abs=0)


if __name__ == "__main__":
    global_options = ["caffe2"]
    core.GlobalInit(global_options)
    unittest.main()
