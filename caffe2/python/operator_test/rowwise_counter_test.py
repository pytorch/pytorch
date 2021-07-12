

import unittest

import caffe2.python.hypothesis_test_util as hu
import numpy as np
from caffe2.python import core, workspace


def update_counter_ref(prev_iter, update_counter, indices, curr_iter, counter_halflife):
    prev_iter_out = prev_iter.copy()
    update_counter_out = update_counter.copy()

    counter_neg_log_rho = np.log(2) / counter_halflife
    for i in indices:
        iter_diff = curr_iter[0] - prev_iter_out[i]
        prev_iter_out[i] = curr_iter[0]
        update_counter_out[i] = (
            1.0 + np.exp(-iter_diff * counter_neg_log_rho) * update_counter_out[i]
        )
    return prev_iter_out, update_counter_out


class TestRowWiseCounter(hu.HypothesisTestCase):
    def test_rowwise_counter(self):
        h = 8 * 20
        n = 5
        curr_iter = np.array([100], dtype=np.int64)

        update_counter = np.random.randint(99, size=h).astype(np.float64)
        prev_iter = np.random.rand(h, 1).astype(np.int64)
        indices = np.unique(np.random.randint(0, h, size=n))
        indices.sort(axis=0)
        counter_halflife = 1

        net = core.Net("test_net")
        net.Proto().type = "dag"

        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("curr_iter", curr_iter)
        workspace.FeedBlob("update_counter", update_counter)
        workspace.FeedBlob("prev_iter", prev_iter)

        net.RowWiseCounter(
            ["prev_iter", "update_counter", "indices", "curr_iter"],
            ["prev_iter", "update_counter"],
            counter_halflife=counter_halflife,
        )

        workspace.RunNetOnce(net)

        prev_iter_out = workspace.FetchBlob("prev_iter")
        update_counter_out = workspace.FetchBlob("update_counter")

        prev_iter_out_ref, update_counter_out_ref = update_counter_ref(
            prev_iter,
            update_counter,
            indices,
            curr_iter,
            counter_halflife=counter_halflife,
        )
        assert np.allclose(prev_iter_out, prev_iter_out_ref, rtol=1e-3)
        assert np.allclose(update_counter_out, update_counter_out_ref, rtol=1e-3)


if __name__ == "__main__":
    global_options = ["caffe2"]
    core.GlobalInit(global_options)
    unittest.main()
