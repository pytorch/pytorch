




import numpy as np
import unittest

from caffe2.python import core, workspace, tt_core
import caffe2.python.hypothesis_test_util as hu


class TestTTSVD(hu.HypothesisTestCase):
    def test_full_tt_svd(self):
        size = 256
        np.random.seed(1234)
        X = np.expand_dims(
            np.random.rand(size).astype(np.float32), axis=0)
        W = np.random.rand(size, size).astype(np.float32)
        b = np.zeros(size).astype(np.float32)
        inp_sizes = [4, 4, 4, 4]
        out_sizes = [4, 4, 4, 4]

        op_fc = core.CreateOperator(
            "FC",
            ["X", "W", "b"],
            ["Y"],
        )
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.FeedBlob("b", b)
        workspace.RunOperatorOnce(op_fc)
        Y_fc = workspace.FetchBlob("Y").flatten()

        # Testing TT-decomposition with high ranks
        full_tt_ranks = [1, 16, 256, 16, 1]
        full_cores = tt_core.matrix_to_tt(W, inp_sizes, out_sizes,
                                          full_tt_ranks)

        full_op_tt = core.CreateOperator(
            "TT",
            ["X", "b", "cores"],
            ["Y"],
            inp_sizes=inp_sizes,
            out_sizes=out_sizes,
            tt_ranks=full_tt_ranks,
        )
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("b", b)
        workspace.FeedBlob("cores", full_cores)
        workspace.RunOperatorOnce(full_op_tt)
        Y_full_tt = workspace.FetchBlob("Y").flatten()

        assert(len(Y_fc) == len(Y_full_tt))
        self.assertAlmostEqual(np.linalg.norm(Y_fc - Y_full_tt), 0, delta=1e-3)

        # Testing TT-decomposition with minimal ranks
        sparse_tt_ranks = [1, 1, 1, 1, 1]
        sparse_cores = tt_core.matrix_to_tt(W, inp_sizes, out_sizes,
                                            sparse_tt_ranks)

        sparse_op_tt = core.CreateOperator(
            "TT",
            ["X", "b", "cores"],
            ["Y"],
            inp_sizes=inp_sizes,
            out_sizes=out_sizes,
            tt_ranks=sparse_tt_ranks,
        )
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("b", b)
        workspace.FeedBlob("cores", sparse_cores)
        workspace.RunOperatorOnce(sparse_op_tt)
        Y_sparse_tt = workspace.FetchBlob("Y").flatten()

        assert(len(Y_fc) == len(Y_sparse_tt))
        self.assertAlmostEqual(np.linalg.norm(Y_fc - Y_sparse_tt),
                                39.974, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
