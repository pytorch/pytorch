from __future__ import absolute_import, division, print_function, unicode_literals

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPBatchPermutationOpTest(hu.HypothesisTestCase):
    @given(N=st.integers(min_value=0, max_value=100), **hu.gcs_cpu_only)
    def test_batch_permutation(self, N, gc, dc):
        N = 0
        X = np.round(np.random.rand(N, 10, 20, 3) * 255).astype(np.float32)
        indices = np.arange(N).astype(np.int32)
        np.random.shuffle(indices)

        quantize = core.CreateOperator("Quantize", ["X"], ["X_q"], engine="DNNLOWP")
        batch_perm = core.CreateOperator(
            "BatchPermutation", ["X_q", "indices"], ["Y_q"], engine="DNNLOWP"
        )

        net = core.Net("test_net")
        net.Proto().op.extend([quantize, batch_perm])

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("indices", indices)
        workspace.RunNetOnce(net)
        X_q = workspace.FetchInt8Blob("X_q").data
        Y_q = workspace.FetchInt8Blob("Y_q").data

        def batch_permutation_ref(X, indices):
            return np.array([X[i] for i in indices])

        Y_q_ref = batch_permutation_ref(X_q, indices)
        np.testing.assert_allclose(Y_q, Y_q_ref)
