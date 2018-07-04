from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import unittest


class TestSpecializedSegmentOps(hu.HypothesisTestCase):

    @given(batchsize=st.integers(1, 20),
           fptype=st.sampled_from([np.float16, np.float32]),
           fp16asint=st.booleans(),
           blocksize=st.sampled_from([8, 17, 32, 64, 85, 96, 128, 163]),
           normalize_by_lengths=st.booleans(), **hu.gcs)
    def test_sparse_lengths_sum_cpu(
            self, batchsize, fptype, fp16asint, blocksize, normalize_by_lengths, gc, dc):

        if normalize_by_lengths == False:
            print("<test_sparse_lengths_sum_cpu>")
        else:
            print("<test_sparse_lengths_sum_mean_cpu>")

        tblsize = 300
        if fptype == np.float32:
            Tbl = np.random.rand(tblsize, blocksize).astype(np.float32)
            atol = 1e-5
        else:
            if fp16asint:
                Tbl = (10.0 * np.random.rand(tblsize, blocksize)
                       ).round().astype(np.float16)
                atol = 1e-3
            else:
                Tbl = np.random.rand(tblsize, blocksize).astype(np.float16)
                atol = 1e-1

        # array of each row length
        Lengths = np.random.randint(1, 30, size=batchsize).astype(np.int32)
        # flat indices
        Indices = np.random.randint(
            0, tblsize, size=sum(Lengths)).astype(np.int64)

        if normalize_by_lengths == False:
            op = core.CreateOperator("SparseLengthsSum", [
                                     "Tbl", "Indices", "Lengths"], "out")
        else:
            op = core.CreateOperator("SparseLengthsMean", [
                                     "Tbl", "Indices", "Lengths"], "out")

        self.ws.create_blob("Tbl").feed(Tbl)
        self.ws.create_blob("Indices").feed(Indices)
        self.ws.create_blob("Lengths").feed(Lengths)
        self.ws.run(op)

        def sparse_lengths_sum_ref(Tbl, Indices, Lengths):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            if normalize_by_lengths == False:
                for i in range(0, len(rptr[0:-1])):
                    out[i] = Tbl[Indices[rptr[i]:rptr[i + 1]]].sum(axis=0)
            else:
                for i in range(0, len(rptr[0:-1])):
                    out[i] = Tbl[Indices[rptr[i]:rptr[i + 1]]
                                 ].sum(axis=0) * 1.0 / float(Lengths[i])

            return out

        np.testing.assert_allclose(self.ws.blobs[("out")].fetch(),
                                   sparse_lengths_sum_ref(Tbl, Indices, Lengths), rtol=1e-3, atol=atol)

    @given(batchsize=st.integers(1, 20),
           fptype=st.sampled_from([np.float16, np.float32]),
           fp16asint=st.booleans(),
           blocksize=st.sampled_from([8, 17, 32, 64, 85, 96, 128, 163]),
           **hu.gcs)
    def test_sparse_lengths_weightedsum_cpu(
            self, batchsize, fptype, fp16asint, blocksize, gc, dc):

        print("<test_sparse_lengths_weightedsum_cpu>")

        tblsize = 300
        if fptype == np.float32:
            Tbl = np.random.rand(tblsize, blocksize).astype(np.float32)
            atol = 1e-5
        else:
            if fp16asint:
                Tbl = (10.0 * np.random.rand(tblsize, blocksize)
                       ).round().astype(np.float16)
                atol = 1e-3
            else:
                Tbl = np.random.rand(tblsize, blocksize).astype(np.float16)
                atol = 1e-1

        # array of each row length
        Lengths = np.random.randint(1, 30, size=batchsize).astype(np.int32)
        # flat indices
        Indices = np.random.randint(
            0, tblsize, size=sum(Lengths)).astype(np.int64)
        Weights = np.random.rand(sum(Lengths)).astype(np.float32)

        op = core.CreateOperator("SparseLengthsWeightedSum", [
                                 "Tbl", "Weights", "Indices", "Lengths"], "out")

        self.ws.create_blob("Tbl").feed(Tbl)
        self.ws.create_blob("Indices").feed(Indices)
        self.ws.create_blob("Lengths").feed(Lengths)
        self.ws.create_blob("Weights").feed(Weights)
        self.ws.run(op)

        def sparse_lengths_weightedsum_ref(Tbl, Weights, Indices, Lengths):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            for i in range(0, len(rptr[0:-1])):
                w = Weights[rptr[i]:rptr[i + 1]]
                out[i] = (Tbl[Indices[rptr[i]:rptr[i + 1]]]
                          * w[:, np.newaxis]).sum(axis=0)
            return out

        np.testing.assert_allclose(self.ws.blobs[("out")].fetch(),
                                   sparse_lengths_weightedsum_ref(Tbl, Weights, Indices, Lengths), rtol=1e-3, atol=atol)


    @given(batchsize=st.integers(1, 20),
           blocksize=st.sampled_from([8, 16, 17, 26, 32, 64, 85, 96, 128, 148, 163]),
           normalize_by_lengths=st.booleans(), **hu.gcs)
    def test_sparse_lengths_weightedsum_8BitsRowwiseOp_cpu(
            self, batchsize, blocksize, normalize_by_lengths, gc, dc):

        if normalize_by_lengths == False:
            print("<test_sparse_lengths_weightedsum_SparseLengthsWeightedSum8BitsRowwise_cpu>")
        else:
            print("<test_sparse_lengths_weightedsum_SparseLengthsWeightedMean8BitsRowwise_cpu>")

        tblsize = 300
        Tbl = np.random.randint(7, size = (tblsize, blocksize)).astype(np.uint8)
        atol = 1e-5

        # array of each row length
        Lengths = np.random.randint(1, 30, size=batchsize).astype(np.int32)
        # flat indices
        Indices = np.random.randint(
            0, tblsize, size=sum(Lengths)).astype(np.int64)
        Weights = np.random.rand(sum(Lengths)).astype(np.float32)
        Scale_Bias = np.random.rand(tblsize, 2).astype(np.float32)

        if normalize_by_lengths == False:
            op = core.CreateOperator("SparseLengthsWeightedSum8BitsRowwise", [
                                     "Tbl", "Weights", "Indices", "Lengths", "Scale_Bias"], "out")
        else:
            op = core.CreateOperator("SparseLengthsWeightedMean8BitsRowwise", [
                                     "Tbl", "Weights", "Indices", "Lengths", "Scale_Bias"], "out")

        self.ws.create_blob("Tbl").feed(Tbl)
        self.ws.create_blob("Weights").feed(Weights)
        self.ws.create_blob("Indices").feed(Indices)
        self.ws.create_blob("Lengths").feed(Lengths)
        self.ws.create_blob("Scale_Bias").feed(Scale_Bias)
        self.ws.run(op)

        def sparse_lengths_weightedsum_8BitsRowwiseOp_cpu_ref(Tbl, Weights, Indices, Lengths, Scale_Bias):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            for i in range(0, len(rptr[0:-1])):
                w = Weights[rptr[i]:rptr[i + 1]]
                s = Scale_Bias[Indices[rptr[i]:rptr[i + 1]], 0][:, np.newaxis]
                b = Scale_Bias[Indices[rptr[i]:rptr[i + 1]], 1][:, np.newaxis]
                f = 1.0
                if normalize_by_lengths == True:
                    f = 1.0 / float(Lengths[i])
                out[i] = (w[:, np.newaxis] *
                          (s * Tbl[Indices[rptr[i]:rptr[i + 1]]] + b)).sum(axis=0) * f
            return out

        np.testing.assert_allclose(self.ws.blobs[("out")].fetch(),
                                   sparse_lengths_weightedsum_8BitsRowwiseOp_cpu_ref(Tbl, Weights, Indices, Lengths, Scale_Bias),
                                   rtol=1e-3, atol=atol)



    @given(batchsize=st.integers(1, 20),
           blocksize=st.sampled_from([8, 16, 17, 26, 32, 64, 85, 96, 128, 148, 163]),
           normalize_by_lengths=st.booleans(), **hu.gcs)
    def test_sparse_lengths_sum_8BitsRowwiseOp_cpu(
            self, batchsize, blocksize, normalize_by_lengths, gc, dc):

        if normalize_by_lengths == False:
            print("<test_sparse_lengths_sum_SparseLengthsSum8BitsRowwise_cpu>")
        else:
            print("<test_sparse_lengths_sum_SparseLengthsMean8BitsRowwise_cpu>")

        tblsize = 300
        Tbl = np.random.randint(7, size = (tblsize, blocksize)).astype(np.uint8)
        atol = 1e-5

        # array of each row length
        Lengths = np.random.randint(1, 30, size=batchsize).astype(np.int32)
        # flat indices
        Indices = np.random.randint(
            0, tblsize, size=sum(Lengths)).astype(np.int64)
        Scale_Bias = np.random.rand(tblsize, 2).astype(np.float32)

        if normalize_by_lengths == False:
            op = core.CreateOperator("SparseLengthsSum8BitsRowwise", [
                                     "Tbl", "Indices", "Lengths", "Scale_Bias"], "out")
        else:
            op = core.CreateOperator("SparseLengthsMean8BitsRowwise", [
                                     "Tbl", "Indices", "Lengths", "Scale_Bias"], "out")

        self.ws.create_blob("Tbl").feed(Tbl)
        self.ws.create_blob("Indices").feed(Indices)
        self.ws.create_blob("Lengths").feed(Lengths)
        self.ws.create_blob("Scale_Bias").feed(Scale_Bias)
        self.ws.run(op)

        def sparse_lengths_sum_8BitsRowwiseOp_cpu_reg(Tbl, Indices, Lengths, Scale_Bias):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            for i in range(0, len(rptr[0:-1])):
                s = Scale_Bias[Indices[rptr[i]:rptr[i + 1]], 0][:, np.newaxis]
                b = Scale_Bias[Indices[rptr[i]:rptr[i + 1]], 1][:, np.newaxis]
                f = 1.0
                if normalize_by_lengths == True:
                    f = 1.0 / float(Lengths[i])
                out[i] = (s * Tbl[Indices[rptr[i]:rptr[i + 1]]] + b).sum(axis=0) * f
            return out

        np.testing.assert_allclose(self.ws.blobs[("out")].fetch(),
                                   sparse_lengths_sum_8BitsRowwiseOp_cpu_reg(Tbl, Indices, Lengths, Scale_Bias),
                                   rtol=1e-3, atol=atol)


    @given(batchsize=st.integers(1, 20),
           blocksize=st.sampled_from([8, 16, 17, 26, 32, 64, 85, 96, 128, 148, 163]),
           normalize_by_lengths=st.booleans(), **hu.gcs)
    def test_sparse_lengths_sum_8BitsRowwiseOp_cpu_invalid_index(
            self, batchsize, blocksize, normalize_by_lengths, gc, dc):

        tblsize = 300
        Tbl = np.random.randint(7, size = (tblsize, blocksize)).astype(np.uint8)

        # array of each row length
        Lengths = np.random.randint(1, 30, size=batchsize).astype(np.int32)
        # flat indices
        Indices = np.random.randint(
            0, tblsize, size=sum(Lengths)).astype(np.int64)
        Indices[0] += 1000
        Scale_Bias = np.random.rand(tblsize, 2).astype(np.float32)

        if normalize_by_lengths == False:
            op = core.CreateOperator("SparseLengthsSum8BitsRowwise", [
                                     "Tbl", "Indices", "Lengths", "Scale_Bias"], "out")
        else:
            op = core.CreateOperator("SparseLengthsMean8BitsRowwise", [
                                     "Tbl", "Indices", "Lengths", "Scale_Bias"], "out")

        self.ws.create_blob("Tbl").feed(Tbl)
        self.ws.create_blob("Indices").feed(Indices)
        self.ws.create_blob("Lengths").feed(Lengths)
        self.ws.create_blob("Scale_Bias").feed(Scale_Bias)
        with self.assertRaises(RuntimeError):
            self.ws.run(op)


if __name__ == "__main__":
    unittest.main()
