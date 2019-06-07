from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hip_test_util as hiputl
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, assume


class TestSpecializedSegmentOps(hu.HypothesisTestCase):
    @given(
        batchsize=st.integers(1, 20),
        fptype=st.sampled_from([np.float16, np.float32]),
        fp16asint=st.booleans(),
        blocksize=st.sampled_from([8, 16, 32, 64, 85, 96, 128, 163]),
        normalize_by_lengths=st.booleans(),
        empty_indices=st.booleans(),
        **hu.gcs
    )
    def test_sparse_lengths_sum_cpu(
        self,
        batchsize,
        fptype,
        fp16asint,
        blocksize,
        normalize_by_lengths,
        empty_indices,
        gc,
        dc,
    ):
        if fptype != np.float32:
            assume(gc.device_type == caffe2_pb2.CPU)
            assume(not hiputl.run_in_hip(gc, dc))
            assume(caffe2_pb2.CUDA not in {d.device_type for d in dc})

        if normalize_by_lengths:
            print("<test_sparse_lengths_sum_mean_cpu>")
        else:
            print("<test_sparse_lengths_sum_cpu>")

        tblsize = 300
        if fptype == np.float32:
            Tbl = np.random.rand(tblsize, blocksize).astype(np.float32)
            atol = 1e-5
        else:
            if fp16asint:
                Tbl = (
                    (10.0 * np.random.rand(tblsize, blocksize))
                    .round()
                    .astype(np.float16)
                )
                atol = 1e-3
            else:
                Tbl = np.random.rand(tblsize, blocksize).astype(np.float16)
                atol = 1e-1

        # array of each row length
        if empty_indices:
            Lengths = np.zeros(batchsize, dtype=np.int32)
        else:
            Lengths = np.random.randint(1, 30, size=batchsize, dtype=np.int32)
        # flat indices
        Indices = np.random.randint(0, tblsize, size=sum(Lengths), dtype=np.int64)

        op = core.CreateOperator(
            "SparseLengths" + ("Mean" if normalize_by_lengths else "Sum"),
            ["Tbl", "Indices", "Lengths"],
            "out",
        )

        def sparse_lengths_sum_ref(Tbl, Indices, Lengths):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            if normalize_by_lengths:
                for i in range(0, len(rptr[0:-1])):
                    if Lengths[i] != 0:
                        out[i] = (
                            Tbl[Indices[rptr[i] : rptr[i + 1]]].sum(axis=0)
                            * 1.0
                            / float(Lengths[i])
                        )
            else:
                for i in range(0, len(rptr[0:-1])):
                    out[i] = Tbl[Indices[rptr[i] : rptr[i + 1]]].sum(axis=0)

            return [out.astype(np.float32)]

        self.assertReferenceChecks(
            gc,
            op,
            [Tbl, Indices, Lengths],
            sparse_lengths_sum_ref,
            threshold=1e-3,
            atol=atol,
        )

    @given(
        batchsize=st.integers(1, 20),
        fptype=st.sampled_from([np.float16, np.float32]),
        fp16asint=st.booleans(),
        blocksize=st.sampled_from([8, 16, 32, 64, 85, 96, 128, 163]),
        empty_indices=st.booleans(),
        **hu.gcs
    )
    def test_sparse_lengths_weightedsum_cpu(
        self, batchsize, fptype, fp16asint, blocksize, empty_indices, gc, dc
    ):
        if fptype != np.float32:
            assume(gc.device_type == caffe2_pb2.CPU)
            assume(not hiputl.run_in_hip(gc, dc))
            assume(caffe2_pb2.CUDA not in {d.device_type for d in dc})

        print("<test_sparse_lengths_weightedsum_cpu>")

        tblsize = 300
        if fptype == np.float32:
            Tbl = np.random.rand(tblsize, blocksize).astype(np.float32)
            atol = 1e-5
        else:
            if fp16asint:
                Tbl = (
                    (10.0 * np.random.rand(tblsize, blocksize))
                    .round()
                    .astype(np.float16)
                )
                atol = 1e-3
            else:
                Tbl = np.random.rand(tblsize, blocksize).astype(np.float16)
                atol = 1e-1

        # array of each row length
        if empty_indices:
            Lengths = np.zeros(batchsize, dtype=np.int32)
        else:
            Lengths = np.random.randint(1, 30, size=batchsize, dtype=np.int32)
        # flat indices
        Indices = np.random.randint(0, tblsize, size=sum(Lengths), dtype=np.int64)
        Weights = np.random.rand(sum(Lengths)).astype(np.float32)

        op = core.CreateOperator(
            "SparseLengthsWeightedSum", ["Tbl", "Weights", "Indices", "Lengths"], "out"
        )

        def sparse_lengths_weightedsum_ref(Tbl, Weights, Indices, Lengths):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            for i in range(0, len(rptr[0:-1])):
                w = Weights[rptr[i] : rptr[i + 1]]
                out[i] = (Tbl[Indices[rptr[i] : rptr[i + 1]]] * w[:, np.newaxis]).sum(
                    axis=0
                )
            return [out.astype(np.float32)]

        self.assertReferenceChecks(
            gc,
            op,
            [Tbl, Weights, Indices, Lengths],
            sparse_lengths_weightedsum_ref,
            threshold=1e-3,
            atol=atol,
        )

    @given(
        batchsize=st.integers(1, 20),
        blocksize=st.sampled_from([8, 16, 17, 26, 32, 64, 85, 96, 128, 148, 163]),
        normalize_by_lengths=st.booleans(),
        empty_indices=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_sparse_lengths_weightedsum_8BitsRowwiseOp_cpu(
        self, batchsize, blocksize, normalize_by_lengths, empty_indices, gc, dc
    ):
        if normalize_by_lengths:
            print(
                "<test_sparse_lengths_weightedsum_SparseLengthsWeightedMean8BitsRowwise_cpu>"
            )
        else:
            print(
                "<test_sparse_lengths_weightedsum_SparseLengthsWeightedSum8BitsRowwise_cpu>"
            )

        tblsize = 300
        Tbl = np.random.randint(7, size=(tblsize, blocksize), dtype=np.uint8)
        atol = 1e-5

        # array of each row length
        if empty_indices:
            Lengths = np.zeros(batchsize, dtype=np.int32)
        else:
            Lengths = np.random.randint(1, 30, size=batchsize, dtype=np.int32)
        # flat indices
        Indices = np.random.randint(0, tblsize, size=sum(Lengths), dtype=np.int64)
        Weights = np.random.rand(sum(Lengths)).astype(np.float32)
        Scale_Bias = np.random.rand(tblsize, 2).astype(np.float32)

        op = core.CreateOperator(
            "SparseLengthsWeighted"
            + ("Mean" if normalize_by_lengths else "Sum")
            + "8BitsRowwise",
            ["Tbl", "Weights", "Indices", "Lengths", "Scale_Bias"],
            "out",
        )

        def sparse_lengths_weightedsum_8BitsRowwiseOp_cpu_ref(
            Tbl, Weights, Indices, Lengths, Scale_Bias
        ):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            for i in range(0, len(rptr[0:-1])):
                w = Weights[rptr[i] : rptr[i + 1]]
                s = Scale_Bias[Indices[rptr[i] : rptr[i + 1]], 0][:, np.newaxis]
                b = Scale_Bias[Indices[rptr[i] : rptr[i + 1]], 1][:, np.newaxis]
                f = 1.0
                if normalize_by_lengths and Lengths[i] != 0:
                    f = 1.0 / float(Lengths[i])
                out[i] = (
                    w[:, np.newaxis] * (s * Tbl[Indices[rptr[i] : rptr[i + 1]]] + b)
                ).sum(axis=0) * f
            return [out.astype(np.float32)]

        self.assertReferenceChecks(
            gc,
            op,
            [Tbl, Weights, Indices, Lengths, Scale_Bias],
            sparse_lengths_weightedsum_8BitsRowwiseOp_cpu_ref,
            threshold=1e-3,
            atol=atol,
        )

    @given(
        batchsize=st.integers(1, 20),
        blocksize=st.sampled_from([8, 16, 17, 26, 32, 64, 85, 96, 128, 148, 163]),
        normalize_by_lengths=st.booleans(),
        empty_indices=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_sparse_lengths_sum_8BitsRowwiseOp_cpu(
        self, batchsize, blocksize, normalize_by_lengths, empty_indices, gc, dc
    ):
        if normalize_by_lengths:
            print("<test_sparse_lengths_sum_SparseLengthsMean8BitsRowwise_cpu>")
        else:
            print("<test_sparse_lengths_sum_SparseLengthsSum8BitsRowwise_cpu>")

        tblsize = 300
        Tbl = np.random.randint(7, size=(tblsize, blocksize), dtype=np.uint8)
        atol = 1e-5

        # array of each row length
        if empty_indices:
            Lengths = np.zeros(batchsize, dtype=np.int32)
        else:
            Lengths = np.random.randint(1, 30, size=batchsize, dtype=np.int32)
        # flat indices
        Indices = np.random.randint(0, tblsize, size=sum(Lengths), dtype=np.int64)
        Scale_Bias = np.random.rand(tblsize, 2).astype(np.float32)

        op = core.CreateOperator(
            "SparseLengths"
            + ("Mean" if normalize_by_lengths else "Sum")
            + "8BitsRowwise",
            ["Tbl", "Indices", "Lengths", "Scale_Bias"],
            "out",
        )

        def sparse_lengths_sum_8BitsRowwiseOp_cpu_reg(
            Tbl, Indices, Lengths, Scale_Bias
        ):
            rptr = np.cumsum(np.insert(Lengths, [0], [0]))
            out = np.zeros((len(Lengths), blocksize))
            for i in range(0, len(rptr[0:-1])):
                s = Scale_Bias[Indices[rptr[i] : rptr[i + 1]], 0][:, np.newaxis]
                b = Scale_Bias[Indices[rptr[i] : rptr[i + 1]], 1][:, np.newaxis]
                f = 1.0
                if normalize_by_lengths and Lengths[i] != 0:
                    f = 1.0 / float(Lengths[i])
                out[i] = (s * Tbl[Indices[rptr[i] : rptr[i + 1]]] + b).sum(axis=0) * f
            return [out.astype(np.float32)]

        self.assertReferenceChecks(
            gc,
            op,
            [Tbl, Indices, Lengths, Scale_Bias],
            sparse_lengths_sum_8BitsRowwiseOp_cpu_reg,
            threshold=1e-3,
            atol=atol,
        )

    @given(
        batchsize=st.integers(1, 20),
        blocksize=st.sampled_from([8, 16, 17, 26, 32, 64, 85, 96, 128, 148, 163]),
        normalize_by_lengths=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_sparse_lengths_sum_8BitsRowwiseOp_cpu_invalid_index(
        self, batchsize, blocksize, normalize_by_lengths, gc, dc
    ):

        tblsize = 300
        Tbl = np.random.randint(7, size=(tblsize, blocksize), dtype=np.uint8)

        # array of each row length
        Lengths = np.random.randint(1, 30, size=batchsize, dtype=np.int32)
        # flat indices
        Indices = np.random.randint(0, tblsize, size=sum(Lengths), dtype=np.int64)
        Indices[0] += 1000
        Scale_Bias = np.random.rand(tblsize, 2).astype(np.float32)

        op = core.CreateOperator(
            "SparseLengths"
            + ("Mean" if normalize_by_lengths else "Sum")
            + "8BitsRowwise",
            ["Tbl", "Indices", "Lengths", "Scale_Bias"],
            "out",
        )

        self.ws.create_blob("Tbl").feed(Tbl)
        self.ws.create_blob("Indices").feed(Indices)
        self.ws.create_blob("Lengths").feed(Lengths)
        self.ws.create_blob("Scale_Bias").feed(Scale_Bias)
        with self.assertRaises(RuntimeError):
            self.ws.run(op)


if __name__ == "__main__":
    unittest.main()
