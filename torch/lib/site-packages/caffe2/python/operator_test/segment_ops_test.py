from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import partial
from hypothesis import given

import numpy as np
import unittest
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TesterBase:
    def segment_reduce_op(self, data, segment_ids, reducer, indices=None):
        segments = self.split(data, segment_ids, indices)
        output = np.zeros((len(segments), ) + data.shape[1:])
        for i, segment in enumerate(segments):
            if len(segment) > 0:
                output[i] = reducer(segment)
            else:
                output[i] = 0.0
        return output

    def segment_reduce_grad_op(
        self,
        data,
        segment_ids,
        reducer_grad,
        grad_out,
        output,
        indices=None
    ):
        segments = self.split(data, segment_ids, indices)
        segment_grads = [
            reducer_grad(grad_out[i], [output[i]], [segment])
            for i, segment in enumerate(segments)
        ]
        return self.unsplit(data.shape[1:], segment_grads, segment_ids)

    def _test(self, prefix, input_strategy, refs, gpu=False, **kwargs):
        tester = self
        operator_args = kwargs.pop('operator_args', {})
        threshold = kwargs.pop('threshold', 1e-4)
        grad_check = kwargs.pop('grad_check', True)

        @given(X=input_strategy, **hu.gcs)
        def test_segment_ops(self, X, gc, dc):
            if not gpu and gc.device_type > 0:
                return
            for op_name, ref, grad_ref in refs:
                inputs = ['input%d' % i for i in range(0, len(X))]
                op = core.CreateOperator(
                    prefix + op_name, inputs, ['output'], **operator_args
                )
                print('Operator %s, ' % op.type, gc.device_type)

                def seg_reduce(data, *args):
                    indices, segments = (
                        args if len(args) == 2 else (None, args[0])
                    )
                    out = tester.segment_reduce_op(
                        data=data,
                        segment_ids=segments,
                        indices=indices,
                        reducer=ref
                    )
                    return (out, )

                def seg_reduce_grad(grad_out, outputs, inputs):
                    data = inputs[0]
                    args = inputs[1:]
                    indices, segments = (
                        args if len(args) == 2 else (None, args[0])
                    )
                    # grad r.t. data
                    grad_val = tester.segment_reduce_grad_op(
                        data, segments, grad_ref, grad_out, outputs[0], indices
                    )
                    # if sparse, include indices along with data gradient
                    data_grad_slice = (
                        (grad_val, indices) if indices is not None else grad_val
                    )
                    # other inputs don't have gradient
                    return (data_grad_slice, ) + (None, ) * (len(inputs) - 1)

                kwargs = {}
                if grad_check:
                    kwargs['output_to_grad'] = 'output'
                    kwargs['grad_reference'] = seg_reduce_grad
                self.assertReferenceChecks(
                    device_option=gc,
                    op=op,
                    inputs=X,
                    reference=seg_reduce,
                    threshold=threshold,
                    **kwargs
                )
        return test_segment_ops


class SegmentsTester(TesterBase):
    def split(self, data, segment_ids, indices=None):
        """
        Given:
          data[M1 x M2 x ... x Md]
                          the input data
          indices[N]      the index of each entry of segment_ids into data,
                          where 0 <= index[i] < M1,
                          with default indices=[0,1,...N]
          segment_ids[N]  the segment_id for each entry of indices,

        returns K outputs, each one containing data entries corresponding
        to one of the segments present in `segment_ids`.
        """
        if segment_ids.size == 0:
            return []
        K = max(segment_ids) + 1
        outputs = [
            np.zeros(
                (np.count_nonzero(segment_ids == seg_id), ) + data.shape[1:],
                dtype=data.dtype
            ) for seg_id in range(0, K)
        ]
        counts = np.zeros(K, dtype=int)
        for i, seg_id in enumerate(segment_ids):
            data_idx = i if indices is None else indices[i]
            outputs[seg_id][counts[seg_id]] = data[data_idx]
            counts[seg_id] += 1
        return outputs

    def unsplit(self, extra_shape, inputs, segment_ids):
        """ Inverse operation to `split`, with indices=None """
        output = np.zeros((len(segment_ids), ) + extra_shape)
        if len(segment_ids) == 0:
            return output
        K = max(segment_ids) + 1
        counts = np.zeros(K, dtype=int)
        for i, seg_id in enumerate(segment_ids):
            output[i] = inputs[seg_id][counts[seg_id]]
            counts[seg_id] += 1
        return output


class LengthsTester(TesterBase):
    def split(self, data, lengths, indices=None):
        K = len(lengths)
        outputs = [
            np.zeros((lengths[seg_id], ) + data.shape[1:],
                     dtype=data.dtype) for seg_id in range(0, K)
        ]
        start = 0
        for i in range(0, K):
            for j in range(0, lengths[i]):
                data_index = start + j
                if indices is not None:
                    data_index = indices[data_index]
                outputs[i][j] = data[data_index]
            start += lengths[i]
        return outputs

    def unsplit(self, extra_shape, inputs, lengths):
        N = sum(lengths)
        output = np.zeros((N, ) + extra_shape)
        K = len(lengths)
        assert len(inputs) == K
        current = 0
        for i in range(0, K):
            for j in range(0, lengths[i]):
                output[current] = inputs[i][j]
                current += 1
        return output


def sum_grad(grad_out, outputs, inputs):
    return np.repeat(
        np.expand_dims(grad_out, axis=0),
        inputs[0].shape[0],
        axis=0
    )


def logsumexp(x):
    return np.log(np.sum(np.exp(x), axis=0))


def logsumexp_grad(grad_out, outputs, inputs):
    sum_exps = np.sum(np.exp(inputs[0]), axis=0)
    return np.repeat(
        np.expand_dims(grad_out / sum_exps, 0),
        inputs[0].shape[0],
        axis=0
    ) * np.exp(inputs[0])


def logmeanexp(x):
    return np.log(np.mean(np.exp(x), axis=0))


def mean(x):
    return np.mean(x, axis=0)


def mean_grad(grad_out, outputs, inputs):
    return np.repeat(
        np.expand_dims(grad_out / inputs[0].shape[0], 0),
        inputs[0].shape[0],
        axis=0
    )


def max_fwd(x):
    return np.amax(x, axis=0)


def max_grad(grad_out, outputs, inputs):
    flat_inputs = inputs[0].flatten()
    flat_outputs = np.array(outputs[0]).flatten()
    flat_grad_in = np.zeros(flat_inputs.shape)
    flat_grad_out = np.array(grad_out).flatten()
    blocks = inputs[0].shape[0]
    if blocks == 0:
        return np.zeros(inputs[0].shape)
    block_size = flat_inputs.shape[0] // blocks

    for i in range(block_size):
        out_grad = flat_grad_out[i]
        out = flat_outputs[i]
        for j in range(blocks):
            idx = j * block_size + i
            # we can produce multiple outputs for max
            if out == flat_inputs[idx]:
                flat_grad_in[idx] = out_grad

    return np.resize(flat_grad_in, inputs[0].shape)


REFERENCES_ALL = [
    ('Sum', partial(np.sum, axis=0), sum_grad),
    ('Mean', partial(np.mean, axis=0), mean_grad),
]

REFERENCES_SORTED = [
    ('RangeSum', partial(np.sum, axis=0), sum_grad),
    ('RangeLogSumExp', logsumexp, logsumexp_grad),
    # gradient is the same as sum
    ('RangeLogMeanExp', logmeanexp, logsumexp_grad),
    ('RangeMean', mean, mean_grad),
    ('RangeMax', max_fwd, max_grad),
]

REFERENCES_LENGTHS_ONLY = [
    ('Max', partial(np.amax, axis=0), max_grad),
]


def sparse_lengths_weighted_sum_ref(D, W, I, L):
    R = np.zeros(shape=(len(L), ) + D.shape[1:], dtype=D.dtype)
    line = 0
    for g in range(len(L)):
        for _ in range(L[g]):
            if len(D.shape) > 1:
                R[g, :] += W[line] * D[I[line], :]
            else:
                R[g] += W[line] * D[I[line]]
            line += 1
    return [R]


def sparse_lengths_weighted_sum_grad_ref(
        GO, fwd_out, fwd_in, grad_on_weights=False):
    D, W, I, L = fwd_in
    GI = np.zeros(shape=(len(I), ) + D.shape[1:], dtype=D.dtype)
    GW = np.zeros(shape=W.shape, dtype=W.dtype) if grad_on_weights else None
    line = 0
    for g in range(len(L)):
        for _ in range(L[g]):
            if len(GO.shape) > 1:
                GI[line, :] = W[line] * GO[g, :]
            else:
                GI[line] = W[line] * GO[g]
            if GW is not None:
                if len(GO.shape) > 1:
                    GW[line] = np.dot(GO[g].flatten(), D[I[line], :].flatten())
                else:
                    GW[line] = np.dot(GO[g].flatten(), D[I[line]].flatten())
            line += 1
    print(GW)
    return [(GI, I), GW, None, None]


class TestSegmentOps(hu.HypothesisTestCase):
    def test_sorted_segment_ops(self):
        SegmentsTester()._test(
            'SortedSegment',
            hu.segmented_tensor(
                dtype=np.float32,
                is_sorted=True,
                allow_empty=True
            ),
            REFERENCES_ALL + REFERENCES_SORTED
        )(self)

    def test_unsorted_segment_ops(self):
        SegmentsTester()._test(
            'UnsortedSegment',
            hu.segmented_tensor(
                dtype=np.float32,
                is_sorted=False,
                allow_empty=True
            ),
            REFERENCES_ALL,
        )(self)

    def test_unsorted_segment_ops_gpu(self):
        SegmentsTester()._test(
            'UnsortedSegment',
            hu.segmented_tensor(
                dtype=np.float32,
                is_sorted=False,
                allow_empty=True,
            ),
            REFERENCES_ALL,
            gpu=workspace.has_gpu_support,
            grad_check=False,
        )(self)

    def test_sparse_sorted_segment_ops(self):
        SegmentsTester()._test(
            'SparseSortedSegment',
            hu.sparse_segmented_tensor(
                dtype=np.float32,
                is_sorted=True,
                allow_empty=True
            ),
            REFERENCES_ALL
        )(self)

    def test_sparse_unsorted_segment_ops(self):
        SegmentsTester()._test(
            'SparseUnsortedSegment',
            hu.sparse_segmented_tensor(
                dtype=np.float32,
                is_sorted=False,
                allow_empty=True
            ),
            REFERENCES_ALL
        )(self)

    def test_lengths_ops(self):
        LengthsTester()._test(
            'Lengths',
            hu.lengths_tensor(
                dtype=np.float32,
                min_value=1,
                max_value=5,
                allow_empty=True
            ),
            REFERENCES_ALL + REFERENCES_LENGTHS_ONLY,
        )(self)

    def test_sparse_lengths_ops(self):
        for itype in [np.int32, np.int64]:
            LengthsTester()._test(
                'SparseLengths',
                hu.sparse_lengths_tensor(
                    dtype=np.float32,
                    min_value=1,
                    max_value=5,
                    allow_empty=True,
                    itype=itype,
                ),
                REFERENCES_ALL,
            )(self)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(**hu.gcs)
    def test_unsorted_sums_large(self, gc, dc):
        X = np.random.rand(10000, 32, 12).astype(np.float32)
        segments = np.random.randint(0, 10000, size=10000).astype(np.int32)
        op = core.CreateOperator("UnsortedSegmentSum", ["X", "segments"], "out")
        self.assertDeviceChecks(dc, op, [X, segments], [0])

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(**hu.gcs)
    def test_sorted_segment_range_mean(self, gc, dc):
        X = np.random.rand(6, 32, 12).astype(np.float32)
        segments = np.array([0, 0, 1, 1, 2, 3]).astype(np.int32)
        op = core.CreateOperator(
            "SortedSegmentRangeMean",
            ["X", "segments"],
            "out"
        )
        self.assertDeviceChecks(dc, op, [X, segments], [0])
        self.assertGradientChecks(gc, op, [X, segments], 0, [0])

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(**hu.gcs)
    def test_sorted_segment_range_log_mean_exp(self, gc, dc):
        X = np.random.rand(7, 32, 12).astype(np.float32)
        segments = np.array([0, 0, 1, 1, 2, 2, 3]).astype(np.int32)
        op = core.CreateOperator(
            "SortedSegmentRangeLogMeanExp",
            ["X", "segments"],
            "out"
        )
        self.assertDeviceChecks(dc, op, [X, segments], [0])
        self.assertGradientChecks(gc, op, [X, segments], 0, [0])

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(**hu.gcs)
    def test_unsorted_means_large(self, gc, dc):
        X = np.random.rand(10000, 31, 19).astype(np.float32)
        segments = np.random.randint(0, 10000, size=10000).astype(np.int32)
        op = core.CreateOperator("UnsortedSegmentMean", ["X", "segments"], "out")
        self.assertDeviceChecks(dc, op, [X, segments], [0])

    @serial.given(
        inputs=hu.lengths_tensor(
            dtype=np.float32,
            min_value=1,
            max_value=5,
            allow_empty=True,
        ),
        **hu.gcs
    )
    def test_lengths_sum(self, inputs, gc, dc):
        X, Y = inputs
        op = core.CreateOperator("LengthsSum", ["X", "Y"], "out")

        def ref(D, L):
            R = np.zeros(shape=(L.size, ) + D.shape[1:], dtype=D.dtype)
            line = 0
            for g in range(L.size):
                for _ in range(L[g]):
                    if len(D.shape) > 1:
                        R[g, :] += D[line, :]
                    else:
                        R[g] += D[line]
                    line += 1
            return [R]

        self.assertReferenceChecks(gc, op, [X, Y], ref)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])

    @serial.given(
        inputs=hu.sparse_lengths_tensor(
            dtype=np.float32,
            min_value=1,
            max_value=5,
            allow_empty=True
        ),
        **hu.gcs
    )
    def test_sparse_lengths_sum(self, inputs, gc, dc):
        X, Y, Z = inputs
        op = core.CreateOperator("SparseLengthsSum", ["X", "Y", "Z"], "out")

        def ref(D, I, L):
            R = np.zeros(shape=(L.size, ) + D.shape[1:], dtype=D.dtype)
            line = 0
            for g in range(L.size):
                for _ in range(L[g]):
                    if len(D.shape) > 1:
                        R[g, :] += D[I[line], :]
                    else:
                        R[g] += D[I[line]]
                    line += 1
            return [R]

        self.assertReferenceChecks(gc, op, [X, Y, Z], ref)
        self.assertDeviceChecks(dc, op, [X, Y, Z], [0])
        self.assertGradientChecks(gc, op, [X, Y, Z], 0, [0])

    @serial.given(
        inputs=hu.lengths_tensor(
            dtype=np.float32,
            min_value=1,
            max_value=5,
            allow_empty=True,
        ),
        **hu.gcs
    )
    def test_lengths_mean(self, inputs, gc, dc):
        X, Y = inputs
        op = core.CreateOperator("LengthsMean", ["X", "Y"], "out")

        def ref(D, L):
            R = np.zeros(shape=(L.size, ) + D.shape[1:], dtype=D.dtype)
            line = 0
            for g in range(L.size):
                for _ in range(L[g]):
                    if len(D.shape) > 1:
                        R[g, :] += D[line, :]
                    else:
                        R[g] += D[line]
                    line += 1
                if L[g] > 1:
                    if len(D.shape) > 1:
                        R[g, :] = R[g, :] / L[g]
                    else:
                        R[g] = R[g] / L[g]

            return [R]

        self.assertReferenceChecks(gc, op, [X, Y], ref)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])

    @serial.given(
        inputs=hu.sparse_lengths_tensor(
            dtype=np.float32,
            min_value=1,
            max_value=5,
            allow_empty=True
        ),
        **hu.gcs
    )
    def test_sparse_lengths_mean(self, inputs, gc, dc):
        X, Y, Z = inputs
        op = core.CreateOperator("SparseLengthsMean", ["X", "Y", "Z"], "out")

        def ref(D, I, L):
            R = np.zeros(shape=(L.size, ) + D.shape[1:], dtype=D.dtype)
            line = 0
            for g in range(L.size):
                for _ in range(L[g]):
                    if len(D.shape) > 1:
                        R[g, :] += D[I[line], :]
                    else:
                        R[g] += D[I[line]]
                    line += 1

                if L[g] > 1:
                    if len(D.shape) > 1:
                        R[g, :] = R[g, :] / L[g]
                    else:
                        R[g] = R[g] / L[g]

            return [R]

        self.assertReferenceChecks(gc, op, [X, Y, Z], ref)
        self.assertDeviceChecks(dc, op, [X, Y, Z], [0])
        self.assertGradientChecks(gc, op, [X, Y, Z], 0, [0])

    @serial.given(
        grad_on_weights=st.booleans(),
        inputs=hu.sparse_lengths_tensor(
            dtype=np.float32,
            min_value=1,
            max_value=5,
            allow_empty=True
        ),
        seed=st.integers(min_value=0, max_value=100),
        **hu.gcs
    )
    def test_sparse_lengths_weighted_sum(
            self, grad_on_weights, inputs, seed, gc, dc):
        D, I, L = inputs

        np.random.seed(int(seed))

        W = np.random.rand(I.size).astype(np.float32)
        op = core.CreateOperator(
            "SparseLengthsWeightedSum",
            ["D", "W", "I", "L"],
            "out",
            grad_on_weights=grad_on_weights)
        self.assertDeviceChecks(dc, op, [D, W, I, L], [0])
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[D, W, I, L],
            reference=sparse_lengths_weighted_sum_ref,
            threshold=1e-4,
            output_to_grad='out',
            grad_reference=partial(
                sparse_lengths_weighted_sum_grad_ref,
                grad_on_weights=grad_on_weights),
        )
        self.assertGradientChecks(gc, op, [D, W, I, L], 0, [0])
        if grad_on_weights:
            self.assertGradientChecks(gc, op, [D, W, I, L], 1, [0])

    @given(**hu.gcs)
    def test_sparse_lengths_indices_in_gradient_sum_gpu(self, gc, dc):
        X = np.random.rand(3, 3, 4, 5).astype(np.float32)
        Y = np.asarray([3, 3, 2]).astype(np.int32)
        Z = np.random.randint(0, 50, size=8).astype(np.int64)
        op = core.CreateOperator(
            "SparseLengthsIndicesInGradientSumGradient", ["X", "Y", "Z"], "out"
        )
        self.assertDeviceChecks(dc, op, [X, Y, Z], [0])

    @given(**hu.gcs)
    def test_sparse_lengths_indices_in_gradient_mean_gpu(self, gc, dc):
        X = np.random.rand(3, 3, 4, 5).astype(np.float32)
        Y = np.asarray([3, 3, 2]).astype(np.int32)
        Z = np.random.randint(0, 50, size=8).astype(np.int64)
        op = core.CreateOperator(
            "SparseLengthsIndicesInGradientMeanGradient", ["X", "Y", "Z"], "out"
        )
        self.assertDeviceChecks(dc, op, [X, Y, Z], [0])

    @given(**hu.gcs_cpu_only)
    def test_legacy_sparse_and_lengths_sum_gradient(self, gc, dc):
        X = np.random.rand(3, 64).astype(np.float32)
        Y = np.asarray([20, 20, 10]).astype(np.int32)
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("Y", Y)
        test_net = core.Net("test_net")
        test_net.SparseLengthsSumGradient(["X", "Y"], "out1")
        test_net.LengthsSumGradient(["X", "Y"], "out2")
        workspace.RunNetOnce(test_net)
        out1 = workspace.FetchBlob("out1")
        out2 = workspace.FetchBlob("out2")
        self.assertTrue((out1 == out2).all())

    @given(**hu.gcs)
    def test_sparse_lengths_sum_invalid_index(self, gc, dc):
        D = np.random.rand(50, 3, 4, 5).astype(np.float32)
        I = (np.random.randint(0, 10000, size=10) + 10000).astype(np.int64)
        L = np.asarray([4, 4, 2]).astype(np.int32)
        op = core.CreateOperator(
            "SparseLengthsSum",
            ["D", "I", "L"],
            "out")
        workspace.FeedBlob('D', D)
        workspace.FeedBlob('I', I)
        workspace.FeedBlob('L', L)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

    @serial.given(**hu.gcs_cpu_only)
    def test_sparse_lengths_positional_weighted_sum(
            self, gc, dc):
        D = np.random.rand(50, 3, 4, 5).astype(np.float32)
        W = np.random.rand(50).astype(np.float32)
        indices = np.random.randint(0, 50, size=10).astype(np.int64)
        L = np.asarray([4, 4, 2]).astype(np.int32)
        op = core.CreateOperator(
            "SparseLengthsPositionalWeightedSum",
            ["D", "W", "indices", "L"],
            "out")

        def ref_sparse(D, W, indices, L):
            workspace.FeedBlob("L", L)
            lengths_range_fill_op = core.CreateOperator(
                "LengthsRangeFill", ["L"], ["L_pos_seq"])
            workspace.RunOperatorOnce(lengths_range_fill_op)

            workspace.FeedBlob("W", W)
            gather_op = core.CreateOperator(
                "Gather", ["W", "L_pos_seq"], ["W_gathered"])
            workspace.RunOperatorOnce(gather_op)

            workspace.FeedBlob("D", D)
            workspace.FeedBlob("indices", indices)
            sparse_op = core.CreateOperator(
                "SparseLengthsWeightedSum",
                ["D", "W_gathered", "indices", "L"],
                "out_ref")
            workspace.RunOperatorOnce(sparse_op)

            return (workspace.FetchBlob("out_ref"),)

        self.assertReferenceChecks(
            gc, op, [D, W, indices, L], ref_sparse)

   # @given(
   #     inputs=hu.lengths_tensor(
   #         dtype=np.float32,
   #         min_value=1,
   #         max_value=5,
   #         min_dim=1,
   #         max_dim=1,
   #         allow_empty=False,
   #     ),
   #     **hu.gcs
   # )
   # def test_lengths_max_gpu(self, inputs, gc, dc):
   #     def lengths_max_ref(I, L):
   #         R = np.zeros(shape=(len(L)), dtype=I.dtype)
   #         line = 0
   #         for g in range(len(L)):
   #             for i in range(L[g]):
   #                 if i == 0:
   #                     R[g] = I[line]
   #                 else:
   #                     R[g] = max(R[g], I[line])
   #                 line += 1
   #         return [R]

   #     X, lengths = inputs
   #     op = core.CreateOperator("LengthsMax", ["X", "lengths"], "out")
   #     self.assertDeviceChecks(dc, op, [X, lengths], [0])
   #     self.assertReferenceChecks(
   #         device_option=gc,
   #         op=op,
   #         inputs=[X, lengths],
   #         reference=lengths_max_ref,
   #         threshold=1e-4,
   #         output_to_grad='out',
   #     )


if __name__ == "__main__":
    import unittest
    unittest.main()
