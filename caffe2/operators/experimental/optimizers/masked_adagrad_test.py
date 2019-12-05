from __future__ import absolute_import, division, print_function

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary(
    "//caffe2/caffe2/operators/experimental/optimizers:sgd_masked_ops"
)


# Create a mask for magnitude-based block-structured pruning of param with
# prune_ratio. Block sparsity structure is applied per row not across rows.
def _get_mask(param, block_size, prune_ratio):
    num_rows = param.shape[0]
    row_size = int(np.prod(param.shape[1:]))
    num_blocks_per_row = (row_size + block_size - 1) // block_size

    # Pad and reshape to have block structure
    row_size_padded = (row_size + block_size - 1) // block_size * block_size
    param_padded = np.pad(
        param.reshape(num_rows, -1),
        ((0, 0), (0, row_size_padded - row_size)),
        mode="constant",
    )
    param_padded_blocked = param_padded.reshape(
        num_rows, row_size_padded // block_size, block_size
    )

    # Compute norm of each block and find threshold
    norm_of_blocks = np.linalg.norm(param_padded_blocked, axis=2)
    num_blocks_to_prune = int(np.floor(num_rows * num_blocks_per_row * prune_ratio))
    threshold = (
        0
        if num_blocks_to_prune == 0
        else np.partition(norm_of_blocks.flatten(), num_blocks_to_prune - 1)[
            num_blocks_to_prune - 1
        ]
    )

    # Create mask
    mask = norm_of_blocks >= threshold
    mask = np.broadcast_to(
        mask.reshape(mask.shape + (1,)), mask.shape + (block_size,)
    ).reshape(num_rows, row_size_padded)[:, :row_size]
    return mask


class TestMaskedAdagrad(hu.HypothesisTestCase):
    @given(
        inputs=hu.tensors(n=3),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
    )
    def test_masked_adagrad(self, inputs, lr, epsilon):
        param, moment, grad = inputs
        moment = np.abs(moment)
        lr = np.array([lr], dtype=np.float32)

        mask = np.random.randint(2, size=param.shape).astype(np.float32)

        workspace.FeedBlob("param", param)
        workspace.FeedBlob("moment", moment)
        workspace.FeedBlob("grad", grad)
        workspace.FeedBlob("lr", lr)
        workspace.FeedBlob("mask", mask)

        ref_op = core.CreateOperator(
            "Adagrad",
            ["param", "moment", "grad", "lr"],
            ["out_param_ref", "out_moment_ref"],
            epsilon=epsilon,
        )
        op = core.CreateOperator(
            "MaskedAdagrad",
            ["param", "moment", "grad", "lr", "mask"],
            ["out_param", "out_moment"],
            epsilon=epsilon,
        )

        workspace.RunOperatorOnce(ref_op)
        workspace.RunOperatorOnce(op)

        out_param_ref = workspace.FetchBlob("out_param_ref")
        out_moment_ref = workspace.FetchBlob("out_moment_ref")
        out_param_ref = np.multiply(mask, out_param_ref)
        out_moment_ref = np.multiply(mask, out_moment_ref)

        out_param = workspace.FetchBlob("out_param")
        out_moment = workspace.FetchBlob("out_moment")

        np.testing.assert_array_equal(out_param_ref, out_param)
        np.testing.assert_array_equal(out_moment_ref, out_moment)

    @given(
        inputs=hu.tensors(n=3),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        has_mask_input=st.booleans(),
        has_mask_out=st.booleans(),
        block_size=st.integers(1, 4),
        row_wise=st.booleans(),
    )
    def test_masked_sparse_adagrad(
        self,
        inputs,
        lr,
        epsilon,
        has_mask_input,
        has_mask_out,
        block_size,
        row_wise,
    ):
        param, moment, grad = inputs
        num_rows = param.shape[0]
        if row_wise:
            moment = np.resize(moment, num_rows)
        moment = np.abs(moment)
        lr = np.array([lr], dtype=np.float32)
        param_ref = np.copy(param)
        moment_ref = np.copy(moment)

        indices = np.random.randint(num_rows, size=grad.shape[0])

        workspace.ResetWorkspace()

        row_size = int(np.prod(param.shape[1:]))
        num_blocks_per_row = (row_size + block_size - 1) // block_size
        bitmask_bytes_per_row = (num_blocks_per_row + 7) // 8
        if has_mask_input:
            # Generate a random bit pattern
            mask = np.random.randint(
                np.iinfo(np.uint8).min,
                np.iinfo(np.uint8).max + 1,
                size=[num_rows, bitmask_bytes_per_row],
                dtype=np.uint8,
            )
            workspace.FeedBlob("mask", mask)
        else:
            delays = np.array([1, 2, 3]).astype(np.int32)
            # Make sure to use numbers that can be exactly represented in
            # float32 to avoid potentially different ways of handling floats
            # between Python and C++.
            prune_ratios = np.array([0.5, 0.75, 0.875]).astype(np.float32)

            # Feed empty mask
            workspace.FeedBlob("mask", np.array([]).astype(np.uint8))

        workspace.FeedBlob("param_ref", param_ref)
        workspace.FeedBlob("moment_ref", moment_ref)
        workspace.FeedBlob("param", param)
        workspace.FeedBlob("moment", moment)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("grad", grad)
        workspace.FeedBlob("lr", lr)

        net = core.Net("test_net")

        prefix = "RowWise" if row_wise else ""
        ref_op = core.CreateOperator(
            prefix + "SparseAdagrad",
            ["param_ref", "moment_ref", "indices", "grad", "lr"],
            ["param_ref", "moment_ref"],
            epsilon=epsilon,
        )

        inputs = ["param", "moment", "indices", "grad", "lr", "mask", "mask_changed"]
        outputs = ["param", "moment"]
        if not has_mask_input:
            inputs += ["iter"]
            if has_mask_out:
                outputs += ["mask_out"]

        op = core.CreateOperator(
            "Masked" + prefix + "SparseAdagrad",
            inputs,
            outputs,
            epsilon=epsilon,
            block_size=block_size,
            delays=[] if has_mask_input else delays,
            prune_ratios=[] if has_mask_input else prune_ratios,
        )
        net.Proto().op.extend([ref_op, op])

        workspace.FeedBlob("mask_changed", np.array([0]).astype(np.bool))
        workspace.FeedBlob("iter", np.array([0]).astype(np.int64))
        workspace.CreateNet(net)

        if has_mask_input:
            # Test1: if mask_changed == false, only the rows we're updating are masked
            workspace.RunNet(net)

            param_ref = workspace.FetchBlob("param_ref")
            moment_ref = workspace.FetchBlob("moment_ref")
            param = workspace.FetchBlob("param")
            moment = workspace.FetchBlob("moment")

            param_ref = param_ref.reshape(num_rows, -1)
            if not row_wise:
                moment_ref = moment_ref.reshape(num_rows, -1)

            for i in range(grad.shape[0]):
                row = indices[i]
                for j in range(row_size):
                    j_block = j // block_size
                    byte = j_block // 8
                    bit = j_block % 8
                    m = mask[row][byte] & (1 << bit)
                    if not m:
                        param_ref[row, j] = 0
                        if not row_wise:
                            moment_ref[row, j] = 0

            np.testing.assert_array_equal(param_ref, param.reshape(num_rows, -1))
            np.testing.assert_array_equal(
                moment_ref, moment if row_wise else moment.reshape(num_rows, -1)
            )

            # Test2: mask_changed == true
            workspace.FeedBlob("param_ref", param_ref)
            workspace.FeedBlob("moment_ref", moment_ref)
            workspace.FeedBlob("mask_changed", np.array([1]).astype(np.bool))
            workspace.RunNet(net)

            param_ref = workspace.FetchBlob("param_ref")
            moment_ref = workspace.FetchBlob("moment_ref")

            for i in range(num_rows):
                for j in range(row_size):
                    j_block = j // block_size
                    byte = j_block // 8
                    bit = j_block % 8
                    m = mask[i][byte] & (1 << bit)
                    if not m:
                        param_ref[i, j] = 0
                        if not row_wise:
                            moment_ref[i, j] = 0

            param = workspace.FetchBlob("param")
            moment = workspace.FetchBlob("moment")

            np.testing.assert_array_equal(param_ref, param.reshape(num_rows, -1))
            np.testing.assert_array_equal(
                moment_ref, moment if row_wise else moment.reshape(num_rows, -1)
            )
        else:
            # Test1: in the first iteration, there shouldn't be any masking
            workspace.RunNet(net)

            param_ref = workspace.FetchBlob("param_ref")
            moment_ref = workspace.FetchBlob("moment_ref")

            param = workspace.FetchBlob("param")
            moment = workspace.FetchBlob("moment")

            np.testing.assert_array_equal(param_ref, param)
            np.testing.assert_array_equal(moment_ref, moment)

            # Test2: for each pruning delay, masks should be updated accordingly
            for i in range(len(delays)):
                mask = _get_mask(param_ref, block_size, prune_ratios[i])

                workspace.FeedBlob("iter", np.array([delays[i]]).astype(np.int64))
                workspace.RunNet(net)

                param_ref = workspace.FetchBlob("param_ref")
                moment_ref = workspace.FetchBlob("moment_ref")

                param = workspace.FetchBlob("param")
                moment = workspace.FetchBlob("moment")

                param_ref = mask * param_ref.reshape(num_rows, row_size)
                if not row_wise:
                    moment_ref = mask * moment_ref.reshape(num_rows, row_size)

                np.testing.assert_array_equal(param_ref.flatten(), param.flatten())
                np.testing.assert_array_equal(moment_ref.flatten(), moment.flatten())

            # Test3: after finishing delay, mask should be fixed
            workspace.FeedBlob("iter", np.array([delays[-1] + 1]).astype(np.int64))
            workspace.RunNet(net)

            param_ref = workspace.FetchBlob("param_ref")
            moment_ref = workspace.FetchBlob("moment_ref")

            param = workspace.FetchBlob("param")
            moment = workspace.FetchBlob("moment")

            param_ref = mask * param_ref.reshape(num_rows, row_size)
            if not row_wise:
                moment_ref = mask * moment_ref.reshape(num_rows, row_size)

            np.testing.assert_array_equal(param_ref.flatten(), param.flatten())
            np.testing.assert_array_equal(moment_ref.flatten(), moment.flatten())
