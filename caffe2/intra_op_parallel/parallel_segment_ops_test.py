# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import unittest
from functools import partial

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
# Add gflag options to use async_scheduling in nets created inside assertReferenceChecks
core.GlobalInit(
    [
        "parallel_adagrad_test",
        "--caffe2_override_executor=simple,async_scheduling",
        "--caffe2_net_async_thread_pool_size=7",
    ]
)


def assertEngineChecks(
    device_option,
    op,
    inputs,
    engines,
    threshold=1e-4,
    output_to_grad=None,
    grad_reference=None,
    atol=None,
    outputs_to_check=None,
):
    """
    This runs multiple engines and compares them
    to the output of default engine, with an absolute/relative tolerance
    given by the `threshold` parameter.

    Useful for checking the implementation matches the Python
    (typically NumPy) implementation of the same functionality.

    Usage example:

        @given(X=hu.tensor(), inplace=st.booleans(), **hu.gcs)
        def test_softsign(self, X, inplace, gc, dc):
            op = core.CreateOperator(
                "Softsign", ["X"], ["X" if inplace else "Y"])

            self.assertEngineChecks(gc, op, [X], ["MY_AESOME_ENGINE"])
    """
    op = copy.deepcopy(op)

    outputs_to_check = outputs_to_check or list(range(len(op.output)))

    results = []
    workspace.SwitchWorkspace("_engine_check_", True)
    engines = [""] + engines
    for engine in engines:
        if len(op.input) > len(inputs):
            raise ValueError(
                "must supply an input for each input on the op: %s vs %s"
                % (op.input, inputs)
            )
        for (n, b) in zip(op.input, inputs):
            workspace.FeedBlob(n, b)
        net = core.Net("test_net")
        op.engine = engine
        net.Proto().op.extend([op])
        workspace.RunNetOnce(net)
        results.append(
            [workspace.FetchBlob(op.output[idx]) for idx in outputs_to_check]
        )
        workspace.ResetWorkspace()

    for i, engine in enumerate(engines):
        for j in range(len(outputs_to_check)):
            output_blob_name = op.output[j]
            output = results[i][j]
            ref = results[0][j]
            atol = atol or threshold
            np.testing.assert_allclose(
                output,
                ref,
                atol=atol,
                rtol=threshold,
                err_msg=(
                    "Output {} from engine {} is not matching the reference".format(
                        engine, output_blob_name
                    )
                ),
            )


def sparse_lengths_weighted_sum_ref(D, W, I, L):
    R = np.zeros(shape=(len(L),) + D.shape[1:], dtype=D.dtype)
    line = 0
    for g in range(len(L)):
        for _ in range(L[g]):
            R[g, :] += W[line] * D[I[line], :]
            line += 1
    return [R]


def sparse_lengths_weighted_sum_grad_ref(GO, fwd_out, fwd_in, grad_on_weights=False):
    D, W, I, L = fwd_in
    GI = np.zeros(shape=(len(I),) + D.shape[1:], dtype=D.dtype)
    GW = np.zeros(shape=W.shape, dtype=W.dtype) if grad_on_weights else None
    line = 0
    for g in range(len(L)):
        for _ in range(L[g]):
            GI[line, :] = W[line] * GO[g, :]
            if GW is not None:
                GW[line] = np.dot(GO[g].flatten(), D[I[line], :].flatten())
            line += 1
    return [(GI, I), GW, None, None]


class TestIntralOpParallelSegmentOps(hu.HypothesisTestCase):
    @given(
        input=hu.tensor(min_dim=2, max_dim=2, max_value=50),
        data_strategy=st.data(),
        **hu.gcs_cpu_only
    )
    def test_sparse_lengths_sum(self, input, data_strategy, gc, dc):
        m = input.shape[0]

        lengths = data_strategy.draw(
            hu.tensor(
                max_dim=1,
                max_value=input.shape[0],
                dtype=np.int32,
                elements=st.integers(min_value=0, max_value=27),
            )
        )
        lengths_sum = np.sum(lengths)

        indices = data_strategy.draw(
            hu.arrays(
                [lengths_sum], dtype=np.int64, elements=st.sampled_from(np.arange(m))
            )
        )

        op = core.CreateOperator(
            "SparseLengthsSum", ["input", "indices", "lengths"], "out"
        )
        assertEngineChecks(
            dc, op, [input, indices, lengths], ["INTRA_OP_PARALLEL", "TBB"]
        )

    @given(
        input=hu.tensor(min_dim=2, max_dim=2, max_value=50),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        data_strategy=st.data(),
        **hu.gcs_cpu_only
    )
    def test_sparse_lengths_weighted_sum(self, input, engine, data_strategy, gc, dc):
        for grad_on_weights in (False, True):
            m = input.shape[0]

            lengths = data_strategy.draw(
                hu.tensor(
                    max_dim=1,
                    max_value=input.shape[0],
                    dtype=np.int32,
                    elements=st.integers(min_value=0, max_value=27),
                )
            )
            lengths_sum = np.sum(lengths)

            indices = data_strategy.draw(
                hu.arrays(
                    [lengths_sum],
                    dtype=np.int64,
                    elements=st.sampled_from(np.arange(m)),
                )
            )

            weight = data_strategy.draw(hu.arrays([lengths_sum], dtype=np.float32))

            op = core.CreateOperator(
                "SparseLengthsWeightedSum",
                ["input", "weight", "indices", "lengths"],
                "out",
                engine=engine,
                grad_on_weights=grad_on_weights,
            )
            assertEngineChecks(dc, op, [input, weight, indices, lengths], [engine])
            self.assertReferenceChecks(
                device_option=gc,
                op=op,
                inputs=[input, weight, indices, lengths],
                reference=sparse_lengths_weighted_sum_ref,
                threshold=1e-4,
                output_to_grad="out",
                grad_reference=partial(
                    sparse_lengths_weighted_sum_grad_ref,
                    grad_on_weights=grad_on_weights,
                ),
            )

    @given(
        input=hu.tensor(min_dim=2, max_dim=2, max_value=50),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        data_strategy=st.data(),
        **hu.gcs_cpu_only
    )
    def test_sparse_lengths_positional_weighted_sum(
        self, input, engine, data_strategy, gc, dc
    ):
        m = input.shape[0]

        lengths = data_strategy.draw(
            hu.tensor(
                max_dim=1,
                max_value=input.shape[0],
                dtype=np.int32,
                elements=st.integers(min_value=0, max_value=27),
            )
        )
        lengths_sum = np.sum(lengths)

        indices = data_strategy.draw(
            hu.arrays(
                [lengths_sum], dtype=np.int64, elements=st.sampled_from(np.arange(m))
            )
        )

        weight = data_strategy.draw(hu.arrays([lengths.max()], dtype=np.float32))

        op = core.CreateOperator(
            "SparseLengthsPositionalWeightedSum",
            ["input", "weight", "indices", "lengths"],
            "out",
            engine=engine,
        )

        def ref_sparse_lengths_positional_weighted_sum(input, weight, indices, lengths):
            workspace.FeedBlob("lengths", lengths)
            lengths_range_fill_op = core.CreateOperator(
                "LengthsRangeFill", ["lengths"], ["lengths_pos_seq"]
            )
            workspace.RunOperatorOnce(lengths_range_fill_op)

            workspace.FeedBlob("weight", weight)
            gather_op = core.CreateOperator(
                "Gather", ["weight", "lengths_pos_seq"], ["weight_gathered"]
            )
            workspace.RunOperatorOnce(gather_op)

            workspace.FeedBlob("input", input)
            workspace.FeedBlob("indices", indices)
            sparse_op = core.CreateOperator(
                "SparseLengthsWeightedSum",
                ["input", "weight_gathered", "indices", "lengths"],
                "out_ref",
            )
            workspace.RunOperatorOnce(sparse_op)

            return (workspace.FetchBlob("out_ref"),)

        self.assertReferenceChecks(
            gc,
            op,
            [input, weight, indices, lengths],
            ref_sparse_lengths_positional_weighted_sum,
        )

    @given(**hu.gcs_cpu_only)
    def test_sparse_lengths_indices_in_gradient_sum(self, gc, dc):
        grad = np.random.rand(3, 3, 4, 5).astype(np.float32)
        lengths = np.asarray([3, 3, 2]).astype(np.int32)
        indices = np.random.randint(0, 50, size=8).astype(np.int64)
        op = core.CreateOperator(
            "SparseLengthsIndicesInGradientSumGradient",
            ["grad", "lengths", "indices"],
            "out",
        )
        assertEngineChecks(
            dc, op, [grad, lengths, indices], ["INTRA_OP_PARALLEL", "TBB"]
        )

    @given(**hu.gcs_cpu_only)
    def test_sparse_lengths_indices_in_gradient_weighted_sum_with_main_input_gradient(
        self, gc, dc
    ):
        aux_in = np.random.rand(3, 3, 4, 5).astype(np.float32)
        grad = np.random.rand(3, 3, 4, 5).astype(np.float32)
        lengths = np.asarray([3, 3, 2]).astype(np.int32)
        param = np.random.rand(50, 3, 4, 5).astype(np.float32)
        indices = np.random.randint(0, 50, size=8).astype(np.int64)
        op = core.CreateOperator(
            "SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient",
            ["aux_in", "grad", "lengths", "param", "indices"],
            ["grad_out", "aux_grad"],
        )
        assertEngineChecks(
            dc,
            op,
            [aux_in, grad, lengths, param, indices],
            ["INTRA_OP_PARALLEL", "TBB"],
        )

    @given(engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]), **hu.gcs_cpu_only)
    def test_sparse_lengths_sum_invalid_index(self, engine, gc, dc):
        D = np.random.rand(50, 3, 4, 5).astype(np.float32)
        Indices = (np.random.randint(0, 10000, size=10) + 10000).astype(np.int64)
        L = np.asarray([4, 4, 2]).astype(np.int32)
        op = core.CreateOperator(
            "SparseLengthsSum", ["D", "Indices", "L"], "out", engine=engine
        )
        workspace.FeedBlob("D", D)
        workspace.FeedBlob("Indices", Indices)
        workspace.FeedBlob("L", L)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)


if __name__ == "__main__":
    unittest.main()
