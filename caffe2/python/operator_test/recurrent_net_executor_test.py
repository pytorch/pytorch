from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import model_helper, workspace, core, rnn_cell
import numpy as np

import unittest
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from hypothesis import given


class TestRNNExecutor(unittest.TestCase):
    @given(
        num_layers=st.integers(1, 8),
        T=st.integers(4, 100),
        forward_only=st.booleans(),
        **hu.gcs)
    def test_lstm_equal_simplenet(self, num_layers, T, forward_only, gc, dc):
        '''
        Test that the RNN executor produces same results as
        the non-RNN versino.
        '''
        workspace.ResetWorkspace()
        with core.DeviceScope(gc):
            print("Run with device: {}, forward only: {}".format(
                gc, forward_only))
            batch_size = 4
            input_dim = 20
            hidden_dim = 30
            Tseq = [T, T // 2, T // 2 + T // 4, T, T // 2 + 1]

            workspace.FeedBlob(
                "seq_lengths",
                np.array([T] * batch_size, dtype=np.int32)
            )
            workspace.FeedBlob("target", np.random.rand(
                T, batch_size, hidden_dim).astype(np.float32))
            workspace.FeedBlob("hidden_init", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))
            workspace.FeedBlob("cell_init", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))

            model = model_helper.ModelHelper(name="lstm")
            model.net.AddExternalInputs(["input"])

            init_blobs = []
            for i in range(num_layers):
                hidden_init, cell_init = model.net.AddExternalInputs(
                    "hidden_init_{}".format(i),
                    "cell_init_{}".format(i)
                )
                init_blobs.extend([hidden_init, cell_init])

            output, last_hidden, _, last_state = rnn_cell.LSTM(
                model=model,
                input_blob="input",
                seq_lengths="seq_lengths",
                initial_states=init_blobs,
                dim_in=input_dim,
                dim_out=[hidden_dim] * num_layers,
                scope="",
                drop_states=True,
                forward_only=forward_only,
                return_last_layer_only=True,
            )
            loss = model.AveragedLoss(
                model.SquaredL2Distance([output, "target"], "dist"),
                "loss"
            )

            # init
            for init_blob in init_blobs:
                workspace.FeedBlob(init_blob, np.zeros(
                    [1, batch_size, hidden_dim], dtype=np.float32
                ))

            # Add gradient ops
            model.AddGradientOperators([loss])

            # Store list of blobs that exist in the beginning
            workspace.RunNetOnce(model.param_init_net)
            init_ws = {k: workspace.FetchBlob(k) for k in workspace.Blobs()}

            # Run with executor
            for enable_executor in [0, 1]:
                self.enable_rnn_executor(model.net, enable_executor)
                workspace.ResetWorkspace()

                # Reset original state
                for k, v in init_ws.items():
                    workspace.FeedBlob(k, v)

                np.random.seed(10022015)
                ws = {}
                for j in range(len(Tseq)):
                    input_shape = [Tseq[j], batch_size, input_dim]
                    workspace.FeedBlob(
                        "input", np.random.rand(*input_shape).astype(np.float32))
                    workspace.FeedBlob("target", np.random.rand(
                        Tseq[j], batch_size, hidden_dim).astype(np.float32))
                    if j == 0:
                        workspace.CreateNet(model.net, overwrite=True)

                    workspace.RunNet(model.net.Proto().name)

                    # Store results for each iteration
                    for k in workspace.Blobs():
                        ws[k + "." + str(j)] = workspace.FetchBlob(k)

                if enable_executor:
                    rnn_exec_ws = ws
                else:
                    non_exec_ws = ws

            # Test that all blobs are equal after running with executor
            # or without.
            self.assertEqual(list(non_exec_ws.keys()), list(rnn_exec_ws.keys()))

            for k in rnn_exec_ws.keys():
                non_exec_v = non_exec_ws[k]
                rnn_exec_v = rnn_exec_ws[k]
                if type(non_exec_v) is np.ndarray:
                    self.assertTrue(
                        np.array_equal(non_exec_v, rnn_exec_v),
                        "Mismatch: {}".format(k)
                    )

    def enable_rnn_executor(self, net, value):
        num_found = 0
        for op in net.Proto().op:
            if op.type.startswith("RecurrentNetwork"):
                for arg in op.arg:
                    if arg.name == 'enable_rnn_executor':
                        arg.i = value
                        num_found += 1
        # This sanity check is so that if someone changes the
        # enable_rnn_executor parameter name, the test will
        # start failing as this function will become defective.
        self.assertEqual(2, num_found)

    if __name__ == "__main__":
        import unittest
        import random
        random.seed(2603)
        workspace.GlobalInit([
            'caffe2',
            '--caffe2_log_level=0',
            '--caffe2_rnn_executor=1'])
        unittest.main()
