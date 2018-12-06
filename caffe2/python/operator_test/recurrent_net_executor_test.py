from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import model_helper, workspace, core, rnn_cell
from caffe2.python.attention import AttentionType

import numpy as np

import unittest
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from hypothesis import given


class TestRNNExecutor(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.input_dim = 20
        self.hidden_dim = 30
        self.encoder_dim = 40

    @given(
        T=st.integers(10, 100),
        forward_only=st.booleans(),
        **hu.gcs)
    def test_lstm_with_attention_equal_simplenet(self, T, forward_only, gc, dc):
        self.Tseq = [T, T // 2, T // 2 + T // 4, T, T // 2 + 1]
        workspace.ResetWorkspace()
        with core.DeviceScope(gc):
            print("Run with device: {}, forward only: {}".format(
                gc, forward_only))

            workspace.FeedBlob(
                "seq_lengths",
                np.array([T] * self.batch_size, dtype=np.int32)
            )
            workspace.FeedBlob("target", np.random.rand(
                T, self.batch_size, self.hidden_dim).astype(np.float32))
            workspace.FeedBlob("hidden_init", np.zeros(
                [1, self.batch_size, self.hidden_dim], dtype=np.float32
            ))
            workspace.FeedBlob("cell_init", np.zeros(
                [1, self.batch_size, self.hidden_dim], dtype=np.float32
            ))

            model = model_helper.ModelHelper(name="lstm")
            model.net.AddExternalInputs(["input"])

            init_blobs = []
            hidden_init, cell_init, encoder_outputs = model.net.AddExternalInputs(
                "hidden_init",
                "cell_init",
                "encoder_outputs"
            )

            awec_init = model.net.AddExternalInputs([
                'initial_attention_weighted_encoder_context',
            ])
            init_blobs.extend([hidden_init, cell_init])

            workspace.FeedBlob(
                awec_init,
                np.random.rand(1, self.batch_size, self.encoder_dim).astype(
                    np.float32),
            )
            workspace.FeedBlob(
                encoder_outputs,
                np.random.rand(1, self.batch_size, self.encoder_dim).astype(
                    np.float32),
            )

            outputs = rnn_cell.LSTMWithAttention(
                model=model,
                decoder_inputs="input",
                decoder_input_lengths="seq_lengths",
                initial_decoder_hidden_state=hidden_init,
                initial_decoder_cell_state=cell_init,
                initial_attention_weighted_encoder_context=awec_init,
                encoder_output_dim=self.encoder_dim,
                encoder_outputs=encoder_outputs,
                encoder_lengths=None,
                decoder_input_dim=self.input_dim,
                decoder_state_dim=self.hidden_dim,
                scope="",
                attention_type=AttentionType.Recurrent,
                forward_only=forward_only,
                outputs_with_grads=[0],
            )
            output = outputs[0]

            print(outputs)
            loss = model.AveragedLoss(
                model.SquaredL2Distance([output, "target"], "dist"),
                "loss"
            )
            # Add gradient ops
            if not forward_only:
                model.AddGradientOperators([loss])

            # init
            for init_blob in init_blobs:
                workspace.FeedBlob(init_blob, np.zeros(
                    [1, self.batch_size, self.hidden_dim], dtype=np.float32
                ))

            self._compare(model, forward_only)

    def init_lstm_model(self, T, num_layers, forward_only, use_loss=True):
        workspace.FeedBlob(
            "seq_lengths",
            np.array([T] * self.batch_size, dtype=np.int32)
        )
        workspace.FeedBlob("target", np.random.rand(
            T, self.batch_size, self.hidden_dim).astype(np.float32))
        workspace.FeedBlob("hidden_init", np.zeros(
            [1, self.batch_size, self.hidden_dim], dtype=np.float32
        ))
        workspace.FeedBlob("cell_init", np.zeros(
            [1, self.batch_size, self.hidden_dim], dtype=np.float32
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
            dim_in=self.input_dim,
            dim_out=[self.hidden_dim] * num_layers,
            scope="",
            drop_states=True,
            forward_only=forward_only,
            return_last_layer_only=True,
        )

        if use_loss:
            loss = model.AveragedLoss(
                model.SquaredL2Distance([output, "target"], "dist"),
                "loss"
            )
            # Add gradient ops
            if not forward_only:
                model.AddGradientOperators([loss])

        # init
        for init_blob in init_blobs:
            workspace.FeedBlob(init_blob, np.zeros(
                [1, self.batch_size, self.hidden_dim], dtype=np.float32
            ))

        return model, output

    def test_empty_sequence(self):
        '''
        Test the RNN executor's handling of empty input sequences
        '''
        Tseq = [0, 1, 2, 3, 0, 1]
        workspace.ResetWorkspace()
        with core.DeviceScope(caffe2_pb2.DeviceOption()):
            model, output = self.init_lstm_model(
                T=4, num_layers=1, forward_only=True, use_loss=False)

            workspace.RunNetOnce(model.param_init_net)

            self.enable_rnn_executor(model.net, 1, True)

            np.random.seed(10022015)
            first_call = True
            for seq_len in Tseq:
                input_shape = [seq_len, self.batch_size, self.input_dim]
                workspace.FeedBlob(
                    "input", np.random.rand(*input_shape).astype(np.float32))
                workspace.FeedBlob(
                    "target",
                    np.random.rand(
                        seq_len, self.batch_size, self.hidden_dim
                    ).astype(np.float32))
                if first_call:
                    workspace.CreateNet(model.net, overwrite=True)
                    first_call = False

                workspace.RunNet(model.net.Proto().name)
                val = workspace.FetchBlob(output)
                self.assertEqual(val.shape[0], seq_len)

    @given(
        num_layers=st.integers(1, 8),
        T=st.integers(4, 100),
        forward_only=st.booleans(),
        **hu.gcs)
    def test_lstm_equal_simplenet(self, num_layers, T, forward_only, gc, dc):
        '''
        Test that the RNN executor produces same results as
        the non-executor (i.e running step nets as sequence of simple nets).
        '''
        self.Tseq = [T, T // 2, T // 2 + T // 4, T, T // 2 + 1]

        workspace.ResetWorkspace()
        with core.DeviceScope(gc):
            print("Run with device: {}, forward only: {}".format(
                gc, forward_only))

            model, _ = self.init_lstm_model(T, num_layers, forward_only)
            self._compare(model, forward_only)

    def _compare(self, model, forward_only):
        # Store list of blobs that exist in the beginning
        workspace.RunNetOnce(model.param_init_net)
        init_ws = {k: workspace.FetchBlob(k) for k in workspace.Blobs()}

        # Run with executor
        for enable_executor in [0, 1]:
            self.enable_rnn_executor(model.net, enable_executor, forward_only)
            workspace.ResetWorkspace()

            # Reset original state
            for k, v in init_ws.items():
                workspace.FeedBlob(k, v)

            np.random.seed(10022015)
            ws = {}
            for j in range(len(self.Tseq)):
                input_shape = [self.Tseq[j], self.batch_size, self.input_dim]
                workspace.FeedBlob(
                    "input", np.random.rand(*input_shape).astype(np.float32))
                workspace.FeedBlob(
                    "target",
                    np.random.rand(
                        self.Tseq[j], self.batch_size, self.hidden_dim
                    ).astype(np.float32))
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

        mismatch = False
        for k in rnn_exec_ws.keys():
            non_exec_v = non_exec_ws[k]
            rnn_exec_v = rnn_exec_ws[k]
            if type(non_exec_v) is np.ndarray:
                if not np.allclose(non_exec_v, rnn_exec_v):
                    print("Mismatch: {}".format(k))
                    nv = non_exec_v.flatten()
                    rv = rnn_exec_v.flatten()
                    c = 0
                    for j in range(len(nv)):
                        if rv[j] != nv[j]:
                            print(j, rv[j], nv[j])
                            c += 1
                            if c == 10:
                                break

                    mismatch = True

        self.assertFalse(mismatch)

    def enable_rnn_executor(self, net, value, forward_only):
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
        self.assertEqual(1 if forward_only else 2, num_found)

    if __name__ == "__main__":
        import unittest
        import random
        random.seed(2603)
        workspace.GlobalInit([
            'caffe2',
            '--caffe2_log_level=0',
            '--caffe2_rnn_executor=1'])
        unittest.main()
