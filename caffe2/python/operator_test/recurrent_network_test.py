from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, recurrent, workspace
from caffe2.python.attention import AttentionType
from caffe2.python.model_helper import ModelHelperBase
from caffe2.python.cnn import CNNModelHelper
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return 2.0 * sigmoid(2.0 * x) - 1


def lstm_unit(hidden_t_prev, cell_t_prev, gates, seq_lengths, timestep):
    D = cell_t_prev.shape[2]
    G = gates.shape[2]
    N = gates.shape[1]
    t = (timestep * np.ones(shape=(N, D))).astype(np.int32)
    assert t.shape == (N, D)
    seq_lengths = (np.ones(shape=(N, D)) *
                   seq_lengths.reshape(N, 1)).astype(np.int32)
    assert seq_lengths.shape == (N, D)
    assert G == 4 * D
    # Resize to avoid broadcasting inconsistencies with NumPy
    gates = gates.reshape(N, 4, D)
    cell_t_prev = cell_t_prev.reshape(N, D)
    i_t = gates[:, 0, :].reshape(N, D)
    f_t = gates[:, 1, :].reshape(N, D)
    o_t = gates[:, 2, :].reshape(N, D)
    g_t = gates[:, 3, :].reshape(N, D)
    i_t = sigmoid(i_t)
    f_t = sigmoid(f_t)
    o_t = sigmoid(o_t)
    g_t = tanh(g_t)
    valid = (t < seq_lengths).astype(np.int32)
    assert valid.shape == (N, D)
    cell_t = ((f_t * cell_t_prev) + (i_t * g_t)) * (valid) + \
        (1 - valid) * cell_t_prev
    assert cell_t.shape == (N, D)
    hidden_t = (o_t * tanh(cell_t)) * valid + hidden_t_prev * (1 - valid)
    hidden_t = hidden_t.reshape(1, N, D)
    cell_t = cell_t.reshape(1, N, D)
    return hidden_t, cell_t


def lstm_reference(input, hidden_input, cell_input,
                   gates_w, gates_b, seq_lengths):
    T = input.shape[0]
    N = input.shape[1]
    G = input.shape[2]
    D = hidden_input.shape[hidden_input.ndim - 1]
    hidden = np.zeros(shape=(T + 1, N, D))
    cell = np.zeros(shape=(T + 1, N, D))
    assert hidden.shape[0] == T + 1
    assert cell.shape[0] == T + 1
    assert hidden.shape[1] == N
    assert cell.shape[1] == N
    cell[0, :, :] = cell_input
    hidden[0, :, :] = hidden_input
    for t in range(T):
        input_t = input[t].reshape(1, N, G)
        hidden_t_prev = hidden[t].reshape(1, N, D)
        cell_t_prev = cell[t].reshape(1, N, D)
        gates = np.dot(hidden_t_prev, gates_w.T) + gates_b
        gates = gates + input_t
        hidden_t, cell_t = lstm_unit(
            hidden_t_prev,
            cell_t_prev,
            gates,
            seq_lengths,
            t,
        )
        hidden[t + 1] = hidden_t
        cell[t + 1] = cell_t
    return (
        hidden[1:],
        hidden[-1].reshape(1, N, D),
        cell[1:],
        cell[-1].reshape(1, N, D)
    )



def milstm_reference(
        input,
        hidden_input,
        cell_input,
        gates_w,
        gates_b,
        alpha,
        beta1,
        beta2,
        b,
        seq_lengths):
    T = input.shape[0]
    N = input.shape[1]
    G = input.shape[2]
    D = hidden_input.shape[hidden_input.ndim - 1]
    hidden = np.zeros(shape=(T + 1, N, D))
    cell = np.zeros(shape=(T + 1, N, D))
    assert hidden.shape[0] == T + 1
    assert cell.shape[0] == T + 1
    assert hidden.shape[1] == N
    assert cell.shape[1] == N
    cell[0, :, :] = cell_input
    hidden[0, :, :] = hidden_input
    for t in range(T):
        input_t = input[t].reshape(1, N, G)
        hidden_t_prev = hidden[t].reshape(1, N, D)
        cell_t_prev = cell[t].reshape(1, N, D)
        gates = np.dot(hidden_t_prev, gates_w.T) + gates_b
        gates = (alpha * gates * input_t) + \
                    (beta1 * gates) + \
                    (beta2 * input_t) + \
                    b
        hidden_t, cell_t = lstm_unit(
            hidden_t_prev,
            cell_t_prev,
            gates,
            seq_lengths,
            t,
        )
        hidden[t + 1] = hidden_t
        cell[t + 1] = cell_t
    return (
        hidden[1:],
        hidden[-1].reshape(1, N, D),
        cell[1:],
        cell[-1].reshape(1, N, D)
    )


def lstm_with_attention_reference(
    input,
    initial_hidden_state,
    initial_cell_state,
    initial_attention_weighted_encoder_context,
    encoder_outputs_transposed,
    weighted_encoder_outputs,
    gates_w,
    gates_b,
    decoder_input_lengths,
    weighted_decoder_hidden_state_t_w,
    weighted_decoder_hidden_state_t_b,
    attention_v,
    attention_zeros,
):
    encoder_outputs = np.transpose(encoder_outputs_transposed, axes=[2, 0, 1])
    decoder_input_length = input.shape[0]
    batch_size = input.shape[1]
    decoder_input_dim = input.shape[2]
    decoder_state_dim = initial_hidden_state.shape[2]
    encoder_output_dim = weighted_encoder_outputs.shape[2]
    hidden = np.zeros(
        shape=(decoder_input_length + 1, batch_size, decoder_state_dim))
    cell = np.zeros(
        shape=(decoder_input_length + 1, batch_size, decoder_state_dim))
    attention_weighted_encoder_context = np.zeros(
        shape=(decoder_input_length + 1, batch_size, encoder_output_dim))
    cell[0, :, :] = initial_cell_state
    hidden[0, :, :] = initial_hidden_state
    attention_weighted_encoder_context[0, :, :] = (
        initial_attention_weighted_encoder_context
    )
    for t in range(decoder_input_length):
        input_t = input[t].reshape(1, batch_size, decoder_input_dim)
        hidden_t_prev = hidden[t].reshape(1, batch_size, decoder_state_dim)
        cell_t_prev = cell[t].reshape(1, batch_size, decoder_state_dim)
        attention_weighted_encoder_context_t_prev = (
            attention_weighted_encoder_context[t].reshape(
                1, batch_size, encoder_output_dim)
        )
        gates_input = np.concatenate(
            (hidden_t_prev, attention_weighted_encoder_context_t_prev),
            axis=2,
        )
        gates = np.dot(gates_input, gates_w.T) + gates_b
        gates = gates + input_t
        hidden_t, cell_t = lstm_unit(hidden_t_prev, cell_t_prev, gates,
                                     decoder_input_lengths, t)
        hidden[t + 1] = hidden_t
        cell[t + 1] = cell_t
        weighted_hidden_t = np.dot(
            hidden_t,
            weighted_decoder_hidden_state_t_w.T,
        ) + weighted_decoder_hidden_state_t_b
        attention_v = attention_v.reshape([-1])
        attention_logits_t = np.sum(
            attention_v * np.tanh(weighted_encoder_outputs + weighted_hidden_t),
            axis=2,
        )
        attention_logits_t_exp = np.exp(attention_logits_t)
        attention_weights_t = (
            attention_logits_t_exp /
            np.sum(attention_logits_t_exp, axis=0).reshape([1, -1])
        )
        attention_weighted_encoder_context[t + 1] = np.sum(
            (
                encoder_outputs *
                attention_weights_t.reshape([-1, batch_size, 1])
            ),
            axis=0,
        )
    return (
        hidden[1:],
        hidden[-1].reshape(1, batch_size, decoder_state_dim),
        cell[1:],
        cell[-1].reshape(1, batch_size, decoder_state_dim),
        attention_weighted_encoder_context[1:],
        attention_weighted_encoder_context[-1].reshape(
            1,
            batch_size,
            encoder_output_dim,
        )
    )


def lstm_with_recurrent_attention_reference(
    input,
    initial_hidden_state,
    initial_cell_state,
    initial_attention_weighted_encoder_context,
    encoder_outputs_transposed,
    weighted_encoder_outputs,
    gates_w,
    gates_b,
    decoder_input_lengths,
    weighted_prev_attention_context_w,
    weighted_prev_attention_context_b,
    weighted_decoder_hidden_state_t_w,
    weighted_decoder_hidden_state_t_b,
    attention_v,
    attention_zeros,
):
    encoder_outputs = np.transpose(encoder_outputs_transposed, axes=[2, 0, 1])
    decoder_input_length = input.shape[0]
    batch_size = input.shape[1]
    decoder_input_dim = input.shape[2]
    decoder_state_dim = initial_hidden_state.shape[2]
    encoder_output_dim = weighted_encoder_outputs.shape[2]
    hidden = np.zeros(
        shape=(decoder_input_length + 1, batch_size, decoder_state_dim))
    cell = np.zeros(
        shape=(decoder_input_length + 1, batch_size, decoder_state_dim))
    attention_weighted_encoder_context = np.zeros(
        shape=(decoder_input_length + 1, batch_size, encoder_output_dim))
    cell[0, :, :] = initial_cell_state
    hidden[0, :, :] = initial_hidden_state
    attention_weighted_encoder_context[0, :, :] = (
        initial_attention_weighted_encoder_context
    )
    for t in range(decoder_input_length):
        input_t = input[t].reshape(1, batch_size, decoder_input_dim)
        hidden_t_prev = hidden[t].reshape(1, batch_size, decoder_state_dim)
        cell_t_prev = cell[t].reshape(1, batch_size, decoder_state_dim)
        attention_weighted_encoder_context_t_prev = (
            attention_weighted_encoder_context[t].reshape(
                1, batch_size, encoder_output_dim)
        )
        gates_input = np.concatenate(
            (hidden_t_prev, attention_weighted_encoder_context_t_prev),
            axis=2,
        )
        gates = np.dot(gates_input, gates_w.T) + gates_b
        gates = gates + input_t
        hidden_t, cell_t = lstm_unit(hidden_t_prev, cell_t_prev, gates,
                                     decoder_input_lengths, t)
        hidden[t + 1] = hidden_t
        cell[t + 1] = cell_t

        weighted_hidden_t = np.dot(
            hidden_t,
            weighted_decoder_hidden_state_t_w.T,
        ) + weighted_decoder_hidden_state_t_b
        weighted_prev_attention_context = np.dot(
            attention_weighted_encoder_context_t_prev,
            weighted_prev_attention_context_w.T
        ) + weighted_prev_attention_context_b
        attention_v = attention_v.reshape([-1])
        attention_logits_t = np.sum(
            attention_v * np.tanh(
                weighted_encoder_outputs + weighted_hidden_t +
                weighted_prev_attention_context
            ),
            axis=2,
        )

        attention_logits_t_exp = np.exp(attention_logits_t)
        attention_weights_t = (
            attention_logits_t_exp /
            np.sum(attention_logits_t_exp, axis=0).reshape([1, -1])
        )
        attention_weighted_encoder_context[t + 1] = np.sum(
            (
                encoder_outputs *
                attention_weights_t.reshape([-1, batch_size, 1])
            ),
            axis=0,
        )
    return (
        hidden[1:],
        hidden[-1].reshape(1, batch_size, decoder_state_dim),
        cell[1:],
        cell[-1].reshape(1, batch_size, decoder_state_dim),
        attention_weighted_encoder_context[1:],
        attention_weighted_encoder_context[-1].reshape(
            1,
            batch_size,
            encoder_output_dim,
        )
    )


class RecurrentNetworkTest(hu.HypothesisTestCase):

    @given(t=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_lstm(self, t, n, d):
        for outputs_with_grads in [[0], [1], [0, 1, 2, 3]]:
            self.lstm(
                recurrent.LSTM, t, n, d, lstm_reference, outputs_with_grads)


    @given(t=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_milstm(self, t, n, d):
        for outputs_with_grads in [[0], [1], [0, 1, 2, 3]]:
            self.lstm(
                recurrent.MILSTM, t, n, d, milstm_reference, outputs_with_grads)

    def lstm(self, create_lstm, t, n, d, ref, outputs_with_grads):
        model = CNNModelHelper(name='external')
        input_blob, seq_lengths, hidden_init, cell_init = (
            model.net.AddExternalInputs(
                'input_blob', 'seq_lengths', 'hidden_init', 'cell_init'))

        create_lstm(
            model, input_blob, seq_lengths, (hidden_init, cell_init),
            d, d, scope="external/recurrent",
            outputs_with_grads=outputs_with_grads)

        op = model.net._net.op[-1]

        workspace.RunNetOnce(model.param_init_net)
        input_blob = op.input[0]

        def generate_random_state(n, d):
            ndim = int(np.random.choice(3, 1)) + 1
            if ndim == 1:
                return np.random.randn(1, n, d).astype(np.float32)
            random_state = np.random.randn(n, d).astype(np.float32)
            if ndim == 3:
                random_state = random_state.reshape([1, n, d])
            return random_state

        workspace.FeedBlob(
            str(input_blob), np.random.randn(t, n, d * 4).astype(np.float32))
        workspace.FeedBlob("hidden_init", generate_random_state(n, d))
        workspace.FeedBlob("cell_init", generate_random_state(n, d))
        workspace.FeedBlob(
            "seq_lengths",
            np.random.randint(1, t + 1, size=(n,)).astype(np.int32)
        )
        inputs = [workspace.FetchBlob(name) for name in op.input]

        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            inputs,
            ref,
            outputs_to_check=range(4),
        )

        # Checking for input, gates_t_w and gates_t_b gradients
        for param in range(5):
            self.assertGradientChecks(
                device_option=hu.cpu_do,
                op=op,
                inputs=inputs,
                outputs_to_check=param,
                outputs_with_grads=outputs_with_grads,
                threshold=0.01,
                stepsize=0.005,
            )

    @given(T=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_sum_mul(self, T, n, d):
        model = ModelHelperBase(name='external')

        input_blob, initial_input_blob = model.net.AddExternalInputs(
            'input', 'initial_input')

        step = ModelHelperBase(name='step', param_model=model)
        input_t, output_t_prev = step.net.AddExternalInput(
            'input_t', 'output_t_prev')
        output_t_internal = step.net.Sum([input_t, output_t_prev])
        output_t = step.net.Mul([input_t, output_t_internal])
        step.net.AddExternalOutput(output_t)

        self.simple_rnn(T, n, d, model, step, input_t, output_t, output_t_prev,
                        input_blob, initial_input_blob)

    @given(T=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_mul(self, T, n, d):
        model = ModelHelperBase(name='external')

        input_blob, initial_input_blob = model.net.AddExternalInputs(
            'input', 'initial_input')

        step = ModelHelperBase(name='step', param_model=model)
        input_t, output_t_prev = step.net.AddExternalInput(
            'input_t', 'output_t_prev')
        output_t = step.net.Mul([input_t, output_t_prev])
        step.net.AddExternalOutput(output_t)

        self.simple_rnn(T, n, d, model, step, input_t, output_t, output_t_prev,
                        input_blob, initial_input_blob)

    def simple_rnn(self, T, n, d, model, step, input_t, output_t, output_t_prev,
                   input_blob, initial_input_blob):

        input = np.random.randn(T, n, d).astype(np.float32)
        initial_input = np.random.randn(1, n, d).astype(np.float32)

        recurrent.recurrent_net(
            net=model.net,
            cell_net=step.net,
            inputs=[(input_t, input_blob)],
            initial_cell_inputs=[(output_t_prev, initial_input_blob)],
            links={output_t_prev: output_t},
            scope="test_rnn_sum_mull",
        )

        workspace.blobs[input_blob] = input
        workspace.blobs[initial_input_blob] = initial_input

        op = model.net._net.op[-1]
        # Just conviniently store all inputs in an array in the same
        # order as op.input
        inputs = [workspace.blobs[name] for name in op.input]

        def reference(input, initial_input):
            global_ws_name = workspace.CurrentWorkspace()
            input_all = workspace.blobs[input_blob]

            workspace.SwitchWorkspace("ref", create_if_missing=True)
            workspace.blobs[input_blob] = input
            workspace.blobs[output_t_prev] = initial_input.reshape(n, d)
            res_all = np.zeros(shape=input.shape, dtype=np.float32)

            for t_cur in range(T):
                workspace.blobs[input_t] = input_all[t_cur]
                workspace.RunNetOnce(step.net)
                result_t = workspace.blobs[output_t]
                workspace.blobs[output_t_prev] = result_t
                res_all[t_cur] = result_t

            workspace.SwitchWorkspace(global_ws_name)

            shape = list(input.shape)
            shape[0] = 1
            return (res_all, res_all[-1].reshape(shape))

        self.assertReferenceChecks(
            device_option=hu.cpu_do,
            op=op,
            inputs=inputs,
            reference=reference,
            output_to_grad=op.output[0],
            outputs_to_check=[0, 1],
        )

        self.assertGradientChecks(
            device_option=hu.cpu_do,
            op=op,
            inputs=inputs,
            outputs_to_check=0,
            outputs_with_grads=[0],
            threshold=0.01,
            stepsize=0.005,
        )


    @given(n=st.integers(1, 10),
           d=st.integers(1, 10),
           t=st.integers(1, 10),
           **hu.gcs)
    def test_lstm_unit_recurrent_network(self, n, d, t, dc, gc):
        op = core.CreateOperator(
            "LSTMUnit",
            [
                "hidden_t_prev",
                "cell_t_prev",
                "gates_t",
                "seq_lengths",
                "timestep",
            ],
            ["hidden_t", "cell_t"])
        cell_t_prev = np.random.randn(1, n, d).astype(np.float32)
        hidden_t_prev = np.random.randn(1, n, d).astype(np.float32)
        gates = np.random.randn(1, n, 4 * d).astype(np.float32)
        seq_lengths = np.random.randint(1, t + 1, size=(n,)).astype(np.int32)
        timestep = np.random.randint(0, t, size=(1,)).astype(np.int32)
        inputs = [hidden_t_prev, cell_t_prev, gates, seq_lengths, timestep]
        input_device_options = {"timestep": hu.cpu_do}
        self.assertDeviceChecks(
            dc, op, inputs, [0],
            input_device_options=input_device_options)
        self.assertReferenceChecks(
            gc, op, inputs, lstm_unit,
            input_device_options=input_device_options)
        for i in range(2):
            self.assertGradientChecks(
                gc, op, inputs, i, [0, 1],
                input_device_options=input_device_options)

    @given(encoder_output_length=st.integers(1, 3),
           encoder_output_dim=st.integers(1, 3),
           decoder_input_length=st.integers(1, 3),
           decoder_state_dim=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           **hu.gcs)
    def test_lstm_with_attention(
        self,
        encoder_output_length,
        encoder_output_dim,
        decoder_input_length,
        decoder_state_dim,
        batch_size,
        gc,
        dc,
    ):
        with core.DeviceScope(gc):
            model = CNNModelHelper(name="external")
            (
                encoder_outputs,
                decoder_inputs,
                decoder_input_lengths,
                initial_decoder_hidden_state,
                initial_decoder_cell_state,
                initial_attention_weighted_encoder_context,
            ) = model.net.AddExternalInputs(
                "encoder_outputs",
                "decoder_inputs",
                "decoder_input_lengths",
                "initial_decoder_hidden_state",
                "initial_decoder_cell_state",
                "initial_attention_weighted_encoder_context",
            )
            recurrent.LSTMWithAttention(
                model=model,
                decoder_inputs=decoder_inputs,
                decoder_input_lengths=decoder_input_lengths,
                initial_decoder_hidden_state=initial_decoder_hidden_state,
                initial_decoder_cell_state=initial_decoder_cell_state,
                initial_attention_weighted_encoder_context=(
                    initial_attention_weighted_encoder_context
                ),
                encoder_output_dim=encoder_output_dim,
                encoder_outputs=encoder_outputs,
                decoder_input_dim=decoder_state_dim,
                decoder_state_dim=decoder_state_dim,
                scope='external/LSTMWithAttention',
            )
            op = model.net._net.op[-1]
        workspace.RunNetOnce(model.param_init_net)

        # This is original decoder_inputs after linear layer
        decoder_input_blob = op.input[0]

        workspace.FeedBlob(
            decoder_input_blob,
            np.random.randn(
                decoder_input_length,
                batch_size,
                decoder_state_dim * 4,
            ).astype(np.float32))
        workspace.FeedBlob(
            "external/LSTMWithAttention/encoder_outputs_transposed",
            np.random.randn(
                batch_size,
                encoder_output_dim,
                encoder_output_length,
            ).astype(np.float32),
        )
        workspace.FeedBlob(
            "external/LSTMWithAttention/weighted_encoder_outputs",
            np.random.randn(
                encoder_output_length,
                batch_size,
                encoder_output_dim,
            ).astype(np.float32),
        )
        workspace.FeedBlob(
            decoder_input_lengths,
            np.random.randint(
                0,
                decoder_input_length + 1,
                size=(batch_size,)
            ).astype(np.int32))
        workspace.FeedBlob(
            initial_decoder_hidden_state,
            np.random.randn(1, batch_size, decoder_state_dim).astype(np.float32)
        )
        workspace.FeedBlob(
            initial_decoder_cell_state,
            np.random.randn(1, batch_size, decoder_state_dim).astype(np.float32)
        )
        workspace.FeedBlob(
            initial_attention_weighted_encoder_context,
            np.random.randn(
                1, batch_size, encoder_output_dim).astype(np.float32)
        )
        inputs = [workspace.FetchBlob(name) for name in op.input]
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=lstm_with_attention_reference,
            grad_reference=None,
            output_to_grad=None,
            outputs_to_check=range(6),
        )
        gradients_to_check = [
            index for (index, input_name) in enumerate(op.input)
            if input_name != "decoder_input_lengths"
        ]
        for param in gradients_to_check:
            self.assertGradientChecks(
                device_option=gc,
                op=op,
                inputs=inputs,
                outputs_to_check=param,
                outputs_with_grads=[0, 4],
                threshold=0.01,
                stepsize=0.001,
            )

    @given(encoder_output_length=st.integers(1, 3),
           encoder_output_dim=st.integers(1, 3),
           decoder_input_length=st.integers(1, 3),
           decoder_state_dim=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           **hu.gcs)
    def test_lstm_with_recurrent_attention(
        self,
        encoder_output_length,
        encoder_output_dim,
        decoder_input_length,
        decoder_state_dim,
        batch_size,
        gc,
        dc,
    ):
        with core.DeviceScope(gc):
            model = CNNModelHelper(name="external")
            (
                encoder_outputs,
                decoder_inputs,
                decoder_input_lengths,
                initial_decoder_hidden_state,
                initial_decoder_cell_state,
                initial_attention_weighted_encoder_context,
            ) = model.net.AddExternalInputs(
                "encoder_outputs",
                "decoder_inputs",
                "decoder_input_lengths",
                "initial_decoder_hidden_state",
                "initial_decoder_cell_state",
                "initial_attention_weighted_encoder_context",
            )
            recurrent.LSTMWithAttention(
                model=model,
                decoder_inputs=decoder_inputs,
                decoder_input_lengths=decoder_input_lengths,
                initial_decoder_hidden_state=initial_decoder_hidden_state,
                initial_decoder_cell_state=initial_decoder_cell_state,
                initial_attention_weighted_encoder_context=(
                    initial_attention_weighted_encoder_context
                ),
                encoder_output_dim=encoder_output_dim,
                encoder_outputs=encoder_outputs,
                decoder_input_dim=decoder_state_dim,
                decoder_state_dim=decoder_state_dim,
                scope='external/LSTMWithAttention',
                attention_type=AttentionType.Recurrent
            )
            op = model.net._net.op[-1]
        workspace.RunNetOnce(model.param_init_net)

        # This is original decoder_inputs after linear layer
        decoder_input_blob = op.input[0]

        workspace.FeedBlob(
            decoder_input_blob,
            np.random.randn(
                decoder_input_length,
                batch_size,
                decoder_state_dim * 4,
            ).astype(np.float32))
        workspace.FeedBlob(
            "external/LSTMWithAttention/encoder_outputs_transposed",
            np.random.randn(
                batch_size,
                encoder_output_dim,
                encoder_output_length,
            ).astype(np.float32),
        )
        workspace.FeedBlob(
            "external/LSTMWithAttention/weighted_encoder_outputs",
            np.random.randn(
                encoder_output_length,
                batch_size,
                encoder_output_dim,
            ).astype(np.float32),
        )
        workspace.FeedBlob(
            decoder_input_lengths,
            np.random.randint(
                0,
                decoder_input_length + 1,
                size=(batch_size,)
            ).astype(np.int32))
        workspace.FeedBlob(
            initial_decoder_hidden_state,
            np.random.randn(1, batch_size, decoder_state_dim).astype(np.float32)
        )
        workspace.FeedBlob(
            initial_decoder_cell_state,
            np.random.randn(1, batch_size, decoder_state_dim).astype(np.float32)
        )
        workspace.FeedBlob(
            initial_attention_weighted_encoder_context,
            np.random.randn(
                1, batch_size, encoder_output_dim).astype(np.float32)
        )
        inputs = [workspace.FetchBlob(name) for name in op.input]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=lstm_with_recurrent_attention_reference,
            grad_reference=None,
            output_to_grad=None,
            outputs_to_check=range(6),
        )
        gradients_to_check = [
            index for (index, input_name) in enumerate(op.input)
            if input_name != "decoder_input_lengths"
        ]
        for param in gradients_to_check:
            self.assertGradientChecks(
                device_option=gc,
                op=op,
                inputs=inputs,
                outputs_to_check=param,
                outputs_with_grads=[0, 4],
                threshold=0.01,
                stepsize=0.001,
            )
