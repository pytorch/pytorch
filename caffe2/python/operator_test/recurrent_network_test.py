from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, recurrent, workspace
from caffe2.python.utils import debug
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


def old_lstm_reference(
        input, seq_lengths, gates_w, gates_b, hidden_init, cell_init):
    output, last_output, cell_states, last_state = lstm_reference(
        input, hidden_init, cell_init, gates_w, gates_b, seq_lengths)
    return (output, last_output, last_state)


class RecurrentNetworkTest(hu.HypothesisTestCase):

    @given(t=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_lstm_new(self, t, n, d):
        for outputs_with_grads in [[0], [1], [0, 1, 2, 3]]:
            model = CNNModelHelper(name='external')

            def create_lstm(
                    model, input_blob, seq_lengths,
                    init, dim_in, dim_out, scope):
                recurrent.LSTM(
                    model, input_blob, seq_lengths, init,
                    dim_in, dim_out, scope="external/recurrent",
                    outputs_with_grads=outputs_with_grads)

            self.lstm(model, create_lstm, t, n, d, lstm_reference,
                      gradients_to_check=[0, 1, 2, 3, 4],
                      outputs_to_check=[0, 1, 2, 3],
                      outputs_with_grads=outputs_with_grads)

    @given(t=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_lstm_old(self, t, n, d):
        model = CNNModelHelper(name='external')

        def create_lstm(
                model, input_blob, seq_lengths, init, dim_in, dim_out, scope):
            model.LSTM(
                input_blob, seq_lengths, init,
                dim_in, dim_out, scope="external/recurrent")

        # CNNModelHelper.LSTM returns only 3 outputs. But the operator itself
        # returns 5. We ignore the rest.
        self.lstm(model, create_lstm, t, n, d, old_lstm_reference,
                  gradients_to_check=[0, 2, 3, 4, 5],
                  outputs_to_check=[0, 3, 4])

    @debug
    def lstm(self, model, create_lstm, t, n, d, ref, gradients_to_check,
             outputs_to_check=None, outputs_with_grads=(0,)):
        input_blob, seq_lengths, hidden_init, cell_init = (
            model.net.AddExternalInputs(
                'input_blob', 'seq_lengths', 'hidden_init', 'cell_init'))

        create_lstm(
            model, input_blob, seq_lengths, (hidden_init, cell_init),
            d, d, scope="external/recurrent")

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
            "seq_lengths", np.random.randint(1, t + 1, size=(n,)).astype(np.int32))
        inputs = [workspace.FetchBlob(name) for name in op.input]

        print(op.input)
        print(inputs)

        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            inputs,
            ref,
            outputs_to_check=outputs_to_check,
        )

        # Checking for input, gates_t_w and gates_t_b gradients
        for param in gradients_to_check:
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
    def test_mul_rnn(self, T, n, d):
        model = ModelHelperBase(name='external')

        one_blob = model.param_init_net.ConstantFill(
            [], value=1.0, shape=[1, n, d])
        input_blob = model.net.AddExternalInput('input')

        step = ModelHelperBase(name='step', param_model=model)
        input_t, output_t_prev = step.net.AddExternalInput(
            'input_t', 'output_t_prev')
        output_t = step.net.Mul([input_t, output_t_prev])
        step.net.AddExternalOutput(output_t)

        recurrent.recurrent_net(
            net=model.net,
            cell_net=step.net,
            inputs=[(input_t, input_blob)],
            initial_cell_inputs=[(output_t_prev, one_blob)],
            links={output_t_prev: output_t},
            scope="test_mul_rnn",
        )

        workspace.FeedBlob(
            str(input_blob), np.random.randn(T, n, d).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)

        op = model.net._net.op[-1]

        def reference(input, initial_input):
            recurrent_input = initial_input
            result = np.zeros(shape=input.shape)

            for t_cur in range(T):
                recurrent_input = recurrent_input * input[t_cur]
                result[t_cur] = recurrent_input

            shape = list(input.shape)
            shape[0] = 1
            return (result, result[-1].reshape(shape))

        def grad_reference(output_grad, ref_output, inputs):
            input = inputs[0]
            output = ref_output[0]
            initial_input = inputs[1]
            input_grad = np.zeros(shape=input.shape)
            right_grad = 0

            for t_cur in range(T - 1, -1, -1):
                prev_output = output[t_cur - 1] if t_cur > 0 else initial_input
                input_grad[t_cur] = (output_grad[t_cur] +
                                     right_grad) * prev_output
                right_grad = input[t_cur] * (output_grad[t_cur] + right_grad)
            return (input_grad, right_grad.reshape([1, n, d]))

        self.assertReferenceChecks(
            device_option=hu.cpu_do,
            op=op,
            inputs=[
                workspace.FetchBlob(name)
                for name in [input_blob, one_blob]
            ],
            reference=reference,
            grad_reference=grad_reference,
            output_to_grad=op.output[0],
            outputs_to_check=[0, 1],
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
