from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, recurrent, workspace
from caffe2.python.model_helper import ModelHelperBase
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return 2.0 * sigmoid(2.0 * x) - 1


def lstm_unit(cell_t_prev, gates, seq_lengths, timestep):
    D = cell_t_prev.shape[2]
    G = gates.shape[2]
    N = gates.shape[1]
    t = (timestep[0].reshape(1, 1) * np.ones(shape=(N, D))).astype(np.int32)
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
    hidden_t = (o_t * tanh(cell_t)) * valid
    hidden_t = hidden_t.reshape(1, N, D)
    cell_t = cell_t.reshape(1, N, D)
    return hidden_t, cell_t


class RecurrentNetworkTest(hu.HypothesisTestCase):

    @given(t=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_lstm(self, t, n, d):
        model = ModelHelperBase(name='external')

        input_blob, seq_lengths, hidden_init, cell_init = (
            model.net.AddExternalInputs(
                'input_blob', 'seq_lengths', 'hidden_init', 'cell_init'))

        recurrent.LSTM(
            model, input_blob, seq_lengths, (hidden_init, cell_init),
            d, d, scope="external/recurrent")

        op = model.net._net.op[-1]

        def extract_param_name(model, param_substr):
            result = []
            for p in model.params:
                if param_substr in str(p):
                    result.append(str(p))

            assert len(result) == 1
            return result[0]

        gates = {gate: extract_param_name(model, gate)
                 for gate in ["gates_t_b", "gates_t_w"]}
        workspace.RunNetOnce(model.param_init_net)

        def reference(input, hidden_input, cell_input,
                      gates_w, gates_b, seq_lengths):
            T = input.shape[0]
            N = input.shape[1]
            G = input.shape[2]
            D = hidden_input.shape[2]
            hidden = np.zeros(shape=(T + 1, N, D))
            cell = np.zeros(shape=(T + 1, N, D))
            assert hidden.shape[0] == T + 1
            assert cell.shape[0] == T + 1
            assert hidden.shape[1] == N
            assert cell.shape[1] == N
            cell[0, :, :] = cell_input
            hidden[0, :, :] = hidden_input
            for t in range(T):
                timestep = np.asarray([t]).astype(np.int32)
                input_t = input[t].reshape(1, N, G)
                hidden_t_prev = hidden[t].reshape(1, N, D)
                cell_t_prev = cell[t].reshape(1, N, D)
                gates = np.dot(hidden_t_prev, gates_w.T) + gates_b
                gates = gates + input_t
                hidden_t, cell_t = lstm_unit(cell_t_prev, gates, seq_lengths,
                                             timestep)
                hidden[t + 1] = hidden_t
                cell[t + 1] = cell_t
            return (
                hidden[1:],
                hidden[-1].reshape(1, N, D),
                cell[1:],
                cell[-1].reshape(1, N, D)
            )

        input_blob = op.input[0]

        workspace.FeedBlob(
            str(input_blob), np.random.randn(t, n, d * 4).astype(np.float32))
        workspace.FeedBlob(
            "hidden_init", np.random.randn(1, n, d).astype(np.float32))
        workspace.FeedBlob(
            "cell_init", np.random.randn(1, n, d).astype(np.float32))
        workspace.FeedBlob(
            "seq_lengths", np.random.randint(0, t, size=(n,)).astype(np.int32))

        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [
                workspace.FetchBlob(name)
                for name in [
                    input_blob,
                    "hidden_init", "cell_init",
                    gates["gates_t_w"],
                    gates["gates_t_b"],
                    "seq_lengths"
                ]
            ],
            reference,
        )

        # Checking for input, gates_t_w and gates_t_b gradients
        for param in [0, 3, 4]:
            self.assertGradientChecks(
                hu.cpu_do,
                op,
                [
                    workspace.FetchBlob(name)
                    for name in [
                        input_blob,
                        "hidden_init", "cell_init",
                        gates["gates_t_w"],
                        gates["gates_t_b"],
                        "seq_lengths"
                    ]
                ],
                param,
                [0],
                threshold=0.01,
            )

    @given(t=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_mul_rnn(self, t, n, d):
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
            initial_cell_inputs=[(output_t_prev, one_blob, d)],
            links={output_t_prev: output_t},
            scratch_sizes=[],
            scope="test_mul_rnn",
        )

        workspace.FeedBlob(
            str(input_blob), np.random.randn(t, n, d).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)

        op = model.net._net.op[-1]

        def reference(input, initial_input):
            T = input.shape[0]
            recurrent_input = initial_input
            result = np.zeros(shape=input.shape)

            for t in range(T):
                recurrent_input = recurrent_input * input[t]
                result[t] = recurrent_input

            shape = list(input.shape)
            shape[0] = 1
            return (result, result[-1].reshape(shape))

        def grad_reference(output_grad, ref_output, inputs):
            input = inputs[0]
            output = ref_output[0]
            initial_input = inputs[1]
            input_grad = np.zeros(shape=input.shape)
            T = input.shape[0]
            right_grad = 0

            for t in range(T - 1, -1, -1):
                prev_output = output[t - 1] if t > 0 else initial_input
                input_grad[t] = (output_grad[t] + right_grad) * prev_output
                right_grad = input[t] * (output_grad[t] + right_grad)

            return (input_grad, [0.])

        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [workspace.FetchBlob(name)
             for name in [input_blob, one_blob]],
            reference,
            grad_reference=grad_reference,
            output_to_grad=op.output[0],
        )

    @given(n=st.integers(1, 10),
           d=st.integers(1, 10),
           t=st.integers(1, 10),
           **hu.gcs)
    def test_lstm_unit_recurrent_network(self, n, d, t, dc, gc):
        op = core.CreateOperator(
            "LSTMUnit",
            ["cell_t_prev", "gates_t", "seq_lengths", "timestep"],
            ["hidden_t", "cell_t"])
        cell_t_prev = np.random.randn(1, n, d).astype(np.float32)
        gates = np.random.randn(1, n, 4 * d).astype(np.float32)
        seq_lengths = np.random.randint(0, t, size=(n,)).astype(np.int32)
        timestep = np.random.randint(0, t, size=(1,)).astype(np.int32)
        inputs = [cell_t_prev, gates, seq_lengths, timestep]
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
