from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace, core, scope, gru_cell
from caffe2.python.model_helper import ModelHelper
from caffe2.python.rnn.rnn_cell_test_util import sigmoid, tanh, _prepare_rnn
import caffe2.python.hypothesis_test_util as hu
from caffe2.proto import caffe2_pb2

from functools import partial
from hypothesis import given
from hypothesis import settings as ht_settings
import hypothesis.strategies as st
import numpy as np
import unittest


def gru_unit(*args, **kwargs):
    '''
    Implements one GRU unit, for one time step

    Shapes:
    hidden_t_prev.shape     = (1, N, D)
    gates_out_t.shape       = (1, N, G)
    seq_lenths.shape        = (N,)
    '''

    drop_states = kwargs.get('drop_states', False)
    sequence_lengths = kwargs.get('sequence_lengths', True)

    if sequence_lengths:
        hidden_t_prev, gates_out_t, seq_lengths, timestep = args
    else:
        hidden_t_prev, gates_out_t, timestep = args

    N = hidden_t_prev.shape[1]
    D = hidden_t_prev.shape[2]
    G = gates_out_t.shape[2]
    t = (timestep * np.ones(shape=(N, D))).astype(np.int32)
    assert t.shape == (N, D)
    assert G == 3 * D
    # Calculate reset, update, and output gates separately
    # because output gate depends on reset gate.
    gates_out_t = gates_out_t.reshape(N, 3, D)
    reset_gate_t = gates_out_t[:, 0, :].reshape(N, D)
    update_gate_t = gates_out_t[:, 1, :].reshape(N, D)
    output_gate_t = gates_out_t[:, 2, :].reshape(N, D)

    # Calculate gate outputs.
    reset_gate_t = sigmoid(reset_gate_t)
    update_gate_t = sigmoid(update_gate_t)
    output_gate_t = tanh(output_gate_t)

    if sequence_lengths:
        seq_lengths = (np.ones(shape=(N, D)) *
                       seq_lengths.reshape(N, 1)).astype(np.int32)
        assert seq_lengths.shape == (N, D)
        valid = (t < seq_lengths).astype(np.int32)
    else:
        valid = np.ones(shape=(N, D))
    assert valid.shape == (N, D)
    hidden_t = update_gate_t * hidden_t_prev + (1 - update_gate_t) * output_gate_t
    hidden_t = hidden_t * valid + hidden_t_prev * (1 - valid) * (1 - drop_states)
    hidden_t = hidden_t.reshape(1, N, D)

    return (hidden_t, )


def gru_reference(input, hidden_input,
                   reset_gate_w, reset_gate_b,
                   update_gate_w, update_gate_b,
                   output_gate_w, output_gate_b,
                   seq_lengths, drop_states=False,
                   linear_before_reset=False):
    D = hidden_input.shape[hidden_input.ndim - 1]
    T = input.shape[0]
    N = input.shape[1]
    G = input.shape[2]
    print("Dimensions: T= ", T, " N= ", N, " G= ", G, " D= ", D)
    hidden = np.zeros(shape=(T + 1, N, D))
    hidden[0, :, :] = hidden_input

    for t in range(T):
        input_t = input[t].reshape(1, N, G)
        hidden_t_prev = hidden[t].reshape(1, N, D)

        # Split input contributions for three gates.
        input_t = input_t.reshape(N, 3, D)
        input_reset = input_t[:, 0, :].reshape(N, D)
        input_update = input_t[:, 1, :].reshape(N, D)
        input_output = input_t[:, 2, :].reshape(N, D)

        reset_gate = np.dot(hidden_t_prev, reset_gate_w.T) + reset_gate_b
        reset_gate = reset_gate + input_reset

        update_gate = np.dot(hidden_t_prev, update_gate_w.T) + update_gate_b
        update_gate = update_gate + input_update

        if linear_before_reset:
            with_linear = np.dot(hidden_t_prev, output_gate_w.T) + output_gate_b
            output_gate = sigmoid(reset_gate) * with_linear
        else:
            with_reset = hidden_t_prev * sigmoid(reset_gate)
            output_gate = np.dot(with_reset, output_gate_w.T) + output_gate_b
        output_gate = output_gate + input_output

        gates_out_t = np.concatenate(
            (reset_gate, update_gate, output_gate),
            axis=2,
        )
        print(reset_gate, update_gate, output_gate, gates_out_t, sep="\n")

        (hidden_t, ) = gru_unit(
            hidden_t_prev,
            gates_out_t,
            seq_lengths,
            t,
            drop_states=drop_states
        )
        hidden[t + 1] = hidden_t

    return (
        hidden[1:],
        hidden[-1].reshape(1, N, D),
    )


def gru_unit_op_input():
    '''
    Create input tensor where each dimension is from 1 to 4, ndim=3 and
    last dimension size is a factor of 3

    hidden_t_prev.shape     = (1, N, D)
    '''
    dims_ = st.tuples(
        st.integers(min_value=1, max_value=1),  # 1, one timestep
        st.integers(min_value=1, max_value=4),  # n
        st.integers(min_value=1, max_value=4),  # d
    )

    def create_input(dims):
        dims = list(dims)
        dims[2] *= 3
        return hu.arrays(dims)

    return dims_.flatmap(create_input)


def gru_input():
    '''
    Create input tensor where each dimension is from 1 to 4, ndim=3 and
    last dimension size is a factor of 3
    '''
    dims_ = st.tuples(
        st.integers(min_value=1, max_value=4),  # t
        st.integers(min_value=1, max_value=4),  # n
        st.integers(min_value=1, max_value=4),  # d
    )

    def create_input(dims):
        dims = list(dims)
        dims[2] *= 3
        return hu.arrays(dims)

    return dims_.flatmap(create_input)


def _prepare_gru_unit_op(gc, n, d, outputs_with_grads,
                         forward_only=False, drop_states=False,
                         sequence_lengths=False,
                         two_d_initial_states=None):
    print("Dims: (n,d) = ({},{})".format(n, d))

    def generate_input_state(n, d):
        if two_d_initial_states:
            return np.random.randn(n, d).astype(np.float32)
        else:
            return np.random.randn(1, n, d).astype(np.float32)

    model = ModelHelper(name='external')

    with scope.NameScope("test_name_scope"):
        if sequence_lengths:
            hidden_t_prev, gates_t, seq_lengths, timestep = \
                model.net.AddScopedExternalInputs(
                    "hidden_t_prev",
                    "gates_t",
                    'seq_lengths',
                    "timestep",
                )
        else:
            hidden_t_prev, gates_t, timestep = \
                model.net.AddScopedExternalInputs(
                    "hidden_t_prev",
                    "gates_t",
                    "timestep",
                )
        workspace.FeedBlob(
            hidden_t_prev,
            generate_input_state(n, d).astype(np.float32),
            device_option=gc
        )
        workspace.FeedBlob(
            gates_t,
            generate_input_state(n, 3 * d).astype(np.float32),
            device_option=gc
        )

        if sequence_lengths:
            inputs = [hidden_t_prev, gates_t, seq_lengths, timestep]
        else:
            inputs = [hidden_t_prev, gates_t, timestep]

        hidden_t = model.net.GRUUnit(
            inputs,
            ['hidden_t'],
            forget_bias=0.0,
            drop_states=drop_states,
            sequence_lengths=sequence_lengths,
        )
        model.net.AddExternalOutputs(hidden_t)
        workspace.RunNetOnce(model.param_init_net)

        if sequence_lengths:
            # 10 is used as a magic number to simulate some reasonable timestep
            # and generate some reasonable seq. lengths
            workspace.FeedBlob(
                seq_lengths,
                np.random.randint(1, 10, size=(n,)).astype(np.int32),
                device_option=gc
            )

        workspace.FeedBlob(
            timestep,
            np.random.randint(1, 10, size=(1,)).astype(np.int32),
            device_option=core.DeviceOption(caffe2_pb2.CPU),
        )
        print("Feed {}".format(timestep))

    return hidden_t, model.net


class GRUCellTest(hu.HypothesisTestCase):

    # Test just for GRUUnitOp
    @given(
        seed=st.integers(0, 2**32 - 1),
        input_tensor=gru_unit_op_input(),
        fwd_only=st.booleans(),
        drop_states=st.booleans(),
        sequence_lengths=st.booleans(),
        **hu.gcs
    )
    @ht_settings(max_examples=15)
    def test_gru_unit_op(self, seed, input_tensor, fwd_only,
                         drop_states, sequence_lengths, gc, dc):
        np.random.seed(seed)
        outputs_with_grads = [0]
        ref = gru_unit
        ref = partial(ref)

        t, n, d = input_tensor.shape
        assert d % 3 == 0
        d = d // 3
        ref = partial(ref, drop_states=drop_states,
                      sequence_lengths=sequence_lengths)

        with core.DeviceScope(gc):
            net = _prepare_gru_unit_op(gc, n, d,
                                       outputs_with_grads=outputs_with_grads,
                                       forward_only=fwd_only,
                                       drop_states=drop_states,
                                       sequence_lengths=sequence_lengths)[1]
        # here we don't provide a real input for the net but just for one of
        # its ops (RecurrentNetworkOp). So have to hardcode this name
        workspace.FeedBlob("test_name_scope/external/recurrent/i2h",
                           input_tensor,
                           device_option=gc)
        print(str(net.Proto()))
        op = net._net.op[-1]
        inputs = [workspace.FetchBlob(name) for name in op.input]

        self.assertReferenceChecks(
            gc,
            op,
            inputs,
            ref,
            input_device_options={op.input[-1]: hu.cpu_do},
            outputs_to_check=[0],
        )

        # Checking for hidden_prev and gates gradients
        if not fwd_only:
            for param in range(2):
                print("Check param {}".format(param))
                self.assertGradientChecks(
                    device_option=gc,
                    op=op,
                    inputs=inputs,
                    outputs_to_check=param,
                    outputs_with_grads=outputs_with_grads,
                    threshold=0.0001,
                    stepsize=0.005,
                    input_device_options={op.input[-1]: hu.cpu_do},
                )

    @given(
        seed=st.integers(0, 2**32 - 1),
        input_tensor=gru_input(),
        fwd_only=st.booleans(),
        drop_states=st.booleans(),
        linear_before_reset=st.booleans(),
        **hu.gcs
    )
    @ht_settings(max_examples=20)
    def test_gru_main(self, seed, **kwargs):
        np.random.seed(seed)
        for outputs_with_grads in [[0], [1], [0, 1]]:
            self.gru_base(gru_cell.GRU, gru_reference,
                           outputs_with_grads=outputs_with_grads,
                           **kwargs)

    def gru_base(self, create_rnn, ref, outputs_with_grads,
                  input_tensor, fwd_only, drop_states, linear_before_reset, gc, dc):

        print("GRU test parameters: ", locals())
        t, n, d = input_tensor.shape
        assert d % 3 == 0
        d = d // 3
        ref = partial(ref,
                      drop_states=drop_states,
                      linear_before_reset=linear_before_reset)
        with core.DeviceScope(gc):
            net = _prepare_rnn(
                t, n, d, create_rnn,
                outputs_with_grads=outputs_with_grads,
                memory_optim=False,
                forget_bias=0.0,
                forward_only=fwd_only,
                drop_states=drop_states,
                linear_before_reset=linear_before_reset,
                num_states=1,
            )[1]
        # here we don't provide a real input for the net but just for one of
        # its ops (RecurrentNetworkOp). So have to hardcode this name
        workspace.FeedBlob("test_name_scope/external/recurrent/i2h",
                           input_tensor,
                           device_option=gc)
        op = net._net.op[-1]
        inputs = [workspace.FetchBlob(name) for name in op.input]

        self.assertReferenceChecks(
            gc,
            op,
            inputs,
            ref,
            input_device_options={"timestep": hu.cpu_do},
            outputs_to_check=list(range(2)),
        )

        # Checking for input, gates_t_w and gates_t_b gradients
        if not fwd_only:
            for param in range(2):
                print("Check param {}".format(param))
                self.assertGradientChecks(
                    device_option=gc,
                    op=op,
                    inputs=inputs,
                    outputs_to_check=param,
                    outputs_with_grads=outputs_with_grads,
                    threshold=0.001,
                    stepsize=0.005,
                    input_device_options={"timestep": hu.cpu_do},
                )


if __name__ == "__main__":
    workspace.GlobalInit([
        'caffe2',
        '--caffe2_log_level=0',
    ])
    unittest.main()
