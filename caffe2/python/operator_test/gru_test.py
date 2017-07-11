from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace, scope
from caffe2.python.model_helper import ModelHelper
from caffe2.python.rnn.rnn_cell_test_util import sigmoid, tanh
import caffe2.python.hypothesis_test_util as hu


from functools import partial
from hypothesis import given
from hypothesis import settings as ht_settings
import hypothesis.strategies as st
import numpy as np


def gru_unit(hidden_t_prev, gates_out_t,
             seq_lengths, timestep, drop_states=False):
    '''
    Implements one GRU unit, for one time step

    Shapes:
    hidden_t_prev.shape     = (1, N, D)
    gates_out_t.shape       = (1, N, G)
    seq_lenths.shape        = (N,)
    '''
    N = hidden_t_prev.shape[1]
    D = hidden_t_prev.shape[2]
    G = gates_out_t.shape[2]
    t = (timestep * np.ones(shape=(N, D))).astype(np.int32)
    assert t.shape == (N, D)
    seq_lengths = (np.ones(shape=(N, D)) *
                   seq_lengths.reshape(N, 1)).astype(np.int32)
    assert seq_lengths.shape == (N, D)
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

    valid = (t < seq_lengths).astype(np.int32)
    assert valid.shape == (N, D)
    hidden_t = update_gate_t * hidden_t_prev + (1 - update_gate_t) * output_gate_t
    hidden_t = hidden_t * valid + hidden_t_prev * (1 - valid) * (1 - drop_states)
    hidden_t = hidden_t.reshape(1, N, D)

    return (hidden_t, )


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


def _prepare_gru_unit_op(n, d, outputs_with_grads,
                         forward_only=False, drop_states=False,
                         two_d_initial_states=None):
    print("Dims: (n,d) = ({},{})".format(n, d))

    def generate_input_state(n, d):
        if two_d_initial_states:
            return np.random.randn(n, d).astype(np.float32)
        else:
            return np.random.randn(1, n, d).astype(np.float32)

    model = ModelHelper(name='external')

    with scope.NameScope("test_name_scope"):
        hidden_t_prev, gates_t, seq_lengths, timestep = \
            model.net.AddScopedExternalInputs(
                "hidden_t_prev",
                "gates_t",
                'seq_lengths',
                "timestep",
            )
        workspace.FeedBlob(
            hidden_t_prev,
            generate_input_state(n, d).astype(np.float32)
        )
        workspace.FeedBlob(
            gates_t,
            generate_input_state(n, 3 * d).astype(np.float32)
        )

        hidden_t = model.net.GRUUnit(
            [
                hidden_t_prev,
                gates_t,
                seq_lengths,
                timestep,
            ],
            ['hidden_t'],
            forget_bias=0.0,
            drop_states=drop_states,
        )
        model.net.AddExternalOutputs(hidden_t)
        workspace.RunNetOnce(model.param_init_net)

        # 10 is used as a magic number to simulate some reasonable timestep
        # and generate some reasonable seq. lengths
        workspace.FeedBlob(
            seq_lengths,
            np.random.randint(1, 10, size=(n,)).astype(np.int32)
        )
        workspace.FeedBlob(
            timestep,
            np.random.randint(1, 10, size=(1,)).astype(np.int32)
        )

    return hidden_t, model.net


class GRUCellTest(hu.HypothesisTestCase):

    # Test just for GRUUnitOp
    @given(
        input_tensor=gru_unit_op_input(),
        fwd_only=st.booleans(),
        drop_states=st.booleans(),
    )
    @ht_settings(max_examples=15)
    def test_gru_unit_op(self, input_tensor, fwd_only, drop_states, **kwargs):
        outputs_with_grads = [0]
        ref = gru_unit
        ref = partial(ref)

        t, n, d = input_tensor.shape
        assert d % 3 == 0
        d = d // 3
        ref = partial(ref, drop_states=drop_states)

        net = _prepare_gru_unit_op(n, d,
                                   outputs_with_grads=outputs_with_grads,
                                   forward_only=fwd_only,
                                   drop_states=drop_states)[1]
        # here we don't provide a real input for the net but just for one of
        # its ops (RecurrentNetworkOp). So have to hardcode this name
        workspace.FeedBlob("test_name_scope/external/recurrent/i2h",
                           input_tensor)
        op = net._net.op[-1]
        inputs = [workspace.FetchBlob(name) for name in op.input]

        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            inputs,
            ref,
            outputs_to_check=[0],
        )

        # Checking for hidden_prev and gates gradients
        if not fwd_only:
            for param in range(2):
                print("Check param {}".format(param))
                self.assertGradientChecks(
                    device_option=hu.cpu_do,
                    op=op,
                    inputs=inputs,
                    outputs_to_check=param,
                    outputs_with_grads=outputs_with_grads,
                    threshold=0.0001,
                    stepsize=0.005,
                )
