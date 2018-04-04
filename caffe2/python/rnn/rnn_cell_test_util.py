from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace, scope
from caffe2.python.model_helper import ModelHelper

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return 2.0 * sigmoid(2.0 * x) - 1


def _prepare_rnn(
    t, n, dim_in, create_rnn, outputs_with_grads,
    forget_bias, memory_optim=False,
    forward_only=False, drop_states=False, T=None,
    two_d_initial_states=None, dim_out=None,
    num_states=2,
    **kwargs
):
    if dim_out is None:
        dim_out = [dim_in]
    print("Dims: ", t, n, dim_in, dim_out)

    model = ModelHelper(name='external')

    if two_d_initial_states is None:
        two_d_initial_states = np.random.randint(2)

    def generate_input_state(n, d):
        if two_d_initial_states:
            return np.random.randn(n, d).astype(np.float32)
        else:
            return np.random.randn(1, n, d).astype(np.float32)

    states = []
    for layer_id, d in enumerate(dim_out):
        for i in range(num_states):
            state_name = "state_{}/layer_{}".format(i, layer_id)
            states.append(model.net.AddExternalInput(state_name))
            workspace.FeedBlob(
                states[-1], generate_input_state(n, d).astype(np.float32))

    # Due to convoluted RNN scoping logic we make sure that things
    # work from a namescope
    with scope.NameScope("test_name_scope"):
        input_blob, seq_lengths = model.net.AddScopedExternalInputs(
            'input_blob', 'seq_lengths')

        outputs = create_rnn(
            model, input_blob, seq_lengths, states,
            dim_in=dim_in, dim_out=dim_out, scope="external/recurrent",
            outputs_with_grads=outputs_with_grads,
            memory_optimization=memory_optim,
            forget_bias=forget_bias,
            forward_only=forward_only,
            drop_states=drop_states,
            static_rnn_unroll_size=T,
            **kwargs
        )

    workspace.RunNetOnce(model.param_init_net)

    workspace.FeedBlob(
        seq_lengths,
        np.random.randint(1, t + 1, size=(n,)).astype(np.int32)
    )
    return outputs, model.net, states + [input_blob]
