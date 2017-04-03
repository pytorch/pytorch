## @package recurrent
# Module caffe2.python.recurrent
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.python.scope import CurrentNameScope
from caffe2.python.cnn import CNNModelHelper
from caffe2.python.attention import (
    apply_regular_attention,
    apply_recurrent_attention,
    AttentionType,
)

_workspace_seq = 0

def recurrent_net(
        net, cell_net, inputs, initial_cell_inputs,
        links, timestep=None, scope=None, outputs_with_grads=(0,)
):
    '''
    net: the main net operator should be added to

    cell_net: cell_net which is executed in a recurrent fasion

    inputs: sequences to be fed into the recurrent net. Currently only one input
    is supported. It has to be in a format T x N x (D1...Dk) where T is lengths
    of the sequence. N is a batch size and (D1...Dk) are the rest of dimentions

    initial_cell_inputs: inputs of the cell_net for the 0 timestamp.
    Format for each input is:
        (cell_net_input_name, external_blob_with_data)

    links: a dictionary from cell_net input names in moment t+1 and
    output names of moment t. Currently we assume that each output becomes
    an input for the next timestep.

    timestep: name of the timestep blob to be used. If not provided "timestep"
    is used.

    scope: Internal blobs are going to be scoped in a format
    <scope_name>/<blob_name>
    If not provided we generate a scope name automatically

    outputs_with_grads : position indices of output blobs which will receive
    error gradient (from outside recurrent network) during backpropagation
    '''
    assert len(inputs) == 1, "Only one input blob is supported so far"

    # Validate scoping
    for einp in cell_net.Proto().external_input:
        assert einp.startswith(CurrentNameScope()), \
            '''
            Cell net external inputs are not properly scoped, use
            AddScopedExternalInputs() when creating them
            '''

    input_blobs = [str(i[0]) for i in inputs]
    initial_input_blobs = [str(x[1]) for x in initial_cell_inputs]
    op_name = net.NextName('recurrent')

    def s(name):
        # We have to manually scope due to our internal/external blob
        # relationships.
        scope_name = op_name if scope is None else scope
        return "{}/{}".format(str(scope_name), str(name))

    # determine inputs that are considered to be references
    # it is those that are not referred to in inputs or initial_cell_inputs
    known_inputs = map(str, input_blobs + initial_input_blobs)
    known_inputs += [str(x[0]) for x in initial_cell_inputs]
    if timestep is not None:
        known_inputs.append(str(timestep))
    references = [
        core.BlobReference(b) for b in cell_net.Proto().external_input
        if b not in known_inputs]

    inner_outputs = list(cell_net.Proto().external_output)
    # These gradients are expected to be available during the backward pass
    inner_outputs_map = {o: o + '_grad' for o in inner_outputs}

    # compute the backward pass of the cell net
    backward_ops, backward_mapping = core.GradientRegistry.GetBackwardPass(
        cell_net.Proto().op, inner_outputs_map)
    backward_mapping = {str(k): v for k, v in backward_mapping.items()}
    backward_cell_net = core.Net("RecurrentBackwardStep")

    del backward_cell_net.Proto().op[:]
    backward_cell_net.Proto().op.extend(backward_ops)
    # compute blobs used but not defined in the backward pass
    backward_ssa, backward_blob_versions = core.get_ssa(
        backward_cell_net.Proto())
    undefined = core.get_undefined_blobs(backward_ssa)

    # also add to the output list the intermediate outputs of fwd_step that
    # are used by backward.
    ssa, blob_versions = core.get_ssa(cell_net.Proto())
    scratches = [
        blob for (blob, ver) in blob_versions.items()
        if ver > 0 and
        blob in undefined and
        blob not in cell_net.Proto().external_output]
    backward_cell_net.Proto().external_input.extend(scratches)

    all_inputs = [i[1] for i in inputs] + [
        x[1] for x in initial_cell_inputs] + references
    all_outputs = []

    cell_net.Proto().type = 'simple'
    backward_cell_net.Proto().type = 'simple'

    # Internal arguments used by RecurrentNetwork operator

    # Links are in the format blob_name, recurrent_states, offset.
    # In the moment t we know that corresponding data block is at
    # t + offset position in the recurrent_states tensor
    forward_links = []
    backward_links = []

    # Aliases are used to expose outputs to external world
    # Format (internal_blob, external_blob, offset)
    # Negative offset stands for going from the end,
    # positive - from the beginning
    aliases = []

    # States held inputs to the cell net
    recurrent_states = []

    for cell_input, _ in initial_cell_inputs:
        cell_input = str(cell_input)
        # Recurrent_states is going to be (T + 1) x ...
        # It stores all inputs and outputs of the cell net over time.
        # Or their gradients in the case of the backward pass.
        state = s(cell_input + "_states")
        states_grad = state + "_grad"
        cell_output = links[str(cell_input)]
        forward_links.append((cell_input, state, 0))
        forward_links.append((cell_output, state, 1))
        backward_links.append((cell_output + "_grad", states_grad, 1))

        backward_cell_net.Proto().external_input.append(
            str(cell_output) + "_grad")
        aliases.append((state, cell_output + "_all", 1))
        aliases.append((state, cell_output + "_last", -1))
        all_outputs.extend([cell_output + "_all", cell_output + "_last"])

        recurrent_states.append(state)

        recurrent_input_grad = cell_input + "_grad"
        if not backward_blob_versions.get(recurrent_input_grad, 0):
            # If nobody writes to this recurrent input gradient, we need
            # to make sure it gets to the states grad blob after all.
            # We do this by using backward_links which triggers an alias
            # This logic is being used for example in a SumOp case
            backward_links.append(
                (backward_mapping[cell_input], states_grad, 0))
        else:
            backward_links.append((cell_input + "_grad", states_grad, 0))

    for reference in references:
        # Similar to above, in a case of a SumOp we need to write our parameter
        # gradient to an external blob. In this case we can be sure that
        # reference + "_grad" is a correct parameter name as we know how
        # RecurrentNetworkOp gradient schema looks like.
        reference_grad = reference + "_grad"
        if (reference in backward_mapping and
                reference_grad != str(backward_mapping[reference])):
            # We can use an Alias because after each timestep
            # RNN op adds value from reference_grad into and _acc blob
            # which accumulates gradients for corresponding parameter accross
            # timesteps. Then in the end of RNN op these two are being
            # swaped and reference_grad blob becomes a real blob instead of
            # being an alias
            backward_cell_net.Alias(
                backward_mapping[reference], reference_grad)

    for input_t, input_blob in inputs:
        forward_links.append((str(input_t), str(input_blob), 0))
        backward_links.append((
            backward_mapping[str(input_t)], str(input_blob) + "_grad", 0
        ))
    backward_cell_net.Proto().external_input.extend(
        cell_net.Proto().external_input)
    backward_cell_net.Proto().external_input.extend(
        cell_net.Proto().external_output)

    def unpack_triple(x):
        if x:
            a, b, c = zip(*x)
            return a, b, c
        return [], [], []

    # Splitting to separate lists so we can pass them to c++
    # where we ensemle them back
    link_internal, link_external, link_offset = unpack_triple(forward_links)
    backward_link_internal, backward_link_external, backward_link_offset = \
        unpack_triple(backward_links)
    alias_src, alias_dst, alias_offset = unpack_triple(aliases)

    params = [x for x in references if x in backward_mapping.keys()]
    recurrent_inputs = [str(x[1]) for x in initial_cell_inputs]

    global _workspace_seq
    results = net.RecurrentNetwork(
        all_inputs,
        all_outputs + [s("step_workspaces_{}".format(_workspace_seq))],
        param=map(all_inputs.index, params),
        alias_src=alias_src,
        alias_dst=map(str, alias_dst),
        alias_offset=alias_offset,
        recurrent_states=recurrent_states,
        initial_recurrent_state_ids=map(all_inputs.index, recurrent_inputs),
        link_internal=map(str, link_internal),
        link_external=map(str, link_external),
        link_offset=link_offset,
        backward_link_internal=map(str, backward_link_internal),
        backward_link_external=map(str, backward_link_external),
        backward_link_offset=backward_link_offset,
        step_net=str(cell_net.Proto()),
        backward_step_net=str(backward_cell_net.Proto()),
        timestep="timestep" if timestep is None else str(timestep),
        outputs_with_grads=outputs_with_grads,
    )
    _workspace_seq += 1
    # The last output is a list of step workspaces,
    # which is only needed internally for gradient propogation
    return results[:-1]


def LSTM(model, input_blob, seq_lengths, initial_states, dim_in, dim_out,
         scope, outputs_with_grads=(0,)):
    '''
    Adds a standard LSTM recurrent network operator to a model.

    model: CNNModelHelper object new operators would be added to

    input_blob: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimention

    seq_lengths: blob containing sequence lengths which would be passed to
    LSTMUnit operator

    initial_states: a tupple of (hidden_input_blob, cell_input_blob)
    which are going to be inputs to the cell net on the first iteration

    dim_in: input dimention

    dim_out: output dimention

    outputs_with_grads : position indices of output blobs which will receive
    external error gradient during backpropagation
    '''
    def s(name):
        # We have to manually scope due to our internal/external blob
        # relationships.
        return "{}/{}".format(str(scope), str(name))

    """ initial bulk fully-connected """
    input_blob = model.FC(
        input_blob, s('i2h'), dim_in=dim_in, dim_out=4 * dim_out, axis=2)

    """ the step net """
    step_model = CNNModelHelper(name='lstm_cell', param_model=model)
    input_t, timestep, cell_t_prev, hidden_t_prev = (
        step_model.net.AddScopedExternalInputs(
            'input_t', 'timestep', 'cell_t_prev', 'hidden_t_prev'))
    gates_t = step_model.FC(
        hidden_t_prev, s('gates_t'), dim_in=dim_out,
        dim_out=4 * dim_out, axis=2)
    step_model.net.Sum([gates_t, input_t], gates_t)
    hidden_t, cell_t = step_model.net.LSTMUnit(
        [hidden_t_prev, cell_t_prev, gates_t, seq_lengths, timestep],
        [s('hidden_t'), s('cell_t')],
    )
    step_model.net.AddExternalOutputs(cell_t, hidden_t)

    """ recurrent network """
    (hidden_input_blob, cell_input_blob) = initial_states
    output, last_output, all_states, last_state = recurrent_net(
        net=model.net,
        cell_net=step_model.net,
        inputs=[(input_t, input_blob)],
        initial_cell_inputs=[
            (hidden_t_prev, hidden_input_blob),
            (cell_t_prev, cell_input_blob),
        ],
        links={
            hidden_t_prev: hidden_t,
            cell_t_prev: cell_t,
        },
        timestep=timestep,
        scope=scope,
        outputs_with_grads=outputs_with_grads,
    )
    return output, last_output, all_states, last_state


def LSTMWithAttention(
    model,
    decoder_inputs,
    decoder_input_lengths,
    initial_decoder_hidden_state,
    initial_decoder_cell_state,
    initial_attention_weighted_encoder_context,
    encoder_output_dim,
    encoder_outputs,
    decoder_input_dim,
    decoder_state_dim,
    scope,
    attention_type=AttentionType.Regular,
    outputs_with_grads=(0, 4),
    weighted_encoder_outputs=None,
):
    '''
    Adds a LSTM with attention mechanism to a model.

    The implementation is based on https://arxiv.org/abs/1409.0473, with
    a small difference in the order
    how we compute new attention context and new hidden state, similarly to
    https://arxiv.org/abs/1508.04025.

    The model uses encoder-decoder naming conventions,
    where the decoder is the sequence the op is iterating over,
    while computing the attention context over the encoder.

    model: CNNModelHelper object new operators would be added to

    decoder_inputs: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimention

    decoder_input_lengths: blob containing sequence lengths
    which would be passed to LSTMUnit operator

    initial_decoder_hidden_state: initial hidden state of LSTM

    initial_decoder_cell_state: initial cell state of LSTM

    initial_attention_weighted_encoder_context: initial attention context

    encoder_output_dim: dimension of encoder outputs

    encoder_outputs: the sequence, on which we compute the attention context
    at every iteration

    decoder_input_dim: input dimention (last dimension on decoder_inputs)

    decoder_state_dim: size of hidden states of LSTM

    attention_type: One of: AttentionType.Regular, AttentionType.Recurrent.
    Determines which type of attention mechanism to use.

    outputs_with_grads : position indices of output blobs which will receive
    external error gradient during backpropagation

    weighted_encoder_outputs: encoder outputs to be used to compute attention
    weights. In the basic case it's just linear transformation of
    encoder outputs (that the default, when weighted_encoder_outputs is None).
    However, it can be something more complicated - like a separate
    encoder network (for example, in case of convolutional encoder)
    '''

    def s(name):
        # We have to manually scope due to our internal/external blob
        # relationships.
        return "{}/{}".format(str(scope), str(name))

    decoder_inputs = model.FC(
        decoder_inputs,
        s('i2h'),
        dim_in=decoder_input_dim,
        dim_out=4 * decoder_state_dim,
        axis=2,
    )
    # [batch_size, encoder_output_dim, encoder_length]
    encoder_outputs_transposed = model.Transpose(
        encoder_outputs,
        s('encoder_outputs_transposed'),
        axes=[1, 2, 0],
    )
    if weighted_encoder_outputs is None:
        weighted_encoder_outputs = model.FC(
            encoder_outputs,
            s('weighted_encoder_outputs'),
            dim_in=encoder_output_dim,
            dim_out=encoder_output_dim,
            axis=2,
        )
    step_model = CNNModelHelper(
        name='lstm_with_attention_cell',
        param_model=model,
    )
    (
        input_t,
        timestep,
        cell_t_prev,
        hidden_t_prev,
        attention_weighted_encoder_context_t_prev,
    ) = (
        step_model.net.AddScopedExternalInputs(
            'input_t',
            'timestep',
            'cell_t_prev',
            'hidden_t_prev',
            'attention_weighted_encoder_context_t_prev',
        )
    )
    step_model.net.AddExternalInputs(
        encoder_outputs_transposed,
        weighted_encoder_outputs
    )

    gates_concatenated_input_t, _ = step_model.net.Concat(
        [hidden_t_prev, attention_weighted_encoder_context_t_prev],
        [
            s('gates_concatenated_input_t'),
            s('_gates_concatenated_input_t_concat_dims'),
        ],
        axis=2,
    )
    gates_t = step_model.FC(
        gates_concatenated_input_t,
        s('gates_t'),
        dim_in=decoder_state_dim + encoder_output_dim,
        dim_out=4 * decoder_state_dim,
        axis=2,
    )
    step_model.net.Sum([gates_t, input_t], gates_t)

    hidden_t_intermediate, cell_t = step_model.net.LSTMUnit(
        [hidden_t_prev, cell_t_prev, gates_t, decoder_input_lengths, timestep],
        ['hidden_t_intermediate', s('cell_t')],
    )
    if attention_type == AttentionType.Recurrent:
        attention_weighted_encoder_context_t, _ = apply_recurrent_attention(
            model=step_model,
            encoder_output_dim=encoder_output_dim,
            encoder_outputs_transposed=encoder_outputs_transposed,
            weighted_encoder_outputs=weighted_encoder_outputs,
            decoder_hidden_state_t=hidden_t_intermediate,
            decoder_hidden_state_dim=decoder_state_dim,
            scope=scope,
            attention_weighted_encoder_context_t_prev=(
                attention_weighted_encoder_context_t_prev
            ),
        )
    else:
        attention_weighted_encoder_context_t, _ = apply_regular_attention(
            model=step_model,
            encoder_output_dim=encoder_output_dim,
            encoder_outputs_transposed=encoder_outputs_transposed,
            weighted_encoder_outputs=weighted_encoder_outputs,
            decoder_hidden_state_t=hidden_t_intermediate,
            decoder_hidden_state_dim=decoder_state_dim,
            scope=scope,
        )
    hidden_t = step_model.Copy(hidden_t_intermediate, s('hidden_t'))
    step_model.net.AddExternalOutputs(
        cell_t,
        hidden_t,
        attention_weighted_encoder_context_t,
    )

    return recurrent_net(
        net=model.net,
        cell_net=step_model.net,
        inputs=[
            (input_t, decoder_inputs),
        ],
        initial_cell_inputs=[
            (hidden_t_prev, initial_decoder_hidden_state),
            (cell_t_prev, initial_decoder_cell_state),
            (
                attention_weighted_encoder_context_t_prev,
                initial_attention_weighted_encoder_context,
            ),
        ],
        links={
            hidden_t_prev: hidden_t,
            cell_t_prev: cell_t,
            attention_weighted_encoder_context_t_prev: (
                attention_weighted_encoder_context_t
            ),
        },
        timestep=timestep,
        scope=scope,
        outputs_with_grads=outputs_with_grads,
    )


def MILSTM(model, input_blob, seq_lengths, initial_states, dim_in, dim_out,
           scope, outputs_with_grads=(0,)):
    '''
    Adds MI flavor of standard LSTM recurrent network operator to a model.
    See https://arxiv.org/pdf/1606.06630.pdf

    model: CNNModelHelper object new operators would be added to

    input_blob: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimention

    seq_lengths: blob containing sequence lengths which would be passed to
    LSTMUnit operator

    initial_states: a tupple of (hidden_input_blob, cell_input_blob)
    which are going to be inputs to the cell net on the first iteration

    dim_in: input dimention

    dim_out: output dimention

    outputs_with_grads : position indices of output blobs which will receive
    external error gradient during backpropagation
    '''
    def s(name):
        # We have to manually scope due to our internal/external blob
        # relationships.
        return "{}/{}".format(str(scope), str(name))

    """ initial bulk fully-connected """
    input_blob = model.FC(
        input_blob, s('i2h'), dim_in=dim_in, dim_out=4 * dim_out, axis=2)

    """ the step net """
    step_model = CNNModelHelper(name='milstm_cell', param_model=model)
    input_t, timestep, cell_t_prev, hidden_t_prev = (
        step_model.net.AddScopedExternalInputs(
            'input_t', 'timestep', 'cell_t_prev', 'hidden_t_prev'))
    # hU^T
    # Shape: [1, batch_size, 4 * hidden_size]
    prev_t = step_model.FC(
        hidden_t_prev, s('prev_t'), dim_in=dim_out,
        dim_out=4 * dim_out, axis=2)
    # defining MI parameters
    alpha = step_model.param_init_net.ConstantFill(
        [],
        [s('alpha')],
        shape=[4 * dim_out],
        value=1.0
    )
    beta1 = step_model.param_init_net.ConstantFill(
        [],
        [s('beta1')],
        shape=[4 * dim_out],
        value=1.0
    )
    beta2 = step_model.param_init_net.ConstantFill(
        [],
        [s('beta2')],
        shape=[4 * dim_out],
        value=1.0
    )
    b = step_model.param_init_net.ConstantFill(
        [],
        [s('b')],
        shape=[4 * dim_out],
        value=0.0
    )
    # alpha * (xW^T * hU^T)
    # Shape: [1, batch_size, 4 * hidden_size]
    alpha_tdash = step_model.net.Mul(
        [prev_t, input_t],
        s('alpha_tdash')
    )
    # Shape: [batch_size, 4 * hidden_size]
    alpha_tdash_rs, _ = step_model.net.Reshape(
        alpha_tdash,
        [s('alpha_tdash_rs'), s('alpha_tdash_old_shape')],
        shape=[-1, 4 * dim_out],
    )
    alpha_t = step_model.net.Mul(
        [alpha_tdash_rs, alpha],
        s('alpha_t'),
        broadcast=1,
        use_grad_hack=1
    )
    # beta1 * hU^T
    # Shape: [batch_size, 4 * hidden_size]
    prev_t_rs, _ = step_model.net.Reshape(
        prev_t,
        [s('prev_t_rs'), s('prev_t_old_shape')],
        shape=[-1, 4 * dim_out],
    )
    beta1_t = step_model.net.Mul(
        [prev_t_rs, beta1],
        s('beta1_t'),
        broadcast=1,
        use_grad_hack=1
    )
    # beta2 * xW^T
    # Shape: [batch_szie, 4 * hidden_size]
    input_t_rs, _ = step_model.net.Reshape(
        input_t,
        [s('input_t_rs'), s('input_t_old_shape')],
        shape=[-1, 4 * dim_out],
    )
    beta2_t = step_model.net.Mul(
        [input_t_rs, beta2],
        s('beta2_t'),
        broadcast=1,
        use_grad_hack=1
    )
    # Add 'em all up
    gates_tdash = step_model.net.Sum(
        [alpha_t, beta1_t, beta2_t],
        s('gates_tdash')
    )
    gates_t = step_model.net.Add(
        [gates_tdash, b],
        s('gates_t'),
        broadcast=1,
        use_grad_hack=1
    )
    # # Shape: [1, batch_size, 4 * hidden_size]
    gates_t_rs, _ = step_model.net.Reshape(
        gates_t,
        [s('gates_t_rs'), s('gates_t_old_shape')],
        shape=[1, -1, 4 * dim_out],
    )

    hidden_t, cell_t = step_model.net.LSTMUnit(
        [hidden_t_prev, cell_t_prev, gates_t_rs, seq_lengths, timestep],
        [s('hidden_t'), s('cell_t')],
    )
    step_model.net.AddExternalOutputs(cell_t, hidden_t)

    """ recurrent network """
    (hidden_input_blob, cell_input_blob) = initial_states
    output, last_output, all_states, last_state = recurrent_net(
        net=model.net,
        cell_net=step_model.net,
        inputs=[(input_t, input_blob)],
        initial_cell_inputs=[
            (hidden_t_prev, hidden_input_blob),
            (cell_t_prev, cell_input_blob),
        ],
        links={
            hidden_t_prev: hidden_t,
            cell_t_prev: cell_t,
        },
        timestep=timestep,
        scope=scope,
        outputs_with_grads=outputs_with_grads,
    )
    return output, last_output, all_states, last_state
