from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.python.cnn import CNNModelHelper


def recurrent_net(
        net, cell_net, inputs, initial_cell_inputs,
        links, scratch_sizes,
        timestep=None, scope=None
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

    scratch_sizes: sizes of the scratch blobs. Scratch blobs are those
    intermidiate blobs of the cell_net which are used in backward pass.
    We use sizes iformation to preallocate memory for them over time.

    For example in case of LSTM we have FC -> Sum ->LSTMUnit sequence of
    operations in each iteration of the cell net. Output of Sum is an
    intermidiate blob. Also it is going to be part of the backward pass.
    Thus it is a scratch blob size of which we must to pvovide.

    timestep: name of the timestep blob to be used. If not provided "timestep"
    is used.

    scope: Internal blobs are going to be scoped in a format
    <scope_name>/<blob_name>
    If not provided we generate a scope name automatically
    '''
    assert len(inputs) == 1, "Only one input blob is supported so far"

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
        b for b in cell_net.Proto().external_input
        if b not in known_inputs]

    inner_outputs = list(cell_net.Proto().external_output)
    # These gradients are expected to be available during the backward pass
    inner_outputs_map = {o: o + '_grad' for o in inner_outputs}

    # compute the backward pass of the cell net
    backward_ops, backward_mapping = core.GradientRegistry.GetBackwardPass(
        cell_net.Proto().op, inner_outputs_map)
    backward_mapping = {str(k): str(v) for k, v in backward_mapping.items()}
    backward_cell_net = core.Net("RecurrentBackwardStep")

    del backward_cell_net.Proto().op[:]
    backward_cell_net.Proto().op.extend(backward_ops)

    # compute blobs used but not defined in the backward pass
    ssa, _ = core.get_ssa(backward_cell_net.Proto())
    undefined = core.get_undefined_blobs(ssa)

    # also add to the output list the intermediate outputs of fwd_step that
    # are used by backward.
    ssa, blob_versions = core.get_ssa(cell_net.Proto())
    scratches = [
        blob for (blob, ver) in blob_versions.items()
        if ver > 0 and
        blob in undefined and
        blob not in cell_net.Proto().external_output]

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
    backward_aliases = []

    # States held inputs to the cell net
    recurrent_states = []

    # a map from gradient blob name to blob with its value over time
    grad_to_state = {}

    # A mapping from a blob to its gradient state blob

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
        backward_links.append((cell_input + "_grad", states_grad, 0))
        backward_links.append((cell_output + "_grad", states_grad, 1))

        backward_cell_net.Proto().external_input.append(
            str(cell_output) + "_grad")
        aliases.append((state, cell_output + "_last", -1))
        aliases.append((state, cell_output + "_all", 1))
        all_outputs.extend([cell_output + "_all", cell_output + "_last"])

        recurrent_states.append(state)

    for scratch in scratches:
        # no scoping as scratches should be already scoped
        forward_links.append((scratch, scratch + "_states", 0))
        grad_blob = scratch + "_grad"
        states_grad_blob = scratch + "_states_grad"
        backward_links.append((grad_blob, states_grad_blob, 0))
        backward_cell_net.Proto().external_input.append(scratch)
        grad_to_state[grad_blob] = states_grad_blob

    input_gradient_ids = []
    for input_id, (input_t, input_blob) in enumerate(inputs):
        forward_links.append((str(input_t), str(input_blob), 0))

        input_blob_grad = str(input_blob) + "_grad"
        if backward_mapping[str(input_t)] != str(input_t) + "_grad":
            # Some scratch (internal blob) ends up being an input gradient
            # So we avoid extra copy and reuse it by applying this alias
            backward_aliases.append((
                grad_to_state[backward_mapping[str(input_t)]],
                input_blob_grad,
                0
            ))
        else:
            # This is a general case - we have to explicitly create input
            # gradient blob as it doesn't match any of internal gradients
            backward_links.append(
                (str(input_t) + "_grad", input_blob_grad, 0))
            input_gradient_ids.append(input_id)

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
    backward_alias_src, backward_alias_dst, backward_alias_offset = \
        unpack_triple(backward_aliases)

    params = [x for x in references if x in backward_mapping.keys()]

    recurrent_inputs = [str(x[1]) for x in initial_cell_inputs]

    return net.RecurrentNetwork(
        all_inputs,
        all_outputs,
        param=map(all_inputs.index, params),
        alias_src=alias_src,
        alias_dst=map(str, alias_dst),
        alias_offset=alias_offset,
        recurrent_states=recurrent_states,
        recurrent_inputs=recurrent_inputs,
        recurrent_input_ids=map(all_inputs.index, recurrent_inputs),
        link_internal=map(str, link_internal),
        link_external=map(str, link_external),
        link_offset=link_offset,
        backward_link_internal=map(str, backward_link_internal),
        backward_link_external=map(str, backward_link_external),
        backward_link_offset=backward_link_offset,
        backward_alias_src=backward_alias_src,
        backward_alias_dst=backward_alias_dst,
        backward_alias_offset=backward_alias_offset,
        scratch=[sc + "_states" for sc in scratches],
        backward_scratch=[sc + "_states_grad" for sc in scratches],
        scratch_sizes=scratch_sizes,
        step_net=str(cell_net.Proto()),
        backward_step_net=str(backward_cell_net.Proto()),
        timestep="timestep" if timestep is None else str(timestep),
        input_gradient_ids=input_gradient_ids,
    )


def LSTM(model, input_blob, seq_lengths, initial_states, dim_in, dim_out,
         scope):
    '''
    Adds a standard LSTM recurrent network operator to a model.

    model: ModelHelperBase object new operators would be added to

    input_blob: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimention

    seq_lengths: blob containing sequence lengths which would be passed to
    LSTMUnit operator

    initial_states: a tupple of (hidden_input_blob, cell_input_blob)
    which are going to be inputs to the cell net on the first iteration

    dim_in: input dimention

    dim_out: output dimention
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
        step_model.net.AddExternalInputs(
            'input_t', 'timestep', 'cell_t_prev', 'hidden_t_prev'))
    gates_t = step_model.FC(
        hidden_t_prev, s('gates_t'), dim_in=dim_out,
        dim_out=4 * dim_out, axis=2)
    step_model.net.Sum([gates_t, input_t], gates_t)
    hidden_t, cell_t = step_model.net.LSTMUnit(
        [cell_t_prev, gates_t, seq_lengths, timestep],
        ['hidden_t', 'cell_t'],
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
        scratch_sizes=[dim_out * 4],
        scope=scope,
    )
    return output, last_output, all_states, last_state
