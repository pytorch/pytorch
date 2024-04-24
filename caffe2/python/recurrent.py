## @package recurrent
# Module caffe2.python.recurrent





from caffe2.python import core, workspace

def recurrent_net(
        net, cell_net, inputs, initial_cell_inputs,
        links, timestep=None, scope=None, outputs_with_grads=(0,),
        recompute_blobs_on_backward=None, forward_only=False,
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

    recompute_blobs_on_backward: specify a list of blobs that will be
                 recomputed for backward pass, and thus need not to be
                 stored for each forward timestep.

    forward_only: if True, only forward steps are executed
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
    known_inputs = [str(b) for b in input_blobs + initial_input_blobs]
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
    if not forward_only:
        backward_ops, backward_mapping = core.GradientRegistry.GetBackwardPass(
            cell_net.Proto().op, inner_outputs_map)
        backward_mapping = {str(k): v for k, v in backward_mapping.items()}

        backward_cell_net = core.Net("RecurrentBackwardStep")
        del backward_cell_net.Proto().op[:]

        if recompute_blobs_on_backward is not None:
            # Insert operators to re-compute the specified blobs.
            # They are added in the same order as for the forward pass, thus
            # the order is correct.
            recompute_blobs_on_backward = {str(b) for b in
                                           recompute_blobs_on_backward}

            for op in cell_net.Proto().op:
                if not recompute_blobs_on_backward.isdisjoint(set(op.output)):
                    backward_cell_net.Proto().op.extend([op])
                    # This fires if other outputs than the declared
                    # are computed by the ops that are recomputed
                    assert set(op.output).issubset(recompute_blobs_on_backward)

        backward_cell_net.Proto().op.extend(backward_ops)
        # compute blobs used but not defined in the backward pass
        backward_ssa, backward_blob_versions = core.get_ssa(
            backward_cell_net.Proto())
        undefined = core.get_undefined_blobs(backward_ssa)

        # also add to the output list the intermediate outputs of fwd_step that
        # are used by backward.
        ssa, blob_versions = core.get_ssa(cell_net.Proto())
        scratches = [
            blob
            for blob, ver in blob_versions.items()
            if (ver > 0 and
                blob in undefined and
                blob not in cell_net.Proto().external_output)
        ]
        backward_cell_net.Proto().external_input.extend(scratches)
        backward_cell_net.Proto().type = 'simple'
    else:
        backward_cell_net = None

    all_inputs = [i[1] for i in inputs] + [
        x[1] for x in initial_cell_inputs] + references
    all_outputs = []

    cell_net.Proto().type = 'simple'

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

        aliases.append((state, cell_output + "_all", 1))
        aliases.append((state, cell_output + "_last", -1))
        all_outputs.extend([cell_output + "_all", cell_output + "_last"])

        recurrent_states.append(state)

        if backward_cell_net is not None:
            backward_links.append((cell_output + "_grad", states_grad, 1))
            backward_cell_net.Proto().external_input.append(
                str(cell_output) + "_grad")

            recurrent_input_grad = cell_input + "_grad"
            if not backward_blob_versions.get(recurrent_input_grad, 0):
                # If nobody writes to this recurrent input gradient, we need
                # to make sure it gets to the states grad blob after all.
                # We do this by using backward_links which triggers an alias
                # This logic is being used for example in a SumOp case
                backward_links.append(
                    (backward_mapping[cell_input], states_grad, 0))
            else:
                backward_links.append((recurrent_input_grad, states_grad, 0))


    for input_t, input_blob in inputs:
        forward_links.append((str(input_t), str(input_blob), 0))

    if backward_cell_net is not None:
        for input_t, input_blob in inputs:
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
    alias_src, alias_dst, alias_offset = unpack_triple(aliases)

    recurrent_inputs = [str(x[1]) for x in initial_cell_inputs]

    # Make sure that recurrent gradients accumulate with internal gradients
    # (if a blob in the backward_cell_net receives gradient from both an
    # external connection as well as from within the backward_cell_net,
    # those gradients need to be added together, rather than one overwriting
    # the other)
    if backward_cell_net is not None:
        proto = backward_cell_net.Proto()
        operators = []
        while len(proto.op) > 0:
            op = proto.op[-1]
            proto.op.remove(op)
            operators.append(op)
        for op in operators[::-1]:
            proto.op.extend([op])
            for j, output_blob in enumerate(op.output):
                if output_blob in proto.external_input:
                    # In place operation won't cause issues because it takes
                    # existing value of a blob into account
                    if output_blob in op.input:
                        continue
                    output_blob = core.BlobReference(output_blob)
                    accum_blob = output_blob + "_accum"
                    proto.op[-1].output[j] = str(accum_blob)
                    backward_cell_net.Sum(
                        [output_blob, accum_blob],
                        [output_blob],
                    )

    def map_to_dual_list(m):
        return [str(x) for x in list(m.keys())] + \
               [str(x) for x in list(m.values())]

    backward_args = {}
    if backward_cell_net is not None:
        backward_mapping_keys = set(backward_mapping.keys())
        backward_link_internal, backward_link_external, backward_link_offset = \
            unpack_triple(backward_links)
        params = [x for x in references if x in backward_mapping_keys]
        param_grads = [
            str(backward_mapping[x])
            for x in references
            if x in backward_mapping_keys
        ]
        if recompute_blobs_on_backward is None:
            recompute_blobs_on_backward = set()
        backward_args = {
            'param': [all_inputs.index(p) for p in params],
            'backward_link_internal': [str(l) for l in backward_link_internal],
            'backward_link_external': [str(l) for l in backward_link_external],
            'backward_link_offset': backward_link_offset,
            'outputs_with_grads': outputs_with_grads,
            'recompute_blobs_on_backward': [
                str(b) for b in recompute_blobs_on_backward
            ],
            'param_grads': param_grads,
        }
        if len(backward_cell_net.Proto().op) != 0:
            backward_args['backward_step_net'] = backward_cell_net.Proto()


    results = net.RecurrentNetwork(
        all_inputs,
        all_outputs + [s("step_workspaces")],
        alias_src=alias_src,
        alias_dst=[str(a) for a in alias_dst],
        alias_offset=alias_offset,
        recurrent_states=recurrent_states,
        initial_recurrent_state_ids=[
            all_inputs.index(i) for i in recurrent_inputs
        ],
        link_internal=[str(l) for l in link_internal],
        link_external=[str(l) for l in link_external],
        link_offset=link_offset,
        enable_rnn_executor=1,
        step_net=cell_net.Proto(),
        timestep="timestep" if timestep is None else str(timestep),
        **backward_args
    )

    # Restore net type since 'rnn' is not recognized outside RNNs
    cell_net.Proto().type = 'simple'

    # The last output is a list of step workspaces,
    # which is only needed internally for gradient propagation
    return results[:-1]


def set_rnn_executor_config(rnn_op, num_threads=None, max_cuda_streams=None):
    from caffe2.proto import caffe2_pb2
    assert rnn_op.type in {'RecurrentNetwork', 'RecurrentNetworkGradient'}

    def add_arg(s, v):
        a = caffe2_pb2.Argument()
        a.name = "rnn_executor." + s
        a.i = v
        rnn_op.arg.extend([a])

    if num_threads is not None:
        add_arg('num_threads', num_threads)
    if max_cuda_streams is not None:
        add_arg('max_cuda_streams', max_cuda_streams)


def retrieve_step_blobs(net, prefix='rnn'):
    '''
    Retrieves blobs from step workspaces (which contain intermediate recurrent
    network computation for each timestep) and puts them in the global
    workspace. This allows access to the contents of this intermediate
    computation in python. Returns the list of extracted blob names.

    net: the net from which the step workspace blobs should be extracted

    prefix: prefix to append to extracted blob names when placing them in the
    global workspace
    '''
    count = 1
    output_list = []
    for op in net.Proto().op:
        if op.type == "RecurrentNetwork":
            blob_name = prefix + "_" + str(count)
            count = count + 1
            scratch_workspaces_blob_name = op.output[-1]
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "RecurrentNetworkBlobFetcher",
                    [scratch_workspaces_blob_name],
                    [blob_name],
                    prefix=prefix
                )
            )
            output_list += workspace.FetchBlob(blob_name).tolist()
    return output_list
