## @package control_ops_grad
# Module caffe2.python.control_ops_grad





from caffe2.proto import caffe2_pb2


def gen_do_gradient(op, g_output):
    """
    Generates gradient Do operator, given forward Do op and a list
    of gradient blobs corresponding to forward op's outputs
    Returns a gradient op and a list of blobs corresponding to input gradients
    """
    from caffe2.python.core import BlobReference
    subnet, outer_to_inner_map, inner_to_outer_map, workspace_blob_name = \
        _do_op_sanity_check_and_process(op)

    assert len(g_output) == len(op.output), \
        "Different number of gradient blobs and Do op outputs"

    grad_ops, deduped_g_output = dedupe_g_output(op, g_output)
    g_output = deduped_g_output

    # From the outer net point of view:
    #  Do is an operator that has some number of inputs and outputs;
    #  we have to generate a gradient operator that writes into
    #  corresponding input gradient blobs and has access to inputs, outputs
    #  and gradient output blobs
    # From the inner net point of view:
    #  Do is an operator with a subnet and blob bindings,
    #  we need to forward Do's output blob gradients into inner workspace,
    #  use them to run backward pass generation and forward Do's input blob
    #  gradients back into outer workspace

    op_output = [str(o) for o in op.output]
    op_output = op_output[:-1]  # remove workspace pointer blob
    op_input = [str(i) for i in op.input]
    op_input = op_input[:-1]  # remove workspace pointer blob

    ordered_inner_output_blob_names = [outer_to_inner_map[o] for o in op_output]

    backward_pass_initial_grad_map = {}
    initial_grad_map = {}
    for inner_output_name, outer_grad_output_name in \
            zip(ordered_inner_output_blob_names, g_output):
        # link inner_output_name to corresponding inner_grad_output_name for
        # backward pass generation;
        if outer_grad_output_name:
            inner_grad_output_name = inner_output_name + "/_DO_OPERATOR_INNER_GRAD_"
            backward_pass_initial_grad_map[BlobReference(inner_output_name)] = \
                BlobReference(inner_grad_output_name)
            initial_grad_map[inner_grad_output_name] = str(outer_grad_output_name)
    assert len(initial_grad_map) > 0, "Empty initial gradient map for Do op"

    inner_grad_ops, inner_grad_names_map = _gen_subgradient_pass(
        subnet, backward_pass_initial_grad_map)

    if len(inner_grad_ops) == 0:
        return [], []

    grad_copy_ops = []
    g_input = []
    new_op_outputs = []
    new_blob_bindings = {}
    for outer_input_name in op_input:
        inner_input_name = outer_to_inner_map[outer_input_name]
        if inner_input_name in inner_grad_names_map:
            inner_grad_input_name = inner_grad_names_map[inner_input_name]
            outer_grad_input_name = outer_input_name + "_grad"

            # It is possible that inner_grad_input_name will need to be
            # linked to another outer blob. For example:
            #
            #    // y - param initialized in init_net
            #    x = ...
            #    z = ...
            #    with ops.IfNet(...):
            #        ops.Add([z, x], y) # inner Do block
            #    loss = f(..., y, ...)
            #
            # In this case x, y and z are external for the inner Do block,
            # the inputs of the Do block are z and x and the output is y.
            # When computing the gradient of input x given the gradient
            # of output y it's easy to see that they are equal.
            # During the generation of gradient Do operator, we link
            # external gradient y (y_grad) to the internal name
            # (y/_DO_OPERATOR_INNER_GRAD_) and generate the backward pass
            # for the internal Do net. As a result we get gradient operators
            # for the gradient Do and gradient map that maps internal Do
            # blobs to their computed gradients.
            # In this example, gradient map may have blob x linked to
            # gradient blob y/_DO_OPERATOR_INNER_GRAD_.
            # We should export gradient for x outside of Do, so
            # we add a blob mapping from inner gradient blob
            # (y/_DO_OPERATOR_INNER_GRAD_) to a new outer name (x_grad).
            #
            # (Note: since we use transparent blob mapping between outer and
            # inner (Do's) workspace, these operations do not involve copying
            # but are merely using blobs in outer workspace in the Do's operator
            # workspace under (possibly) different names)
            #
            # At the same time, we need to add a blob mapping from inner name
            # y/_DO_OPERATOR_INNER_GRAD_ to the outer blob y_grad
            # Hence in this case, we cannot use existing blob mapping scheme
            # that requires a bijection between subset of inner blob names and
            # a set of all (Do's input and output) outer blob names

            # TODO(iliacher): Remove unnecessary blob copying

            new_inner_grad_input_name = \
                inner_input_name + "/_DO_OPERATOR_INNER_GRAD_COPY_"
            grad_copy_ops.append(_prepare_blob_copy_op(
                inner_grad_input_name, new_inner_grad_input_name))

            new_blob_bindings[new_inner_grad_input_name] = outer_grad_input_name
            new_op_outputs.append(outer_grad_input_name)
            g_input.append(outer_grad_input_name)
        else:
            g_input.append(None)

    new_op_inputs = []
    overwritten_names = set()
    saved_local_blob_names = set()
    for grad_op in inner_grad_ops:
        grad_op_input = [str(i) for i in grad_op.input]
        grad_op_output = [str(o) for o in grad_op.output]
        for grad_op_input_name in grad_op_input:
            if grad_op_input_name in overwritten_names:
                continue
            # check if this is an external blob
            outer_name = inner_to_outer_map.get(grad_op_input_name, None)
            if not outer_name:
                # check if this is an external gradient blob
                outer_name = initial_grad_map.get(grad_op_input_name, None)
            if outer_name:
                outer_name = str(outer_name)
                if outer_name not in new_op_inputs:
                    new_op_inputs.append(outer_name)

                new_blob_bindings[grad_op_input_name] = outer_name
            else:
                # this is a local blob, we'll get it's value from
                # a saved forward op workspace
                saved_local_blob_names.add(grad_op_input_name)
        overwritten_names.update(grad_op_output)

    # add inner gradient copy ops
    inner_grad_ops += grad_copy_ops

    gradient_do_def = _prepare_gradient_do_op(
        fwd_op=op,
        fwd_net=subnet,
        grad_ops=inner_grad_ops,
        inputs=new_op_inputs,
        outputs=new_op_outputs,
        blob_bindings=new_blob_bindings,
        saved_fwd_blobs=saved_local_blob_names,
        workspace_blob_name=workspace_blob_name)
    grad_ops.append(gradient_do_def)

    _do_op_sanity_check_and_process(gradient_do_def)

    return grad_ops, g_input


def dedupe_g_output(op, g_output):
    # When generation a gradient op it's possible to receive the same gradient
    # blob corresponding to different forward op output blobs, Do operator
    # requires a bijection between inner and outer names, make sure we do
    # deduplication
    grad_ops = []
    deduped_g_output = []
    init_grad_map = {}
    for output_name, grad_name in zip(op.output, g_output):
        if not grad_name:
            deduped_g_output.append(grad_name)
            continue

        if output_name in init_grad_map:
            deduped_g_output.append(init_grad_map[output_name])
        else:
            if grad_name not in init_grad_map.values():
                init_grad_map[output_name] = grad_name
                deduped_g_output.append(grad_name)
            else:
                deduped_grad_name = output_name + "_" + grad_name + "_DEDUP"
                assert deduped_grad_name not in init_grad_map.values()
                grad_copy_op = caffe2_pb2.OperatorDef()
                grad_copy_op.type = "Copy"
                grad_copy_op.input.extend([grad_name])
                grad_copy_op.output.extend([deduped_grad_name])
                grad_ops.append(grad_copy_op)
                deduped_g_output.append(deduped_grad_name)
                init_grad_map[output_name] = deduped_grad_name
    return grad_ops, deduped_g_output


def gen_while_gradient(op, g_output):
    """
    Generates gradient While operator
    """
    from caffe2.python.core import BlobReference
    assert op.type == "While", "Expected While op"
    assert len(op.input) > 0, "Expected at least one input in While op"

    assert len(op.output) == len(g_output), \
        "Different number of gradient blobs and While op outputs"

    grad_ops, deduped_g_output = dedupe_g_output(op, g_output)
    g_output = deduped_g_output

    init_grad_map = {}
    op_output = [str(o) for o in op.output]
    for output_name, grad_output_name in zip(op_output, g_output):
        if grad_output_name:
            init_grad_map[BlobReference(output_name)] = \
                BlobReference(grad_output_name)
    assert len(init_grad_map) > 0, "Empty initial gradient map for While op"

    loop_net = _get_net_argument(op, "loop_net")
    assert loop_net, "Expected loop subnet in While op"
    assert len(loop_net.op) == 1 and loop_net.op[0].type == "Do", \
        "Gradient While op requires single Do op as a loop body"
    do_op = loop_net.op[0]
    do_args = _get_do_arguments(do_op)
    assert "reuse_workspace" not in do_args or not do_args["reuse_workspace"], \
        "Gradient While op requires Do loop body op without reuse_workspace set"

    assert len(do_op.output) > 0, "Expected Do op with at least one output"
    workspace_blob = do_op.output[-1]

    loop_grad_net, loop_grad_map, loop_input_names, loop_output_names = \
        _gen_subnet_gradient(loop_net, init_grad_map)
    assert loop_grad_net, "Failed to get gradient net for loop body in While op"

    grad_ops += _prepare_gradient_while_ops(
        fwd_op=op,
        input_names=loop_input_names,
        output_names=loop_output_names,
        loop_grad_net=loop_grad_net,
        workspace_blob=workspace_blob,
        init_grad_map=init_grad_map,
        loop_grad_map=loop_grad_map)

    op_input = [str(i) for i in op.input]
    g_input = [loop_grad_map.get(i, None) for i in op_input]
    return grad_ops, g_input


# Constructs gradient While op, arguments:
#  fwd_op - forward While op
#  input_names - input blob names for a gradient op
#  output_names - output blob names for a gradient op
#  loop_grad_net - gradient loop body net
#  workspace_blob - blob that holds forward workspaces stack
#  init_grad_map - initial gradient to forward blob map
#  loop_grad_map - gradient blob map for loop's body
def _prepare_gradient_while_ops(
        fwd_op, input_names, output_names, loop_grad_net, workspace_blob,
        init_grad_map, loop_grad_map):
    gradient_while_def = caffe2_pb2.OperatorDef()
    gradient_while_def.CopyFrom(fwd_op)
    if gradient_while_def.name:
        gradient_while_def.name += "_grad"

    loop_net_arg = caffe2_pb2.Argument()
    loop_net_arg.name = "loop_net"
    loop_net_arg.n.CopyFrom(loop_grad_net)

    cond_net_arg = caffe2_pb2.Argument()
    cond_net_arg.name = "cond_net"
    from caffe2.python.core import Net, BlobReference
    # Construct condition net - check that there're still forward workspaces
    # left using HasScope op
    cond_net = Net('gradient_loop_cond_net')
    cond_init_net = Net('gradient_loop_cond_net_init')
    cond_blob = cond_net.NextScopedBlob(cond_net.Name() + '/cond')
    cond_init_net.HasScope(workspace_blob, cond_blob)
    cond_net.HasScope(workspace_blob, cond_blob)
    for blob, init_grad_blob in init_grad_map.items():
        blob_name = str(blob)
        init_grad_blob_name = str(init_grad_blob)
        if blob_name in loop_grad_map and \
                loop_grad_map[blob_name] != init_grad_blob_name:
            cond_net.Copy(
                BlobReference(loop_grad_map[blob_name]), init_grad_blob)
            cond_init_net.Copy(
                init_grad_blob, BlobReference(loop_grad_map[blob_name]))
    cond_net_arg.n.CopyFrom(cond_net.Proto())

    del gradient_while_def.arg[:]
    gradient_while_def.arg.extend([loop_net_arg, cond_net_arg])

    del gradient_while_def.control_input[:]
    del gradient_while_def.input[:]
    gradient_while_def.input.extend(
        [str(cond_blob).encode('utf-8')] + list(input_names))
    del gradient_while_def.output[:]
    gradient_while_def.output.extend(output_names)
    gradient_while_def.is_gradient_op = True
    return [o for o in cond_init_net.Proto().op] + [gradient_while_def]


def _get_do_arguments(do_op):
    assert do_op.type == "Do", "Expected Do op"
    args = {}
    for arg in do_op.arg:
        if not arg.name:
            continue
        if arg.name == "net":
            assert arg.n, "Expected non empty net argument"
            args["net"] = arg.n
        elif arg.name == "reuse_workspace":
            assert arg.i, "Expected non empty reuse_workspace argument"
            args["reuse_workspace"] = bool(arg.i)
        elif arg.name == "inner_blobs":
            assert arg.strings, "Expected non empty inner_blobs argument"
            args["inner_blobs"] = arg.strings
        elif arg.name == "outer_blobs_idx":
            assert arg.ints, "Expected non empty outer_blobs_idx argument"
            args["outer_blobs_idx"] = arg.ints
    return args


def gen_if_gradient(op, g_output):
    """
    Generates gradient If operator, given forward If op and a list
    of gradient blobs corresponding to forward op's outputs
    Returns a gradient op and a list of blobs corresponding to input gradients
    """
    from caffe2.python.core import BlobReference
    assert op.type == "If", "Expected If op"
    # first input is the condition blob
    assert len(op.input) > 0, "Expected at least one input in If op"

    assert len(op.output) == len(g_output), \
        "Different number of gradient blobs and If op outputs"

    grad_ops, deduped_g_output = dedupe_g_output(op, g_output)
    g_output = deduped_g_output

    init_grad_map = {}  # map from if's output blob to output gradient blob
    op_input = [str(i) for i in op.input]
    op_output = [str(o) for o in op.output]
    for output_name, grad_output_name in zip(op_output, g_output):
        if grad_output_name:
            init_grad_map[BlobReference(output_name)] = \
                BlobReference(grad_output_name)
    # shouldn't call without at least one output gradient available
    assert len(init_grad_map) > 0, "Empty initial gradient map for If op"

    grad_map = {}  # map from blob to gradient blob
    then_net = _get_net_argument(op, "then_net")
    assert then_net, "Expected then subnet in If op"
    then_grad_net, then_grad_map, then_input_names, then_output_names = \
        _gen_subnet_gradient(then_net, init_grad_map)
    assert then_grad_net, "Failed to get gradient net for then in If op"
    grad_map.update(then_grad_map)

    else_input_names = set()
    else_output_names = set()
    else_grad_map = {}
    else_grad_net = None
    else_net = _get_net_argument(op, "else_net")
    if else_net:
        else_grad_net, else_grad_map, else_input_names, else_output_names = \
            _gen_subnet_gradient(else_net, init_grad_map)
        assert else_grad_net, "Failed to get gradient net for else in If op"
        # consider case: else doesn't update blob's gradient and keeps original
        # from init_grad_map, but then updates the gradient
        for else_blob, else_grad_blob in else_grad_map.items():
            if else_blob in then_grad_map:
                then_grad_blob = then_grad_map[else_blob]
                # if both then and else branches have grad blob name for the same
                # blob and grad names are different, then one of the branches
                # doesn't use blob and has original grad blob name in it's grad map,
                # and another branch uses blob and has <blob_name>_grad name
                # in it's grad map (might be different from original grad blob)
                if then_grad_blob != else_grad_blob:
                    init_grad_name = init_grad_map[else_blob] \
                        if else_blob in init_grad_map else None

                    if then_grad_blob == init_grad_name:
                        grad_map[else_blob] = else_grad_blob
                    elif else_grad_blob == init_grad_name:
                        grad_map[else_blob] = then_grad_blob
                    else:
                        raise "Unexpected grad blob name " + else_blob + ", " + \
                            else_grad_blob + ", " + then_grad_blob
            else:
                grad_map[else_blob] = else_grad_blob

    # make sure gradients of blobs that were not computed
    # by the selected if's branch are initialized with zeros
    then_other_output_names = \
        then_output_names - (then_output_names & else_output_names)
    then_other_grad_output_names = set(
        [o for o in then_other_output_names if o in then_grad_map.values()])
    zero_then = _gen_grad_zero_init_ops(
        init_grad_map, then_grad_map, then_other_grad_output_names)
    if else_grad_net:
        else_grad_net.op.extend(zero_then)
    elif len(zero_then) > 0:
        else_grad_net = caffe2_pb2.NetDef()
        else_grad_net.CopyFrom(then_grad_net)
        if else_grad_net.name:
            else_grad_net.name += "_auto_else_zero_blobs_"
        del else_grad_net.op[:]
        else_grad_net.op.extend(zero_then)
        del else_grad_net.external_input[:]
        del else_grad_net.external_output[:]

    else_other_output_names = \
        else_output_names - (then_output_names & else_output_names)
    else_other_grad_output_names = set(
        [o for o in else_other_output_names if o in else_grad_map.values()])
    zero_else = _gen_grad_zero_init_ops(
        init_grad_map, else_grad_map, else_other_grad_output_names)
    then_grad_net.op.extend(zero_else)

    output_names = list(then_output_names | else_output_names)
    input_names = then_input_names | else_input_names
    # make sure condition blob is the first in the list
    input_names = [op_input[0]] + list(input_names - set(op_input[0]))
    gradient_if_def = _prepare_gradient_if_op(
        fwd_op=op,
        input_names=input_names,
        output_names=output_names,
        then_grad_net=then_grad_net,
        else_grad_net=else_grad_net)
    g_input = [grad_map.get(i, None) for i in op_input]
    return grad_ops + [gradient_if_def], g_input


def _gen_subnet_gradient(subnet, init_grad):
    grad_ops, grad_names_map = _gen_subgradient_pass(
        subnet, init_grad)

    output_names = set()
    input_names = set()
    for grad_op in grad_ops:
        for grad_op_input in grad_op.input:
            if str(grad_op_input) not in output_names:
                input_names.add(str(grad_op_input))
        for grad_op_output in grad_op.output:
            output_names.add(str(grad_op_output))

    gradient_net_def = caffe2_pb2.NetDef()
    gradient_net_def.CopyFrom(subnet)
    if gradient_net_def.name:
        gradient_net_def.name += "_grad"
    del gradient_net_def.op[:]
    gradient_net_def.op.extend(grad_ops)
    del gradient_net_def.external_input[:]
    del gradient_net_def.external_output[:]

    return gradient_net_def, grad_names_map, input_names, output_names


def _get_net_argument(op, net_name):
    for arg in op.arg:
        if arg.name and arg.name == net_name:
            assert arg.n, "Expected non empty net argument " + net_name
            return arg.n
    return None


def getNetArgument(op, net_name):
    """A wrapper for external call"""
    return _get_net_argument(op, net_name)


def _gen_subgradient_pass(subnet, init_grad):
    from caffe2.python.core import IR
    subnet_ir = IR(subnet.op)
    grad_ops, grad_blob_map = \
        subnet_ir.GetBackwardPass(init_grad)
    grad_names_map = {}
    for b, g in grad_blob_map.items():
        grad_names_map[str(b)] = str(g)
    return grad_ops, grad_names_map


def _do_op_sanity_check_and_process(op):
    assert op.type == "Do", "Expected Do op"

    subnet = _get_net_argument(op, "net")
    assert subnet, "No net argument found in Do op"

    inner_blobs = None
    outer_blobs_idx = None
    for arg in op.arg:
        if arg.name and arg.name == "inner_blobs":
            assert not inner_blobs, "inner_blobs redefinition"
            assert arg.strings and len(arg.strings) > 0, \
                "Empty inner_blobs argument in Do op"
            inner_blobs = [s.decode('utf-8') for s in arg.strings]
        if arg.name and arg.name == "outer_blobs_idx":
            assert not outer_blobs_idx, "outer_blobs_idx redefinition"
            assert arg.ints and len(arg.ints) > 0, \
                "Empty outer_blobs_idx argument in Do op"
            outer_blobs_idx = arg.ints
        if inner_blobs and outer_blobs_idx:
            break

    assert inner_blobs, "No inner_blobs argument found in Do op"
    assert outer_blobs_idx, "No outer_blobs_idx argument found in Do op"

    assert len(inner_blobs) == len(outer_blobs_idx), \
        "Arguments inner_blobs and outer_blobs_idx of different length in Do op"

    all_inner_blobs = set(inner_blobs)
    assert len(all_inner_blobs) == len(inner_blobs), \
        "Found duplicates in inner_blobs in Do op"

    op_input = [str(i) for i in op.input]
    assert len(op_input) > 0, "Expected at least one input blob"
    # remove last input blob that holds pointer to workspace
    input_workspace_blob_name = op_input[-1]
    op_input = op_input[:-1]

    op_output = [str(o) for o in op.output]
    assert len(op_output) > 0, "Expected at least one output blob"
    # remove last output blob that holds pointer to workspace
    workspace_blob_name = op_output[-1]
    assert input_workspace_blob_name == workspace_blob_name, \
        "Expected same input/output workspace blob"
    op_output = op_output[:-1]

    all_op_input_blob_names = set(op_input)
    assert len(all_op_input_blob_names) == len(op_input), \
        "Found duplicates in Do op inputs"
    all_op_output_blob_names = set(op_output)
    assert len(all_op_output_blob_names) == len(op_output), \
        "Found duplicates in Do op outputs"

    ordered_outer_blob_names = op_input + op_output
    all_outer_blob_names = set(ordered_outer_blob_names)
    used_outer_blob_names = set()
    outer_to_inner_map = {}
    inner_to_outer_map = {}
    for inner_name, outer_blob_idx in zip(inner_blobs, outer_blobs_idx):
        assert outer_blob_idx >= 0 and \
            outer_blob_idx < len(ordered_outer_blob_names), \
            "Outer blob index is out of bounds in Do op"
        outer_name = ordered_outer_blob_names[outer_blob_idx]
        assert outer_name not in used_outer_blob_names, \
            "Reusage of outer blob name " + outer_name + " in Do op"
        used_outer_blob_names.add(outer_name)
        outer_to_inner_map[outer_name] = inner_name
        inner_to_outer_map[inner_name] = outer_name

    assert len(used_outer_blob_names) == len(all_outer_blob_names), \
        "Not all outer blob names are used in blob bindings in Do op"

    return subnet, outer_to_inner_map, inner_to_outer_map, workspace_blob_name


def _prepare_blob_copy_op(from_name, to_name):
    copy_op_def = caffe2_pb2.OperatorDef()
    copy_op_def.type = "Copy"
    copy_op_def.input.extend([from_name])
    copy_op_def.output.extend([to_name])
    return copy_op_def


def _prepare_gradient_do_op(
        fwd_op, fwd_net, grad_ops, inputs, outputs, blob_bindings, saved_fwd_blobs,
        workspace_blob_name):
    gradient_net_def = caffe2_pb2.NetDef()
    gradient_net_def.CopyFrom(fwd_net)
    if gradient_net_def.name:
        gradient_net_def.name += "_grad"
    del gradient_net_def.op[:]
    gradient_net_def.op.extend(grad_ops)
    del gradient_net_def.external_input[:]
    del gradient_net_def.external_output[:]

    gradient_do_def = caffe2_pb2.OperatorDef()
    gradient_do_def.CopyFrom(fwd_op)
    if gradient_do_def.name and len(gradient_do_def.name) > 0:
        gradient_do_def.name += "_grad"

    del gradient_do_def.input[:]
    gradient_do_def.input.extend(inputs)
    # workspace pointer blob
    gradient_do_def.input.append(workspace_blob_name)
    del gradient_do_def.output[:]
    gradient_do_def.output.extend(outputs)
    # workspace pointer blob
    gradient_do_def.output.append(workspace_blob_name)

    net_arg = caffe2_pb2.Argument()
    net_arg.name = "net"
    net_arg.n.CopyFrom(gradient_net_def)

    ordered_new_outer_names = inputs + outputs
    inner_blobs = blob_bindings.keys()
    new_outer_blobs_idx = [ordered_new_outer_names.index(blob_bindings[b])
                            for b in inner_blobs]

    inner_blobs_arg = caffe2_pb2.Argument()
    inner_blobs_arg.name = "inner_blobs"
    inner_blobs_arg.strings.extend([b.encode('utf-8') for b in inner_blobs])

    outer_blobs_idx_arg = caffe2_pb2.Argument()
    outer_blobs_idx_arg.name = "outer_blobs_idx"
    outer_blobs_idx_arg.ints.extend(new_outer_blobs_idx)

    saved_blobs_arg = caffe2_pb2.Argument()
    saved_blobs_arg.name = "saved_fwd_blobs"
    saved_blobs_arg.strings.extend(
        [b.encode('utf-8') for b in saved_fwd_blobs])

    del gradient_do_def.arg[:]
    gradient_do_def.arg.extend([
        net_arg, inner_blobs_arg, outer_blobs_idx_arg, saved_blobs_arg])
    del gradient_do_def.control_input[:]

    gradient_do_def.is_gradient_op = True

    return gradient_do_def


def _gen_grad_zero_init_ops(init_grad_map, grad_map, grad_output_names):
    grad_init_ops = []
    for grad_output in grad_output_names:
        # get the corresponding output name blob and use it in ConstantFill
        # so that grad_output has the same shape
        output_name = None
        for o, g in grad_map.items():
            if g == grad_output:
                output_name = o
                break
        assert output_name, "Unknown gradient output " + grad_output

        grad_init_op = None
        # make sure that we do not overwrite existing gradients with zeros
        if output_name in init_grad_map:
            init_grad_name = init_grad_map[output_name]
            # in case we use a different gradient blob name, copy gradient
            if init_grad_name != grad_output:
                grad_init_op = caffe2_pb2.OperatorDef()
                grad_init_op.type = "Copy"
                grad_init_op.input.extend([str(init_grad_name)])
                grad_init_op.output.extend([str(grad_output)])
        else:
            grad_init_op = caffe2_pb2.OperatorDef()
            grad_init_op.type = "ConstantFill"
            grad_init_op.input.extend([output_name])
            grad_init_op.output.extend([grad_output])
            value_arg = caffe2_pb2.Argument()
            value_arg.name = "value"
            value_arg.f = 0.0
            grad_init_op.arg.extend([value_arg])

        if grad_init_op:
            grad_init_ops.append(grad_init_op)
    return grad_init_ops


def _prepare_gradient_if_op(
        fwd_op, input_names, output_names, then_grad_net, else_grad_net):
    gradient_if_def = caffe2_pb2.OperatorDef()
    gradient_if_def.CopyFrom(fwd_op)
    del gradient_if_def.input[:]
    gradient_if_def.input.extend(input_names)
    del gradient_if_def.output[:]
    gradient_if_def.output.extend(output_names)

    then_net_arg = caffe2_pb2.Argument()
    then_net_arg.name = "then_net"
    then_net_arg.n.CopyFrom(then_grad_net)
    gradient_args = [then_net_arg]
    if else_grad_net:
        else_net_arg = caffe2_pb2.Argument()
        else_net_arg.name = "else_net"
        else_net_arg.n.CopyFrom(else_grad_net)
        gradient_args.append(else_net_arg)

    del gradient_if_def.arg[:]
    gradient_if_def.arg.extend(gradient_args)
    if gradient_if_def.name:
        gradient_if_def.name += "_grad"
    del gradient_if_def.control_input[:]
    gradient_if_def.is_gradient_op = True
    return gradient_if_def


def disambiguate_grad_if_op_output(grad_op, idx, new_grad_output):
    then_net = _get_net_argument(grad_op, "then_net")
    old_grad_out_match = grad_op.output[idx]
    for op in then_net.op:
        for i, out in enumerate(op.output):
            if out == old_grad_out_match:
                op.output[i] = new_grad_output
    else_net = _get_net_argument(grad_op, "else_net")
    if else_net:
        for op in else_net.op:
            for i, out in enumerate(op.output):
                if out == old_grad_out_match:
                    op.output[i] = new_grad_output
    grad_op.output[idx] = new_grad_output
