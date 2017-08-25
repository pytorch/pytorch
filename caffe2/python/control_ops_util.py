## @package control_ops_util
# Module caffe2.python.control_ops_util
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core


def get_external_blob_names(net, lexical_scope):
    """
    Returns a set of blobs a given net depends on and a set of
    output blobs that are written by the net
    Inputs:
        net - net to return input/output blobs for;
        lexical_scope - all external blob names visible to the net
    """
    net_proto = net.Proto()
    net_ssa, _ = core.get_ssa(net_proto)
    input_names = core.get_undefined_blobs(net_ssa)
    if net_proto.external_input:
        input_names |= set(net_proto.external_input)

    output_names = set()
    if net_proto.external_output:
        output_names = set(net_proto.external_output)
    for op in net_proto.op:
        for output in op.output:
            if output in lexical_scope:
                output_names.add(output)

    return input_names, output_names


def add_if_op(if_net, cond_blob, lexical_scope, then_net, else_net=None):
    """
    A helper function to add an If op to the net.
    Automatically determines whether blobs in the then/else subnets are external
    (from the outer workspace) or local (visible only inside subnet's workspace)
    based on lexical scope - set of all outer blob names visible to the 'If'
    operator. All the blobs in then/else subnets with names matching a name in lexical
    scope and all the blobs that are first used as the operators' inputs are
    considered outer blobs - these blobs must exist in the outer workspace,
    then/else subnets can read their values and new values written into these blobs
    will be visible outside of the 'If' operator. All other blobs are local - exist
    only within inner workspaces for then/else.
    Inputs:
        if_net - net to add an If op to;
        cond_blob - scalar bool blob reference, used as If condition;
        lexical_scope - a set of outer blob names visible to then/else branches;
        then_net/else_net - nets (core.Net) for then/else branches
    """
    then_input_blob_names, then_output_blob_names = get_external_blob_names(
        then_net, lexical_scope)

    else_input_blob_names = set()
    else_output_blob_names = set()
    if else_net:
        else_input_blob_names, else_output_blob_names = get_external_blob_names(
            else_net, lexical_scope)

    input_blob_names = then_input_blob_names | else_input_blob_names

    # find outputs that are not produced by both then and else branches and
    # add them into inputs
    outputs_to_inputs = then_output_blob_names ^ else_output_blob_names
    input_blob_names |= outputs_to_inputs

    output_blob_names = then_output_blob_names | else_output_blob_names

    ext_then_input_blob_names = then_input_blob_names | (
        then_output_blob_names - else_output_blob_names)
    ext_else_input_blob_names = else_input_blob_names | (
        else_output_blob_names - then_output_blob_names)

    if_inputs = [cond_blob]
    if_inputs += [core.BlobReference(name=b, net=None) for b in input_blob_names]
    if_outputs = [core.BlobReference(name=b, net=None) for b in output_blob_names]

    do_then_net = core.Net('do_then_net')

    ext_then_input_blobs = \
        [core.BlobReference(name=b, net=None) for b in ext_then_input_blob_names]
    then_output_blobs = \
        [core.BlobReference(name=b, net=None) for b in then_output_blob_names]
    then_input_output_names_ordered = [
        str(b) for b in (ext_then_input_blobs + then_output_blobs)]

    then_outer_blob_names = list(ext_then_input_blob_names | then_output_blob_names)
    then_outer_blob_names_idx = [
        then_input_output_names_ordered.index(b) for b in then_outer_blob_names]

    do_then_net.Do(
        ext_then_input_blobs,
        then_output_blobs,
        net=then_net.Proto(),
        inner_blobs=then_outer_blob_names,
        outer_blobs_idx=then_outer_blob_names_idx)
    do_then_net.AddExternalOutput(*then_output_blobs)

    if_args = {}
    if_args['then_net'] = do_then_net.Proto()

    if else_net:
        do_else_net = core.Net('do_else_net')

        ext_else_input_blobs = \
            [core.BlobReference(name=b, net=None) for b in ext_else_input_blob_names]
        else_output_blobs = \
            [core.BlobReference(name=b, net=None) for b in else_output_blob_names]
        else_input_output_names_ordered = [
            str(b) for b in (ext_else_input_blobs + else_output_blobs)]

        else_outer_blob_names = list(ext_else_input_blob_names | else_output_blob_names)
        else_outer_blob_names_idx = [
            else_input_output_names_ordered.index(b) for b in else_outer_blob_names]

        do_else_net.Do(
            ext_else_input_blobs,
            else_output_blobs,
            net=else_net.Proto(),
            inner_blobs=else_outer_blob_names,
            outer_blobs_idx=else_outer_blob_names_idx)
        do_else_net.AddExternalOutput(*else_output_blobs)
        if_args['else_net'] = do_else_net.Proto()

    if_net.If(if_inputs, if_outputs, **if_args)
    if_net.AddExternalOutput(*if_outputs)


def add_while_op(
        while_net, cond_blob, lexical_scope, loop_body_net, condition_body_net=None):
    """
    A helper function to add a While op to the net. Same rules for determining
    outer and inner blobs as for the 'If' operator apply for the 'While' operator
    loop and condition subnets. If specified, condition net is executed in a separate
    workspace before the first and after each iteration, the last operator must have
    a single scalar boolean output that is written into the condition blob.
    Inputs:
        while_net - net to add a While op to;
        cond_blob - scalar bool blob reference, used as a stop condition;
        lexical_scope - a set of outer blob names visible to the loop's body;
        loop_body_net - net to execute on each iteration;
        condition_body_net - net to compute condition value
    """
    input_blob_names, output_blob_names = get_external_blob_names(
        loop_body_net, lexical_scope)

    # Since it's possible that loop is not going to run even once
    # we have to add loop's external outputs into inputs
    input_blob_names |= output_blob_names

    while_inputs = [cond_blob]
    while_inputs += [core.BlobReference(name=b, net=None) for b in input_blob_names]
    while_outputs = [core.BlobReference(name=b, net=None) for b in output_blob_names]

    do_loop_body_net = core.Net('do_loop_body_net')

    loop_input_output_names_ordered = [
        str(b) for b in (while_inputs + while_outputs)]
    loop_body_outer_blob_names = list(input_blob_names | output_blob_names)
    loop_body_outer_blob_names_idx = [
        loop_input_output_names_ordered.index(b) for b in loop_body_outer_blob_names]
    do_loop_body_net.Do(
        while_inputs,
        while_outputs,
        net=loop_body_net.Proto(),
        inner_blobs=loop_body_outer_blob_names,
        outer_blobs_idx=loop_body_outer_blob_names_idx)
    do_loop_body_net.AddExternalOutput(*while_outputs)

    while_args = {}
    while_args['loop_net'] = do_loop_body_net.Proto()

    condition_net = None
    if condition_body_net:
        # make sure condition blob is visible outside of condition net
        if str(cond_blob) not in condition_body_net.Proto().external_output:
            condition_body_net.AddExternalOutput(cond_blob)

        cond_input_blob_names, cond_output_blob_names = get_external_blob_names(
            condition_body_net, lexical_scope)

        cond_inputs = [core.BlobReference(name=b, net=None)
                        for b in cond_input_blob_names]
        assert str(cond_blob) in cond_output_blob_names, \
            'Condition blob expected in condition net output'
        cond_outputs = [core.BlobReference(name=b, net=None)
                        for b in cond_output_blob_names]

        cond_input_output_names_ordered = [
            str(b) for b in (cond_inputs + cond_outputs)]
        cond_body_outer_blob_names = \
            list(cond_input_blob_names | cond_output_blob_names)
        cond_body_outer_blob_names_idx = [
            cond_input_output_names_ordered.index(b)
            for b in cond_body_outer_blob_names]
        condition_net = core.Net('do_loop_condition_net')
        condition_net.Do(
            cond_inputs,
            cond_outputs,
            net=condition_body_net.Proto(),
            inner_blobs=cond_body_outer_blob_names,
            outer_blobs_idx=cond_body_outer_blob_names_idx)
        condition_net.AddExternalOutput(*cond_outputs)

        while_args['cond_net'] = condition_net.Proto()

        while_inputs += [b for b in cond_inputs
                            if str(b) not in input_blob_names]
        while_outputs += [b for b in cond_outputs
                            if str(b) not in output_blob_names]

        if str(cond_blob) not in lexical_scope:
            while_net.ConstantFill(
                [],
                cond_blob,
                dtype=core.DataType.BOOL,
                value=False)

    while_net.While(while_inputs, while_outputs, **while_args)
    while_net.AddExternalOutput(*while_outputs)
