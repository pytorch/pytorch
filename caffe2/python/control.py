"""
Implement functions for controlling execution of nets and steps, including
  Do
  DoParallel
  For-loop
  While-loop
  Do-While-loop
  Switch
  If
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core


def GetConditionBlobFromNet(condition_net):
    """
    The condition blob is the last external_output that must
    be a single bool
    """
    assert len(condition_net.Proto().external_output) > 0, (
        "Condition net %s must has at least one external output" %
        condition_net.Proto.name)
    # we need to use a blob reference here instead of a string
    # otherwise, it will add another name_scope to the input later
    # when we create new ops (such as OR of two inputs)
    return core.BlobReference(condition_net.Proto().external_output[-1])

def NotNet(condition_blob_or_net):
    """Not of a condition blob or net

    Args:
    condition_blob_or_net can be either blob or net. If condition_blob_or_net
    is Net, the condition is its last external_output
    that must be a single bool.

    returns
    not_net: the net NOT the input
    out_blob: the output blob of the not_net
    """
    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
    else:
        condition_blob = condition_blob_or_net

    not_net = core.Net('not_net')
    out_blob = not_net.Not(condition_blob)
    not_net.AddExternalOutput(out_blob)

    return not_net, out_blob


def _CopyConditionBlobNet(condition_blob):
    """Make a condition net that copies the condition_blob

    Args:
    condition_blob is a single bool.

    returns
    not_net: the net NOT the input
    out_blob: the output blob of the not_net
    """
    condition_net = core.Net('copy_condition_blob_net')
    out_blob = condition_net.Copy(condition_blob)
    condition_net.AddExternalOutput(out_blob)

    return condition_net, out_blob


def MergeConditionNets(name, condition_nets, relation):
    """
    Merge multi condition nets into a single condition nets.

    Args:
        name: name of the new condition net.
        condition_nets: a list of condition nets. The last external_output
                        of each condition net must be single bool value.
        relation: can be 'And' or 'Or'.

    Returns:
        - A new condition net. Its last external output is relation of all
          condition_nets.
    """
    if not isinstance(condition_nets, list):
        return condition_nets
    if len(condition_nets) <= 1:
        return condition_nets[0] if condition_nets else None

    merged_net = core.Net(name)
    for i in range(len(condition_nets)):
        net_proto = condition_nets[i].Proto()
        assert net_proto.device_option == merged_net.Proto().device_option
        assert net_proto.type == merged_net.Proto().type
        merged_net.Proto().op.extend(net_proto.op)
        merged_net.Proto().external_input.extend(net_proto.external_input)
        # discard external outputs as we're combining them together
        curr_cond = GetConditionBlobFromNet(condition_nets[i])
        if i == 0:
            last_cond = curr_cond
        else:
            last_cond = merged_net.__getattr__(relation)([last_cond, curr_cond])

    merged_net.AddExternalOutput(last_cond)

    return merged_net


def Do(*nets_or_steps):
    """
    Execute the sequence of nets or steps once.

    Examples:
    - Do(net1, net2, ..., net_n)
    - Do(list_of_nets)
    - Do(step1, step2, ..., step_n)
    - Do(list_of_steps)
    """
    if len(nets_or_steps) == 0:
        raise ValueError(
            'nets_or_steps cannot be empty.')
    elif len(nets_or_steps) == 1:
        nets_or_steps = nets_or_steps[0]
    else:
        nets_or_steps = list(nets_or_steps)

    return core.execution_step('Do', nets_or_steps)


def DoParallel(*nets_or_steps):
    """
    Execute the nets or steps in parallel, waiting for all of them to finish

    Examples:
    - DoParallel(net1, net2, ..., net_n)
    - DoParallel(list_of_nets)
    - DoParallel(step1, step2, ..., step_n)
    - DoParallel(list_of_steps)
    """
    if len(nets_or_steps) == 0:
        raise ValueError(
            'nets_or_steps cannot be empty.')
    elif len(nets_or_steps) == 1:
        nets_or_steps = nets_or_steps[0]
    else:
        nets_or_steps = list(nets_or_steps)

    return core.execution_step(
        'DoParallel', nets_or_steps, concurrent_substeps=True)


def _StopNet(stop_blob):
    stop_net = core.Net('stop_net')
    stop_net.ConstantFill(
        [], [stop_blob], shape=[], value=True, dtype=core.DataType.BOOL)
    return stop_net


def _ToExecutionStep(net_or_step):
    if isinstance(net_or_step, core.Net):
        return Do(net_or_step)
    elif isinstance(net_or_step, core.ExecutionStep):
        return net_or_step
    else:
        raise ValueError(
            'net_or_step must be a net or a step.')


def _RunOnceIf(condition_blob_or_net, net_or_step):
    """
    Execute net_or_step once if condition_blob_or_net evaluates as true.

    If condition_blob_or_net is Net, the condition is its last external_output
    that must be a single bool. And this net will be executed before net_or_step
    so as to get the condition.
    """
    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
        return Do(Do(condition_blob_or_net),
                  _RunOnceIf(condition_blob, net_or_step))

    stop_if_not_net, stop_blob = NotNet(condition_blob_or_net)
    stop_net = _StopNet(stop_blob)

    return core.execution_step(
        '_RunOnceIf',
        [Do(stop_if_not_net), _ToExecutionStep(net_or_step), Do(stop_net)],
        should_stop_blob=stop_blob)


def _RunOnceIfNot(condition_blob_or_net, net_or_step):
    """
    Similar to _RunOnceIf() but Execute net_or_step once if
    condition_blob_or_net evaluates as false.
    """
    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
        return Do(Do(condition_blob_or_net),
                  _RunOnceIfNot(condition_blob, net_or_step))

    stop_if_net, stop_blob = _CopyConditionBlobNet(condition_blob_or_net)
    stop_net = _StopNet(stop_blob)

    return core.execution_step(
        '_RunOnceIfNot',
        [Do(stop_if_net), _ToExecutionStep(net_or_step), Do(stop_net)],
        should_stop_blob=stop_blob)


def For(net_or_step, iter_num):
    """
    Execute net_or_step iter_num times.

    Args:
    net_or_step: an instance of a ExecutionStep or a Net.
    iter_num:    the number times to execute the net_or_step.

    Returns:
    A ExecutionStep instance.
    """
    init_net = core.Net('init-net')
    iter_cnt = init_net.CreateCounter([], init_count=iter_num)
    iter_net = core.Net('For-iter')
    iter_done = iter_net.CountDown([iter_cnt])

    if isinstance(net_or_step, core.Net):
        for_step = core.execution_step(
            'For', [iter_net, net_or_step], should_stop_blob=iter_done)
    elif isinstance(net_or_step, core.ExecutionStep):
        for_step = core.execution_step(
            'For', [Do(iter_net), net_or_step], should_stop_blob=iter_done)
    else:
        raise ValueError(
            'net_or_step must be a net or a step.')

    return Do(Do(init_net), for_step)


def While(condition_blob_or_net, net_or_step):
    """
    Execute net_or_step when condition_blob_or_net returns true.

    Args:
    condition_blob_or_net: If it is an instance of Net, its last
      external_output must be a single bool.
    net_or_step: an instance of a ExecutionStep or a Net.

    Returns:
    A ExecutionStep instance.
    """
    condition_not_net, stop_blob = NotNet(condition_blob_or_net)
    if isinstance(condition_blob_or_net, core.Net):
        condition_step = Do(condition_blob_or_net, condition_not_net)
    else:
        condition_step = Do(condition_not_net)

    return core.execution_step(
        'While',
        [condition_step, _ToExecutionStep(net_or_step)],
        should_stop_blob=stop_blob)


def Until(condition_blob_or_net, net_or_step):
    """
    Similar to While() but execute net_or_step when
    condition_blob_or_net returns false
    """
    if isinstance(condition_blob_or_net, core.Net):
        stop_blob = GetConditionBlobFromNet(condition_blob_or_net)
        condition_step = Do(condition_blob_or_net)
    else:
        copy_net, stop_blob = _CopyConditionBlobNet(condition_blob_or_net)
        condition_step = Do(copy_net)

    return core.execution_step(
        'Until',
        [condition_step, _ToExecutionStep(net_or_step)],
        should_stop_blob=stop_blob)


def DoWhile(condition_blob_or_net, net_or_step):
    """
    Execute net_or_step when condition_blob_or_net returns true. It will execute
    net_or_step at least once.

    Args:
    condition_blob_or_net: if it is an instance of Net, tts last external_output
      must be a single bool.
    net_or_step: an instance of a ExecutionStep or a Net.

    Returns:
    A ExecutionStep instance.
    """
    condition_not_net, stop_blob = NotNet(condition_blob_or_net)
    if isinstance(condition_blob_or_net, core.Net):
        condition_step = Do(condition_blob_or_net, condition_not_net)
    else:
        condition_step = Do(condition_not_net)

    return core.execution_step(
        'DoWhile',
        [_ToExecutionStep(net_or_step), condition_step],
        should_stop_blob=stop_blob)


def DoUntil(condition_blob_or_net, net_or_step):
    """
    Similar to DoWhile() but execute net_or_step when
    condition_blob_or_net returns false
    """
    steps = [_ToExecutionStep(net_or_step)]

    if isinstance(condition_blob_or_net, core.Net):
        steps.append(Do(condition_blob_or_net))
        stop_blob = GetConditionBlobFromNet(condition_blob_or_net)
    else:
        stop_blob = condition_blob_or_net

    stop_blob = core.BlobReference(str(stop_blob))
    return core.execution_step('DoUntil', steps, should_stop_blob=stop_blob)


def Switch(*conditions):
    """
    Execute the steps for which the condition is true.
    Each condition is a tuple (condition_blob_or_net, step).
    Note:
      1. Multi steps can be executed if their conditions are true.
      2. The conditions_blob_or_net (if it is Net) of all steps will be
         executed once.

    Examples:
    - Switch((cond_1, net_1), (cond_2, net_2), ..., (cond_n, net_n))
    - Switch([(cond_1, net1), (cond_2, net_2), ..., (cond_n, net_n)])
    - Switch((cond_1, net_1))
    """
    if len(conditions) == 0:
        raise ValueError(
            'conditions cannot be empty.')
    elif len(conditions) == 1:
        conditions = conditions[0]
        if not isinstance(conditions, list):
            conditions = [conditions]
    else:
        conditions = list(conditions)

    return core.execution_step(
        'Switch', [_RunOnceIf(cond, step) for cond, step in conditions])


def If(condition_blob_or_net, true_net_or_step, false_net_or_step=None):
    """
    condition_blob_or_net is first evaluated or executed. If the condition is
    true, true_net_or_step is then executed, otherwise, false_net_or_step
    is executed.

    If condition_blob_or_net is Net, the condition is its last external_output
    that must be a single bool. And this Net will be executred before both
    true/false_net_or_step so as to get the condition.
    """
    if not false_net_or_step:
        return _RunOnceIf(condition_blob_or_net, true_net_or_step)

    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
        return Do(Do(condition_blob_or_net),
                  If(condition_blob, true_net_or_step, false_net_or_step))

    condition_blob = condition_blob_or_net
    not_net, _ = NotNet(condition_blob)

    return Switch(
        (condition_blob, true_net_or_step),
        (not_net, false_net_or_step),
    )


def IfNot(condition_blob_or_net, true_net_or_step, false_net_or_step=None):
    """
    If condition_blob_or_net returns false, executes true_net_or_step,
    otherwise executes false_net_or_step
    """
    if not false_net_or_step:
        return _RunOnceIfNot(condition_blob_or_net, true_net_or_step)

    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
        return Do(Do(condition_blob_or_net),
                  IfNot(condition_blob, true_net_or_step, false_net_or_step))

    condition_blob = condition_blob_or_net
    not_net, _ = NotNet(condition_blob)

    return Switch(
        (condition_blob, false_net_or_step),
        (not_net, true_net_or_step),
    )
