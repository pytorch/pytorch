## @package control
# Module caffe2.python.control
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
from future.utils import viewitems


# Used to generate names of the steps created by the control functions.
# It is actually the internal index of these steps.
_current_idx = 1
_used_step_names = set()


def _get_next_step_name(control_name, base_name):
    global _current_idx, _used_step_names
    concat_name = '%s/%s' % (base_name, control_name)
    next_name = concat_name
    while next_name in _used_step_names:
        next_name = '%s_%d' % (concat_name, _current_idx)
        _current_idx += 1
    _used_step_names.add(next_name)
    return next_name


def _MakeList(input):
    """ input is a tuple.
    Example:
    (a, b, c)   --> [a, b, c]
    (a)         --> [a]
    ([a, b, c]) --> [a, b, c]
    """
    if len(input) == 0:
        raise ValueError(
            'input cannot be empty.')
    elif len(input) == 1:
        output = input[0]
        if not isinstance(output, list):
            output = [output]
    else:
        output = list(input)
    return output


def _IsNets(nets_or_steps):
    if isinstance(nets_or_steps, list):
        return all(isinstance(n, core.Net) for n in nets_or_steps)
    else:
        return isinstance(nets_or_steps, core.Net)


def _PrependNets(nets_or_steps, *nets):
    nets_or_steps = _MakeList((nets_or_steps,))
    nets = _MakeList(nets)
    if _IsNets(nets_or_steps):
        return nets + nets_or_steps
    else:
        return [Do('prepend', nets)] + nets_or_steps


def _AppendNets(nets_or_steps, *nets):
    nets_or_steps = _MakeList((nets_or_steps,))
    nets = _MakeList(nets)
    if _IsNets(nets_or_steps):
        return nets_or_steps + nets
    else:
        return nets_or_steps + [Do('append', nets)]


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


def BoolNet(*blobs_with_bool_value):
    """A net assigning constant bool values to blobs. It is mainly used for
    initializing condition blobs, for example, in multi-task learning, we
    need to access reader_done blobs before reader_net run. In that case,
    the reader_done blobs must be initialized.

    Args:
    blobs_with_bool_value: one or more (blob, bool_value) pairs. The net will
    assign each bool_value to the corresponding blob.

    returns
    bool_net: A net assigning constant bool values to blobs.

    Examples:
    - BoolNet((blob_1, bool_value_1), ..., (blob_n, bool_value_n))
    - BoolNet([(blob_1, net1), ..., (blob_n, bool_value_n)])
    - BoolNet((cond_1, bool_value_1))
    """
    blobs_with_bool_value = _MakeList(blobs_with_bool_value)
    bool_net = core.Net('bool_net')
    for blob, bool_value in blobs_with_bool_value:
        out_blob = bool_net.ConstantFill(
            [],
            [blob],
            shape=[],
            value=bool_value,
            dtype=core.DataType.BOOL)
        bool_net.AddExternalOutput(out_blob)

    return bool_net


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
        # merge attributes
        for k, v in viewitems(condition_nets[i]._attr_dict):
            merged_net._attr_dict[k] += v

    merged_net.AddExternalOutput(last_cond)

    return merged_net


def CombineConditions(name, condition_nets, relation):
    """
    Combine conditions of multi nets into a single condition nets. Unlike
    MergeConditionNets, the actual body of condition_nets is not copied into
    the combine condition net.

    One example is about multi readers. Each reader net has a reader_done
    condition. When we want to check whether all readers are done, we can
    use this function to build a new net.

    Args:
        name: name of the new condition net.
        condition_nets: a list of condition nets. The last external_output
                        of each condition net must be single bool value.
        relation: can be 'And' or 'Or'.

    Returns:
        - A new condition net. Its last external output is relation of all
          condition_nets.
    """
    if not condition_nets:
        return None
    if not isinstance(condition_nets, list):
        raise ValueError('condition_nets must be a list of nets.')

    if len(condition_nets) == 1:
        condition_blob = GetConditionBlobFromNet(condition_nets[0])
        condition_net, _ = _CopyConditionBlobNet(condition_blob)
        return condition_net

    combined_net = core.Net(name)
    for i in range(len(condition_nets)):
        curr_cond = GetConditionBlobFromNet(condition_nets[i])
        if i == 0:
            last_cond = curr_cond
        else:
            last_cond = combined_net.__getattr__(relation)(
                [last_cond, curr_cond])

    combined_net.AddExternalOutput(last_cond)

    return combined_net


def Do(name, *nets_or_steps):
    """
    Execute the sequence of nets or steps once.

    Examples:
    - Do('myDo', net1, net2, ..., net_n)
    - Do('myDo', list_of_nets)
    - Do('myDo', step1, step2, ..., step_n)
    - Do('myDo', list_of_steps)
    """
    nets_or_steps = _MakeList(nets_or_steps)
    if (len(nets_or_steps) == 1 and isinstance(
            nets_or_steps[0], core.ExecutionStep)):
        return nets_or_steps[0]
    else:
        return core.scoped_execution_step(
            _get_next_step_name('Do', name), nets_or_steps)


def DoParallel(name, *nets_or_steps):
    """
    Execute the nets or steps in parallel, waiting for all of them to finish

    Examples:
    - DoParallel('pDo', net1, net2, ..., net_n)
    - DoParallel('pDo', list_of_nets)
    - DoParallel('pDo', step1, step2, ..., step_n)
    - DoParallel('pDo', list_of_steps)
    """
    nets_or_steps = _MakeList(nets_or_steps)
    if (len(nets_or_steps) == 1 and isinstance(
            nets_or_steps[0], core.ExecutionStep)):
        return nets_or_steps[0]
    else:
        return core.scoped_execution_step(
            _get_next_step_name('DoParallel', name),
            nets_or_steps,
            concurrent_substeps=True)


def _RunOnceIf(name, condition_blob_or_net, nets_or_steps):
    """
    Execute nets_or_steps once if condition_blob_or_net evaluates as true.

    If condition_blob_or_net is Net, the condition is its last external_output
    that must be a single bool. And this net will be executed before
    nets_or_steps so as to get the condition.
    """
    condition_not_net, stop_blob = NotNet(condition_blob_or_net)
    if isinstance(condition_blob_or_net, core.Net):
        nets_or_steps = _PrependNets(
            nets_or_steps, condition_blob_or_net, condition_not_net)
    else:
        nets_or_steps = _PrependNets(nets_or_steps, condition_not_net)

    def if_step(control_name):
        return core.scoped_execution_step(
            _get_next_step_name(control_name, name),
            nets_or_steps,
            should_stop_blob=stop_blob,
            only_once=True,
        )

    if _IsNets(nets_or_steps):
        bool_net = BoolNet((stop_blob, False))
        return Do(name + '/_RunOnceIf',
                  bool_net, if_step('_RunOnceIf-inner'))
    else:
        return if_step('_RunOnceIf')


def _RunOnceIfNot(name, condition_blob_or_net, nets_or_steps):
    """
    Similar to _RunOnceIf() but Execute nets_or_steps once if
    condition_blob_or_net evaluates as false.
    """
    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
        nets_or_steps = _PrependNets(nets_or_steps, condition_blob_or_net)
    else:
        copy_net, condition_blob = _CopyConditionBlobNet(condition_blob_or_net)
        nets_or_steps = _PrependNets(nets_or_steps, copy_net)

    return core.scoped_execution_step(
        _get_next_step_name('_RunOnceIfNot', name),
        nets_or_steps,
        should_stop_blob=condition_blob,
        only_once=True,
    )


def For(name, nets_or_steps, iter_num):
    """
    Execute nets_or_steps iter_num times.

    Args:
    nets_or_steps: a ExecutionStep or a Net or a list of ExecutionSteps or
                   a list nets.
    iter_num:    the number times to execute the nets_or_steps.

    Returns:
    A ExecutionStep instance.
    """
    init_net = core.Net('init-net')
    iter_cnt = init_net.CreateCounter([], init_count=iter_num)
    iter_net = core.Net('For-iter')
    iter_done = iter_net.CountDown([iter_cnt])

    for_step = core.scoped_execution_step(
        _get_next_step_name('For-inner', name),
        _PrependNets(nets_or_steps, iter_net),
        should_stop_blob=iter_done)
    return Do(name + '/For',
              Do(name + '/For-init-net', init_net),
              for_step)


def While(name, condition_blob_or_net, nets_or_steps):
    """
    Execute nets_or_steps when condition_blob_or_net returns true.

    Args:
    condition_blob_or_net: If it is an instance of Net, its last
      external_output must be a single bool.
    nets_or_steps: a ExecutionStep or a Net or a list of ExecutionSteps or
                   a list nets.

    Returns:
    A ExecutionStep instance.
    """
    condition_not_net, stop_blob = NotNet(condition_blob_or_net)
    if isinstance(condition_blob_or_net, core.Net):
        nets_or_steps = _PrependNets(
            nets_or_steps, condition_blob_or_net, condition_not_net)
    else:
        nets_or_steps = _PrependNets(nets_or_steps, condition_not_net)

    def while_step(control_name):
        return core.scoped_execution_step(
            _get_next_step_name(control_name, name),
            nets_or_steps,
            should_stop_blob=stop_blob,
        )

    if _IsNets(nets_or_steps):
        # In this case, while_step has sub-nets:
        # [condition_blob_or_net, condition_not_net, nets_or_steps]
        # If stop_blob is pre-set to True (this may happen when While() is
        # called twice), the loop will exit after executing
        # condition_blob_or_net. So we use BootNet to set stop_blob to
        # False.
        bool_net = BoolNet((stop_blob, False))
        return Do(name + '/While', bool_net, while_step('While-inner'))
    else:
        return while_step('While')


def Until(name, condition_blob_or_net, nets_or_steps):
    """
    Similar to While() but execute nets_or_steps when
    condition_blob_or_net returns false
    """
    if isinstance(condition_blob_or_net, core.Net):
        stop_blob = GetConditionBlobFromNet(condition_blob_or_net)
        nets_or_steps = _PrependNets(nets_or_steps, condition_blob_or_net)
    else:
        stop_blob = core.BlobReference(str(condition_blob_or_net))

    return core.scoped_execution_step(
        _get_next_step_name('Until', name),
        nets_or_steps,
        should_stop_blob=stop_blob)


def DoWhile(name, condition_blob_or_net, nets_or_steps):
    """
    Execute nets_or_steps when condition_blob_or_net returns true. It will
    execute nets_or_steps before evaluating condition_blob_or_net.

    Args:
    condition_blob_or_net: if it is an instance of Net, tts last external_output
      must be a single bool.
    nets_or_steps: a ExecutionStep or a Net or a list of ExecutionSteps or
                   a list nets.

    Returns:
    A ExecutionStep instance.
    """
    condition_not_net, stop_blob = NotNet(condition_blob_or_net)
    if isinstance(condition_blob_or_net, core.Net):
        nets_or_steps = _AppendNets(
            nets_or_steps, condition_blob_or_net, condition_not_net)
    else:
        nets_or_steps = _AppendNets(nets_or_steps, condition_not_net)

    # If stop_blob is pre-set to True (this may happen when DoWhile() is
    # called twice), the loop will exit after executing the first net/step
    # in nets_or_steps. This is not what we want. So we use BootNet to
    # set stop_blob to False.
    bool_net = BoolNet((stop_blob, False))
    return Do(name + '/DoWhile', bool_net, core.scoped_execution_step(
        _get_next_step_name('DoWhile-inner', name),
        nets_or_steps,
        should_stop_blob=stop_blob,
    ))


def DoUntil(name, condition_blob_or_net, nets_or_steps):
    """
    Similar to DoWhile() but execute nets_or_steps when
    condition_blob_or_net returns false. It will execute
    nets_or_steps before evaluating condition_blob_or_net.

    Special case: if condition_blob_or_net is a blob and is pre-set to
    true, then only the first net/step of nets_or_steps will be executed and
    loop is exited. So you need to be careful about the initial value the
    condition blob when using DoUntil(), esp when DoUntil() is called twice.
    """
    if not isinstance(condition_blob_or_net, core.Net):
        stop_blob = core.BlobReference(condition_blob_or_net)
        return core.scoped_execution_step(
            _get_next_step_name('DoUntil', name),
            nets_or_steps,
            should_stop_blob=stop_blob)

    nets_or_steps = _AppendNets(nets_or_steps, condition_blob_or_net)
    stop_blob = GetConditionBlobFromNet(condition_blob_or_net)

    # If stop_blob is pre-set to True (this may happen when DoWhile() is
    # called twice), the loop will exit after executing the first net/step
    # in nets_or_steps. This is not what we want. So we use BootNet to
    # set stop_blob to False.
    bool_net = BoolNet((stop_blob, False))
    return Do(name + '/DoUntil', bool_net, core.scoped_execution_step(
        _get_next_step_name('DoUntil-inner', name),
        nets_or_steps,
        should_stop_blob=stop_blob,
    ))


def Switch(name, *conditions):
    """
    Execute the steps for which the condition is true.
    Each condition is a tuple (condition_blob_or_net, nets_or_steps).
    Note:
      1. Multi steps can be executed if their conditions are true.
      2. The conditions_blob_or_net (if it is Net) of all steps will be
         executed once.

    Examples:
    - Switch('name', (cond_1, net_1), (cond_2, net_2), ..., (cond_n, net_n))
    - Switch('name', [(cond_1, net1), (cond_2, net_2), ..., (cond_n, net_n)])
    - Switch('name', (cond_1, net_1))
    """
    conditions = _MakeList(conditions)
    return core.scoped_execution_step(
        _get_next_step_name('Switch', name),
        [_RunOnceIf(name + '/Switch', cond, step) for cond, step in conditions])


def SwitchNot(name, *conditions):
    """
    Similar to Switch() but execute the steps for which the condition is False.
    """
    conditions = _MakeList(conditions)
    return core.scoped_execution_step(
        _get_next_step_name('SwitchNot', name),
        [_RunOnceIfNot(name + '/SwitchNot', cond, step)
         for cond, step in conditions])


def If(name, condition_blob_or_net,
       true_nets_or_steps, false_nets_or_steps=None):
    """
    condition_blob_or_net is first evaluated or executed. If the condition is
    true, true_nets_or_steps is then executed, otherwise, false_nets_or_steps
    is executed.

    If condition_blob_or_net is Net, the condition is its last external_output
    that must be a single bool. And this Net will be executred before both
    true/false_nets_or_steps so as to get the condition.
    """
    if not false_nets_or_steps:
        return _RunOnceIf(name + '/If',
                          condition_blob_or_net, true_nets_or_steps)

    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
    else:
        condition_blob = condition_blob_or_net

    return Do(
        name + '/If',
        _RunOnceIf(name + '/If-true',
                   condition_blob_or_net, true_nets_or_steps),
        _RunOnceIfNot(name + '/If-false', condition_blob, false_nets_or_steps)
    )


def IfNot(name, condition_blob_or_net,
          true_nets_or_steps, false_nets_or_steps=None):
    """
    If condition_blob_or_net returns false, executes true_nets_or_steps,
    otherwise executes false_nets_or_steps
    """
    if not false_nets_or_steps:
        return _RunOnceIfNot(name + '/IfNot',
                             condition_blob_or_net, true_nets_or_steps)

    if isinstance(condition_blob_or_net, core.Net):
        condition_blob = GetConditionBlobFromNet(condition_blob_or_net)
    else:
        condition_blob = condition_blob_or_net

    return Do(
        name + '/IfNot',
        _RunOnceIfNot(name + '/IfNot-true',
                      condition_blob_or_net, true_nets_or_steps),
        _RunOnceIf(name + '/IfNot-false', condition_blob, false_nets_or_steps)
    )
