## @package stat
# Module caffe2.python.stat
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import copy
import logging
import numpy as np

from past.builtins import basestring
from caffe2.proto import caffe2_pb2
from caffe2.python import core, utils, workspace

def GatherStatInfo(predict_def, init_def, data_generator, data_name,
        device_option=None, skip_ops_in=[], skip_ops_out=[], kind='absmax'):
    """
    The gether statistics of abs max for the topology
    Inputs:
        predict_def:    Topology net definition
        init_def:       Topology external inputs
        data_generator: The generator to fetch data as a iterator
        data_name:      The data/input name
        device_option:  DeviceOption to run on
        skip_ops_in:    The operator type to skip gathering input info
        skip_ops_out:   The operator type to skip gathering output info
        kind:           The kind of stat info
    Outputs:
        NetDef:         New topology with stat info
    """
    if (kind == 'absmax'):
        return GatherAbsMax(predict_def, init_def, data_generator, data_name,
                device_option, skip_ops_in, skip_ops_out)
    else:
        raise ValueError("Unsupport stat info type: {}".format(kind))


def GatherAbsMax(predict_def, init_def, data_generator, data_name,
        device_option, skip_ops_in, skip_ops_out):
    def insert_max(mlist, pos, name, absmax):
        if len(mlist) == 0:
            mlist.append([pos, name, absmax])
            return
        for m in mlist:
            if m[0] != pos or m[1] != name:
                continue
            assert(len(absmax.shape) == 1 and m[2].shape == absmax.shape)
            m[2] = np.array([np.max([m[2][p], absmax[p]])
                for p in range(absmax.shape[0])]).astype(np.float32)
            return
        mlist.append([pos, name, absmax])
        return

    def has_arg(op, name):
        for arg in op.arg:
            if arg.name == name:
                return True
        return False

    orig_workspace = workspace.CurrentWorkspace()
    workspace.SwitchWorkspace("__gather_stat_abs_max__", True)
    init_def.device_option.CopyFrom(device_option)
    workspace.RunNetOnce(init_def)

    run_predict_def = copy.deepcopy(predict_def)
    for op in run_predict_def.op:
        op.device_option.CopyFrom(device_option)

    max_list = []
    for data in data_generator():
        workspace.FeedBlob(data_name, data, device_option)
        workspace.RunOperatorOnce(run_predict_def.op[0])
        for i, op in enumerate(run_predict_def.op[1:]):
            op_pos = i + 1
            if op.type not in skip_ops_in:
                for j, input_name in enumerate(op.input):
                    if op.type == 'Conv' and j == 1: continue
                    input_blob = workspace.FetchBlob(input_name)
                    abs_max = np.array(
                            [np.absolute(input_blob).max()]).astype(np.float32)
                    max_name = 'absmax_input_' + str(j)
                    insert_max(max_list, op_pos, max_name, abs_max)

            workspace.RunOperatorOnce(op)
            if op.type not in skip_ops_out:
                for m, output_name in enumerate(op.output):
                    output_blob = workspace.FetchBlob(output_name)
                    abs_max = np.array(
                            [np.absolute(output_blob).max()]).astype(np.float32)
                    max_name = 'absmax_output_' + str(m)
                    insert_max(max_list, op_pos, max_name, abs_max)

    absmax_predict_def = copy.deepcopy(predict_def)
    for m in max_list:
        op = absmax_predict_def.op[m[0]]
        max_arg = utils.MakeArgument(m[1], m[2])
        # save max vaules in predict_def as operator arguments
        op.arg.extend([max_arg])
        if op.type == 'Conv' and not has_arg(op, 'need_quantize'):
            qflag_arg = utils.MakeArgument('need_quantize', 1)
            op.arg.extend([qflag_arg])

    workspace.ResetWorkspace()
    workspace.SwitchWorkspace(orig_workspace)
    return absmax_predict_def
