# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package onnx
# Module caffe2.python.onnx.frontend

"""Caffe2 Protobuf to ONNX converter

To run this, you will need to have Caffe2 installed as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import collections
import logging
import re

from caffe2.python import core as caffe2_core
from enum import Enum
from onnx import (defs, checker, helper, numpy_helper, mapping,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorSetIdProto)
from onnx.helper import make_tensor, make_tensor_value_info
import numpy as np

from caffe2.python.onnx.helper import make_model, c2_native_run_net, dummy_name
from caffe2.python.onnx.error import Unsupported

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Caffe2Frontend(object):
    # This number controls the semantics of the operators we target.  Whenever
    # ONNX makes a BC breaking change to semantics of operators, having this set
    # to an accurate number will prevent our models form exporting.  However,
    # we should strive to keep this up-to-date as much as possible.
    _target_opset_version = 3

    _renamed_operators = {
        'SpatialBN': 'BatchNormalization',
        'Conv1D': 'Conv',
        'Conv2D': 'Conv',
        'Conv3D': 'Conv',
        'ConvTranspose1D': 'ConvTranspose',
        'ConvTranspose2D': 'ConvTranspose',
        'ConvTranspose3D': 'ConvTranspose',
        'MaxPool1D': 'MaxPool',
        'MaxPool2D': 'MaxPool',
        'MaxPool3D': 'MaxPool',
        'AveragePool1D': 'AveragePool',
        'AveragePool2D': 'AveragePool',
        'AveragePool3D': 'AveragePool',
    }

    # caffe2 arguments that are completely removed in onnx
    _blacklist_caffe2_args = {
        'order': {b'NCHW'},
        'cudnn_exhaustive_search': {0, 1},
        'use_cudnn': {0, 1},
    }

    _global_renamed_args = {
        'kernels': 'kernel_shape',
    }

    _per_op_renamed_args = {
        'Squeeze': {'dims': 'axes'},
        'Transpose': {'axes': 'perm'},
    }

    _special_operators = {
        'Conv': '_create_conv_pool_op',
        'ConvTranspose': '_create_conv_pool_op',
        'ChannelShuffle': '_create_channel_shuffle',
        'MaxPool': '_create_conv_pool_op',
        'AveragePool': '_create_conv_pool_op',
        'Concat': '_create_concat',
        'FC': '_create_gemm',
        'LRN': '_create_lrn',
        'Slice': '_create_slice',
        'Reshape': '_create_reshape',
    }

    @classmethod
    def _common_caffe2_arg_to_onnx_attr(cls, op_def, arg):
        # name
        op_type = op_def.type
        if op_type in cls._per_op_renamed_args:
            name = cls._per_op_renamed_args[op_type].get(
                arg.name, arg.name)
        else:
            name = cls._global_renamed_args.get(arg.name, arg.name)

        # value
        if arg.HasField('f'):
            value = arg.f
        elif arg.HasField('i'):
            value = arg.i
        elif arg.HasField('s'):
            value = arg.s
        elif arg.floats:
            value = arg.floats
        elif arg.ints:
            value = arg.ints
        elif arg.strings:
            value = arg.strings
        else:
            raise ValueError('Could not find data field in arg: {}'.format(arg))

        if name in cls._blacklist_caffe2_args:
            assert value in cls._blacklist_caffe2_args[arg.name]
            return None

        return helper.make_attribute(name, value)

    @classmethod
    def caffe2_arg_to_onnx_attr(cls, op_def, arg):
        return cls._common_caffe2_arg_to_onnx_attr(op_def, arg)

    @classmethod
    def _common_caffe2_op_to_onnx_node(cls, op_def, shapes):
        node_def = NodeProto()
        node_def.name = op_def.name

        node_def.op_type = cls._renamed_operators.get(op_def.type, op_def.type)

        node_def.input.extend(op_def.input)
        node_def.output.extend(op_def.output)

        attrs = filter(None, [cls.caffe2_arg_to_onnx_attr(op_def, arg)
                              for arg in op_def.arg])
        node_def.attribute.extend(attrs)

        return node_def

    @classmethod
    def _create_concat(cls, op_def, shapes):
        node = cls._common_caffe2_op_to_onnx_node(op_def, shapes)
        if len(node.output) == 2:
            del node.output[1]
        explicit_axis = any(arg.name == 'axis' for arg in op_def.arg)
        if not explicit_axis:
            node.attribute.extend([helper.make_attribute('axis', 1)])
        return node

    @classmethod
    def _create_reshape(cls, op_def, shapes):
        node = cls._common_caffe2_op_to_onnx_node(op_def, shapes)
        if len(node.output) == 2:
            del node.output[1]
        return node

    @classmethod
    def _create_conv_pool_op(cls, op_def, shapes):
        node = cls._common_caffe2_op_to_onnx_node(op_def, shapes)

        if node.op_type in ['MaxPool', 'AveragePool']:
            for i, attr in enumerate(node.attribute):
                if attr.name == 'global_pooling' and attr.i:
                    node.op_type = 'Global{}'.format(node.op_type)
                    del node.attribute[i]
                    break

        attrs = {attr.name: attr for attr in node.attribute}
        def apply_trans(k, dim=2, ks=None):
            ks = ks or (k + 's')
            if dim == 2:
                k_h, k_w = k + '_h', k + '_w'
            else:
                k_t, k_l, k_b, k_r = k + '_t', k + '_l', k + '_b', k + '_r'

            vals = None
            if (dim == 2 and k_h in attrs and k_w in attrs):
                vals = [attrs[k_h].i, attrs[k_w].i]
                del attrs[k_h]
                del attrs[k_w]
            elif (dim == 4 and
                  k_t in attrs and k_l in attrs and k_b in attrs and k_r in attrs):
                vals = [attrs[k_t].i,
                        attrs[k_l].i,
                        attrs[k_b].i,
                        attrs[k_r].i]
                del attrs[k_t]
                del attrs[k_l]
                del attrs[k_b]
                del attrs[k_r]
            elif k in attrs:
                vals = [attrs[k].i] * dim
                del attrs[k]

            if vals and not node.op_type.startswith('Global'):
                attrs[ks] = helper.make_attribute(ks, vals)

        apply_trans('kernel', ks='kernel_shape')
        apply_trans('stride')
        apply_trans('dilation')
        apply_trans('adj')
        apply_trans('pad', 4)

        del node.attribute[:]
        node.attribute.extend(attrs.values())
        return node

    @classmethod
    def _create_gemm(cls, op_def, shapes):
        x, w, b = op_def.input
        args = {arg.name: arg for arg in op_def.arg}
        y, = op_def.output
        x_shape = list(shapes[x])

        nodes = []
        if 'axis' in args:
            axis = args['axis'].i
            outer = np.prod(x_shape[:axis]).astype(int)
            inner = np.prod(x_shape[axis:]).astype(int)
            reshaped_x = dummy_name()
            nodes.append(helper.make_node(
                'Reshape',
                inputs=[x],
                outputs=[reshaped_x],
                shape=[outer, inner],
            ))
            x = reshaped_x

        if 'axis_w' in args:
            axis_w = args['axis_w'].i
            w_shape = shapes[w]
            outer = np.prod(w_shape[:axis_w]).astype(int).item()
            inner = np.prod(w_shape[axis_w:]).astype(int).item()
            reshaped_w = dummy_name()
            nodes.append(helper.make_node(
                'Reshape',
                inputs=[w],
                outputs=[reshaped_w],
                shape=[outer, inner],
            ))
            w = reshaped_w

        gemm_y_output = dummy_name() if 'axis' in args else y
        nodes.append(helper.make_node(
            'Gemm',
            inputs=[x, w, b],
            outputs=[gemm_y_output],
            name=op_def.name,
            transB=1,
            broadcast=1,
        ))

        if 'axis' in args:
            axis = args['axis'].i
            nodes.append(helper.make_node(
                'Reshape',
                inputs=[gemm_y_output],
                outputs=[y],
                shape=x_shape[:axis] + [-1],
            ))

        return nodes

    @classmethod
    def _create_lrn(cls, op_def, shapes):
        node = cls._common_caffe2_op_to_onnx_node(op_def, shapes)
        if len(node.output) == 2:
            del node.output[1]
        return node

    @classmethod
    def _create_slice(cls, op_def, shapes):
        if len(op_def.input) > 1:
            raise Unsupported(
                'ONNX Slice operator does not support dynamic slice.')
        node = cls._common_caffe2_op_to_onnx_node(op_def, shapes)
        attrs = {attr.name: attr for attr in node.attribute}
        ndims = len(attrs['starts'].ints)

        node.attribute.extend([helper.make_attribute('axes', range(ndims))])

        data, = node.input
        shape = shapes[data]

        ends = attrs['ends'].ints
        for i, end in enumerate(ends):
            if end >= 0:
                continue
            if end == -1:
                end = shape[i]
            else:
                end = end + 1
            ends[i] = end

        return node

    @classmethod
    def _create_channel_shuffle(cls, op_def, shapes):
        x, = op_def.input
        y, = op_def.output
        n, c, h, w = shapes[x]
        args = {arg.name: arg for arg in op_def.arg}
        g = args['group'].i
        assert c % g == 0

        nodes = []

        tmp1 = dummy_name()
        nodes.append(helper.make_node(
            'Reshape',
            inputs=[x],
            outputs=[tmp1],
            shape=[n, g, c // g, h, w],
        ))

        tmp2 = dummy_name()
        nodes.append(helper.make_node(
            'Transpose',
            inputs=[tmp1],
            outputs=[tmp2],
            perm=[0, 2, 1, 3, 4],
        ))

        nodes.append(helper.make_node(
            'Reshape',
            inputs=[tmp2],
            outputs=[y],
            shape=[n, c, h, w],
        ))
        return nodes

    @classmethod
    def caffe2_op_to_onnx_node(cls, op_def, shapes):
        if op_def.type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[op_def.type])
        else:
            translator = cls._common_caffe2_op_to_onnx_node
        nodes = translator(op_def, shapes)
        if not isinstance(nodes, collections.Iterable):
            nodes = [nodes]
        return nodes

    @staticmethod
    def _all_names_in_net(net):
        if net is None:
            return set()

        names = set()
        names.update(net.external_input)
        names.update(net.external_output)
        for op in net.op:
            names.update(op.input)
            names.update(op.output)
        return names

    @classmethod
    def caffe2_net_to_onnx_graph(cls,
                                 predict_net,
                                 init_net=None,
                                 value_info=None):
        if value_info is None:
            value_info = {}
        if not isinstance(value_info, dict):
            raise ValueError('Please pass value_info as a '
                             'name -> (type, shape) dictionary')

        cls._ssa_rewrite(predict_net, init_net, value_info)

        if init_net:
            initializer = cls.caffe2_init_net_to_initializer(init_net)
            value_info.update({init.name: (init.data_type, init.dims)
                               for init in initializer})
        else:
            initializer = []

        # Check whether we have got type shape info of all input
        missing = (set(list(predict_net.external_input)) -
                   set(value_info.keys()))
        if missing:
            raise RuntimeError('Could not find value info of inputs: {}'.format(
                ', '.join(missing)))

        inputs = {}
        for name in predict_net.external_input:
            elem_type, shape = value_info[name]
            inputs[name] = np.random.randn(*shape).astype(
                mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type])

        ws, outputs = c2_native_run_net(
            init_net,
            predict_net,
            inputs)

        for name in predict_net.external_output:
            output = outputs[name]
            elem_type = mapping.NP_TYPE_TO_TENSOR_TYPE[output.dtype]
            shape = output.shape
            value_info[name] = (elem_type, shape)

        graph_def = GraphProto()
        graph_def.name = predict_net.name
        graph_def.initializer.extend(initializer)
        # This is a mapping from Caffe2 names to ONNX names
        graph_def.input.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in predict_net.external_input)

        dummy_name(cls._all_names_in_net(predict_net) |
                   cls._all_names_in_net(init_net))

        for op in predict_net.op:
            shapes = {}
            for name in itertools.chain(op.input, op.output):
                blob = ws.FetchBlob(name)
                if hasattr(blob, 'shape'):
                    shapes[name] = blob.shape
            graph_def.node.extend(
                cls.caffe2_op_to_onnx_node(
                    op, shapes=shapes))

        all_output = set(sum((list(node.output) for node in graph_def.node),
                             [init.name for init in graph_def.initializer]))
        redundant_output = set(vi.name for vi in graph_def.output) - all_output
        if redundant_output:
            logger.warning(
                'There are graph output not produced by any node or initializer: {}'
                '! Will drop them.'.format(', '.join(redundant_output)))
        graph_def.output.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in predict_net.external_output
            if name in all_output)

        cls._annotate_consumed(graph_def)
        checker.check_graph(graph_def)
        return graph_def

    @classmethod
    def caffe2_init_net_to_initializer(cls, init_net):
        initializer = []
        for op in init_net.op:
            assert not op.input
            try:
                data_type, field_name = {
                    'GivenTensorFill': (TensorProto.FLOAT, 'floats'),
                    'GivenTensorInt64Fill': (TensorProto.INT64, 'ints'),
                    'GivenTensorIntFill': (TensorProto.INT32, 'ints'),
                    'GivenTensorBoolFill': (TensorProto.BOOL, 'ints'),
                    'GivenTensorStringFill': (TensorProto.STRING, 'strings'),
                }[op.type]
            except KeyError:
                raise RuntimeError(
                    "Can not translate init_net with operator '{}' "
                    "to initializer".format(op.type)
                )
            raw = (data_type != TensorProto.STRING)
            args = {a.name: a for a in op.arg}
            vals = getattr(args['values'], field_name)
            if raw:
                vals = np.asarray(
                    vals,
                    dtype=mapping.TENSOR_TYPE_TO_NP_TYPE[data_type]).tobytes()
            initializer.append(make_tensor(
                name=op.output[0],
                data_type=data_type,
                dims=args['shape'].ints,
                vals=vals,
                raw=raw,
            ))
        return initializer

    @classmethod
    def _annotate_consumed(cls, graph_def):
        for node in graph_def.node:
            schema = defs.get_schema(node.op_type)
            consumes = []
            for i, _input_name in enumerate(node.input):
                consume_type, output_idx = schema.consumed(i)
                if consume_type == defs.OpSchema.UseType.CONSUME_ENFORCED:
                    consumes.append(1)
                else:
                    consumes.append(0)

            if any(consumes):
                node.attribute.extend([helper.make_attribute(
                    'consumed_inputs',
                    consumes,
                )])

    @classmethod
    def _ssa_rewrite(cls, net, init_net, value_info):
        def ssa_name(name, version):
            return '{}_{}'.format(name, version)

        if init_net:
            for op in init_net.op:
                assert re.match('GivenTensor.*Fill', op.type)
                assert len(op.output) == 1
                op.output[0] = ssa_name(op.output[0], 0)
            init_net.external_input[:] = [ssa_name(name, 0)
                                          for name in init_net.external_input]
            init_net.external_output[:] = [ssa_name(name, 0)
                                           for name in init_net.external_output]
        if value_info:
            ssa_value_info = {ssa_name(name, 0): value
                              for name, value in value_info.items()}
            value_info.clear()
            value_info.update(ssa_value_info)
        net.external_input[:] = [ssa_name(name, 0)
                                 for name in net.external_input]
        ssa, blob_versions = caffe2_core.get_ssa(net)
        assert len(net.op) == len(ssa)
        for op, (versioned_inputs, versioned_outputs) in zip(net.op, ssa):
            op.input[:] = [ssa_name(name, version)
                           for name, version in versioned_inputs]
            op.output[:] = [ssa_name(name, version)
                            for name, version in versioned_outputs]
        net.external_output[:] = [ssa_name(name, blob_versions[name])
                                  for name in net.external_output]

    @classmethod
    def caffe2_net_to_onnx_model(cls, *args, **kwargs):
        model = make_model(cls.caffe2_net_to_onnx_graph(*args, **kwargs))
        opset_id = OperatorSetIdProto()
        opset_id.domain = ''  # ONNX
        opset_id.version = cls._target_opset_version
        model.opset_import.extend([opset_id])
        checker.check_model(model)
        return model


caffe2_net_to_onnx_graph = Caffe2Frontend.caffe2_net_to_onnx_graph
caffe2_net_to_onnx_model = Caffe2Frontend.caffe2_net_to_onnx_model
caffe2_init_net_to_initializer = Caffe2Frontend.caffe2_init_net_to_initializer
