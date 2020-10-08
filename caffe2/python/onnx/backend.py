## @package onnx
# Module caffe2.python.onnx.backend

"""Backend for running ONNX on Caffe2

To run this, you will need to have Caffe2 installed as well.
"""





import os
import collections
from subprocess import Popen, PIPE
import sys
import zipfile
import itertools

# When onnx is built against a version of protobuf that is older than
# that which is vendored with caffe2, onnx will crash if caffe2's
# vendored protobuf is loaded first. We can work around this by
# importing onnx first, which will cause it to go out and pick up the
# system protobuf.
import onnx.backend

import caffe2
from caffe2.python import core, workspace, rnn_cell, gru_cell
from caffe2.python.compatibility import container_abcs
from caffe2.python.model_helper import ModelHelper
from caffe2.proto import caffe2_pb2
import caffe2.python.utils
import numpy as np
import onnx
from onnx import checker, GraphProto, TensorProto, AttributeProto, ModelProto
import onnx.numpy_helper
import onnx.defs
import onnx.optimizer
import onnx.shape_inference
import onnx.utils
from onnx.backend.base import Backend, Device, DeviceType, namedtupledict

from caffe2.python.onnx.workspace import Workspace
from caffe2.python.onnx.backend_rep import Caffe2Rep
from caffe2.python.onnx.backend_cpp_rep import Caffe2CppRep

import caffe2.python._import_c_extension as C

import warnings

def force_unicode(s):
    try:
        return s.decode('utf-8')
    except AttributeError:
        return s

def get_device_option(device):
    m = {DeviceType.CPU: caffe2_pb2.CPU,
         DeviceType.CUDA: workspace.GpuDeviceType}
    return core.DeviceOption(m[device.type], device.device_id)


class OnnxAttributes(dict):
    """
    This is a more convenient way to work with ONNX/Caffe2 attributes
    that is not the protobuf representation.
    """
    @staticmethod
    def from_onnx(args):
        d = OnnxAttributes()
        for arg in args:
            d[arg.name] = convertAttributeProto(arg)
        return d

    def caffe2(self, kmap=lambda k: k):
        for k, v in self.items():
            if kmap(k) != '':
                yield caffe2.python.utils.MakeArgument(kmap(k), v)

# TODO: Move this into ONNX main library
def convertAttributeProto(onnx_arg):
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.

    NB: Tensor attribute gets returned as the straight proto.
    """
    if onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        return onnx_arg.t  # this is a proto!
    elif onnx_arg.HasField('g'):
        return Caffe2Backend._graph_to_net(onnx_arg.g, Caffe2Backend._known_opset_version)
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    elif len(onnx_arg.graphs):
        retval = []
        # TODO: this doesn't work with RNN ops
        for g in onnx_arg.graphs:
            retval.append(Caffe2Backend._graph_to_net(g, Caffe2Backend._known_opset_version))
        return retval
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


# TODO: Move this into ONNX main library
class OnnxNode(object):
    """
    Reimplementation of NodeProto from ONNX, but in a form
    more convenient to work with from Python.

    We may temporarily edit these nodes to get them into Caffe2 form,
    before actually translating into the Caffe2 protobuf, since this
    is easier than decomposing everything, and putting it back together
    when we're ready.
    """
    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.attrs = OnnxAttributes.from_onnx(node.attribute)
        self.inputs = list(node.input)
        self.outputs = list(node.output)


Caffe2Ops = collections.namedtuple('Caffe2Ops', ['ops', 'init_ops', 'interface_blobs'])


class Caffe2Backend(Backend):

    # The greatest version of the ONNX operator set which we are aware of.
    # Models whose version is larger than this will cause us to emit a warning
    # that we are attempting to translate on a "best effort" basis.
    #
    # If you increase this, make SURE you cross-reference all BC-breaking
    # changes from one version to the next, and any that you did not
    # implement, mark as broken in _broken_operators
    _known_opset_version = 9

    # This dictionary will record operators which are KNOWN to be
    # broken, so we give a good error message rather than do something
    # bogus and then fail.
    _broken_operators = {
        # 'BrokenOp': version_it_was_broken_in
    }

    # Operators that are different between Caffe2 and
    # ONNX but only in their name.
    # In most cases, this should be empty - as the effort of ONNX is
    # to unify the operator definitions.
    _renamed_operators = {
        'GlobalMaxPool':         'MaxPool',
        'GlobalAveragePool':     'AveragePool',
        'Pad':                   'PadImage',
        'Neg':                   'Negative',
        'BatchNormalization':    'SpatialBN',
        'InstanceNormalization': 'InstanceNorm',
        'MatMul':                'BatchMatMul',
        'Upsample':              'ResizeNearest',
        'Identity':              'Copy',
        'InstanceNormalization': 'InstanceNorm',
        'Equal':                 'EQ',
        'Less':                  'LT',
        'Greater':               'GT',
        'Unsqueeze':             'ExpandDims',
        'Loop':                  'ONNXWhile',
        'Tile':                  'NumpyTile',
        'RandomNormal':          'GaussianFill',
        'RandomUniform':         'UniformFill',
    }

    _global_renamed_attrs = {'kernel_shape': 'kernels'}
    _per_op_renamed_attrs = {
        'Squeeze':              {'axes': 'dims'},
        'Unsqueeze':            {'axes': 'dims'},
        'Transpose':            {'perm': 'axes'},
        'Upsample':             {'mode': '',
                                 'scales': ''},
        'ConvTranspose':        {'output_padding': 'adjs'},
        'Selu':                 {'gamma': 'scale'},
        'If':                   {'then_branch': 'then_net',
                                 'else_branch': 'else_net'},
        'RandomUniform':        {'low': 'min',
                                 'high': 'max'}
    }

    # operators whose behavior is different beyond renaming
    # the value is an attribute of this class that is a
    # function from ToffeIR node_def to caffe2 op_def
    _special_operators = {
        'LSTM': '_create_rnn_variant',
        'GRU': '_create_rnn_variant',
        'RNN': '_create_rnn_variant',
        'Loop': '_create_loop',
        'If': '_create_if',
        'Upsample': '_create_upsample',
        'RandomNormal': '_create_gaussian_fill'
    }

    # Dummy name generator
    _dummy_name = C.DummyName()

    @classmethod
    def dummy_name(cls):
        return cls._dummy_name.new_dummy_name()

    # NB: By default, you will use the LATEST definition of the operator,
    # so this interface MAY make BC-breaking changes.  Specify an
    # opset_version if you don't want this to version.
    @classmethod
    def run_node(cls, node, inputs, device='CPU', opset_version=_known_opset_version, outputs_info=None):
        super(Caffe2Backend, cls).run_node(node, inputs, device=device,
                                           outputs_info=outputs_info, opset_version=opset_version)

        value_infos = []
        device_option = get_device_option(Device(device))
        ws = Workspace()
        with core.DeviceScope(device_option):  # temporary!
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    ws.FeedBlob(key, value)
                    value_infos.append(onnx.helper.make_tensor_value_info(
                        name=key,
                        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype],
                        shape=value.shape).SerializeToString())
            else:
                assert len(node.input) == len(inputs), "{}: expected {} but got {}".format(
                    node.op_type, len(node.input), len(inputs))
                for key, value in zip(node.input, inputs):
                    ws.FeedBlob(key, value)
                    value_infos.append(onnx.helper.make_tensor_value_info(
                        name=key,
                        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype],
                        shape=value.shape).SerializeToString())

            ops = []
            cbackend = C.Caffe2Backend(cls._dummy_name)
            ops_str = cbackend.convert_node(node.SerializeToString(), value_infos, opset_version)
            for s in ops_str[0] + ops_str[1]:
                op = caffe2_pb2.OperatorDef()
                op.ParseFromString(s)
                op.device_option.CopyFrom(device_option)
                ops.append(op)
            ws.RunOperatorsOnce(ops)
            output_values = [ws.FetchBlob(name) for name in node.output]
            return namedtupledict('Outputs', node.output)(*output_values)

    @classmethod
    def _create_tensor_filling_op(cls, onnx_tensor, name=None):
        """
        Given an Onnx TensorProto, translate it into a Caffe2 operator
        which produces the given tensor filling op.
        """
        assert name or onnx_tensor.name
        name = name or onnx_tensor.name

        c2_op = caffe2_pb2.OperatorDef()

        c2_values = c2_op.arg.add()
        c2_values.name = "values"

        def tensor2list(onnx_tensor):
            # Use the onnx.numpy_helper because the data may be raw
            return onnx.numpy_helper.to_array(onnx_tensor).flatten().tolist()

        if onnx_tensor.data_type in [TensorProto.FLOAT]:
            c2_op.type = 'GivenTensorFill'
            c2_values.floats.extend(tensor2list(onnx_tensor))
        elif onnx_tensor.data_type in [TensorProto.DOUBLE]:
            c2_op.type = 'GivenTensorDoubleFill'
            c2_values.floats.extend(tensor2list(onnx_tensor))
        elif onnx_tensor.data_type in [TensorProto.INT64,
                                       TensorProto.UINT32]:
            c2_op.type = 'GivenTensorInt64Fill'
            c2_values.ints.extend(tensor2list(onnx_tensor))
        elif onnx_tensor.data_type in [TensorProto.UINT8,
                                       TensorProto.INT8,
                                       TensorProto.UINT16,
                                       TensorProto.INT16,
                                       TensorProto.INT32]:
            c2_op.type = 'GivenTensorIntFill'
            c2_values.ints.extend(tensor2list(onnx_tensor))
        elif onnx_tensor.data_type == TensorProto.BOOL:
            c2_op.type = 'GivenTensorBoolFill'
            c2_values.ints.extend(tensor2list(onnx_tensor))
        elif onnx_tensor.data_type == TensorProto.STRING:
            c2_op.type = 'GivenTensorStringFill'
            c2_values.strings.extend(onnx_tensor.string_data)
        else:
            raise RuntimeError(
                "unrecognized tensor type {}".format(onnx_tensor.data_type))

        c2_shape = c2_op.arg.add()
        c2_shape.name = "shape"
        c2_shape.ints.extend(onnx_tensor.dims)

        c2_op.output.append(name)

        return c2_op

    @classmethod
    def _rnn_reform_weights(cls, reforms, name, hidden_size, init_net, gates, reorder_indices):
        for name_from, name_to, do_concat, extra_dims in reforms:
            gate_blobs = ['%s/%s_%s' % (name, prefix, name_to) for prefix in gates]
            for i, x in enumerate(gate_blobs):
                dim0 = i * hidden_size, (i+1) * hidden_size
                starts, ends = zip(dim0, *extra_dims)
                init_net.Slice(name_from, x, starts=starts, ends=ends)
            if do_concat:
                reordered_gate_blobs = [gate_blobs[i] for i in reorder_indices]
                init_net.Concat(reordered_gate_blobs, ['%s/%s' % (name, name_to), cls.dummy_name()], axis=0)

    @classmethod
    def _make_rnn_direction(cls, input_blob, B, W, R, initial_states_and_names, sequence_lens,
                            pred_mh, init_net,
                            input_size, hidden_size, num_gates, direction_offset,
                            Bi, Br, W_, R_,
                            reform, make_cell, keep_outputs):
        name = cls.dummy_name()

        # input and recurrence biases are squashed together in onnx
        # but not in caffe2
        gates_hidden_size = num_gates * hidden_size
        bias_offset = 2 * direction_offset * gates_hidden_size
        weight_offset = direction_offset * gates_hidden_size
        Bi = init_net.Slice(B, name + Bi,
                            starts=[bias_offset + 0 * gates_hidden_size],
                            ends  =[bias_offset + 1 * gates_hidden_size])
        Br = init_net.Slice(B, name + Br,
                            starts=[bias_offset + 1 * gates_hidden_size],
                            ends  =[bias_offset + 2 * gates_hidden_size])
        W_ = init_net.Slice(W, name + W_,
                            starts=[weight_offset + 0 * gates_hidden_size, 0],
                            ends  =[weight_offset + 1 * gates_hidden_size,-1])
        R_ = init_net.Slice(R, name + R_,
                            starts=[weight_offset + 0 * gates_hidden_size, 0],
                            ends  =[weight_offset + 1 * gates_hidden_size,-1])

        initial_states_sliced = []
        for initial_state, name_suffix in initial_states_and_names:
            initial_states_sliced.append(
                pred_mh.net.Slice(initial_state, name + name_suffix,
                                  starts=[direction_offset + 0, 0, 0],
                                  ends  =[direction_offset + 1,-1,-1]))

        if direction_offset == 1:
            if sequence_lens is not None:
                seq_lens_for_reverse = sequence_lens
            else:
                input_shape = pred_mh.net.Shape(input_blob, name + '/input_shape')
                batch_size = pred_mh.net.Slice(input_shape, name + '/batch_size_slice', starts=[1], ends=[2])
                seq_len = pred_mh.net.Slice(input_shape, name + '/seq_len_slice', starts=[0], ends=[1])
                dummy_sequence_lens = pred_mh.net.Tile([seq_len, batch_size], name + '/dummy_sequence_lens', axis=0)
                pred_mh.net.Reshape(dummy_sequence_lens, [dummy_sequence_lens, cls.dummy_name()], shape=[-1])
                seq_lens_for_reverse = pred_mh.net.Cast(dummy_sequence_lens, name + '/seq_lens_for_reverse', to=core.DataType.INT32)
        reform(Bi, Br, W_, R_, name, hidden_size, init_net)

        if direction_offset == 1:
            input = pred_mh.net.ReversePackedSegs(
                [input_blob, seq_lens_for_reverse], name + "/input-reversed")
        else:
            input = input_blob

        outputs = keep_outputs(list(make_cell(
            pred_mh,
            input,
            sequence_lens,
            initial_states_sliced,
            input_size,
            hidden_size,
            name,
            drop_states=False,
            forward_only=True,
        )))

        if direction_offset == 1:
            outputs[0] = pred_mh.net.ReversePackedSegs(
                [outputs[0], seq_lens_for_reverse], name + "/output-reversed")

        return outputs

    @classmethod
    def _create_rnn_variant(cls, init_model, pred_model, n, opset_version):
        assert init_model is not None, "cannot convert RNNs without access to the full model"
        assert pred_model is not None, "cannot convert RNNs without access to the full model"

        attrs = dict(n.attrs) # make a copy, which is safe to mutate
        hidden_size = attrs.pop('hidden_size')
        direction = force_unicode(attrs.pop('direction', 'forward'))

        if n.op_type == 'RNN':
            activation = force_unicode(attrs.pop('activations', ('tanh',))[0].lower())
        elif n.op_type == 'GRU':
            linear_before_reset = attrs.pop('linear_before_reset', 0)

        assert not attrs, "unsupported RNN attributes: " + str(attrs.keys())
        assert direction in ['forward', 'bidirectional'], "unsupported backwards RNN/GRU/LSTM"

        if n.op_type in ['RNN', 'GRU']:
            input_blob, W, R, B, sequence_lens, initial_h = n.inputs
        elif n.op_type == 'LSTM':
            input_blob, W, R, B, sequence_lens, initial_h, initial_c = n.inputs

        if sequence_lens == "":
            sequence_lens = None

        for x in itertools.chain(init_model.graph.input,
                                 init_model.graph.value_info,
                                 pred_model.graph.input,
                                 pred_model.graph.value_info):
            if x.name == W:
                input_size = x.type.tensor_type.shape.dim[2].dim_value
                break
        else:
            raise RuntimeError("best-effort shape inference for RNN/GRU/LSTM failed")

        pred_mh = ModelHelper()
        init_net = core.Net("init-net")

        init_net.Reshape(W, [W, cls.dummy_name()], shape=[1,-1,0])
        init_net.Squeeze(W, W, dims=[0])
        init_net.Reshape(R, [R, cls.dummy_name()], shape=[1,-1,0])
        init_net.Squeeze(R, R, dims=[0])
        init_net.Reshape(B, [B, cls.dummy_name()], shape=[1,-1])
        init_net.Squeeze(B, B, dims=[0])

        if n.op_type == 'RNN':
            def reform(*args):
                pass

            def make_cell(*args, **kwargs):
                return rnn_cell.BasicRNN(*args, activation=activation, **kwargs)

            def make_rnn(direction_offset):
                return cls._make_rnn_direction(
                    input_blob, B, W, R, [(initial_h, '/initial_h')], sequence_lens,
                    pred_mh, init_net, input_size, hidden_size, 1, direction_offset,
                    "/i2h_b", "/gates_t_b", "/i2h_w", "/gates_t_w",
                    reform, make_cell, lambda x: x)

        elif n.op_type == 'GRU':
            def reform(Bi, Br, W_, R_, name, hidden_size, init_net):
                # caffe2 has a different order from onnx. We need to rearrange
                #  z r h  -> r z h
                reforms = ((W_, 'i2h_w',    True,  [(0,-1)]),
                           (R_, 'gate_t_w', False, [(0,-1)]),
                           (Bi, 'i2h_b',    True,  []),
                           (Br, 'gate_t_b', False, []))
                cls._rnn_reform_weights(reforms, name, hidden_size, init_net,
                                        ['update', 'reset', 'output'], [1, 0, 2])

            def make_cell(*args, **kwargs):
                return gru_cell.GRU(*args, linear_before_reset=linear_before_reset, **kwargs)

            def make_rnn(direction_offset):
                return cls._make_rnn_direction(
                    input_blob, B, W, R, [(initial_h, '/initial_h')], sequence_lens,
                    pred_mh, init_net, input_size, hidden_size, 3, direction_offset,
                    "_bias_i2h", "_bias_gates", "/i2h_w_pre", "/gates_t_w_pre",
                    reform, make_cell, lambda x: x)

        elif n.op_type == 'LSTM':
            def reform(Bi, Br, W_, R_, name, hidden_size, init_net):
                # caffe2 has a different order from onnx. We need to rearrange
                #   i o f c -> i f o c
                reforms = ((W_, 'i2h_w',     True, [(0, -1)]),
                           (R_, 'gates_t_w', True, [(0, -1)]),
                           (Bi, 'i2h_b'    , True, []),
                           (Br, 'gates_t_b', True, []))
                cls._rnn_reform_weights(reforms, name, hidden_size, init_net,
                                        ['input', 'output', 'forget', 'cell'], [0, 2, 1, 3])

            def make_cell(*args, **kwargs):
                return rnn_cell.LSTM(*args, **kwargs)

            def make_rnn(direction_offset):
                return cls._make_rnn_direction(
                    input_blob, B, W, R, [(initial_h, '/initial_h'), (initial_c, '/initial_c')], sequence_lens,
                    pred_mh, init_net, input_size, hidden_size, 4, direction_offset,
                    "/i2h_b", "/gates_t_b", "/i2h_w", "/gates_t_w",
                    reform, make_cell, lambda x: [x[0], x[1], x[3]])

        if direction == 'forward':
            outputs = make_rnn(0)

            # in the forward case, storage is shared between the
            # last outputs. We need to decouple them so that the
            # VariableLengthSequencePadding only mutates
            # n.outputs[0]
            for i in range(1, len(outputs)):
                pred_mh.net.Copy(outputs[i], n.outputs[i])

            if sequence_lens is not None:
                pred_mh.net.VariableLengthSequencePadding(
                    [outputs[0], sequence_lens], [outputs[0]])
            pred_mh.net.ExpandDims([outputs[0]], [n.outputs[0]], dims=[1])
        elif direction == 'bidirectional':
            outputs_f = make_rnn(0)
            outputs_b = make_rnn(1)

            concatted_output, _ = pred_mh.net.Concat(
                [outputs_f[0], outputs_b[0]], [cls.dummy_name(), cls.dummy_name()], axis=2)
            if sequence_lens is not None:
                pred_mh.net.VariableLengthSequencePadding(
                    [concatted_output, sequence_lens], [concatted_output])
            reshaped_output, _ = pred_mh.net.Reshape(concatted_output, [cls.dummy_name(), cls.dummy_name()], shape=[0,0,-1,2])
            pred_mh.net.Transpose(reshaped_output, n.outputs[0], axes=[0,2,1,3])
            for i in range(1, len(n.outputs)):
                pred_mh.net.Concat([outputs_f[i], outputs_b[i]],
                                   [n.outputs[i], cls.dummy_name()], axis=0)

        # We want to decide whether to put all of our weight-reshaping
        # operators in the init net or the predict net. We can put
        # them in the init net iff the inputs to those operators are
        # already available, either as graph initializers, or as the
        # output of other operators in the init net. The latter case
        # occurs, for example, when exporting from pytorch to onnx.
        # In most production use, we expect has_initializers to be
        # true.
        initializers = {i.name for i in init_model.graph.initializer}
        outputs = {output for node in init_model.graph.node for output in node.output}
        has_initializers = all(x in initializers or x in outputs for x in (W, R, B))

        pred_ops = []
        init_ops = []
        (init_ops if has_initializers else pred_ops).extend(init_net.Proto().op)
        pred_ops.extend(pred_mh.Proto().op)

        return Caffe2Ops(pred_ops, init_ops, list(pred_mh.Proto().external_input))

    @classmethod
    def _create_control_op(cls, init_model, pred_model, n, opset_version):
        control_inputs = []
        if '__control_inputs' in n.attrs:
            control_inputs.extend(n.attrs['__control_inputs'])
        node = cls._common_onnx_node_to_caffe2_op(init_model, pred_model, n, opset_version)
        node.control_input.extend(control_inputs)
        return Caffe2Ops([node], [], [])

    @classmethod
    def _remove_ssa(cls, net, remap_dict):
        for op in net.op:
            for i, name in enumerate(op.output):
                if name in remap_dict:
                    op.output[i] = remap_dict[name]
        for i, out in enumerate(net.external_output):
            if out in remap_dict:
                net.external_output[i] = remap_dict[out]

    @classmethod
    def _create_if(cls, init_model, pred_model, n, opset_version):
        ops = cls._create_control_op(init_model, pred_model, n, opset_version)
        assert ops[0][0].type == 'If'
        if_op = ops[0][0]
        then_net = else_net = None
        control_inputs = []
        for arg in if_op.arg:
            if arg.name == 'then_net':
                then_net = arg.n
            if arg.name == 'else_net':
                else_net = arg.n
            if arg.name == '__control_inputs':
                control_inputs = arg.strings

        assert then_net and else_net
        then_net_outs = then_net.external_output
        else_net_outs = else_net.external_output
        op_outputs = if_op.output
        assert len(then_net_outs) == len(else_net_outs)
        assert len(else_net_outs) == len(op_outputs)

        for arg in if_op.arg:
            if arg.name == 'then_net':
                arg.n.external_input.extend(control_inputs)
            if arg.name == 'else_net':
                arg.n.external_input.extend(control_inputs)

        return ops

    @classmethod
    def _create_loop(cls, init_model, pred_model, n, opset_version):
        ops = cls._create_control_op(init_model, pred_model, n, opset_version)
        assert ops[0][0].type == 'ONNXWhile'
        while_op = ops[0][0]
        while_op.arg.extend([caffe2.python.utils.MakeArgument('has_trip_count', True)])
        while_op.arg.extend([caffe2.python.utils.MakeArgument('has_cond', True)])
        while_op.arg.extend([caffe2.python.utils.MakeArgument('disable_scopes', True)])
        control_inputs = []
        for arg in while_op.arg:
            if arg.name == '__control_inputs':
                control_inputs = arg.strings
        num_loop_carried_deps = 0
        for arg in while_op.arg:
            if arg.name == 'body':
                num_loop_carried_deps = len(arg.n.external_input) - 2
                arg.n.external_input.extend(control_inputs)
        while_op.arg.extend([
            caffe2.python.utils.MakeArgument('num_loop_carried_deps',
                                             num_loop_carried_deps)
        ])

        return ops

    @classmethod
    def _substitute_raw_value(cls, tp, raw_values_dict):
        if tp.HasField('raw_data') and tp.raw_data == bytes(b'__EXTERNAL'):
            if tp.name not in raw_values_dict:
                raise RuntimeError('TensorProto for value {} referenced raw data but it was not found!'.format(tp.name))
            else:
                tp.raw_data = raw_values_dict[tp.name]

    @classmethod
    def _visit_and_substitute_raw_values(cls, nodes, raw_values_dict):
        for node in nodes:
            for attr in node.attribute:
                if attr.HasField('t'):
                    cls._substitute_raw_value(attr.t, raw_values_dict)
                for t in attr.tensors:
                    cls._substitute_raw_value(t, raw_values_dict)
                if attr.HasField('g'):
                    cls._visit_and_substitute_raw_values(attr.g.node, raw_values_dict)
                for g in attr.graphs:
                    cls._visit_and_substitute_raw_values(g.node, raw_values_dict)

    @classmethod
    def _external_value_resolution_pass(cls, model, raw_values_dict):
        for init in model.graph.initializer:
            cls._substitute_raw_value(init, raw_values_dict)

        cls._visit_and_substitute_raw_values(model.graph.node, raw_values_dict)


    @classmethod
    def _direct_initialize_parameters(cls, initializer, ws, device_option):
        for tp in initializer:
            ws.FeedBlob(tp.name, onnx.numpy_helper.to_array(tp), device_option)

    @classmethod
    def _direct_initialize_inputs(cls, inputs, initialized, ws, device_option):
        for value_info in inputs:
            if value_info.name in initialized:
                continue
            shape = list(d.dim_value for d in value_info.type.tensor_type.shape.dim)
            ws.FeedBlob(
                value_info.name,
                np.ones(shape, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[value_info.type.tensor_type.elem_type]),
                device_option)

    @staticmethod
    def optimize_onnx(input, init=False, predict=False):
        passes =  ['fuse_consecutive_transposes',
                   'eliminate_nop_transpose',
                   'fuse_transpose_into_gemm',
                   'lift_lexical_references']
        if init:
            passes.append('split_init')
        if predict:
            passes.append('split_predict')
        out = onnx.optimizer.optimize(input, passes)
        return out

    @classmethod
    def prepare_zip_archive(cls, file, device='CPU', **kwargs):
        with zipfile.ZipFile(file, mode='r') as z:
            with z.open('__MODEL_PROTO', 'r') as f:
                model = onnx.load(f);
            blob_names = set(z.namelist()) - set('__MODEL_PROTO')
            # TODO: make this more efficient
            raw_values_dict = {}
            for name in blob_names:
                with z.open(name, 'r') as blob_file:
                    raw_values_dict[name] = blob_file.read()

        return cls.prepare(model, device, raw_values_dict=raw_values_dict, **kwargs)

    @classmethod
    def prepare(cls, model, device='CPU', raw_values_dict=None, **kwargs):
        '''
        For Onnx Caffe2Backend, we require that init_graph don't initialize the actual input of the predict_graph,

        for example, if "img" is the input blob for the predict_net, we require that in init_graph and in
        initializer of the predict_graph, "img" is not initalized. We don't have a check for this, since
        there is no way we can know which blob is the input of the predict_graph.
        '''
        if not kwargs.pop('no_check_UNSAFE', False):
            super(Caffe2Backend, cls).prepare(model, device, **kwargs)
        opset_version = None
        for imp in model.opset_import:
            if not imp.HasField("domain") or imp.domain == "":
                opset_version = imp.version
                if imp.version > cls._known_opset_version:
                    warnings.warn("This version of onnx-caffe2 targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
            else:
                warnings.warn("Unrecognized operator set {}".format(imp.domain))
        if opset_version is None:
            if model.ir_version >= 0x00000003:
                raise RuntimeError("Model with IR version >= 3 did not specify ONNX operator set version (onnx-caffe2 requires it)")
            else:
                opset_version = 1

        model = onnx.shape_inference.infer_shapes(model)

        ws = Workspace()
        device_option = get_device_option(Device(device))

        init_net, predict_net = cls._onnx_model_to_caffe2_net(model, device, opset_version, False)

        if raw_values_dict:
            cls._external_value_resolution_pass(model, raw_values_dict)

        # Directly load initializer data into blobs in workspace
        cls._direct_initialize_parameters(
            model.graph.initializer,
            ws,
            device_option,
        )

        initialized = {init.name for init in model.graph.initializer}

        cls._direct_initialize_inputs(
            model.graph.input,
            initialized,
            ws,
            device_option,
        )

        uninitialized = [value_info.name for value_info in model.graph.input if value_info.name not in initialized]

        retval = Caffe2Rep(init_net, predict_net, ws, uninitialized)
        return retval


    @classmethod
    # TODO: This method needs a refactor for clarity
    def _onnx_node_to_caffe2_op(cls, init_model, pred_model, node_def, opset_version):
        cbackend = C.Caffe2Backend(cls._dummy_name)
        if cbackend.support_onnx_import(node_def.op_type):

            # extract value infos from pred model (value infos of
            # node's inputs that are in init model should be all
            # available in pred model)
            value_infos = []
            for name in node_def.input:
                if pred_model is not None:
                    for vi in itertools.chain(pred_model.graph.input,
                                              pred_model.graph.output,
                                              pred_model.graph.value_info):
                        if vi.name == name:
                            value_infos.append(vi.SerializeToString())

            op_strs = cbackend.convert_node(node_def.SerializeToString(), value_infos, opset_version)
            init_ops = []
            for s in op_strs[0]:
                op = caffe2_pb2.OperatorDef()
                op.ParseFromString(s)
                init_ops.append(op)
            ops = []
            for s in op_strs[1]:
                op = caffe2_pb2.OperatorDef()
                op.ParseFromString(s)
                ops.append(op)
            return Caffe2Ops(ops, init_ops, [])

        if node_def.op_type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[node_def.op_type])
        else:
            translator = cls._common_onnx_node_to_caffe2_op
        ops = translator(init_model, pred_model, OnnxNode(node_def), opset_version)
        if isinstance(ops, Caffe2Ops):
            return ops
        if not isinstance(ops, container_abcs.Iterable):
            ops = [ops]
        return Caffe2Ops(ops, [], [])

    _broadcast_operators = {
        'Add',
        'Sub',
    }

    @classmethod
    def _common_onnx_node_to_caffe2_op(cls, init_model, pred_model, onnx_node, opset_version):
        """
        This translator performs the basic translation of ONNX nodes into
        Caffe2 operators.  Besides doing a straightforward marshalling from
        one format to another, it also does these extra things:

          - Renames operators based on '_renamed_operators'
          - Renames attributes based on '_global_renamed_attrs' and
            '_per_op_renamed_attrs'

        If you're writing a custom translator, consider calling this first,
        and then fixing things up further.
        """
        c2_op = caffe2_pb2.OperatorDef()

        c2_op.input.extend(onnx_node.inputs)
        c2_op.output.extend(onnx_node.outputs)
        c2_op.name = onnx_node.name


        onnx_op_type = onnx_node.op_type
        broken_version = cls._broken_operators.get(onnx_op_type, float('Inf'))
        if broken_version <= opset_version:
            raise ValueError(
                "Don't know how to translate op {} in ONNX operator set v{} (I only support prior to v{})".format(onnx_op_type, opset_version, broken_version))
        c2_op.type = cls._renamed_operators.get(onnx_op_type, onnx_op_type)
        if not core.IsOperator(c2_op.type):
            raise ValueError(
                "Don't know how to translate op {}".format(onnx_op_type))

        def kmap(k):
            if (onnx_op_type in cls._per_op_renamed_attrs and
                    k in cls._per_op_renamed_attrs[onnx_op_type]):
                return cls._per_op_renamed_attrs[onnx_op_type][k]
            if k in cls._global_renamed_attrs:
                return cls._global_renamed_attrs[k]
            return k
        c2_op.arg.extend(onnx_node.attrs.caffe2(kmap=kmap))

        if opset_version < 7:
            # onnx opset 7 and newest caffe2 have adopted full onnx broadcast semantics
            # so we don't need this hack anymore
            if c2_op.type in cls._broadcast_operators:
                already_broadcast = False
                for arg in c2_op.arg:
                    if arg.name == 'broadcast':
                        already_broadcast = True
                if not already_broadcast:
                    c2_op.arg.extend([caffe2.python.utils.MakeArgument('broadcast', 1)])

        return c2_op

    @staticmethod
    def _all_names_in_graph(graph):
        if graph is None:
            return set()

        names = set()
        names.update(value_info.name for value_info in graph.input)
        names.update(value_info.name for value_info in graph.output)
        for node in graph.node:
            names.update(node.input)
            names.update(node.output)
        return names

    @classmethod
    def _graph_to_net(cls, onnx_graph, opset_version):
        net = caffe2_pb2.NetDef()
        for node in onnx_graph.node:
            try:
                c2ops = cls._onnx_node_to_caffe2_op(
                    None, None, node, opset_version)
            except Exception as e:
                print('ONNX FATAL:', e)
                continue
            net.op.extend(c2ops.init_ops)
            net.op.extend(c2ops.ops)
            net.external_input.extend(c2ops.interface_blobs)
        net.external_output.extend(
            value_info.name for value_info in onnx_graph.output)
        net.external_input.extend(
            value_info.name for value_info in onnx_graph.input)
        return net

    @classmethod
    def _onnx_model_to_caffe2_net(cls, onnx_model, device, opset_version, include_initializers):
        device_option = get_device_option(Device(device))

        onnx_model = onnx.utils.polish_model(onnx_model)
        init_model = cls.optimize_onnx(onnx_model, init=True)
        pred_model = cls.optimize_onnx(onnx_model, predict=True)

        init_net = caffe2_pb2.NetDef()
        pred_net = caffe2_pb2.NetDef()

        init_net.name = onnx_model.graph.name + '_init'
        pred_net.name = onnx_model.graph.name + '_predict'

        if include_initializers:
            init_net.op.extend(cls._create_tensor_filling_op(tp) for tp in onnx_model.graph.initializer)

        cls._dummy_name.reset(cls._all_names_in_graph(init_model.graph) | cls._all_names_in_graph(pred_model.graph))

        errors = []
        for net, model in ( (init_net, init_model), (pred_net, pred_model) ):
            net.device_option.CopyFrom(device_option)
            for node in model.graph.node:
                try:
                    c2ops = cls._onnx_node_to_caffe2_op(
                        init_model, pred_model, node, opset_version)
                except Exception as e:
                    msg = 'Error while processing node: {}. Exception: {}'.format(node, e)
                    errors.append(msg)
                    print('ONNX FATAL:', msg, file=sys.stderr)
                    continue
                init_net.op.extend(c2ops.init_ops)
                net.op.extend(c2ops.ops)
                net.external_input.extend(c2ops.interface_blobs)
            net.external_output.extend(
                value_info.name for value_info in model.graph.output)
            net.external_input.extend(
                value_info.name for value_info in model.graph.input)

        if len(errors) > 0:
            raise RuntimeError(
                "ONNX conversion failed, encountered {} errors:\n\n{}".format(
                    len(errors), "\n\n".join(errors)))

        return init_net, pred_net

    # wrapper for backwards compatibility
    @classmethod
    def onnx_graph_to_caffe2_net(cls, model, device="CPU", opset_version=_known_opset_version):
        return cls._onnx_model_to_caffe2_net(model, device=device, opset_version=opset_version, include_initializers=True)

    @classmethod
    def supports_device(cls, device_str):
        device = Device(device_str)
        if device.type == DeviceType.CPU:
            return True
        elif core.IsGPUDeviceType(device.type):
            return workspace.has_gpu_support
        return False

    @classmethod
    def is_compatible(cls, model, device='CPU', **kwargs):
        if hasattr(super(Caffe2Backend, cls), 'is_compatible') \
           and callable(super(Caffe2Backend, cls).is_compatible):
            if not super(Caffe2Backend, cls).is_compatible(model, device, **kwargs):
                return False
        # TODO: should have an unspported list of operators, be optimistic for now
        return True

prepare = Caffe2Backend.prepare

prepare_zip_archive = Caffe2Backend.prepare_zip_archive

run_node = Caffe2Backend.run_node

run_model = Caffe2Backend.run_model

supports_device = Caffe2Backend.supports_device  # noqa

is_compatible = Caffe2Backend.is_compatible
