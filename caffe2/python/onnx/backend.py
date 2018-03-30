## @package onnx
# Module caffe2.python.onnx.backend

"""Backend for running ONNX on Caffe2

To run this, you will need to have Caffe2 installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import collections
from subprocess import Popen, PIPE
import zipfile

import caffe2
from caffe2.python import core, workspace, rnn_cell, gru_cell
from caffe2.python.model_helper import ModelHelper
from caffe2.proto import caffe2_pb2
import caffe2.python.utils
import numpy as np
import onnx
from onnx import checker, GraphProto, TensorProto, AttributeProto, ModelProto
import onnx.numpy_helper
import onnx.defs
import onnx.optimizer
from onnx.backend.base import Backend, Device, DeviceType, namedtupledict

from caffe2.python.onnx.workspace import Workspace
from caffe2.python.onnx.backend_rep import Caffe2Rep
from caffe2.python.onnx.backend_cpp_rep import Caffe2CppRep
from caffe2.python.onnx.helper import dummy_name

import caffe2.python._import_c_extension as C

import warnings

def force_unicode(s):
    try:
        return s.decode('utf-8')
    except AttributeError:
        return s

def get_device_option(device):
    m = {DeviceType.CPU: caffe2_pb2.CPU,
         DeviceType.CUDA: caffe2_pb2.CUDA}
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
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
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
    _known_opset_version = 6

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
        'Caffe2ConvTranspose':   'ConvTranspose',
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
    }

    _global_renamed_attrs = {'kernel_shape': 'kernels'}
    _per_op_renamed_attrs = {
        'Squeeze':              {'axes': 'dims'},
        'Unsqueeze':            {'axes': 'dims'},
        'Transpose':            {'perm': 'axes'},
        'Upsample':             {'mode': ''},
        'ConvTranspose':        {'output_padding': 'adjs'},
        'Selu':                 {'gamma': 'scale'},
    }

    # operators whose behavior is different beyond renaming
    # the value is an attribute of this class that is a
    # function from ToffeIR node_def to caffe2 op_def
    _special_operators = {
        'LSTM': '_create_lstm',
        'GRU': '_create_gru',
        'RNN': '_create_rnn',
    }

    # NB: By default, you will use the LATEST definition of the operator,
    # so this interface MAY make BC-breaking changes.  Specify an
    # opset_version if you don't want this to version.
    @classmethod
    def run_node(cls, node, inputs, device='CPU', opset_version=_known_opset_version, outputs_info=None):
        super(Caffe2Backend, cls).run_node(node, inputs, device=device, outputs_info=outputs_info)

        device_option = get_device_option(Device(device))
        with Workspace(), core.DeviceScope(device_option):  # temporary!
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    workspace.FeedBlob(key, value)
            else:
                assert len(node.input) == len(inputs), "{}: expected {} but got {}".format(
                    node.op_type, len(node.input), len(inputs))
                for key, value in zip(node.input, inputs):
                    workspace.FeedBlob(key, value)

            ops = []
            cbackend = C.Caffe2Backend()
            ops_str = cbackend.convert_node(node.SerializeToString(), opset_version)
            for s in ops_str[0] + ops_str[1]:
                op = caffe2_pb2.OperatorDef()
                op.ParseFromString(s)
                op.device_option.CopyFrom(device_option)
                ops.append(op)
            # For testing
            if "ONNX_CAFFE2_DEBUG" in os.environ:
                init_ops, ops2, _ = cls._onnx_node_to_caffe2_op(
                    None, None, node, opset_version or cls._known_opset_version)
                ops2 = init_ops + ops2
                for op in ops2:
                    op.device_option.CopyFrom(device_option)
                print("\nC++:\n{}\nPython:\n{}".format(ops, ops2))
            workspace.RunOperatorsOnce(ops)
            output_values = [workspace.FetchBlob(name) for name in node.output]
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
    def _rnn_shape_inference(cls, init_model, pred_model, n, input_blob, W):
        # ad-hoc, informally-specified, bug-ridden, slow
        # implementation of shape inference

        # if the weight matrices are directly provided as
        # initializers, their dimensions should be available in the
        # init net model.
        for x in init_model.graph.input:
            if x.name == W:
                return x.type.tensor_type.shape.dim[1].dim_value

        # otherwise, assume that the input_blob is either a direct
        # graph input, or another rnn op of the same type. This
        # matches the pattern produced by exporting from pytorch
        # (where the weight matrices are unusable for this purpose due
        # to reshaping operations that lose shape information).
        for x in pred_model.graph.input:
            if x.name == input_blob:
                return x.type.tensor_type.shape.dim[2].dim_value

        curr = n
        while True:
            for x in pred_model.graph.input:
                if x.name == curr.inputs[0] and curr.op_type == 'Gather':
                    return x.type.tensor_type.shape.dim[1].dim_value
            prev = [x for x in map(OnnxNode, pred_model.graph.node) if x.outputs[0] == curr.inputs[0]]
            if len(prev) != 1:
                return
            prev = prev[0]
            if prev.op_type == n.op_type:
                return prev.attrs['hidden_size']
            if prev.op_type == 'Transpose':
                for x in pred_model.graph.input:
                    if x.name == prev.inputs[0]:
                        return x.type.tensor_type.shape.dim[2].dim_value
            curr = prev

    @classmethod
    def _create_rnn(cls, init_model, pred_model, n, opset_version):
        assert init_model is not None, "cannot convert RNNs without access to the full model"
        assert pred_model is not None, "cannot convert RNNs without access to the full model"

        attrs = dict(n.attrs) # make a copy, which is safe to mutate
        hidden_size = attrs.pop('hidden_size')
        activation = force_unicode(attrs.pop('activations', ('tanh',))[0])
        direction = force_unicode(attrs.pop('direction', 'forward'))
        assert not attrs, "unsupported RNN attributes: " + str(attrs.keys())
        assert direction in ['forward', 'bidirectional'], "unsupported backwards RNN"

        input_blob, W, R, B, sequence_lens, initial_h = n.inputs

        if sequence_lens == "":
            sequence_lens = None

        input_size = cls._rnn_shape_inference(init_model, pred_model, n, input_blob, W)
        if input_size is None:
            raise RuntimeError("best-effort shape inference for RNN input failed")

        init_net = core.Net("init-net")
        pred_mh = ModelHelper()

        def make_rnn(direction_offset):
            name = dummy_name()

            # input and recurrence biases are squashed together in
            # onnx but not in caffe2

            bias_offset = 2 * direction_offset * hidden_size
            init_net.Slice(B, name + "/i2h_b",
                           starts=[bias_offset + 0 * hidden_size],
                           ends  =[bias_offset + 1 * hidden_size])
            init_net.Slice(B, name + "/gates_t_b",
                           starts=[bias_offset + 1 * hidden_size],
                           ends  =[bias_offset + 2 * hidden_size])

            weight_offset = direction_offset * hidden_size
            init_net.Slice(W, name + '/i2h_w',
                           starts=[weight_offset + 0 * hidden_size, 0],
                           ends  =[weight_offset + 1 * hidden_size,-1])
            init_net.Slice(R, name + '/gates_t_w',
                           starts=[weight_offset + 0 * hidden_size, 0],
                           ends  =[weight_offset + 1 * hidden_size,-1])

            initial_h_sliced = name + '/initial_h'
            init_net.Slice(initial_h, initial_h_sliced,
                           starts=[direction_offset + 0, 0, 0],
                           ends  =[direction_offset + 1,-1,-1])

            if direction_offset == 1:
                input = pred_mh.net.ReversePackedSegs(
                    [input_blob, sequence_lens], name + "/input-reversed")
            else:
                input = input_blob

            hidden_t_all, hidden_t_last = rnn_cell.BasicRNN(
                pred_mh,
                input,
                sequence_lens,
                [initial_h_sliced],
                input_size,
                hidden_size,
                name,
                drop_states=False,
                forward_only=True,
                activation=activation
            )

            if direction_offset == 1:
                hidden_t_all = pred_mh.net.ReversePackedSegs(
                    [hidden_t_all, sequence_lens], name + "/output-reversed")

            return hidden_t_all, hidden_t_last

        if direction == 'forward':
            hidden_t_all, hidden_t_last = make_rnn(0)

            # in the forward case, storage is shared between the two
            # outputs. We need to decouple them so that the
            # VariableLengthSequencePadding only mutates n.outputs[0]
            pred_mh.net.Copy(hidden_t_last, n.outputs[1])

            pred_mh.net = pred_mh.net.Clone(
                "dummy-clone-net",
                blob_remap={ hidden_t_all: n.outputs[0] }
            )
        elif direction == 'bidirectional':
            hidden_t_all_f, hidden_t_last_f = make_rnn(0)
            hidden_t_all_b, hidden_t_last_b = make_rnn(1)
            pred_mh.net.Concat([hidden_t_all_f, hidden_t_all_b],
                               [n.outputs[0], dummy_name()], axis=2)
            pred_mh.net.Concat([hidden_t_last_f, hidden_t_last_b],
                               [n.outputs[1], dummy_name()], axis=0)

        if sequence_lens is not None:
            pred_mh.net.VariableLengthSequencePadding(
                [n.outputs[0], sequence_lens], [n.outputs[0]])

        return Caffe2Ops(list(pred_mh.Proto().op),
                         list(init_net.Proto().op),
                         list(pred_mh.Proto().external_input))

    @classmethod
    def _create_lstm(cls, init_model, pred_model, n, opset_version):
        assert init_model is not None, "cannot convert LSTMs without access to the full model"
        assert pred_model is not None, "cannot convert LSTMs without access to the full model"

        attrs = dict(n.attrs) # make a copy, which is safe to mutate
        hidden_size = attrs.pop('hidden_size')
        direction = force_unicode(attrs.pop('direction', 'forward'))
        assert not attrs, "unsupported LSTM attributes: " + str(attrs.keys())
        assert direction in ['forward', 'bidirectional'], "unsupported backwards LSTM"

        input_blob, W, R, B, sequence_lens, initial_h, initial_c = n.inputs

        if sequence_lens == "":
            sequence_lens = None

        input_size = cls._rnn_shape_inference(init_model, pred_model, n, input_blob, W)
        if input_size is None:
            raise RuntimeError("best-effort shape inference for LSTM input failed")

        init_net = core.Net("init-net")
        pred_mh = ModelHelper()

        def make_lstm(direction_offset):
            name = dummy_name()

            # input and recurrence biases are squashed together in
            # onnx but not in caffe2

            bias_offset = 8 * direction_offset * hidden_size
            Bi = init_net.Slice(B, name + "_bias_i2h",
                                starts=[bias_offset + 0 * hidden_size],
                                ends  =[bias_offset + 4 * hidden_size])
            Br = init_net.Slice(B, name + "_bias_gates",
                                starts=[bias_offset + 4 * hidden_size],
                                ends  =[bias_offset + 8 * hidden_size])

            weight_offset = 4 * direction_offset * hidden_size
            W_ = init_net.Slice(W, name + '/i2h_w_pre',
                                starts=[weight_offset + 0 * hidden_size, 0],
                                ends  =[weight_offset + 4 * hidden_size,-1])
            R_ = init_net.Slice(R, name + '/gates_t_w_pre',
                                starts=[weight_offset + 0 * hidden_size, 0],
                                ends  =[weight_offset + 4 * hidden_size,-1])

            # caffe2 has a different order from onnx. We need to rearrange
            #   i o f c -> i f o c
            reforms = ((W_, 'i2h_w',     [(0, -1)]),
                       (R_, 'gates_t_w', [(0, -1)]),
                       (Bi, 'i2h_b'    , []),
                       (Br, 'gates_t_b', []))
            for name_from, name_to, extra_dims in reforms:
                xi, xo, xf, xc = [name_from + suffix for suffix in ("_i", "_o", "_f", "_c")]
                for i, x in enumerate([xi, xo, xf, xc]):
                    dim0 = i * hidden_size, (i+1) * hidden_size
                    starts, ends = zip(dim0, *extra_dims)
                    init_net.Slice(name_from, x, starts=starts, ends=ends)
                init_net.Concat([xi, xf, xo, xc], ['%s/%s' % (name, name_to), dummy_name()], axis=0)

            initial_h_sliced = name + '/initial_h'
            init_net.Slice(initial_h, initial_h_sliced,
                           starts=[direction_offset + 0, 0, 0],
                           ends  =[direction_offset + 1,-1,-1])
            initial_c_sliced = name + '/initial_c'
            init_net.Slice(initial_c, initial_c_sliced,
                           starts=[direction_offset + 0, 0, 0],
                           ends  =[direction_offset + 1,-1,-1])

            if direction_offset == 1:
                input = pred_mh.net.ReversePackedSegs(
                    [input_blob, sequence_lens], name + "/input-reversed")
            else:
                input = input_blob

            hidden_t_all, hidden_t_last, _, cell_last, params = rnn_cell.LSTM(
                pred_mh,
                input,
                sequence_lens,
                [initial_h_sliced, initial_c_sliced],
                input_size,
                hidden_size,
                name,
                drop_states=False,
                forward_only=True,
                return_params=True
            )

            if direction_offset == 1:
                hidden_t_all = pred_mh.net.ReversePackedSegs(
                    [hidden_t_all, sequence_lens], name + "/output-reversed")

            return hidden_t_all, hidden_t_last, cell_last

        if direction == 'forward':
            hidden_t_all, hidden_t_last, cell_last = make_lstm(0)

            # in the forward case, storage is shared between the three
            # outputs. We need to decouple them so that the
            # VariableLengthSequencePadding only mutates n.outputs[0]
            pred_mh.net.Copy(hidden_t_last, n.outputs[1])
            pred_mh.net.Copy(cell_last, n.outputs[2])

            pred_mh.net = pred_mh.net.Clone(
                "dummy-clone-net",
                blob_remap={ hidden_t_all: n.outputs[0] }
            )
        elif direction == 'bidirectional':
            hidden_t_all_f, hidden_t_last_f, cell_last_f = make_lstm(0)
            hidden_t_all_b, hidden_t_last_b, cell_last_b = make_lstm(1)
            pred_mh.net.Concat([hidden_t_all_f, hidden_t_all_b],
                               [n.outputs[0], dummy_name()], axis=2)
            pred_mh.net.Concat([hidden_t_last_f, hidden_t_last_b],
                               [n.outputs[1], dummy_name()], axis=0)
            pred_mh.net.Concat([cell_last_f, cell_last_b],
                               [n.outputs[2], dummy_name()], axis=0)

        if sequence_lens is not None:
            pred_mh.net.VariableLengthSequencePadding(
                [n.outputs[0], sequence_lens], [n.outputs[0]])

        return Caffe2Ops(list(pred_mh.Proto().op),
                         list(init_net.Proto().op),
                         list(pred_mh.Proto().external_input))

    @classmethod
    def _create_gru(cls, init_model, pred_model, n, opset_version):
        assert init_model is not None, "cannot convert GRUs without access to the full model"
        assert pred_model is not None, "cannot convert GRUs without access to the full model"

        attrs = dict(n.attrs) # make a copy, which is safe to mutate
        hidden_size = attrs.pop('hidden_size')
        linear_before_reset = attrs.pop('linear_before_reset', 0)
        direction = force_unicode(attrs.pop('direction', 'forward'))
        assert not attrs, "unsupported GRU attributes: " + str(attrs.keys())
        assert direction in ['forward', 'bidirectional'], "unsupported backwards GRU"

        input_blob, W, R, B, sequence_lens, initial_h = n.inputs

        if sequence_lens == "":
            sequence_lens = None

        input_size = cls._rnn_shape_inference(init_model, pred_model, n, input_blob, W)
        if input_size is None:
            raise RuntimeError("best-effort shape inference for GRU input failed")

        init_net = core.Net("init-net")
        pred_mh = ModelHelper()

        def make_gru(direction_offset):
            name = dummy_name()

            # input and recurrence biases are squashed together in
            # onnx but not in caffe2

            bias_offset = 6 * direction_offset * hidden_size
            Bi = init_net.Slice(B, name + "_bias_i2h",
                                starts=[bias_offset + 0 * hidden_size],
                                ends  =[bias_offset + 3 * hidden_size])
            Br = init_net.Slice(B, name + "_bias_gates",
                                starts=[bias_offset + 3 * hidden_size],
                                ends  =[bias_offset + 6 * hidden_size])

            weight_offset = 3 * direction_offset * hidden_size
            W_ = init_net.Slice(W, name + '/i2h_w_pre',
                                starts=[weight_offset + 0 * hidden_size, 0],
                                ends  =[weight_offset + 3 * hidden_size,-1])
            R_ = init_net.Slice(R, name + '/gates_t_w_pre',
                                starts=[weight_offset + 0 * hidden_size, 0],
                                ends  =[weight_offset + 3 * hidden_size,-1])

            # caffe2 has a different order from onnx. We need to rearrange
            #  z r h  -> r z h
            reforms = ((W_, 'i2h_w',    True,  [(0,-1)]),
                       (R_, 'gate_t_w', False, [(0,-1)]),
                       (Bi, 'i2h_b',    True,  []),
                       (Br, 'gate_t_b', False, []))
            for name_from, name_to, do_concat, extra_dims in reforms:
                xz, xr, xh = ['%s/%s_%s' % (name, prefix, name_to) for prefix in ('update', 'reset', 'output')]
                for i, x in enumerate([xz, xr, xh]):
                    dim0 = i * hidden_size, (i+1) * hidden_size
                    starts, ends = zip(dim0, *extra_dims)
                    init_net.Slice(name_from, x, starts=starts, ends=ends)
                if do_concat:
                    init_net.Concat([xr, xz, xh], ['%s/%s' % (name, name_to), dummy_name()], axis=0)

            initial_h_sliced = name + '/initial_h'
            init_net.Slice(initial_h, initial_h_sliced,
                           starts=[direction_offset + 0, 0, 0],
                           ends  =[direction_offset + 1,-1,-1])

            if direction_offset == 1:
                input = pred_mh.net.ReversePackedSegs(
                    [input_blob, sequence_lens], name + "/input-reversed")
            else:
                input = input_blob

            hidden_t_all, hidden_t_last = gru_cell.GRU(
                pred_mh,
                input,
                sequence_lens,
                [initial_h_sliced],
                input_size,
                hidden_size,
                name,
                drop_states=False,
                forward_only=True,
                linear_before_reset=linear_before_reset
            )

            if direction_offset == 1:
                hidden_t_all = pred_mh.net.ReversePackedSegs(
                    [hidden_t_all, sequence_lens], name + "/output-reversed")

            return hidden_t_all, hidden_t_last

        if direction == 'forward':
            hidden_t_all, hidden_t_last = make_gru(0)

            # in the forward case, storage is shared between the two
            # outputs. We need to decouple them so that the
            # VariableLengthSequencePadding only mutates n.outputs[0]
            pred_mh.net.Copy(hidden_t_last, n.outputs[1])

            pred_mh.net = pred_mh.net.Clone(
                "dummy-clone-net",
                blob_remap={ hidden_t_all: n.outputs[0] }
            )
        elif direction == 'bidirectional':
            hidden_t_all_f, hidden_t_last_f = make_gru(0)
            hidden_t_all_b, hidden_t_last_b = make_gru(1)
            pred_mh.net.Concat([hidden_t_all_f, hidden_t_all_b],
                               [n.outputs[0], dummy_name()], axis=2)
            pred_mh.net.Concat([hidden_t_last_f, hidden_t_last_b],
                               [n.outputs[1], dummy_name()], axis=0)

        if sequence_lens is not None:
            pred_mh.net.VariableLengthSequencePadding(
                [n.outputs[0], sequence_lens], [n.outputs[0]])

        return Caffe2Ops(list(pred_mh.Proto().op),
                         list(init_net.Proto().op),
                         list(pred_mh.Proto().external_input))

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
            ws.FeedBlob(value_info.name, np.ones(shape), device_option)

    @staticmethod
    def optimize_onnx(input, init=False, predict=False):
        passes =  ['fuse_consecutive_transposes',
                   'eliminate_nop_transpose',
                   'fuse_transpose_into_gemm']
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

        cls._external_value_resolution_pass(model, raw_values_dict)
        return cls.prepare(model, device, **kwargs)

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        '''
        For Onnx Caffe2Backend, we require that init_graph don't initialize the actual input of the predict_graph,

        for example, if "img" is the input blob for the predict_net, we require that in init_graph and in
        initializer of the predict_graph, "img" is not initalized. We don't have a check for this, since
        there is no way we can know which blob is the input of the predict_graph.
        '''
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

        # Check whether we have RNN related ops
        pred_model = ModelProto()
        pred_model.ParseFromString(cls.optimize_onnx(model.SerializeToString(), predict=True))
        rnn_nodes = []
        for node in pred_model.graph.node:
            if node.op_type in {'LSTM', 'GRU', 'RNN'}:
                rnn_nodes.append(node)

        # Build the C++ backend
        # TODO: build a predictor that supports GPU
        #       And for RNN nets, we need to avoid adding init_net
        use_cpp_backend = device == 'CPU' and not rnn_nodes
        # use python backend for now
        use_cpp_backend = False
        if use_cpp_backend:
            c2_rnn_ops = []
            if rnn_nodes:
                init_model = ModelProto()
                init_model.ParseFromString(cls.optimize_onnx(model.SerializeToString(), init=True))
                for node in rnn_nodes:
                    c2ops = cls._onnx_node_to_caffe2_op(
                        init_model, pred_model, node, opset_version)
                    init_ops = [x.SerializeToString() for x in c2ops.init_ops]
                    ops = [x.SerializeToString() for x in c2ops.ops]
                    external_inputs = c2ops.interface_blobs
                    c2_rnn_ops.append(C.Caffe2Ops(init_ops, ops, external_inputs))
                del init_model

            cbackend = C.Caffe2Backend()
            rep = cbackend.prepare(model.SerializeToString(), device, c2_rnn_ops)
            # For testing
            # Dump the net descriptions to file for comparison with the Python ones
            if "ONNX_CAFFE2_DEBUG" in os.environ:
                pred_net_str = rep.pred_net()
                pn = caffe2_pb2.NetDef()
                pn.ParseFromString(pred_net_str)
                init_net_str = rep.init_net()
                inn = caffe2_pb2.NetDef()
                inn.ParseFromString(init_net_str)
                with open("cpp.txt", "w") as f:
                    f.write("pred_net: \n{}".format(pn))

            rep_wrapper = Caffe2CppRep(rep)
            return rep_wrapper
        else:
            ws = Workspace()
            device_option = get_device_option(Device(device))

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

            init_net, predict_net = cls._onnx_model_to_caffe2_net(model, device, opset_version, False)
            if "ONNX_CAFFE2_DEBUG" in os.environ:
                with open("python.txt", "w") as f:
                    f.write("pred_net: \n{}".format(predict_net))
            retval = Caffe2Rep(init_net, predict_net, ws, uninitialized)
            return retval


    @classmethod
    # TODO: This method needs a refactor for clarity
    def _onnx_node_to_caffe2_op(cls, init_model, pred_model, node_def, opset_version):
        cbackend = C.Caffe2Backend()
        if cbackend.support_onnx_import(node_def.op_type):
            op_strs = cbackend.convert_node(node_def.SerializeToString(), opset_version)
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
        if not isinstance(ops, collections.Iterable):
            ops = [ops]
        return Caffe2Ops(ops, [], [])

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
    def _onnx_model_to_caffe2_net(cls, onnx_model, device, opset_version, include_initializers):
        device_option = get_device_option(Device(device))

        init_model = ModelProto()
        init_model.ParseFromString(cls.optimize_onnx(onnx_model.SerializeToString(), init=True))

        pred_model = ModelProto()
        pred_model.ParseFromString(cls.optimize_onnx(onnx_model.SerializeToString(), predict=True))

        init_net = caffe2_pb2.NetDef()
        pred_net = caffe2_pb2.NetDef()

        init_net.name = onnx_model.graph.name + '_init'
        pred_net.name = onnx_model.graph.name + '_predict'

        if include_initializers:
            init_net.op.extend(cls._create_tensor_filling_op(tp) for tp in onnx_model.graph.initializer)

        dummy_name(cls._all_names_in_graph(init_model.graph) | cls._all_names_in_graph(pred_model.graph))

        success = True
        for net, model in ( (init_net, init_model), (pred_net, pred_model) ):
            net.device_option.CopyFrom(device_option)
            for node in model.graph.node:
                try:
                    c2ops = cls._onnx_node_to_caffe2_op(
                        init_model, pred_model, node, opset_version)
                except Exception as e:
                    success = False
                    print('ONNX FATAL:', e)
                    continue
                (init_net if include_initializers else net).op.extend(c2ops.init_ops)
                net.op.extend(c2ops.ops)
                net.external_input.extend(c2ops.interface_blobs)
            net.external_output.extend(
                value_info.name for value_info in model.graph.output)
            net.external_input.extend(
                value_info.name for value_info in model.graph.input)

        if not success:
            raise RuntimeError('ONNX conversion failed')

        return init_net, pred_net

    # wrapper for backwards compatability
    @classmethod
    def onnx_graph_to_caffe2_net(cls, model, device="CPU", opset_version=_known_opset_version):
        return cls._onnx_model_to_caffe2_net(model, device=device, opset_version=opset_version, include_initializers=True)

    @classmethod
    def supports_device(cls, device_str):
        device = Device(device_str)
        if device.type == DeviceType.CPU:
            return True
        elif device.type == DeviceType.CUDA:
            return workspace.has_gpu_support
        return False


prepare = Caffe2Backend.prepare

prepare_zip_archive = Caffe2Backend.prepare_zip_archive

run_node = Caffe2Backend.run_node

run_model = Caffe2Backend.run_model

supports_device = Caffe2Backend.supports_device  # noqa
