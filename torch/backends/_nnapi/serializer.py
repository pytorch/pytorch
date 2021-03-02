import enum
import struct
import array
import logging
from typing import (
    Tuple,
    NamedTuple,
)

import torch


# TODO: Add type annotations
# TODO: Check tensor types for ops


LOG = logging.getLogger("nnapi_serialize")


class NNAPI_OperandCode(object):
    FLOAT32 = 0
    INT32 = 1
    UINT32 = 2
    TENSOR_FLOAT32 = 3
    TENSOR_INT32 = 4
    TENSOR_QUANT8_ASYMM = 5
    BOOL = 6
    TENSOR_QUANT16_SYMM = 7
    TENSOR_FLOAT16 = 8
    TENSOR_BOOL8 = 9
    FLOAT16 = 10
    TENSOR_QUANT8_SYMM_PER_CHANNEL = 11
    TENSOR_QUANT16_ASYMM = 12


class NNAPI_OperationCode(object):
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    LOGISTIC = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    BATCH_TO_SPACE_ND = 29
    DIV = 30
    MEAN = 31
    PAD = 32
    SPACE_TO_BATCH_ND = 33
    SQUEEZE = 34
    STRIDED_SLICE = 35
    SUB = 36
    TRANSPOSE = 37
    ABS = 38
    ARGMAX = 39
    ARGMIN = 40
    AXIS_ALIGNED_BBOX_TRANSFORM = 41
    BIDIRECTIONAL_SEQUENCE_LSTM = 42
    BIDIRECTIONAL_SEQUENCE_RNN = 43
    BOX_WITH_NMS_LIMIT = 44
    CAST = 45
    CHANNEL_SHUFFLE = 46
    DETECTION_POSTPROCESSING = 47
    EQUAL = 48
    EXP = 49
    EXPAND_DIMS = 50
    GATHER = 51
    GENERATE_PROPOSALS = 52
    GREATER = 53
    GREATER_EQUAL = 54
    GROUPED_CONV_2D = 55
    HEATMAP_MAX_KEYPOINT = 56
    INSTANCE_NORMALIZATION = 57
    LESS = 58
    LESS_EQUAL = 59
    LOG = 60
    LOGICAL_AND = 61
    LOGICAL_NOT = 62
    LOGICAL_OR = 63
    LOG_SOFTMAX = 64
    MAXIMUM = 65
    MINIMUM = 66
    NEG = 67
    NOT_EQUAL = 68
    PAD_V2 = 69
    POW = 70
    PRELU = 71
    QUANTIZE = 72
    QUANTIZED_16BIT_LSTM = 73
    RANDOM_MULTINOMIAL = 74
    REDUCE_ALL = 75
    REDUCE_ANY = 76
    REDUCE_MAX = 77
    REDUCE_MIN = 78
    REDUCE_PROD = 79
    REDUCE_SUM = 80
    ROI_ALIGN = 81
    ROI_POOLING = 82
    RSQRT = 83
    SELECT = 84
    SIN = 85
    SLICE = 86
    SPLIT = 87
    SQRT = 88
    TILE = 89
    TOPK_V2 = 90
    TRANSPOSE_CONV_2D = 91
    UNIDIRECTIONAL_SEQUENCE_LSTM = 92
    UNIDIRECTIONAL_SEQUENCE_RNN = 93
    RESIZE_NEAREST_NEIGHBOR = 94


class NNAPI_FuseCode(object):
    FUSED_NONE = 0
    FUSED_RELU = 1
    FUSED_RELU1 = 2
    FUSED_RELU6 = 3


class OperandValueSourceType(object):
    IMMEDIATE = 0
    NUMBERED_BUFFER = 2
    NUMBERED_MEMORY = 3


# Scalar types that appear explicitly in models.
# These must be kept in sync with
# AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS.
# TODO: Expose these directly to Python to avoid maintaining this list.
class TorchScalarTypes(enum.Enum):
    QUINT8 = 13


def approx_equal(lhs, rhs, tolerance=1e-6):
    return abs(lhs - rhs) <= tolerance * min(lhs, rhs)


def tensor_size(op_type, dims):
    ITEM_SIZES = {
        NNAPI_OperandCode.TENSOR_FLOAT32: 4,
        NNAPI_OperandCode.TENSOR_INT32: 4,
        NNAPI_OperandCode.TENSOR_QUANT8_ASYMM: 1,
        NNAPI_OperandCode.TENSOR_QUANT16_SYMM: 2,
    }
    size = ITEM_SIZES[op_type]
    for d in dims:
        size *= d
    return size


class ConvPoolArgs2d(NamedTuple):
    """Configuration arguments for a convolution."""
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_t: int
    pad_b: int
    pad_l: int
    pad_r: int
    dilation_h: int
    dilation_w: int
    group: int


class DimOrder(enum.Enum):
    PRESUMED_CONTIGUOUS = 0
    CHANNELS_LAST = 1
    SCALAR_OR_VECTOR = 2
    UNKNOWN_CONSTANT = 999


class Operand(NamedTuple):
    """Represenation of an NNAPI operand."""

    # NNAPI operand type.  One of NNAPI_OperandCode.
    # TODO: Make this an enum.
    op_type: int

    # This is always the PyTorch shape, which is NCHW for feature maps.
    # The actual NNAPI operand might have a transposed shape.
    shape: Tuple[int, ...]

    # Specifies how the shape of the operand that we define in NNAPI
    # relates to the shape we track above.
    # - PRESUMED_CONTIGUOUS: physical NNAPI operand will exactly match
    #   the shape of the PyTorch tensor.
    # - CHANNELS_LAST: The PyTorch tensor is expected to be NCHW, and
    #   the NNAPI operand will be represented explicitly as NHWC.
    dim_order: DimOrder

    # Quantization params
    scale: float
    zero_point: int

    def use_nchw(self):
        if self.dim_order is DimOrder.PRESUMED_CONTIGUOUS:
            return True
        if self.dim_order is DimOrder.CHANNELS_LAST:
            return False
        raise Exception("Unknown dim order")


def broadcast_shapes(shape1, shape2):
    assert len(shape1) > 0
    assert len(shape2) > 0
    s1 = list(shape1)
    s2 = list(shape2)
    # TODO: Support non-equal-rank broadcast where semantics match.
    # This can be tricky for NHWC tensors because dimension orders
    # don't match between PT and NNAPI, even though semantics match.
    if len(s1) > len(s2):
        # s2 = [1] * (len(s1) - len(s2)) + s2
        raise Exception("Non-equal-rank broadcast is not supported yet.")
    if len(s2) > len(s1):
        # s3 = [1] * (len(s2) - len(s1)) + s1
        raise Exception("Non-equal-rank broadcast is not supported yet.")
    ret = []
    for d1, d2 in zip(s1, s2):
        if d1 == 1:
            ret.append(d2)
        elif d2 == 1:
            ret.append(d1)
        elif d1 == d2:
            ret.append(d1)
        else:
            raise Exception("Cannot broadcast shapes: {} and {}".format(shape1, shape2))
    return tuple(ret)


def get_conv_pool_shape(image_shape, args, out_ch, transpose):
    batch, in_c, in_h, in_w = image_shape

    # TODO: Handle dilation
    if args.dilation_h != 1 or args.dilation_w != 1:
        raise Exception("Dilation not supported yet.")

    if transpose:
        out_h = (in_h - 1) * args.stride_h + args.kernel_h - args.pad_t - args.pad_b
        out_w = (in_w - 1) * args.stride_w + args.kernel_w - args.pad_l - args.pad_l
    else:
        out_h = (in_h - args.kernel_h + args.pad_t + args.pad_b) // args.stride_h + 1
        out_w = (in_w - args.kernel_w + args.pad_l + args.pad_r) // args.stride_w + 1

    # Handle variable-sized tensors.
    if in_h == 0:
        out_h = 0
    if in_w == 0:
        out_w = 0

    out_shape = (batch, out_ch, out_h, out_w)
    return out_shape


def fix_shape(shape, dim_order):
    # Return the actual shape that an operand should have in NNAPI,
    # given a PyTorch shape and dimension order.  This is where we
    # convert from PyTorch's "always NCHW" shape to explicit NHWC.
    if dim_order is DimOrder.PRESUMED_CONTIGUOUS:
        return shape
    if dim_order is DimOrder.CHANNELS_LAST:
        return tuple([shape[0]] + list(shape[2:]) + [shape[1]])
    if dim_order is DimOrder.SCALAR_OR_VECTOR:
        assert len(shape) == 0 or len(shape) == 1
        return shape
    if dim_order is DimOrder.UNKNOWN_CONSTANT:
        # XXX think this through
        return shape
    raise Exception(f"Bad dim_order: {dim_order!r}.")


class _NnapiSerializer(object):
    def __init__(self, config):
        self.operands = []
        self.values = []
        self.operations = []
        self.value_data = []
        self.operation_args = []
        self.inputs = []
        self.outputs = []

        self.modules = {}
        self.constants = {}
        self.tensor_tuples = {}
        self.jitval_operand_map = {}
        self.cached_immediates = {}
        self.used_weights = []
        self.weight_offset = 0

        if config is None:
            config = {}

        # XXX get rid of this
        self.solid_weights = config.get("solid_weights", False)

    def add_tensor_operand(self, jitval, oper):
        assert isinstance(oper, Operand)
        if jitval in self.jitval_operand_map:
            raise Exception("Duplicate tensor: %r" % jitval)

        operand_id = len(self.operands)
        self.operands.append(oper)
        self.jitval_operand_map[jitval] = operand_id
        return operand_id

    @staticmethod
    def torch_tensor_to_operand(tensor, dim_order):
        dtype = str(tensor.dtype).replace("torch.", "")
        scale = 0.0
        zero_point = 0
        if dtype == "float32":
            op_type = NNAPI_OperandCode.TENSOR_FLOAT32
        elif dtype == "quint8":
            op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
        elif dtype == "qint32":
            op_type = NNAPI_OperandCode.TENSOR_INT32
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
            assert zero_point == 0
        else:
            raise Exception(f"Can't handle input with dtype '{tensor.dtype}'")
        return Operand(
            shape=tuple(tensor.shape),
            op_type=op_type,
            dim_order=dim_order,
            scale=scale,
            zero_point=zero_point,
        )

    def add_tensor_operand_for_input(self, jitval, tensor):
        dim_order = (
            DimOrder.CHANNELS_LAST if getattr(tensor, "nnapi_nhwc", False)
            else DimOrder.PRESUMED_CONTIGUOUS)
        toper = self.torch_tensor_to_operand(tensor, dim_order)
        operand_id = self.add_tensor_operand(jitval, toper)
        self.inputs.append(operand_id)
        return operand_id

    def add_tensor_operand_for_weight(self, tensor):
        toper = self.torch_tensor_to_operand(tensor, DimOrder.UNKNOWN_CONSTANT)
        operand_id = len(self.operands)
        self.operands.append(toper)
        tsize = tensor_size(toper.op_type, toper.shape)
        psize = ((tsize - 1) | 0x3) + 1
        self.values.append((operand_id, OperandValueSourceType.NUMBERED_BUFFER))
        if self.solid_weights:
            buf_num = 0
            offset = self.weight_offset
            self.weight_offset += psize
        else:
            buf_num = len(self.used_weights)
            offset = 0
        self.value_data.append(struct.pack(
            "iii",
            buf_num,
            offset,
            tsize))
        self.used_weights.append(tensor)
        return operand_id

    def add_immediate_operand(self, code, value, dims):
        assert isinstance(dims, tuple)
        cache_key = (code, value)
        if cache_key not in self.cached_immediates:
            operand_id = len(self.operands)
            self.operands.append(Operand(code, dims, DimOrder.SCALAR_OR_VECTOR, 0.0, 0))
            self.values.append((operand_id, OperandValueSourceType.IMMEDIATE))
            self.value_data.append(value)
            self.cached_immediates[cache_key] = operand_id
        return self.cached_immediates[cache_key]

    def add_immediate_int_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.INT32,
            struct.pack("i", value),
            ())

    def add_immediate_float_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.FLOAT32,
            struct.pack("f", value),
            ())

    def add_immediate_bool_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.BOOL,
            b"\x01" if value else b"\x00",
            ())

    def add_immediate_int_vector(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.TENSOR_INT32,
            array.array("i", value).tobytes(),
            (len(value),))

    def get_tensor_operand_by_jitval(self, jitval):
        operand_id = self.jitval_operand_map[jitval]
        return (operand_id, self.operands[operand_id])

    def get_tensor_operand_or_constant(self, jitval):
        operand_id = self.jitval_operand_map.get(jitval)
        if operand_id is None:
            _, value = self.get_constant_value(jitval, "TensorType")
            operand_id = self.add_tensor_operand_for_weight(value)
        return (operand_id, self.operands[operand_id])

    def get_tensor_operand_for_weight(self, jitval):
        _, value = self.get_constant_value(jitval, "TensorType")
        operand_id = self.add_tensor_operand_for_weight(value)
        return (operand_id, self.operands[operand_id])

    def add_operation(self, opcode, inputs, outputs):
        self.operations.append((opcode, len(inputs), len(outputs)))
        self.operation_args.extend(inputs + outputs)

    def add_tensor_tuple(self, jitval, values):
        assert jitval not in self.tensor_tuples
        self.tensor_tuples[jitval] = values

    def add_constant_value(self, jitval, ctype, value):
        assert jitval not in self.constants
        self.constants[jitval] = (ctype, value)

    def get_constant_value(self, jitval, typekind=None):
        record = self.constants.get(jitval)
        if record is None:
            raise Exception(f"Could not find constant value for '{jitval!r}'.")
        ctype, _ = record
        if typekind is not None and ctype.kind() != typekind:
            raise Exception(
                f"Expected constant value of type {typekind}, but got {ctype.kind()} for value '{jitval!r}'")
        return record

    def get_size_arg(self, jitval):
        ctype, value = self.get_constant_value(jitval)
        if ctype.kind() == "ListType":
            assert ctype.getElementType().kind() == "IntType"
            return value
        raise Exception(f"Can't handle size arg of type '{ctype!r}' for '{jitval!r}'")

    def get_conv_pool_args_2d_from_pack(self, kernel_size, packed_config):
        pc = [i.item() for i in packed_config]
        assert pc[0] == 2
        strides = [pc[1], pc[2]]
        paddings = [pc[3], pc[4]]
        dilations = [pc[5], pc[6]]
        output_padding = [pc[7], pc[8]]
        group_num = pc[9]
        transpose = pc[10]

        assert len(pc) == 11
        assert output_padding == [0, 0]
        assert transpose == 0

        return self.get_conv_pool_args_2d_common(kernel_size, strides, paddings, dilations, group_num)

    def get_conv_pool_args_2d_from_jit(self, kernel_size, stride, padding, dilation, group=None):
        strides = self.get_size_arg(stride)
        paddings = self.get_size_arg(padding)
        dilations = self.get_size_arg(dilation)
        if group is not None:
            _, group_num = self.get_constant_value(group, "IntType")
        else:
            group_num = None
        return self.get_conv_pool_args_2d_common(kernel_size, strides, paddings, dilations, group_num)

    def get_conv_pool_args_2d_common(self, kernel_size, strides, paddings, dilations, group_num):
        kernels = list(kernel_size)

        assert len(kernels) == 2
        assert len(strides) == 2
        assert len(paddings) == 2
        assert len(dilations) == 2

        # NNAPI uses 4 values for padding.
        ph, pw = paddings
        real_paddings = [ph, ph, pw, pw]

        return ConvPoolArgs2d(*(kernels + strides + real_paddings + dilations + [group_num]))

    def serialize_model(self, model, inputs):
        self.add_immediate_bool_scalar(False)
        self.add_immediate_bool_scalar(True)

        inp_dim_orders = []
        out_dim_orders = []

        self_jitval = next(model.graph.inputs())
        self.add_constant_value(self_jitval, self_jitval.type(), model)

        for input_value, input_tensor in zip(list(model.graph.inputs())[1:], inputs):
            op_id = self.add_tensor_operand_for_input(input_value, input_tensor)
            inp_dim_orders.append(self.operands[op_id].dim_order.value)

        for idx, node in enumerate(model.graph.nodes()):
            LOG.debug("Processing node #%d: %r", idx, node)
            self.add_node(node)

        retn = model.graph.return_node()
        assert retn.inputsSize() == 1
        assert retn.outputsSize() == 0
        retn_input = retn.inputsAt(0)
        if retn_input.type().kind() == "TensorType":
            op_id = self.jitval_operand_map[retn_input]
            # TODO: Make outputs a local variable?
            self.outputs.append(op_id)
            out_dim_orders.append(self.operands[op_id].dim_order.value)
        elif retn_input.type().kind() == "TupleType":
            for v in self.tensor_tuples[retn_input]:
                op_id = self.jitval_operand_map[v]
                self.outputs.append(op_id)
                out_dim_orders.append(self.operands[op_id].dim_order.value)

        model = []

        version = 1
        header = struct.pack(
            "iiiiii",
            version,
            len(self.operands),
            len(self.values),
            len(self.operations),
            len(self.inputs),
            len(self.outputs),
        )
        model.append(header)

        serialized_values, serialized_value_data = self.serialize_values()

        model.extend(struct.pack("iifi", t, len(d), s, z) for (t, d, _m, s, z) in self.operands)
        model.extend(serialized_values)
        model.extend(struct.pack("iii", *x) for x in self.operations)
        model.extend(self.serialize_ints(fix_shape(dims, mf)) for (_, dims, mf, _, _) in self.operands)
        model.extend(serialized_value_data)
        model.append(self.serialize_ints(self.operation_args))
        model.append(self.serialize_ints(self.inputs))
        model.append(self.serialize_ints(self.outputs))

        # return (b"".join(model), self.used_weight_tensor_names)
        return (b"".join(model), self.used_weights, inp_dim_orders, out_dim_orders)

    def serialize_values(self):
        serialized_values = []
        serialized_value_data = []
        assert len(self.values) == len(self.value_data)
        for ((op_index, source_type), data) in zip(self.values, self.value_data):
            source_length = len(data)

            # Pad with 0 bytes out to a multiple of 4 for alignment.
            physical_length = ((source_length - 1) | 0x3) + 1
            padded_data = data + (b"\0" * (physical_length - source_length))

            serialized_values.append(struct.pack("iii", op_index, source_type, source_length))
            serialized_value_data.append(padded_data)

        return serialized_values, serialized_value_data

    @staticmethod
    def serialize_ints(ints):
        return struct.pack("i" * len(ints), *ints)

    ADDER_MAP = {
        "prim::GetAttr": lambda self, node:
            self.add_getattr(node),
        "prim::Constant": lambda self, node:
            self.add_constant_node(node),
        "prim::ListConstruct": lambda self, node:
            self.add_list_construct(node),
        "prim::TupleConstruct": lambda self, node:
            self.add_tuple_construct(node),
        "aten::reshape": lambda self, node:
            self.add_reshape(node),
        "aten::size": lambda self, node:
            self.add_size(node),
        "aten::quantize_per_tensor": lambda self, node:
            self.add_quantize(node),
        "aten::dequantize": lambda self, node:
            self.add_dequantize(node),
        "aten::add": lambda self, node:
            self.add_add_sub_op(node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_NONE),
        "aten::sub": lambda self, node:
            self.add_add_sub_op(node, NNAPI_OperationCode.SUB, NNAPI_FuseCode.FUSED_NONE),
        "aten::mul": lambda self, node:
            self.add_pointwise_simple_binary_broadcast_op(node, NNAPI_OperationCode.MUL),
        "aten::relu": lambda self, node:
            self.add_pointwise_simple_unary_op(node, NNAPI_OperationCode.RELU),
        "aten::sigmoid": lambda self, node:
            self.add_pointwise_simple_unary_op(node, NNAPI_OperationCode.LOGISTIC),
        "aten::hardtanh": lambda self, node:
            self.add_hardtanh(node),
        "aten::max_pool2d": lambda self, node:
            self.add_pool2d_node(node, NNAPI_OperationCode.MAX_POOL_2D),
        "aten::adaptive_avg_pool2d": lambda self, node:
            self.add_adaptive_avg_pool2d(node),
        "aten::upsample_nearest2d": lambda self, node:
            self.add_upsample_nearest2d(node),
        "aten::prelu": lambda self, node:
            self.add_prelu_op(node),
        "aten::addmm": lambda self, node:
            self.add_addmm(node),
        "aten::_convolution": lambda self, node:
            self.add_conv_underscore(node),
        "aten::conv2d": lambda self, node:
            self.add_conv2d(node),
        "quantized::linear": lambda self, node:
            self.add_qlinear(node),
        "quantized::conv2d": lambda self, node:
            self.add_qconv2d(node, NNAPI_FuseCode.FUSED_NONE),
        "quantized::conv2d_relu": lambda self, node:
            self.add_qconv2d(node, NNAPI_FuseCode.FUSED_RELU),
        "quantized::add": lambda self, node:
            self.add_qadd(node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_NONE),
        "quantized::add_relu": lambda self, node:
            self.add_qadd(node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_RELU),
    }

    def add_node(self, node):
        adder = self.ADDER_MAP.get(node.kind())
        if not adder:
            raise Exception("Unsupported node kind (%r) in node %r" % (node.kind(), node))
        adder(self, node)

    def add_getattr(self, node):
        assert node.inputsSize() == 1
        assert node.outputsSize() == 1
        obj_ctype, obj = self.get_constant_value(node.inputsAt(0))
        assert str(obj_ctype).startswith("__torch__.")
        name = node.s("name")
        value = getattr(obj, name)
        output = node.outputsAt(0)
        ctype = output.type()
        self.add_constant_value(output, ctype, value)

    def add_constant_node(self, node):
        assert node.inputsSize() == 0
        assert node.outputsSize() == 1
        output = node.outputsAt(0)
        ctype = output.type()
        value = output.toIValue()
        self.add_constant_value(output, ctype, value)

    def add_list_construct(self, node):
        assert node.outputsSize() == 1
        output = node.outputsAt(0)
        ctype = output.type()
        values = []
        for inp in node.inputs():
            _, val = self.get_constant_value(inp)
            values.append(val)
        self.add_constant_value(output, ctype, values)

    def add_tuple_construct(self, node):
        assert node.outputsSize() == 1
        output = node.outputsAt(0)
        values = []
        for inp in node.inputs():
            values.append(inp)
        self.add_tensor_tuple(output, values)

    def add_reshape(self, node):
        assert node.inputsSize() == 2
        assert node.outputsSize() == 1

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        shape_ctype, shape = self.get_constant_value(node.inputsAt(1))
        assert shape_ctype.kind() == "ListType"
        assert shape_ctype.getElementType().kind() == "IntType"
        is_trivial_reshape = len(shape) == 2 and shape[1] == -1

        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and not is_trivial_reshape:
            raise Exception(
                "Currently, reshape is only supported on NHWC tensors if the target size is [X, -1].")

        # Bit of a hack here.  Use a real tensor to infer the output shape.
        out_shape = torch.zeros(1).expand(in_oper.shape).reshape(shape).shape
        out_oper = in_oper._replace(shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS)

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(shape)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)

    def add_size(self, node):
        assert node.inputsSize() == 2
        assert node.outputsSize() == 1

        _, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        _, value = self.constants[node.inputsAt(1)]
        res = in_oper.shape[value]
        output = node.outputsAt(0)
        self.add_constant_value(output, output.type(), res)

    def add_quantize(self, node):
        assert node.inputsSize() == 4
        assert node.outputsSize() == 1

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        if in_oper.dim_order != DimOrder.CHANNELS_LAST:
            raise Exception(
                "Most hardware backends prefer NHWC quantized tensors.  "
                "Try setting `t.nnapi_nhwc = True` on your tensor inputs.  ")
        _, scale = self.get_constant_value(node.inputsAt(1), "FloatType")
        _, zero_point = self.get_constant_value(node.inputsAt(2), "IntType")
        _, scalar_type = self.get_constant_value(node.inputsAt(3), "IntType")
        if scalar_type != TorchScalarTypes.QUINT8.value:
            raise Exception(
                "PyTorch NNAPI export only supports quantized tensors "
                "with the quint8 dtype.")
        op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM

        out_oper = in_oper._replace(
            op_type=op_type,
            scale=scale,
            zero_point=zero_point,
        )

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.QUANTIZE, inputs, outputs)

    def add_dequantize(self, node):
        assert node.inputsSize() == 1
        assert node.outputsSize() == 1

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        out_oper = in_oper._replace(
            op_type=NNAPI_OperandCode.TENSOR_FLOAT32,
            scale=0.0,
            zero_point=0,
        )

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.DEQUANTIZE, inputs, outputs)

    def add_pointwise_simple_unary_op(self, node, opcode):
        assert node.inputsSize() == 1
        assert node.outputsSize() == 1

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), in_oper)

        self.add_operation(opcode, inputs, outputs)

    def _do_add_binary(self, node, opcode, fuse_code, *, qparams=None):
        """Helper for pointwise binary broadcast ops with superfluous extra args"""
        assert node.outputsSize() == 1

        assert node.inputsAt(0).type().kind() == "TensorType"
        assert node.inputsAt(1).type().kind() == "TensorType"

        # TODO: Should support constant as either operand.
        in0_id, in0_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        in1_id, in1_oper = self.get_tensor_operand_by_jitval(node.inputsAt(1))

        assert in0_oper.op_type == in1_oper.op_type
        assert in0_oper.dim_order == in1_oper.dim_order
        # NOTE: PyTorch and NNAPI have the same broadcast semantics.
        out_shape = broadcast_shapes(in0_oper.shape, in1_oper.shape)
        out_oper = in0_oper._replace(shape=out_shape)
        if qparams is not None:
            scale, zp = qparams
            out_oper = out_oper._replace(scale=scale, zero_point=zp)

        inputs = [None] * 3
        inputs[0] = in0_id
        inputs[1] = in1_id
        inputs[2] = self.add_immediate_int_scalar(fuse_code)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(opcode, inputs, outputs)

    def add_pointwise_simple_binary_broadcast_op(self, node, opcode):
        assert node.inputsSize() == 2
        self._do_add_binary(node, opcode)

    def add_add_sub_op(self, node, opcode, fuse_code):
        assert node.inputsSize() == 3

        _, alpha = self.get_constant_value(node.inputsAt(2), "IntType")
        if alpha != 1:
            raise Exception("NNAPI does not support add/sub with alpha.")

        self._do_add_binary(node, opcode, fuse_code)

    def add_qadd(self, node, opcode, fuse_code):
        assert node.inputsSize() == 4

        _, scale = self.get_constant_value(node.inputsAt(2), "FloatType")
        _, zero_point = self.get_constant_value(node.inputsAt(3), "IntType")

        self._do_add_binary(node, opcode, fuse_code, qparams=(scale, zero_point))

    def add_hardtanh(self, node):
        assert node.inputsSize() == 3
        assert node.outputsSize() == 1

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        _, min_val = self.get_constant_value(node.inputsAt(1), "FloatType")
        _, max_val = self.get_constant_value(node.inputsAt(2), "FloatType")

        op_map = {
            1: NNAPI_OperationCode.RELU1,
            6: NNAPI_OperationCode.RELU6,
        }

        if min_val != 0 or max_val not in op_map:
            raise Exception("NNAPI only supports hardtanh with args (0, 1) or (0, 6).")
        opcode = op_map[max_val]

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), in_oper)

        self.add_operation(opcode, inputs, outputs)

    def add_prelu_op(self, node):
        assert node.inputsSize() == 2
        assert node.outputsSize() == 1

        assert node.inputsAt(0).type().kind() == "TensorType"
        assert node.inputsAt(1).type().kind() == "TensorType"

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        w_id, w_oper = self.get_tensor_operand_for_weight(node.inputsAt(1))
        assert len(w_oper.shape) == 1
        assert w_oper.shape[0] > 0
        if w_oper.shape[0] > 1:
            if in_oper.use_nchw():
                # TODO: Support this by adding trailing 1 dims.
                raise Exception("Per-channel PReLU only supports channels_last right now.")

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = w_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), in_oper)

        self.add_operation(NNAPI_OperationCode.PRELU, inputs, outputs)

    def add_pool2d_node(self, node, opcode):
        assert node.inputsSize() == 6
        assert node.outputsSize() == 1
        image, kernel, stride, padding, dilation, ceil_mode = node.inputs()

        stride = stride or kernel

        # TODO: Validate ceil_mode semantics.

        args = self.get_conv_pool_args_2d_from_jit(self.get_size_arg(kernel), stride, padding, dilation)
        if args.dilation_h != 1 or args.dilation_w != 1:
            raise Exception("NNAPI does not support dilated pooling.")

        image_id, image_oper = self.get_tensor_operand_by_jitval(image)
        assert len(image_oper.shape) == 4

        out_shape = get_conv_pool_shape(image_oper.shape, args, image_oper.shape[1], False)
        use_nchw = image_oper.use_nchw()

        inputs = [None] * 11
        inputs[0] = image_id
        inputs[1] = self.add_immediate_int_scalar(args.pad_l)
        inputs[2] = self.add_immediate_int_scalar(args.pad_r)
        inputs[3] = self.add_immediate_int_scalar(args.pad_t)
        inputs[4] = self.add_immediate_int_scalar(args.pad_b)
        inputs[5] = self.add_immediate_int_scalar(args.stride_w)
        inputs[6] = self.add_immediate_int_scalar(args.stride_h)
        inputs[7] = self.add_immediate_int_scalar(args.kernel_w)
        inputs[8] = self.add_immediate_int_scalar(args.kernel_h)
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), image_oper._replace(shape=out_shape))

        self.add_operation(opcode, inputs, outputs)

    def add_adaptive_avg_pool2d(self, node):
        assert node.inputsSize() == 2
        assert node.outputsSize() == 1

        image_id, image_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        assert len(image_oper.shape) == 4

        size_ctype, size_arg = self.get_constant_value(node.inputsAt(1))
        assert size_ctype.kind() == "ListType"
        assert size_ctype.getElementType().kind() == "IntType"
        if size_arg != [1, 1]:
            raise Exception("NNAPI only supports adaptive_avg_pool2d with output size (1, 1).")

        out_shape = image_oper.shape[0:2] + tuple(size_arg)
        use_nchw = image_oper.use_nchw()

        inputs = [None] * 11
        inputs[0] = image_id
        inputs[1] = self.add_immediate_int_scalar(0)
        inputs[2] = self.add_immediate_int_scalar(0)
        inputs[3] = self.add_immediate_int_scalar(0)
        inputs[4] = self.add_immediate_int_scalar(0)
        inputs[5] = self.add_immediate_int_scalar(1)
        inputs[6] = self.add_immediate_int_scalar(1)
        inputs[7] = self.add_immediate_int_scalar(image_oper.shape[3])
        inputs[8] = self.add_immediate_int_scalar(image_oper.shape[2])
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), image_oper._replace(shape=out_shape))

        self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)

    def add_upsample_nearest2d(self, node):
        assert node.inputsSize() == 3
        assert node.outputsSize() == 1
        image, size_jit, scale_jit = node.inputs()
        size_ctype, size_arg = self.get_constant_value(size_jit)
        scale_ctype, scale_arg = self.get_constant_value(scale_jit)

        image_id, image_oper = self.get_tensor_operand_by_jitval(image)
        assert len(image_oper.shape) == 4

        if size_ctype.kind() != "NoneType" and scale_ctype.kind() != "NoneType":
            raise Exception("Size and scale cannot both be non-None.")
        elif size_ctype.kind() != "NoneType":
            assert size_ctype.kind() == "ListType"
            assert size_ctype.getElementType().kind() == "IntType"
            assert scale_ctype.kind() == "NoneType"
            assert scale_arg is None
            assert isinstance(size_arg, list)
            assert size_arg
            assert all(isinstance(val, int) for val in size_arg)
            if len(size_arg) == 1:
                size_arg = size_arg * 2
            assert len(size_arg) == 2
            out_h = size_arg[0]
            out_w = size_arg[1]
            arg_h = self.add_immediate_int_scalar(out_h)
            arg_w = self.add_immediate_int_scalar(out_w)
        elif scale_ctype.kind() != "NoneType":
            assert scale_ctype.kind() == "ListType"
            assert scale_ctype.getElementType().kind() == "FloatType"
            assert size_ctype.kind() == "NoneType"
            assert size_arg is None
            assert isinstance(scale_arg, list)
            assert scale_arg
            assert all(isinstance(val, float) for val in scale_arg)
            if len(scale_arg) == 1:
                scale_arg = scale_arg * 2
            assert len(scale_arg) == 2
            out_h = int(scale_arg[0] * image_oper.shape[2])
            out_w = int(scale_arg[1] * image_oper.shape[3])
            arg_h = self.add_immediate_float_scalar(scale_arg[0])
            arg_w = self.add_immediate_float_scalar(scale_arg[1])
        else:
            raise Exception("Size and scale cannot both be None.")

        out_shape = (image_oper.shape[0], image_oper.shape[1], out_h, out_w)
        use_nchw = image_oper.use_nchw()

        inputs = [None] * 4
        inputs[0] = image_id
        inputs[1] = arg_w
        inputs[2] = arg_h
        inputs[3] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), image_oper._replace(shape=out_shape))

        self.add_operation(NNAPI_OperationCode.RESIZE_NEAREST_NEIGHBOR, inputs, outputs)

    def add_addmm(self, node):
        assert node.inputsSize() == 5
        assert node.outputsSize() == 1
        jit_bias, jit_input, jit_weight, jit_beta, jit_alpha = node.inputs()

        for jitval in (jit_beta, jit_alpha):
            scale_ctype, scale_value = self.get_constant_value(jitval)
            assert scale_ctype.kind() in ("IntType", "FloatType")
            if scale_value != 1:
                raise Exception("NNAPI Fully-Connected does not support alpha and beta.")

        input_id, input_oper = self.get_tensor_operand_by_jitval(jit_input)
        bias_id, bias_oper = self.get_tensor_operand_for_weight(jit_bias)

        assert len(input_oper.shape) == 2
        assert len(bias_oper.shape) == 1

        # TODO: Transform at load time to share weights with CPU model.
        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        assert len(weight_tensor.shape) == 2
        nnapi_weight_tensor = weight_tensor.t().contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        out_shape = (input_oper.shape[0], weight_oper.shape[0])

        inputs = [None] * 4
        inputs[0] = input_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), input_oper._replace(shape=out_shape))

        self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)

    def add_qlinear(self, node):
        assert node.inputsSize() == 4
        assert node.outputsSize() == 1
        (
            jit_input,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        input_id, input_oper = self.get_tensor_operand_by_jitval(jit_input)
        # TODO: Support automatic reshape
        assert len(input_oper.shape) == 2

        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        assert weight_ctype.name() == "LinearPackedParamsBase"
        raw_weight, raw_bias = packed_weight.__getstate__()[0]
        assert raw_bias is not None

        assert len(raw_weight.shape) == 2
        assert len(raw_bias.shape) == 1
        assert raw_bias.shape[0] == raw_weight.shape[0]
        assert raw_weight.shape[1] == input_oper.shape[1]

        assert raw_weight.qscheme() == torch.per_tensor_affine
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            assert raw_weight.dtype == torch.qint8
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128)
        weight_scale = unsigned_weight.q_scale()
        bias_scale = input_oper.scale * weight_scale
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        multiplier = input_oper.scale * weight_scale / out_scale
        assert multiplier > 0
        if multiplier >= 1:
            raise Exception(
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  ")

        # TODO: Transform at load time to share weights with CPU model.
        nnapi_weight_tensor = unsigned_weight.contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        out_shape = (input_oper.shape[0], weight_oper.shape[0])
        out_oper = input_oper._replace(
            shape=out_shape,
            scale=out_scale,
            zero_point=out_zero_point,
        )

        inputs = [None] * 4
        inputs[0] = input_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)

    def get_optional_bias(self, jit_bias, weight_tensor):
        ctype, value = self.get_constant_value(jit_bias)
        if ctype.kind() == "NoneType":
            nnapi_bias_tensor = torch.zeros(weight_tensor.size()[0], dtype=weight_tensor.dtype)
            bias_id = self.add_tensor_operand_for_weight(nnapi_bias_tensor)
            bias_oper = self.operands[bias_id]
            return bias_id, bias_oper
        else:
            return self.get_tensor_operand_for_weight(jit_bias)

    def add_conv2d(self, node):
        assert node.inputsSize() == 7
        assert node.outputsSize() == 1

        (
            jit_image,
            jit_weight,
            jit_bias,
            jit_stride,
            jit_pad,
            jit_dilation,
            jit_groups,
        ) = node.inputs()

        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        bias_id, bias_oper = self.get_optional_bias(jit_bias, weight_tensor)
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups)

        return self.add_conv2d_common(
            node.outputsAt(0),
            0.0,
            0,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            False,  # transpose
            NNAPI_FuseCode.FUSED_NONE,
        )

    def add_conv_underscore(self, node):
        assert node.inputsSize() == 13
        assert node.outputsSize() == 1

        (
            jit_image,
            jit_weight,
            jit_bias,
            jit_stride,
            jit_pad,
            jit_dilation,
            jit_transpose,
            _,
            jit_groups,
            _,
            _,
            _,
            _,
        ) = node.inputs()

        # XXX check jit_transpose

        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        bias_id, bias_oper = self.get_optional_bias(jit_bias, weight_tensor)
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups)

        return self.add_conv2d_common(
            node.outputsAt(0),
            0.0,
            0,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            False,  # transpose
            NNAPI_FuseCode.FUSED_NONE,
        )

    def add_qconv2d(self, node, fuse_code):
        assert node.inputsSize() == 4
        assert node.outputsSize() == 1

        (
            jit_image,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        assert weight_ctype.name() == "Conv2dPackedParamsBase"
        (
            pack_version,
            tensors,
            opt_tensors,
        ) = packed_weight.__getstate__()[0]
        assert pack_version == "2"
        packed_config, raw_weight = tensors
        raw_bias, = opt_tensors
        assert raw_bias is not None
        args = self.get_conv_pool_args_2d_from_pack(raw_weight.shape[2:4], packed_config)

        assert raw_weight.qscheme() == torch.per_tensor_affine
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            assert raw_weight.dtype == torch.qint8
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128)
        weight_scale = unsigned_weight.q_scale()
        _, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        bias_scale = image_oper.scale * weight_scale
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        multiplier = image_oper.scale * weight_scale / out_scale
        assert multiplier > 0
        if multiplier >= 1:
            raise Exception(
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  ")

        return self.add_conv2d_common(
            node.outputsAt(0),
            out_scale,
            out_zero_point,
            jit_image,
            unsigned_weight,
            bias_id,
            args,
            False,  # transpose
            fuse_code,
        )

    def add_conv2d_common(
            self,
            jit_out,
            out_scale,
            out_zero_point,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            transpose,
            fuse_code):
        image_id, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        in_c = image_oper.shape[1]

        if args.group == 1:
            # Full convolution
            depthwise = False
            weight_permutation = (0, 2, 3, 1)
        elif args.group == in_c:
            # Depthwise convolution
            depthwise = True
            weight_permutation = (1, 2, 3, 0)
        else:
            raise Exception("Group convolution not supported yet.")

        # TODO: Transform at load time to share weights with CPU model.
        nnapi_weight_tensor = weight_tensor.permute(*weight_permutation).contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        bias_oper = self.operands[bias_id]

        if image_oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32:
            assert weight_oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32
            assert bias_oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32
        elif image_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
            assert weight_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
            assert bias_oper.op_type == NNAPI_OperandCode.TENSOR_INT32
            assert approx_equal(image_oper.scale * weight_oper.scale, bias_oper.scale)
            assert bias_oper.zero_point == 0
        else:
            raise Exception(
                "Unsupported input type for conv2d: {}"
                .format(image_oper.op_type))

        assert len(image_oper.shape) == 4
        assert len(weight_oper.shape) == 4
        assert len(bias_oper.shape) == 1

        if depthwise:
            # Depthwise convolution
            one, kern_h, kern_w, out_c = weight_oper.shape
            assert one == 1
            assert out_c % in_c == 0
            channel_multiplier = out_c // in_c
            assert channel_multiplier == 1  # Don't support multiplier
            assert out_c == in_c
        else:
            # Full convolution
            kern_nf, kern_h, kern_w, kern_d = weight_oper.shape
            out_c = kern_nf
            assert kern_d == in_c

        assert out_c == bias_oper.shape[0]

        out_shape = get_conv_pool_shape(image_oper.shape, args, out_c, transpose)
        out_oper = image_oper._replace(
            shape=out_shape,
            scale=out_scale,
            zero_point=out_zero_point,
        )

        use_nchw = image_oper.use_nchw()

        if depthwise:
            num_args = 12
            opcode = NNAPI_OperationCode.DEPTHWISE_CONV_2D
        else:
            num_args = 11
            if transpose:
                opcode = NNAPI_OperationCode.TRANSPOSE_CONV_2D
            else:
                opcode = NNAPI_OperationCode.CONV_2D

        inputs = [None] * num_args
        inputs[0] = image_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(args.pad_l)
        inputs[4] = self.add_immediate_int_scalar(args.pad_r)
        inputs[5] = self.add_immediate_int_scalar(args.pad_t)
        inputs[6] = self.add_immediate_int_scalar(args.pad_b)
        inputs[7] = self.add_immediate_int_scalar(args.stride_w)
        inputs[8] = self.add_immediate_int_scalar(args.stride_h)
        if depthwise:
            inputs[9] = self.add_immediate_int_scalar(1)
            inputs[10] = self.add_immediate_int_scalar(fuse_code)
            inputs[11] = self.add_immediate_bool_scalar(use_nchw)
        else:
            inputs[9] = self.add_immediate_int_scalar(fuse_code)
            inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(jit_out, out_oper)

        self.add_operation(opcode, inputs, outputs)


def serialize_model(module, inputs, config=None):
    return _NnapiSerializer(config).serialize_model(module, inputs)
