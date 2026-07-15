# mypy: allow-untyped-defs
import array
import enum
import functools
import logging
import operator
import struct
import sys
from typing import NamedTuple, Optional

import torch


# TODO: Add type annotations
# TODO: Check tensor types for ops


LOG = logging.getLogger("nnapi_serialize")


class NNAPI_OperandCode:
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


class NNAPI_OperationCode:
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


class NNAPI_FuseCode:
    FUSED_NONE = 0
    FUSED_RELU = 1
    FUSED_RELU1 = 2
    FUSED_RELU6 = 3


class OperandValueSourceType:
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
        NNAPI_OperandCode.TENSOR_QUANT16_ASYMM: 2,
    }
    size = ITEM_SIZES[op_type]
    for d in dims:
        size *= d
    return size


def change_element(tup, index, value):
    ls = list(tup)
    ls[index] = value
    return tuple(ls)


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
    """Representation of an NNAPI operand."""

    # NNAPI operand type.  One of NNAPI_OperandCode.
    # TODO: Make this an enum.
    op_type: int

    # This is always the PyTorch shape, which is NCHW for feature maps.
    # The actual NNAPI operand might have a transposed shape.
    # we use 0 for load time dynamic shapes & -1 for runtime dynamic shapes
    shape: tuple[int, ...]

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
        raise Exception("Unknown dim order")  # noqa: TRY002


def broadcast_shapes(shape1, shape2):
    if len(shape1) <= 0:
        raise AssertionError(f"shape1 must have length > 0, got {len(shape1)}")
    if len(shape2) <= 0:
        raise AssertionError(f"shape2 must have length > 0, got {len(shape2)}")
    s1 = list(shape1)
    s2 = list(shape2)
    # TODO: Support non-equal-rank broadcast where semantics match.
    # This can be tricky for NHWC tensors because dimension orders
    # don't match between PT and NNAPI, even though semantics match.
    if len(s1) > len(s2):
        # s2 = [1] * (len(s1) - len(s2)) + s2
        raise Exception(  # noqa: TRY002
            "Non-equal-rank broadcast is not supported yet."
        )  # noqa: TRY002
    if len(s2) > len(s1):
        # s3 = [1] * (len(s2) - len(s1)) + s1
        raise Exception(  # noqa: TRY002
            "Non-equal-rank broadcast is not supported yet."
        )  # noqa: TRY002
    ret = []
    for d1, d2 in zip(s1, s2):
        if d1 == 1:
            ret.append(d2)
        elif d2 == 1:
            ret.append(d1)
        elif d1 == d2:
            ret.append(d1)
        else:
            raise Exception(  # noqa: TRY002
                f"Cannot broadcast shapes: {shape1} and {shape2}"
            )  # noqa: TRY002
    return tuple(ret)


def get_conv_pool_shape(image_shape, args, out_ch, transpose):
    batch, _in_c, in_h, in_w = image_shape

    # TODO: Handle dilation
    if args.dilation_h != 1 or args.dilation_w != 1:
        raise Exception("Dilation not supported yet.")  # noqa: TRY002

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
        if not (len(shape) == 0 or len(shape) == 1):
            raise AssertionError(
                f"SCALAR_OR_VECTOR requires len(shape) == 0 or 1, got {len(shape)}"
            )
        return shape
    if dim_order is DimOrder.UNKNOWN_CONSTANT:
        # XXX think this through
        return shape
    raise Exception(f"Bad dim_order: {dim_order!r}.")  # noqa: TRY002


def reverse_map_dim(dim_order, d):
    # Return the original PyTorch dimension position for a given dimension.
    # d should be the dimension that NNAPI will see.
    # reverse_map_dim(PRESUMED_CONTIGUOUS, x) == x
    # reverse_map_dim(CHANNELS_LAST, 3) == 1
    if dim_order in (DimOrder.PRESUMED_CONTIGUOUS, DimOrder.SCALAR_OR_VECTOR):
        return d
    if dim_order is DimOrder.CHANNELS_LAST:
        return [0, 2, 3, 1][d]
    raise AssertionError(f"expected DimOrder.CHANNELS_LAST, got {dim_order}")


def flex_name(op_id, dim):
    # Return the local variable name for the computed flexible size
    # for a given op and dimension.
    return f"s_{op_id}_{dim}"


class _NnapiSerializer:
    def __init__(self, config, use_int16_for_qint16=False):
        self.operands = []
        self.values = []
        self.operations = []
        self.value_data = []
        self.operation_args = []
        self.inputs = []
        self.outputs = []
        self.flexible_shape_computation_lines = []

        self.modules = {}
        self.constants = {}
        self.tensor_sequences = {}
        self.jitval_operand_map = {}
        self.cached_immediates = {}
        self.used_weights = []
        self.weight_offset = 0
        self.use_int16_for_qint16 = use_int16_for_qint16

        if config is None:
            config = {}

    def get_next_operand_id(self):
        return len(self.operands)

    # Add a tensor operand corresponding to a JIT Value.
    # Returns the NNAPI operand ID.  Can be looked up later with
    # get_tensor_operand_by_jitval.
    def add_tensor_operand(self, jitval, oper):
        if not isinstance(oper, Operand):
            raise AssertionError(f"expected Operand, got {type(oper)}")
        if jitval in self.jitval_operand_map:
            raise Exception(f"Duplicate tensor: {jitval!r}")  # noqa: TRY002

        operand_id = self.get_next_operand_id()
        self.operands.append(oper)
        self.jitval_operand_map[jitval] = operand_id
        return operand_id

    # Add a tensor operand that does not correspond to a JIT Value.
    # Useful for cases where multiple NNAPI operands are required
    # to implement one JIT IR node.  Returns the NNAPI operand ID.
    def add_anonymous_tensor_operand(self, oper):
        if not isinstance(oper, Operand):
            raise AssertionError(f"expected Operand, got {type(oper)}")
        operand_id = self.get_next_operand_id()
        self.operands.append(oper)
        return operand_id

    def torch_tensor_to_operand(self, tensor, dim_order):
        dtype = str(tensor.dtype).replace("torch.", "")
        scale = 0.0
        zero_point = 0
        if dtype == "float32":
            op_type = NNAPI_OperandCode.TENSOR_FLOAT32
        elif dtype == "int32":
            op_type = NNAPI_OperandCode.TENSOR_INT32
        elif dtype == "quint8":
            op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
        elif dtype == "qint32":
            op_type = NNAPI_OperandCode.TENSOR_INT32
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
            if zero_point != 0:
                raise AssertionError(f"qint32 zero_point must be 0, got {zero_point}")
        elif dtype == "int16":
            if self.use_int16_for_qint16:
                nnapi_dtype = getattr(tensor, "nnapi_dtype", None)
                op_codes = (
                    NNAPI_OperandCode.TENSOR_QUANT16_SYMM,
                    NNAPI_OperandCode.TENSOR_QUANT16_ASYMM,
                )
                if nnapi_dtype in op_codes:
                    op_type = nnapi_dtype
                    scale = tensor.nnapi_scale
                    zero_point = tensor.nnapi_zero_point
                else:
                    raise Exception(  # noqa: TRY002
                        f"`nnapi_type` needs to be one of {op_codes} for `int16`"
                    )
            else:
                raise Exception(  # noqa: TRY002
                    "`int16` isn't supported. If you're trying to represent NNAPI"
                    " qint16 with Pytorch int16, set `use_int16_for_qint16 = True`"
                )
        else:
            raise Exception(  # noqa: TRY002
                f"Can't handle input with dtype '{tensor.dtype}'"
            )  # noqa: TRY002
        return Operand(
            shape=tuple(tensor.shape),
            # pyrefly: ignore [bad-argument-type]
            op_type=op_type,
            dim_order=dim_order,
            scale=scale,
            zero_point=zero_point,
        )

    def add_tensor_operand_for_input(self, arg_idx, jitval, tensor):
        dim_order = (
            DimOrder.CHANNELS_LAST
            if getattr(tensor, "nnapi_nhwc", False)
            else DimOrder.PRESUMED_CONTIGUOUS
        )
        toper = self.torch_tensor_to_operand(tensor, dim_order)
        operand_id = self.add_tensor_operand(jitval, toper)
        self.inputs.append(operand_id)
        for dim, size in enumerate(tensor.shape):
            if size == 0:
                self.compute_operand_shape(
                    operand_id, dim, f"args[{arg_idx}].shape[{dim}]"
                )
        return operand_id

    def add_tensor_operand_for_weight(
        self, tensor, dim_order=DimOrder.UNKNOWN_CONSTANT
    ):
        toper = self.torch_tensor_to_operand(tensor, dim_order)
        operand_id = len(self.operands)
        self.operands.append(toper)
        tsize = tensor_size(toper.op_type, toper.shape)
        self.values.append((operand_id, OperandValueSourceType.NUMBERED_BUFFER))
        buf_num = len(self.used_weights)
        offset = 0
        self.value_data.append(struct.pack("iii", buf_num, offset, tsize))
        # For NHWC NNAPI op, lay out data in the same dim order by permuting torch tensor
        if dim_order == DimOrder.CHANNELS_LAST:
            tensor = tensor.permute(0, 2, 3, 1)
        self.used_weights.append(tensor)
        return operand_id

    def add_immediate_operand(self, code, value, dims):
        if not isinstance(dims, tuple):
            raise AssertionError(f"dims must be a tuple, got {type(dims)}")
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
            NNAPI_OperandCode.INT32, struct.pack("i", value), ()
        )

    def add_immediate_float_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.FLOAT32, struct.pack("f", value), ()
        )

    def add_immediate_bool_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.BOOL, b"\x01" if value else b"\x00", ()
        )

    def add_immediate_int_vector(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.TENSOR_INT32,
            array.array("i", value).tobytes(),
            (len(value),),
        )

    def has_operand_for_jitval(self, jitval):
        return jitval in self.jitval_operand_map

    def get_tensor_operand_by_jitval(self, jitval):
        operand_id = self.jitval_operand_map[jitval]
        return (operand_id, self.operands[operand_id])

    def get_tensor_operand_by_jitval_fixed_size(self, jitval):
        op_id, oper = self.get_tensor_operand_by_jitval(jitval)
        for s in oper.shape:
            if s == 0:
                # TODO: Improve this error message, possibly after converting
                # many callsites to support flexible size.
                raise Exception(  # noqa: TRY002
                    "Flexible size is not supported for this operand."
                )  # noqa: TRY002
            if s < 0:
                # runtime flex
                LOG.warning("Operand %s has runtime flex shape", oper)
        return op_id, oper

    def get_tensor_operand_or_constant(
        self, jitval, dim_order=DimOrder.PRESUMED_CONTIGUOUS
    ):
        operand_id = self.jitval_operand_map.get(jitval)
        if operand_id is None:
            _, value = self.get_constant_value(jitval, "TensorType")
            operand_id = self.add_tensor_operand_for_weight(value, dim_order)
        return (operand_id, self.operands[operand_id])

    def get_tensor_operand_for_weight(self, jitval):
        _, value = self.get_constant_value(jitval, "TensorType")
        operand_id = self.add_tensor_operand_for_weight(value)
        return (operand_id, self.operands[operand_id])

    def add_operation(self, opcode, inputs, outputs):
        self.operations.append((opcode, len(inputs), len(outputs)))
        self.operation_args.extend(inputs + outputs)

    def add_tensor_sequence(self, jitval, values):
        if jitval in self.tensor_sequences:
            raise AssertionError(f"jitval {jitval!r} already in tensor_sequences")
        self.tensor_sequences[jitval] = values

    def add_constant_value(self, jitval, ctype, value):
        if jitval in self.constants:
            raise AssertionError(f"jitval {jitval!r} already in constants")
        self.constants[jitval] = (ctype, value)

    def get_constant_value(self, jitval, typekind=None):
        record = self.constants.get(jitval)
        if record is None:
            raise Exception(  # noqa: TRY002
                f"Could not find constant value for '{jitval!r}'."
            )  # noqa: TRY002
        ctype, _ = record
        if typekind is not None and ctype.kind() != typekind:
            raise Exception(  # noqa: TRY002
                f"Expected constant value of type {typekind}, but got {ctype.kind()} for value '{jitval!r}'"
            )
        return record

    def operand_to_template_torchscript(self, op_id, oper, shape=None):
        """Return a TorchScript expression to build a template for a given operand."""
        if shape is None:
            shape = oper.shape
        else:
            if len(shape) != len(oper.shape):
                raise AssertionError(
                    f"shape length {len(shape)} != oper.shape length {len(oper.shape)}"
                )

        shape_parts = ["("]
        for d, s in enumerate(shape):
            if s > 0:
                # Fixed shape dimension: just add the value.
                shape_parts.append(str(s))
            elif s == 0:
                # Load time flexible shape dimension: it should have been computed in a variable.
                shape_parts.append(flex_name(op_id, d))
            elif s == -1:
                # Runtime flexible shape
                shape_parts.append("0")
            else:
                raise Exception(  # noqa: TRY002
                    "Unknown dim value, dimensions should be >= -1"
                )  # noqa: TRY002
            shape_parts.append(",")
        shape_parts.append(")")
        shape_code = "".join(shape_parts)
        if oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32:
            return f"torch.zeros({shape_code}, dtype=torch.float32)"
        elif oper.op_type == NNAPI_OperandCode.TENSOR_INT32:
            return f"torch.zeros({shape_code}, dtype=torch.int32)"
        elif oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
            return (
                f"torch.quantize_per_tensor("
                f"torch.zeros(1), scale={oper.scale}, zero_point={oper.zero_point}, dtype=torch.quint8)"
                f".expand({shape_code}).contiguous()"
            )
        elif oper.op_type in (
            NNAPI_OperandCode.TENSOR_QUANT16_ASYMM,
            NNAPI_OperandCode.TENSOR_QUANT16_SYMM,
        ):
            if self.use_int16_for_qint16:
                return f"torch.zeros({shape_code}, dtype=torch.int16)"
            else:
                raise Exception(  # noqa: TRY002
                    "`int16` isn't supported. If you're trying to represent NNAPI"
                    " qint16 with Pytorch int16, set `use_int16_for_qint16 = True`"
                )

        raise Exception(  # noqa: TRY002
            f"Unsupported output operand type: {oper.op_type}"
        )  # noqa: TRY002

    def forward_operand_shape(self, out_op_id, out_dim, in_op_id, in_dim):
        self.compute_operand_shape(out_op_id, out_dim, flex_name(in_op_id, in_dim))

    def compute_operand_shape(self, op_id, dim, expr):
        self.flexible_shape_computation_lines.append(
            f"{flex_name(op_id, dim)} = {expr}"
        )

    def transpose_to_nhwc(self, in_id, oper):
        if oper.shape[2:] != (1, 1):
            raise Exception(  # noqa: TRY002
                "Automatic transpose only supported for H,W == 1,1"
            )  # noqa: TRY002

        out_oper = oper._replace(dim_order=DimOrder.CHANNELS_LAST)

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector([0, 2, 3, 1])

        outputs = [None] * 1
        outputs[0] = self.add_anonymous_tensor_operand(out_oper)

        self.add_operation(NNAPI_OperationCode.TRANSPOSE, inputs, outputs)

        return outputs[0], out_oper

    # Transpose inputs as necessary to allow broadcasting.
    def transpose_for_broadcast(self, in0_id, in0_oper, in1_id, in1_oper):
        if in0_oper.dim_order == in1_oper.dim_order:
            return in0_id, in0_oper, in1_id, in1_oper

        # Assume NHWC is preferred if there is a mismatch.
        orders = (in0_oper.dim_order, in1_oper.dim_order)
        if orders == (DimOrder.PRESUMED_CONTIGUOUS, DimOrder.CHANNELS_LAST):
            return self.transpose_to_nhwc(in0_id, in0_oper) + (in1_id, in1_oper)
        if orders == (DimOrder.CHANNELS_LAST, DimOrder.PRESUMED_CONTIGUOUS):
            return (in0_id, in0_oper) + self.transpose_to_nhwc(in1_id, in1_oper)

        raise Exception(  # noqa: TRY002
            f"Automatic transpose not supported for dim_orders: {in0_oper.dim_order!r}, {in1_oper.dim_order!r}"
        )

    def get_size_arg(self, jitval):
        ctype, value = self.get_constant_value(jitval)
        if ctype.kind() == "ListType":
            if ctype.getElementType().kind() != "IntType":
                raise AssertionError(
                    f"expected ListType of IntType, got {ctype.getElementType().kind()}"
                )
            return value
        raise Exception(  # noqa: TRY002
            f"Can't handle size arg of type '{ctype!r}' for '{jitval!r}'"
        )  # noqa: TRY002

    def get_conv_pool_args_2d_from_pack(self, kernel_size, packed_config):
        pc = [i.item() for i in packed_config]
        if pc[0] != 2:
            raise AssertionError(f"expected pc[0] == 2, got {pc[0]}")
        strides = [pc[1], pc[2]]
        paddings = [pc[3], pc[4]]
        dilations = [pc[5], pc[6]]
        output_padding = [pc[7], pc[8]]
        group_num = pc[9]

        if len(pc) != 11:
            raise AssertionError(f"expected len(pc) == 11, got {len(pc)}")
        if output_padding != [0, 0]:
            raise AssertionError(
                f"expected output_padding == [0, 0], got {output_padding}"
            )

        return self.get_conv_pool_args_2d_common(
            kernel_size, strides, paddings, dilations, group_num
        )

    def get_conv_pool_args_2d_from_jit(
        self, kernel_size, stride, padding, dilation=None, group=None
    ):
        strides = self.get_size_arg(stride)
        paddings = self.get_size_arg(padding)
        if dilation is None:
            dilations = [1, 1]
        else:
            dilations = self.get_size_arg(dilation)
        if group is not None:
            _, group_num = self.get_constant_value(group, "IntType")
        else:
            group_num = None
        return self.get_conv_pool_args_2d_common(
            kernel_size, strides, paddings, dilations, group_num
        )

    def get_conv_pool_args_2d_common(
        self, kernel_size, strides, paddings, dilations, group_num
    ):
        kernels = list(kernel_size)

        if len(kernels) != 2:
            raise AssertionError(f"expected len(kernels) == 2, got {len(kernels)}")
        if len(strides) != 2:
            raise AssertionError(f"expected len(strides) == 2, got {len(strides)}")
        if len(paddings) != 2:
            raise AssertionError(f"expected len(paddings) == 2, got {len(paddings)}")
        if len(dilations) != 2:
            raise AssertionError(f"expected len(dilations) == 2, got {len(dilations)}")

        # NNAPI uses 4 values for padding.
        ph, pw = paddings
        real_paddings = [ph, ph, pw, pw]

        return ConvPoolArgs2d(
            *(kernels + strides + real_paddings + dilations + [group_num])
        )

    def serialize_model(self, model, inputs, return_shapes=None):
        self.add_immediate_bool_scalar(False)
        self.add_immediate_bool_scalar(True)

        inp_dim_orders = []
        out_dim_orders = []

        self_jitval = next(model.graph.inputs())
        self.add_constant_value(self_jitval, self_jitval.type(), model)

        for arg_idx, (input_value, input_tensor) in enumerate(
            zip(list(model.graph.inputs())[1:], inputs)
        ):
            op_id = self.add_tensor_operand_for_input(
                arg_idx, input_value, input_tensor
            )
            inp_dim_orders.append(self.operands[op_id].dim_order.value)

        for idx, node in enumerate(model.graph.nodes()):
            LOG.debug("Processing node #%d: %r", idx, node)
            self.add_node(node)

        retn = model.graph.return_node()
        if retn.inputsSize() != 1:
            raise AssertionError(
                f"expected retn.inputsSize() == 1, got {retn.inputsSize()}"
            )
        if retn.outputsSize() != 0:
            raise AssertionError(
                f"expected retn.outputsSize() == 0, got {retn.outputsSize()}"
            )
        retn_input = retn.inputsAt(0)
        template_return_lines = ["return ["]
        if retn_input.type().kind() == "TensorType":
            return_values = [retn_input]
            retval_count = -1
        elif retn_input.type().kind() == "TupleType":
            return_values = self.tensor_sequences[retn_input]
            retval_count = len(return_values)
        else:
            raise Exception(  # noqa: TRY002
                f"Unsupported return type: {retn_input.type()}"
            )  # noqa: TRY002

        if return_shapes is not None:
            if len(return_shapes) != len(return_values):
                raise AssertionError(
                    f"return_shapes length {len(return_shapes)} != return_values length {len(return_values)}"
                )
        for i, v in enumerate(return_values):
            op_id = self.jitval_operand_map[v]
            self.outputs.append(op_id)
            out_dim_orders.append(self.operands[op_id].dim_order.value)
            shape = return_shapes[i] if return_shapes else None
            template_return_lines.append(
                self.operand_to_template_torchscript(op_id, self.operands[op_id], shape)
                + ","
            )
        template_return_lines.append("]")

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

        model.extend(
            struct.pack("iifi", t, len(d), s, z) for (t, d, _m, s, z) in self.operands
        )
        model.extend(serialized_values)
        model.extend(struct.pack("iii", *x) for x in self.operations)

        # Compact the model so we can get its length so far.
        model = [b"".join(model)]
        model_offset = len(model[0])
        # Model offset is the index into the model (in 32-bit words, not bytes)
        # of the next dimension we're about to serialize.  If it's 0,
        # generate code to mutate it before passing to NNAPI.
        if model_offset % 4 != 0:
            raise AssertionError(
                f"model_offset must be divisible by 4, got {model_offset}"
            )
        model_offset = int(model_offset / 4)

        for op_id, (_, dims, dim_order, _, _) in enumerate(self.operands):
            shape = fix_shape(dims, dim_order)
            for d, s in enumerate(shape):
                if s == 0:
                    pt_d = reverse_map_dim(dim_order, d)
                    self.flexible_shape_computation_lines.append(
                        f"ser_model[{model_offset}] = {flex_name(op_id, pt_d)}"
                    )
                model_offset += 1

            # convert runtime flex shape from -1 to 0
            shape = tuple(d if d != -1 else 0 for d in shape)
            model.append(self.serialize_ints(shape))

        model.extend(serialized_value_data)
        model.append(self.serialize_ints(self.operation_args))
        model.append(self.serialize_ints(self.inputs))
        model.append(self.serialize_ints(self.outputs))

        self.flexible_shape_computation_lines.extend(template_return_lines)

        return (
            array.array("i", b"".join(model)),
            self.used_weights,
            inp_dim_orders,
            out_dim_orders,
            self.flexible_shape_computation_lines,
            retval_count,
        )

    def serialize_values(self):
        serialized_values = []
        serialized_value_data = []
        if len(self.values) != len(self.value_data):
            raise AssertionError(
                f"values length {len(self.values)} != value_data length {len(self.value_data)}"
            )
        for (op_index, source_type), data in zip(self.values, self.value_data):
            source_length = len(data)

            # Pad with 0 bytes out to a multiple of 4 for alignment.
            physical_length = ((source_length - 1) | 0x3) + 1
            padded_data = data + (b"\0" * (physical_length - source_length))

            serialized_values.append(
                struct.pack("iii", op_index, source_type, source_length)
            )
            serialized_value_data.append(padded_data)

        return serialized_values, serialized_value_data

    @staticmethod
    def serialize_ints(ints):
        return array.array("i", ints).tobytes()

    ADDER_MAP = {
        "prim::GetAttr": lambda self, node: self.add_getattr(node),
        "prim::Constant": lambda self, node: self.add_constant_node(node),
        "prim::ListConstruct": lambda self, node: self.add_list_construct(node),
        "prim::TupleConstruct": lambda self, node: self.add_tuple_construct(node),
        "aten::unsqueeze": lambda self, node: self.add_unsqueeze(node),
        "aten::to": lambda self, node: self.add_to(node),
        "aten::detach": lambda self, node: self._identity(node),
        "aten::reshape": lambda self, node: self.add_reshape(node),
        "aten::flatten": lambda self, node: self.add_flatten(node),
        "aten::slice": lambda self, node: self.add_slice(node),
        "aten::size": lambda self, node: self.add_size(node),
        "aten::cat": lambda self, node: self.add_cat(node),
        "aten::mean": lambda self, node: self.add_mean(node),
        "aten::quantize_per_tensor": lambda self, node: self.add_quantize(node),
        "aten::dequantize": lambda self, node: self.add_dequantize(node),
        "aten::add": lambda self, node: self.add_add_sub_op(
            node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::sub": lambda self, node: self.add_add_sub_op(
            node, NNAPI_OperationCode.SUB, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::mul": lambda self, node: self.add_pointwise_simple_binary_broadcast_op(
            node, NNAPI_OperationCode.MUL, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::div": lambda self, node: self.add_pointwise_simple_binary_broadcast_op(
            node, NNAPI_OperationCode.DIV, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::relu": lambda self, node: self.add_pointwise_simple_unary_op(
            node, NNAPI_OperationCode.RELU
        ),
        "aten::sigmoid": lambda self, node: self.add_pointwise_simple_unary_op(
            node, NNAPI_OperationCode.LOGISTIC
        ),
        "aten::softmax": lambda self, node: self.add_softmax(node),
        "aten::hardtanh": lambda self, node: self.add_hardtanh(node),
        "aten::avg_pool2d": lambda self, node: self.add_avg_pool2d(node),
        "aten::max_pool2d": lambda self, node: self.add_pool2d_node(
            node, NNAPI_OperationCode.MAX_POOL_2D
        ),
        "aten::adaptive_avg_pool2d": lambda self, node: self.add_adaptive_avg_pool2d(
            node
        ),
        "aten::upsample_nearest2d": lambda self, node: self.add_upsample_nearest2d(
            node
        ),
        "aten::prelu": lambda self, node: self.add_prelu_op(node),
        "aten::addmm": lambda self, node: self.add_addmm(node),
        "aten::linear": lambda self, node: self.add_linear(node),
        "aten::_convolution": lambda self, node: self.add_conv_underscore(node),
        "aten::conv2d": lambda self, node: self.add_conv2d(node),
        "aten::log_softmax": lambda self, node: self.add_log_softmax(node),
        "quantized::linear": lambda self, node: self.add_qlinear(node),
        "quantized::conv2d": lambda self, node: self.add_qconv2d(
            node, NNAPI_FuseCode.FUSED_NONE
        ),
        "quantized::conv2d_relu": lambda self, node: self.add_qconv2d(
            node, NNAPI_FuseCode.FUSED_RELU
        ),
        "quantized::conv_transpose2d": lambda self, node: self.add_qconv2d(
            node, NNAPI_FuseCode.FUSED_NONE, transpose=True
        ),
        "quantized::add": lambda self, node: self.add_qadd(
            node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_NONE
        ),
        "quantized::add_relu": lambda self, node: self.add_qadd(
            node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_RELU
        ),
        "quantized::mul": lambda self, node: self.add_qadd(
            node, NNAPI_OperationCode.MUL, NNAPI_FuseCode.FUSED_NONE
        ),
    }

    def add_node(self, node):
        adder = self.ADDER_MAP.get(node.kind())
        if not adder:
            raise Exception(  # noqa: TRY002
                f"Unsupported node kind ({node.kind()!r}) in node {node!r}"
            )  # noqa: TRY002
        adder(self, node)

    def _identity(self, node):
        in_id, _in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        jitval = node.outputsAt(0)
        self.jitval_operand_map[jitval] = in_id

    def add_getattr(self, node):
        if node.inputsSize() != 1:
            raise AssertionError(
                f"expected node.inputsSize() == 1, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        obj_ctype, obj = self.get_constant_value(node.inputsAt(0))
        if not str(obj_ctype).startswith("__torch__."):
            raise AssertionError(
                f"expected obj_ctype to start with '__torch__.', got {obj_ctype}"
            )
        name = node.s("name")
        value = getattr(obj, name)
        output = node.outputsAt(0)
        ctype = output.type()
        self.add_constant_value(output, ctype, value)

    def add_constant_node(self, node):
        if node.inputsSize() != 0:
            raise AssertionError(
                f"expected node.inputsSize() == 0, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        output = node.outputsAt(0)
        ctype = output.type()
        value = output.toIValue()
        self.add_constant_value(output, ctype, value)

    def add_list_construct(self, node):
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        output = node.outputsAt(0)
        ctype = output.type()
        const_vals: Optional[list] = []
        tensors: Optional[list] = []
        for inp in node.inputs():
            if const_vals is not None and inp in self.constants:
                _, val = self.get_constant_value(inp)
                const_vals.append(val)
            else:
                const_vals = None
            if tensors is not None and inp.type().kind() == "TensorType":
                tensors.append(inp)
            else:
                tensors = None

        if const_vals is not None:
            # NOTE: Now that TorchScript supports list constants,
            # this code path might not be used anymore.
            self.add_constant_value(output, ctype, const_vals)
        if tensors is not None:
            self.add_tensor_sequence(output, tensors)
        if const_vals is None and tensors is None:
            raise Exception(  # noqa: TRY002
                f"Unable to handle ListConstruct node.  Neither all constants nor all tensors. {node!r}"
            )

    def add_tuple_construct(self, node):
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        output = node.outputsAt(0)
        values = list(node.inputs())
        self.add_tensor_sequence(output, values)

    def add_unsqueeze(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))

        _, dim = self.get_constant_value(node.inputsAt(1), "IntType")
        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS:
            raise AssertionError(
                f"expected dim_order PRESUMED_CONTIGUOUS, got {in_oper.dim_order}"
            )

        real_dim = dim if dim >= 0 else dim + len(in_oper.shape) + 1
        out_shape_list = list(in_oper.shape)
        out_shape_list.insert(real_dim, 1)
        out_shape = tuple(out_shape_list)
        out_oper = in_oper._replace(shape=out_shape)

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_scalar(dim)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.EXPAND_DIMS, inputs, outputs)

    def add_to(self, node):
        # Handle to("cpu") / to("gpu") case
        self._identity(node)

    def add_reshape(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))

        shape_ctype, shape = self.get_constant_value(node.inputsAt(1))
        if shape_ctype.kind() != "ListType":
            raise AssertionError(
                f"expected shape_ctype ListType, got {shape_ctype.kind()}"
            )
        if shape_ctype.getElementType().kind() != "IntType":
            raise AssertionError(
                f"expected shape element type IntType, got {shape_ctype.getElementType().kind()}"
            )
        is_trivial_reshape = len(shape) == 2 and shape[1] == -1

        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and not is_trivial_reshape:
            raise Exception(  # noqa: TRY002
                "Currently, reshape is only supported on NHWC tensors if the target size is [X, -1]."
            )

        # Bit of a hack here.  Use a real tensor to infer the output shape.
        out_shape = torch.zeros(1).expand(in_oper.shape).reshape(shape).shape
        out_oper = in_oper._replace(
            shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS
        )

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(shape)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)

    def add_flatten(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        _start_ctype, start_dim = self.get_constant_value(node.inputsAt(1), "IntType")
        _end_ctype, end_dim = self.get_constant_value(node.inputsAt(2), "IntType")

        # channels last with channels == 1 or (height & width both 1)
        is_trivial_flatten = len(in_oper.shape) == 4 and (
            in_oper.shape[1] == 1 or (in_oper.shape[2] == 1 and in_oper.shape[3] == 1)
        )
        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and not is_trivial_flatten:
            raise Exception(  # noqa: TRY002
                "Currently, flatten is not supported on NHWC tensors unless C=1 or H=W=1"
            )

        if start_dim < 0:
            start_dim += len(in_oper.shape)
        if end_dim < 0:
            end_dim += len(in_oper.shape)

        out_shape = (
            in_oper.shape[:start_dim]
            + (functools.reduce(operator.mul, in_oper.shape[start_dim : end_dim + 1]),)
            + in_oper.shape[end_dim + 1 :]
        )

        if any(dim == 0 for dim in in_oper.shape[start_dim : end_dim + 1]):
            raise Exception(  # noqa: TRY002
                "Flattening flexible dims is not supported yet"
            )  # noqa: TRY002
        non_flattened_dims = in_oper.shape[:start_dim] + in_oper.shape[end_dim + 1 :]
        if non_flattened_dims.count(0) > 1:
            raise Exception("Only 1 dim can be flexible")  # noqa: TRY002

        out_oper = in_oper._replace(
            shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS
        )
        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)

        for idx, dim in enumerate(out_shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, in_oper.shape.index(0))

        inputs_1 = tuple(dim if dim != 0 else -1 for dim in out_shape)
        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(inputs_1)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)

    def add_slice(self, node):
        if node.inputsSize() != 5:
            raise AssertionError(
                f"expected node.inputsSize() == 5, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        _, dim_value = self.get_constant_value(node.inputsAt(1))
        _, start_value = self.get_constant_value(node.inputsAt(2))
        _, stop_value = self.get_constant_value(node.inputsAt(3))
        _, step_value = self.get_constant_value(node.inputsAt(4))

        if start_value is None:
            start_value = 0
        if stop_value is None:
            stop_value = sys.maxsize

        if start_value < 0:
            start_value += in_oper.shape[dim_value]
        elif start_value == sys.maxsize:
            start_value = 0

        if start_value == 0 and stop_value == sys.maxsize:
            self._identity(node)
            return

        if in_oper.shape[dim_value] == 0:
            raise Exception("Unable to slice with flexible shape")  # noqa: TRY002

        if stop_value < 0:
            stop_value += in_oper.shape[dim_value]
        elif stop_value == sys.maxsize:
            stop_value = in_oper.shape[dim_value]

        if start_value >= stop_value:
            raise Exception(  # noqa: TRY002
                "Slice start value should be less than stop value"
            )  # noqa: TRY002

        out_len = (stop_value - start_value) // step_value
        out_shape = tuple(
            out_len if i == dim_value else dim for i, dim in enumerate(in_oper.shape)
        )
        out_id = self.add_tensor_operand(
            node.outputsAt(0), in_oper._replace(shape=out_shape)
        )

        # flex inputs
        end_mask = 0
        for idx, dim in enumerate(out_shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, idx)
                end_mask |= 1 << idx

        inputs = [None] * 7
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(
            [start_value if i == dim_value else 0 for i in range(len(in_oper.shape))]
        )
        inputs[2] = self.add_immediate_int_vector(
            [
                stop_value if i == dim_value else dim
                for i, dim in enumerate(in_oper.shape)
            ]
        )
        inputs[3] = self.add_immediate_int_vector(
            [step_value if i == dim_value else 1 for i in range(len(in_oper.shape))]
        )
        inputs[4] = self.add_immediate_int_scalar(0)  # begin mask
        inputs[5] = self.add_immediate_int_scalar(end_mask)
        inputs[6] = self.add_immediate_int_scalar(0)  # shrink axis mas

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.STRIDED_SLICE, inputs, outputs)

    def add_size(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        _, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        _, value = self.constants[node.inputsAt(1)]
        res = in_oper.shape[value]
        output = node.outputsAt(0)
        self.add_constant_value(output, output.type(), res)

    def add_cat(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        tensors = self.tensor_sequences[node.inputsAt(0)]
        _, dim = self.get_constant_value(node.inputsAt(1), "IntType")

        if len(tensors) <= 0:
            raise AssertionError(f"expected len(tensors) > 0, got {len(tensors)}")
        in_ids = []
        out_oper = None
        out_dim_size = 0
        for inp in tensors:
            in_id, in_oper = self.get_tensor_operand_by_jitval(inp)
            if out_oper is None:
                out_shape = change_element(in_oper.shape, dim, -1)
                out_oper = in_oper._replace(shape=out_shape)
            if in_oper.op_type != out_oper.op_type:
                raise AssertionError(
                    f"in_oper.op_type {in_oper.op_type} != out_oper.op_type {out_oper.op_type}"
                )
            if in_oper.dim_order != out_oper.dim_order:
                raise AssertionError(
                    f"in_oper.dim_order {in_oper.dim_order} != out_oper.dim_order {out_oper.dim_order}"
                )
            if change_element(in_oper.shape, dim, -1) != change_element(
                out_oper.shape, dim, -1
            ):
                raise AssertionError(
                    f"shape mismatch: {change_element(in_oper.shape, dim, -1)} != {change_element(out_oper.shape, dim, -1)}"
                )
            # TODO: Possibly check scale and zero point.
            in_ids.append(in_id)
            # TODO: Possibly support variable-sized inputs.
            out_dim_size += in_oper.shape[dim]

        if out_oper is None:
            raise AssertionError("out_oper must not be None")
        out_oper = out_oper._replace(
            shape=change_element(out_oper.shape, dim, out_dim_size)
        )

        if in_oper.dim_order == DimOrder.CHANNELS_LAST:  # type: ignore[possibly-undefined]
            if len(out_oper.shape) != 4:
                raise AssertionError(
                    f"expected len(out_oper.shape) == 4 for CHANNELS_LAST, got {len(out_oper.shape)}"
                )
            nnapi_dim = [0, 3, 1, 2][dim]
        else:
            nnapi_dim = dim

        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
        for idx, d in enumerate(out_oper.shape):
            if d == 0:
                if idx == dim:
                    shape = " + ".join(flex_name(ip_id, dim) for ip_id in in_ids)
                    self.compute_operand_shape(out_id, idx, shape)
                else:
                    self.forward_operand_shape(out_id, idx, in_ids[0], idx)

        inputs = in_ids + [self.add_immediate_int_scalar(nnapi_dim)]

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.CONCATENATION, inputs, outputs)

    def add_mean(self, node):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        dim_ctype, dim = self.get_constant_value(node.inputsAt(1))
        if dim_ctype.kind() != "ListType":
            raise AssertionError(f"expected dim_ctype ListType, got {dim_ctype.kind()}")
        if dim_ctype.getElementType().kind() != "IntType":
            raise AssertionError(
                f"expected dim element type IntType, got {dim_ctype.getElementType().kind()}"
            )
        _, keep_dim = self.get_constant_value(node.inputsAt(2), "BoolType")
        # Expect None for dtype
        self.get_constant_value(node.inputsAt(3), "NoneType")

        if in_oper.dim_order == DimOrder.CHANNELS_LAST:
            if len(in_oper.shape) != 4:
                raise AssertionError(
                    f"expected len(in_oper.shape) == 4 for CHANNELS_LAST, got {len(in_oper.shape)}"
                )
            nnapi_dim = [[0, 3, 1, 2][d] for d in dim]
        else:
            nnapi_dim = dim

        collapsed_dims = set()
        for d in dim:
            if d < 0:
                d += len(in_oper.shape)
            collapsed_dims.add(d)

        if in_oper.dim_order == DimOrder.CHANNELS_LAST and not keep_dim:
            if not collapsed_dims.issuperset({2, 3}):
                raise AssertionError(
                    f"expected collapsed_dims to include {{2, 3}}, got {collapsed_dims}"
                )
            out_dim_order = DimOrder.PRESUMED_CONTIGUOUS
        else:
            out_dim_order = in_oper.dim_order

        out_shape = []
        for i, s in enumerate(in_oper.shape):
            if i not in collapsed_dims:
                out_shape.append(s)
            elif keep_dim:
                out_shape.append(1)

        out_oper = in_oper._replace(shape=out_shape, dim_order=out_dim_order)

        inputs = [None] * 3
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(nnapi_dim)
        inputs[2] = self.add_immediate_int_scalar(keep_dim)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.MEAN, inputs, outputs)

    def add_quantize(self, node):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        if in_oper.dim_order != DimOrder.CHANNELS_LAST:
            raise Exception(  # noqa: TRY002
                "Most hardware backends prefer NHWC quantized tensors.  "
                "Try setting `t.nnapi_nhwc = True` on your tensor inputs.  "
            )
        _, scale = self.get_constant_value(node.inputsAt(1), "FloatType")
        _, zero_point = self.get_constant_value(node.inputsAt(2), "IntType")
        _, scalar_type = self.get_constant_value(node.inputsAt(3), "IntType")
        if scalar_type != TorchScalarTypes.QUINT8.value:
            raise Exception(  # noqa: TRY002
                "PyTorch NNAPI export only supports quantized tensors "
                "with the quint8 dtype."
            )
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
        if node.inputsSize() != 1:
            raise AssertionError(
                f"expected node.inputsSize() == 1, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
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
        if node.inputsSize() != 1:
            raise AssertionError(
                f"expected node.inputsSize() == 1, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        out_oper = in_oper
        if opcode == NNAPI_OperationCode.LOGISTIC:
            # NNAPI docs: For ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, the scale
            # must be 1.f / 256 and the zeroPoint must be 0.
            # https://fburl.com/h52stoog
            if in_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
                out_oper = in_oper._replace(zero_point=0, scale=1.0 / 256)

        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)

        for idx, dim in enumerate(in_oper.shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, idx)

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(opcode, inputs, outputs)

    def _do_add_binary(self, node, opcode, fuse_code, *, qparams=None):  # noqa: D401
        """Helper for pointwise binary broadcast ops with superfluous extra args."""
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        if node.inputsAt(0).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(0) TensorType, got {node.inputsAt(0).type().kind()}"
            )
        if node.inputsAt(1).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(1) TensorType, got {node.inputsAt(1).type().kind()}"
            )

        if self.has_operand_for_jitval(node.inputsAt(0)):
            in0_id, in0_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
            in1_id, in1_oper = self.get_tensor_operand_or_constant(
                node.inputsAt(1), in0_oper.dim_order
            )
        elif self.has_operand_for_jitval(node.inputsAt(1)):
            in1_id, in1_oper = self.get_tensor_operand_by_jitval(node.inputsAt(1))
            in0_id, in0_oper = self.get_tensor_operand_or_constant(
                node.inputsAt(0), in1_oper.dim_order
            )
        else:
            raise Exception(  # noqa: TRY002
                f"Can't do a NNAPI binary op: {opcode} on two constants"
            )  # noqa: TRY002

        if in0_oper.op_type != in1_oper.op_type:
            raise AssertionError(
                f"in0_oper.op_type {in0_oper.op_type} != in1_oper.op_type {in1_oper.op_type}"
            )
        in0_id, in0_oper, in1_id, in1_oper = self.transpose_for_broadcast(
            in0_id, in0_oper, in1_id, in1_oper
        )
        # NOTE: PyTorch and NNAPI have the same broadcast semantics.
        out_shape = broadcast_shapes(in0_oper.shape, in1_oper.shape)
        out_oper = in0_oper._replace(shape=out_shape)
        if qparams is not None:
            scale, zp = qparams
            out_oper = out_oper._replace(scale=scale, zero_point=zp)

        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
        for idx, (d0, d1) in enumerate(zip(in0_oper.shape, in1_oper.shape)):
            if d0 == 1 and d1 == 0:
                self.forward_operand_shape(out_id, idx, in1_id, idx)
            elif d0 == 0 and d1 == 1:
                self.forward_operand_shape(out_id, idx, in0_id, idx)
            elif d0 == 0 and d1 == 0:
                self.flexible_shape_computation_lines.append(
                    f"assert {flex_name(in0_id, idx)} == {flex_name(in1_id, idx)}"
                )
                self.forward_operand_shape(out_id, idx, in0_id, idx)

        inputs = [None] * 3
        inputs[0] = in0_id
        inputs[1] = in1_id
        inputs[2] = self.add_immediate_int_scalar(fuse_code)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(opcode, inputs, outputs)

    def add_pointwise_simple_binary_broadcast_op(self, node, opcode, fuse_code):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        self._do_add_binary(node, opcode, fuse_code)

    def add_add_sub_op(self, node, opcode, fuse_code):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )

        _, alpha = self.get_constant_value(node.inputsAt(2), "IntType")
        if alpha != 1:
            raise Exception(  # noqa: TRY002
                "NNAPI does not support add/sub with alpha."
            )  # noqa: TRY002

        self._do_add_binary(node, opcode, fuse_code)

    def add_qadd(self, node, opcode, fuse_code):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )

        _, scale = self.get_constant_value(node.inputsAt(2), "FloatType")
        _, zero_point = self.get_constant_value(node.inputsAt(3), "IntType")

        self._do_add_binary(node, opcode, fuse_code, qparams=(scale, zero_point))

    def add_softmax(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        _, softmax_dim = self.get_constant_value(node.inputsAt(1), "IntType")

        out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
        for dim, size in enumerate(in_oper.shape):
            if size == 0:
                self.forward_operand_shape(out_id, dim, in_id, dim)

        inputs = [None] * 3
        inputs[0] = in_id
        inputs[1] = self.add_immediate_float_scalar(
            1.0
        )  # positive scaling factor of exponent, beta
        inputs[2] = self.add_immediate_int_scalar(softmax_dim)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.SOFTMAX, inputs, outputs)

    def add_hardtanh(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        _, min_val = self.get_constant_value(node.inputsAt(1), "FloatType")
        _, max_val = self.get_constant_value(node.inputsAt(2), "FloatType")

        op_map = {
            (-1, 1): NNAPI_OperationCode.RELU1,
            (0, 6): NNAPI_OperationCode.RELU6,  # noqa: E201
        }

        opcode = op_map.get((min_val, max_val))
        if opcode is None:
            raise Exception(  # noqa: TRY002
                "NNAPI only supports hardtanh with args (-1, 1) or (0, 6)."
            )  # noqa: TRY002

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), in_oper)

        self.add_operation(opcode, inputs, outputs)

    def add_prelu_op(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        if node.inputsAt(0).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(0) TensorType, got {node.inputsAt(0).type().kind()}"
            )
        if node.inputsAt(1).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(1) TensorType, got {node.inputsAt(1).type().kind()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        w_id, w_oper = self.get_tensor_operand_for_weight(node.inputsAt(1))
        if len(w_oper.shape) != 1:
            raise AssertionError(
                f"expected len(w_oper.shape) == 1, got {len(w_oper.shape)}"
            )
        if w_oper.shape[0] <= 0:
            raise AssertionError(f"expected w_oper.shape[0] > 0, got {w_oper.shape[0]}")
        if w_oper.shape[0] > 1:
            if in_oper.use_nchw():
                # TODO: Support this by adding trailing 1 dims.
                raise Exception(  # noqa: TRY002
                    "Per-channel PReLU only supports channels_last right now."
                )

        out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
        for dim, size in enumerate(in_oper.shape):
            if size > 0:
                pass
            elif dim <= 1:
                raise Exception(  # noqa: TRY002
                    "PReLU requires fixed size for dim 0 and dim 1."
                )  # noqa: TRY002
            else:
                self.forward_operand_shape(out_id, dim, in_id, dim)

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = w_id

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.PRELU, inputs, outputs)

    def add_pool2d_node(self, node, opcode):
        if node.inputsSize() != 6:
            raise AssertionError(
                f"expected node.inputsSize() == 6, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        image, kernel, stride, padding, dilation, _ceil_mode = node.inputs()

        stride = stride or kernel

        # TODO: Validate ceil_mode semantics.

        args = self.get_conv_pool_args_2d_from_jit(
            self.get_size_arg(kernel), stride, padding, dilation
        )
        if args.dilation_h != 1 or args.dilation_w != 1:
            raise Exception("NNAPI does not support dilated pooling.")  # noqa: TRY002

        image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(image)
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        out_shape = get_conv_pool_shape(
            image_oper.shape, args, image_oper.shape[1], False
        )
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
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        self.add_operation(opcode, inputs, outputs)

    def add_avg_pool2d(self, node):
        if node.inputsSize() != 7:
            raise AssertionError(
                f"expected node.inputsSize() == 7, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        (
            image,
            kernel,
            stride,
            padding,
            _ceil_mode,
            count_include_pad,
            divisor_override,
        ) = node.inputs()

        _, count_include_pad_value = self.get_constant_value(count_include_pad)
        _, divisor_override_value = self.get_constant_value(divisor_override)
        if not count_include_pad_value or divisor_override_value:
            raise Exception(  # noqa: TRY002
                "NNAPI doesn't support count_include_pad=False or divisor_override"
            )

        args = self.get_conv_pool_args_2d_from_jit(
            self.get_size_arg(kernel), stride, padding
        )

        image_id, image_oper = self.get_tensor_operand_by_jitval(image)
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        out_shape = get_conv_pool_shape(
            image_oper.shape, args, image_oper.shape[1], False
        )
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
        out_id = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )
        self._handle_conv_pool_flexible_input(out_id, image, args, False)
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)

    def add_adaptive_avg_pool2d(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(
            node.inputsAt(0)
        )
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        size_ctype, size_arg = self.get_constant_value(node.inputsAt(1))
        if size_ctype.kind() != "ListType":
            raise AssertionError(
                f"expected size_ctype ListType, got {size_ctype.kind()}"
            )
        if size_ctype.getElementType().kind() != "IntType":
            raise AssertionError(
                f"expected size element type IntType, got {size_ctype.getElementType().kind()}"
            )
        if size_arg != [1, 1]:
            raise Exception(  # noqa: TRY002
                "NNAPI only supports adaptive_avg_pool2d with output size (1, 1)."
            )

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
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)

    def add_upsample_nearest2d(self, node):
        if not (node.inputsSize() == 3 or node.inputsSize() == 4):
            raise AssertionError(
                f"expected node.inputsSize() == 3 or 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        if node.inputsSize() == 3:
            image, size_jit, scale_jit = node.inputs()
        else:
            image, size_jit, scale_h_jit, scale_w_jit = node.inputs()
        size_ctype, size_arg = self.get_constant_value(size_jit)

        if node.inputsSize() == 3:
            scale_ctype, scale_arg = self.get_constant_value(scale_jit)  # type: ignore[possibly-undefined]
        else:
            scale_h_ctype, scale_h_arg = self.get_constant_value(scale_h_jit)  # type: ignore[possibly-undefined]
            scale_w_ctype, _scale_w_arg = self.get_constant_value(scale_w_jit)  # type: ignore[possibly-undefined]

            # The only way for the 4-argument overload of upsample_nearest2d to
            # have been added to the graph without error is if the scale_h and
            # scale_w arguments are None
            if scale_h_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected scale_h_ctype NoneType, got {scale_h_ctype.kind()}"
                )
            if scale_w_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected scale_w_ctype NoneType, got {scale_w_ctype.kind()}"
                )

            scale_ctype = scale_h_ctype
            scale_arg = scale_h_arg

        image_id, image_oper = self.get_tensor_operand_by_jitval(image)
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        if size_ctype.kind() != "NoneType" and scale_ctype.kind() != "NoneType":
            raise Exception("Size and scale cannot both be non-None.")  # noqa: TRY002
        elif size_ctype.kind() != "NoneType":
            if size_ctype.kind() != "ListType":
                raise AssertionError(
                    f"expected size_ctype ListType, got {size_ctype.kind()}"
                )
            if size_ctype.getElementType().kind() != "IntType":
                raise AssertionError(
                    f"expected size element type IntType, got {size_ctype.getElementType().kind()}"
                )
            if scale_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected scale_ctype NoneType, got {scale_ctype.kind()}"
                )
            if scale_arg is not None:
                raise AssertionError(f"expected scale_arg None, got {scale_arg}")
            if not isinstance(size_arg, list):
                raise AssertionError(
                    f"expected size_arg to be list, got {type(size_arg)}"
                )
            if not size_arg:
                raise AssertionError("expected size_arg to be non-empty")
            if not all(isinstance(val, int) for val in size_arg):
                raise AssertionError("expected all size_arg values to be int")
            if len(size_arg) == 1:
                size_arg = size_arg * 2
            if len(size_arg) != 2:
                raise AssertionError(
                    f"expected len(size_arg) == 2, got {len(size_arg)}"
                )
            out_h = size_arg[0]
            out_w = size_arg[1]
            arg_h = self.add_immediate_int_scalar(out_h)
            arg_w = self.add_immediate_int_scalar(out_w)
        elif scale_ctype.kind() != "NoneType":
            if scale_ctype.kind() != "ListType":
                raise AssertionError(
                    f"expected scale_ctype ListType, got {scale_ctype.kind()}"
                )
            if scale_ctype.getElementType().kind() != "FloatType":
                raise AssertionError(
                    f"expected scale element type FloatType, got {scale_ctype.getElementType().kind()}"
                )
            if size_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected size_ctype NoneType, got {size_ctype.kind()}"
                )
            if size_arg is not None:
                raise AssertionError(f"expected size_arg None, got {size_arg}")
            if not isinstance(scale_arg, list):
                raise AssertionError(
                    f"expected scale_arg to be list, got {type(scale_arg)}"
                )
            if not scale_arg:
                raise AssertionError("expected scale_arg to be non-empty")
            if not all(isinstance(val, float) for val in scale_arg):
                raise AssertionError("expected all scale_arg values to be float")
            if len(scale_arg) == 1:
                scale_arg = scale_arg * 2
            if len(scale_arg) != 2:
                raise AssertionError(
                    f"expected len(scale_arg) == 2, got {len(scale_arg)}"
                )
            out_h = int(scale_arg[0] * image_oper.shape[2])
            out_w = int(scale_arg[1] * image_oper.shape[3])
            arg_h = self.add_immediate_float_scalar(scale_arg[0])
            arg_w = self.add_immediate_float_scalar(scale_arg[1])
        else:
            raise Exception("Size and scale cannot both be None.")  # noqa: TRY002

        out_shape = (image_oper.shape[0], image_oper.shape[1], out_h, out_w)
        use_nchw = image_oper.use_nchw()
        out_id = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        if image_oper.shape[0] == 0 or image_oper.shape[1] == 0:
            raise Exception("Flexible batch or channels not supported")  # noqa: TRY002

        # Handle variable input size
        for dim in (2, 3):  # h, w indices
            if image_oper.shape[dim] == 0:
                if size_ctype.kind() != "NoneType":
                    # pyrefly: ignore [unsupported-operation]
                    self.compute_operand_shape(out_id, dim, size_arg[dim - 2])
                elif scale_ctype.kind() != "NoneType":
                    self.compute_operand_shape(
                        out_id,
                        dim,
                        # pyrefly: ignore [unsupported-operation]
                        f"int({scale_arg[dim - 2]} * {flex_name(image_id, dim)})",
                    )
                else:
                    raise Exception(  # noqa: TRY002
                        "Size and scale cannot both be None."
                    )  # noqa: TRY002

        inputs = [None] * 4
        inputs[0] = image_id
        inputs[1] = arg_w
        inputs[2] = arg_h
        inputs[3] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.RESIZE_NEAREST_NEIGHBOR, inputs, outputs)

    def add_addmm(self, node):
        if node.inputsSize() != 5:
            raise AssertionError(
                f"expected node.inputsSize() == 5, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        jit_bias, jit_input, jit_weight, jit_beta, jit_alpha = node.inputs()

        for jitval in (jit_beta, jit_alpha):
            scale_ctype, scale_value = self.get_constant_value(jitval)
            if scale_ctype.kind() not in ("IntType", "FloatType"):
                raise AssertionError(
                    f"expected scale_ctype IntType or FloatType, got {scale_ctype.kind()}"
                )
            if scale_value != 1:
                raise Exception(  # noqa: TRY002
                    "NNAPI Fully-Connected does not support alpha and beta."
                )

        self.add_addmm_or_linear(node, True, jit_input, jit_weight, jit_bias)

    def add_linear(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        jit_input, jit_weight, jit_bias = node.inputs()

        self.add_addmm_or_linear(node, False, jit_input, jit_weight, jit_bias)

    def add_addmm_or_linear(
        self, node, transpose_weight, jit_input, jit_weight, jit_bias
    ):
        input_id, input_oper = self.get_tensor_operand_by_jitval(jit_input)
        bias_id, bias_oper = self.get_tensor_operand_for_weight(jit_bias)

        if len(input_oper.shape) != 2:
            raise AssertionError(
                f"expected len(input_oper.shape) == 2, got {len(input_oper.shape)}"
            )
        if len(bias_oper.shape) != 1:
            raise AssertionError(
                f"expected len(bias_oper.shape) == 1, got {len(bias_oper.shape)}"
            )

        # TODO: Transform at load time to share weights with CPU model.
        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        if len(weight_tensor.shape) != 2:
            raise AssertionError(
                f"expected len(weight_tensor.shape) == 2, got {len(weight_tensor.shape)}"
            )
        if transpose_weight:
            nnapi_weight_tensor = weight_tensor.t().contiguous()
        else:
            nnapi_weight_tensor = weight_tensor.contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        out_shape = (input_oper.shape[0], weight_oper.shape[0])
        out_id = self.add_tensor_operand(
            node.outputsAt(0), input_oper._replace(shape=out_shape)
        )

        if input_oper.shape[0] == 0:
            self.forward_operand_shape(out_id, 0, input_id, 0)

        inputs = [None] * 4
        inputs[0] = input_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)

    def add_qlinear(self, node):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        (
            jit_input,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        input_id, input_oper = self.get_tensor_operand_by_jitval_fixed_size(jit_input)
        # TODO: Support automatic reshape
        if len(input_oper.shape) != 2:
            raise AssertionError(
                f"expected len(input_oper.shape) == 2, got {len(input_oper.shape)}"
            )

        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        if weight_ctype.name() != "LinearPackedParamsBase":
            raise AssertionError(
                f"expected weight_ctype LinearPackedParamsBase, got {weight_ctype.name()}"
            )
        raw_weight, raw_bias = packed_weight.__getstate__()[0]
        if raw_bias is None:
            raise AssertionError("raw_bias must not be None")

        if len(raw_weight.shape) != 2:
            raise AssertionError(
                f"expected len(raw_weight.shape) == 2, got {len(raw_weight.shape)}"
            )
        if len(raw_bias.shape) != 1:
            raise AssertionError(
                f"expected len(raw_bias.shape) == 1, got {len(raw_bias.shape)}"
            )
        if raw_bias.shape[0] != raw_weight.shape[0]:
            raise AssertionError(
                f"raw_bias.shape[0] {raw_bias.shape[0]} != raw_weight.shape[0] {raw_weight.shape[0]}"
            )
        if raw_weight.shape[1] != input_oper.shape[1]:
            raise AssertionError(
                f"raw_weight.shape[1] {raw_weight.shape[1]} != input_oper.shape[1] {input_oper.shape[1]}"
            )

        if raw_weight.qscheme() != torch.per_tensor_affine:
            raise AssertionError(
                f"expected raw_weight.qscheme() per_tensor_affine, got {raw_weight.qscheme()}"
            )
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            if raw_weight.dtype != torch.qint8:
                raise AssertionError(
                    f"expected raw_weight.dtype qint8, got {raw_weight.dtype}"
                )
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128,
            )
        weight_scale = unsigned_weight.q_scale()
        bias_scale = input_oper.scale * weight_scale
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        multiplier = input_oper.scale * weight_scale / out_scale
        if multiplier <= 0:
            raise AssertionError(f"expected multiplier > 0, got {multiplier}")
        if multiplier >= 1:
            raise Exception(  # noqa: TRY002
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  "
            )

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

    def get_optional_bias(self, jit_bias, weight_tensor, transpose=False):
        ctype, _value = self.get_constant_value(jit_bias)
        if ctype.kind() == "NoneType":
            bias_idx = 1 if transpose else 0
            nnapi_bias_tensor = torch.zeros(
                weight_tensor.size()[bias_idx], dtype=weight_tensor.dtype
            )
            bias_id = self.add_tensor_operand_for_weight(nnapi_bias_tensor)
            bias_oper = self.operands[bias_id]
            return bias_id, bias_oper
        else:
            return self.get_tensor_operand_for_weight(jit_bias)

    def add_conv2d(self, node):
        if node.inputsSize() != 7:
            raise AssertionError(
                f"expected node.inputsSize() == 7, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

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
        bias_id, _bias_oper = self.get_optional_bias(jit_bias, weight_tensor)
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups
        )

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
        if node.inputsSize() != 13:
            raise AssertionError(
                f"expected node.inputsSize() == 13, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

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

        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        _, transpose = self.get_constant_value(jit_transpose)
        bias_id, _bias_oper = self.get_optional_bias(jit_bias, weight_tensor, transpose)
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups
        )

        return self.add_conv2d_common(
            node.outputsAt(0),
            0.0,
            0,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            transpose,
            NNAPI_FuseCode.FUSED_NONE,
        )

    def add_log_softmax(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        jit_input, jit_dim, _jit_half_to_float = node.inputs()
        input_id, input_oper = self.get_tensor_operand_by_jitval_fixed_size(jit_input)
        _, dim = self.get_constant_value(jit_dim, "IntType")

        out_shape = input_oper.shape

        inputs = [None] * 3
        inputs[0] = input_id
        # specifying 1 as the scaling factor for the exponent, beta
        inputs[1] = self.add_immediate_float_scalar(1)
        inputs[2] = self.add_immediate_int_scalar(dim)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), input_oper._replace(shape=out_shape)
        )
        self.add_operation(NNAPI_OperationCode.LOG_SOFTMAX, inputs, outputs)

    def add_qconv2d(self, node, fuse_code, transpose=False):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        (
            jit_image,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        if weight_ctype.name() != "Conv2dPackedParamsBase":
            raise AssertionError(
                f"expected weight_ctype Conv2dPackedParamsBase, got {weight_ctype.name()}"
            )
        (
            pack_version,
            tensors,
            opt_tensors,
        ) = packed_weight.__getstate__()[0]
        if pack_version != "2":
            raise AssertionError(f"expected pack_version '2', got {pack_version!r}")
        packed_config, raw_weight = tensors
        (raw_bias,) = opt_tensors
        if raw_bias is None:
            raise AssertionError("raw_bias must not be None")
        args = self.get_conv_pool_args_2d_from_pack(
            raw_weight.shape[2:4], packed_config
        )

        if raw_weight.qscheme() != torch.per_tensor_affine:
            raise AssertionError(
                f"expected raw_weight.qscheme() per_tensor_affine, got {raw_weight.qscheme()}"
            )
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            if raw_weight.dtype != torch.qint8:
                raise AssertionError(
                    f"expected raw_weight.dtype qint8, got {raw_weight.dtype}"
                )
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128,
            )
        weight_scale = unsigned_weight.q_scale()
        _, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        bias_scale = image_oper.scale * weight_scale
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        multiplier = image_oper.scale * weight_scale / out_scale
        if multiplier <= 0:
            raise AssertionError(f"expected multiplier > 0, got {multiplier}")
        if multiplier >= 1:
            raise Exception(  # noqa: TRY002
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  "
            )

        return self.add_conv2d_common(
            node.outputsAt(0),
            out_scale,
            out_zero_point,
            jit_image,
            unsigned_weight,
            bias_id,
            args,
            transpose,
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
        fuse_code,
    ):
        image_id, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        in_c = image_oper.shape[1]

        if args.group == 1:
            # Full convolution
            depthwise = False
            if transpose:
                weight_permutation = (1, 2, 3, 0)
            else:
                weight_permutation = (0, 2, 3, 1)
        elif args.group == in_c:
            # Depthwise convolution
            depthwise = True
            weight_permutation = (1, 2, 3, 0)
        else:
            raise Exception("Group convolution not supported yet.")  # noqa: TRY002

        # TODO: Transform at load time to share weights with CPU model.
        nnapi_weight_tensor = weight_tensor.permute(*weight_permutation).contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        bias_oper = self.operands[bias_id]

        if image_oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32:
            if weight_oper.op_type != NNAPI_OperandCode.TENSOR_FLOAT32:
                raise AssertionError(
                    f"expected weight_oper TENSOR_FLOAT32, got {weight_oper.op_type}"
                )
            if bias_oper.op_type != NNAPI_OperandCode.TENSOR_FLOAT32:
                raise AssertionError(
                    f"expected bias_oper TENSOR_FLOAT32, got {bias_oper.op_type}"
                )
        elif image_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
            if weight_oper.op_type != NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
                raise AssertionError(
                    f"expected weight_oper TENSOR_QUANT8_ASYMM, got {weight_oper.op_type}"
                )
            if bias_oper.op_type != NNAPI_OperandCode.TENSOR_INT32:
                raise AssertionError(
                    f"expected bias_oper TENSOR_INT32, got {bias_oper.op_type}"
                )
            if not approx_equal(image_oper.scale * weight_oper.scale, bias_oper.scale):
                raise AssertionError(
                    f"scale mismatch: image*weight scale {image_oper.scale * weight_oper.scale} != bias scale {bias_oper.scale}"
                )
            if bias_oper.zero_point != 0:
                raise AssertionError(
                    f"expected bias_oper.zero_point == 0, got {bias_oper.zero_point}"
                )
        else:
            raise Exception(  # noqa: TRY002
                f"Unsupported input type for conv2d: {image_oper.op_type}"
            )  # noqa: TRY002

        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )
        if len(weight_oper.shape) != 4:
            raise AssertionError(
                f"expected len(weight_oper.shape) == 4, got {len(weight_oper.shape)}"
            )
        if len(bias_oper.shape) != 1:
            raise AssertionError(
                f"expected len(bias_oper.shape) == 1, got {len(bias_oper.shape)}"
            )

        if depthwise:
            # Depthwise convolution
            one, _kern_h, _kern_w, out_c = weight_oper.shape
            if one != 1:
                raise AssertionError(f"expected weight_oper.shape[0] == 1, got {one}")
            if out_c % in_c != 0:
                raise AssertionError(f"out_c {out_c} must be divisible by in_c {in_c}")
            channel_multiplier = out_c // in_c
            if channel_multiplier != 1:
                raise AssertionError(
                    f"channel_multiplier must be 1, got {channel_multiplier}"
                )
            if out_c != in_c:
                raise AssertionError(f"out_c {out_c} != in_c {in_c}")
        else:
            # Full convolution
            out_c, _kern_h, _kern_w, kern_d = weight_oper.shape
            if kern_d != in_c:
                raise AssertionError(f"kern_d {kern_d} != in_c {in_c}")

        if out_c != bias_oper.shape[0]:
            raise AssertionError(
                f"out_c {out_c} != bias_oper.shape[0] {bias_oper.shape[0]}"
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
        out_shape = get_conv_pool_shape(image_oper.shape, args, out_c, transpose)
        out_oper = image_oper._replace(
            shape=out_shape,
            scale=out_scale,
            zero_point=out_zero_point,
        )
        out_id = self.add_tensor_operand(jit_out, out_oper)
        self._handle_conv_pool_flexible_input(out_id, jit_image, args, transpose)

        outputs[0] = out_id
        self.add_operation(opcode, inputs, outputs)

    def _handle_conv_pool_flexible_input(self, out_id, jit_image, args, transpose):
        image_id, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        batch, in_ch, in_h, in_w = image_oper.shape

        if batch == 0:
            self.forward_operand_shape(out_id, 0, image_id, 0)
        if in_ch == 0:
            raise Exception("Input channels can't be flexible")  # noqa: TRY002
        # H & W
        if transpose:
            if in_h == 0:
                self.compute_operand_shape(
                    out_id,
                    2,
                    f"({flex_name(image_id, 2)} - 1) * {args.stride_h} + {args.kernel_h} - {args.pad_t} - {args.pad_b}",
                )
            if in_w == 0:
                self.compute_operand_shape(
                    out_id,
                    3,
                    f"({flex_name(image_id, 3)} - 1) * {args.stride_w} + {args.kernel_w} - {args.pad_l} - {args.pad_r}",
                )
        else:
            if in_h == 0:
                self.compute_operand_shape(
                    out_id,
                    2,
                    f"({flex_name(image_id, 2)} - {args.kernel_h} + {args.pad_t} + {args.pad_b}) // {args.stride_h} + 1",
                )
            if in_w == 0:
                self.compute_operand_shape(
                    out_id,
                    3,
                    f"({flex_name(image_id, 3)} - {args.kernel_w} + {args.pad_l} + {args.pad_r}) // {args.stride_w} + 1",
                )


def serialize_model(
    module, inputs, *, config=None, return_shapes=None, use_int16_for_qint16=False
):
    """Convert to NNAPI and serialize torchscript module.

    Parameters:
        module: Torchscript module to convert
        inputs: Tensors used to specify input details for NNAPI
        config (optional): Optional config to attach to module
        return_shapes (optional): Specify shape of outputs if
            your module uses runtime flexible shapes to set output
            buffer size for NNAPI
        use_int16_for_qint16 (optional): Use Pytorch int16 to represent NNAPI qint16 values
    """
    return _NnapiSerializer(config, use_int16_for_qint16).serialize_model(
        module, inputs, return_shapes
    )
