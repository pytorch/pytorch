#include "caffe2/operators/lengths_reducer_rowwise_8bit_ops.h"
#include "caffe2/core/registry.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Rowwise8BitQuantizedToFloat,
    Rowwise8BitQuantizedToFloatOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    FloatToRowwiseQuantized8Bits,
    FloatToRowwiseQuantized8BitsOp<CPUContext>);

REGISTER_CPU_OPERATOR(
    SparseLengthsSum8BitsRowwise,
    SparseLengths8BitsRowwiseOp<CPUContext>);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSum8BitsRowwise,
    SparseLengths8BitsRowwiseOp<CPUContext, 1>);

REGISTER_CPU_OPERATOR(
    SparseLengthsMean8BitsRowwise,
    SparseLengths8BitsRowwiseOp<CPUContext, 0, 1>);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedMean8BitsRowwise,
    SparseLengths8BitsRowwiseOp<CPUContext, 1, 1>);

OPERATOR_SCHEMA(SparseLengthsSum8BitsRowwise)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Variation of SparseLengthsSum operator, where DATA is
stored using 8bits. DATA was quantized with 8Bit row-wise
quantization (see doc to FloatToRowwiseQuantized8Bits operator). To
restore DATA from 8Bit, we use additional input that stores scales
and biases.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToRowwiseQuantized8Bits")
    .Input(
        1,
        "INDICES",
        "Integer vector containing indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        2,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA")
    .Input(
        3,
        "scale_bias",
        "Matrix of floats, each row r_i of which stores a pair "
        "s_i, b_i -- scale and bias for i-th row")

    .Output(0, "output", "output");

OPERATOR_SCHEMA(SparseLengthsWeightedSum8BitsRowwise)
    .NumInputs(5)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Variation of SparseLengthsWeightedSum operator, where
DATA is stored using 8bits. DATA was quantized with 8Bit row-wise
quantization (see doc to FloatToRowwiseQuantized8Bits operator). To
restore DATA from 8Bit, we use additional input that stores scales
and biases.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToRowwiseQuantized8Bits")
    .Input(
        1,
        "SCALARS",
        "Scalar multipliers for the input slices. Must "
        "be a vector with the length matching the length of INDICES")
    .Input(
        2,
        "INDICES",
        "Integer vector containing indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        3,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA")
    .Input(
        4,
        "scale_bias",
        "Matrix of floats, each row r_i of which stores a pair "
        "s_i, b_i -- scale and bias for i-th row")
    .Output(0, "output", "output");

OPERATOR_SCHEMA(SparseLengthsMean8BitsRowwise)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Variation of SparseLengthsMean operator, where DATA is
stored using 8bits. DATA was quantized with 8Bit row-wise
quantization (see doc to FloatToRowwiseQuantized8Bits operator). To
restore DATA from 8Bit, we use additional input that stores scales
and biases.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToRowwiseQuantized8Bits")
    .Input(
        1,
        "INDICES",
        "Integer vector containing indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        2,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA")
    .Input(
        3,
        "scale_bias",
        "Matrix of floats, each row r_i of which stores a pair "
        "s_i, b_i -- scale and bias for i-th row")

    .Output(0, "output", "output");

OPERATOR_SCHEMA(SparseLengthsWeightedMean8BitsRowwise)
    .NumInputs(5)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Variation of SparseLengthsWeightedMean operator, where
DATA is stored using 8bits. DATA was quantized with 8Bit row-wise
quantization (see doc to FloatToRowwiseQuantized8Bits operator). To
restore DATA from 8Bit, we use additional input that stores scales
and biases.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToRowwiseQuantized8Bits")
    .Input(
        1,
        "SCALARS",
        "Scalar multipliers for the input slices. Must "
        "be a vector with the length matching the length of INDICES")
    .Input(
        2,
        "INDICES",
        "Integer vector containing indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        3,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA")
    .Input(
        4,
        "scale_bias",
        "Matrix of floats, each row r_i of which stores a pair "
        "s_i, b_i -- scale and bias for i-th row")
    .Output(0, "output", "output");

OPERATOR_SCHEMA(FloatToRowwiseQuantized8Bits)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
This operator applies 8Bit row-wise quantization to
input tensor and returns quantized tensor. Row wise quantization of
input tensor is the following process. We take tensor of size
(m_1, m_2,...,m_n), n >= 2, reshape it into matrix of size
(m_1, m_2 x... x m_n) and apply row-wise quantization. After this,
we compute scale_i= (min_i - max_i) / 255 and  bias_i = min_i for
i-th row r_i of reshaped matrix, where min_i and max_i --  minimum
and maximum elements of i-th row, and quantize each element r_{ij} as
0 <= round(r_ij - bias_i) / scale_i) < 256. Instead of input tensor
we obtain uint8 tensor and auxiliary information as scale and bias to
restore input tensor (with losses).
)DOC")
    .Input(0, "input", "input")
    .Output(0, "quantized_input", "quantized_input")
    .Output(
        1,
        "scale_bias",
        "Matrix of floats, each row r_i of which stores a pair "
        "s_i, b_i");

OPERATOR_SCHEMA(Rowwise8BitQuantizedToFloat)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given uint8 tensor, quantized using 8bit row-wise
quantization, and auxiliary scales and biases, this operator
restores float tensor in the following way. We take input 8bits tensor
of size  (m_1, m_2, ..., m_n), n >= 2, reshape it  into matrix of size
(m_1, m_2 x... x m_n). We compute element r_{ij} of output matrix as
r_{ij} * s_i + b_i and after this we reshape this output matrix into
output tensor of size (m_1, m_2, ..., m_n).
)DOC")
    .Input(0, "quantized_input", "quantized_input")
    .Input(
        1,
        "scale_bias",
        "Matrix of floats, each row r_i of which stores a pair "
        "s_i, b_i -- scale and bias for i-th row")
    .Output(1, "output", "output");

NO_GRADIENT(Rowwise8BitQuantizedToFloat);
NO_GRADIENT(FloatToRowwiseQuantized8Bits);
NO_GRADIENT(SparseLengthsSum8BitsRowwise);
NO_GRADIENT(SparseLengthsWeightedSum8BitsRowwise);
NO_GRADIENT(SparseLengthsMean8BitsRowwise);
NO_GRADIENT(SparseLengthsWeightedMean8BitsRowwise);
}
