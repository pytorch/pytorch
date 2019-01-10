#include "caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.h"
#include "caffe2/core/registry.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<CPUContext>);
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsSum, but operating on
8-bit rowwise quantized matrices with fused storage (where each row
stores quantized values, and then 4-byte scale and 4-byte bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused8BitRowwiseQuantized")
    .Input(
        1,
        "INDICES",
        "Integer vector containing indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        2,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsSumFused8BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<CPUContext, /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwise)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsWeightedSum,
but operating on 8-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 4-byte scale and 4-byte bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused8BitRowwiseQuantized")
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
        "WEIGHTS",
        "Vector of weights to scale rows of DATA with before reduction")
    .Output(0, "output", "output");

NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>);
OPERATOR_SCHEMA(SparseLengthsMeanFused8BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsMean, but
operating on 8-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 4-byte scale and 4-byte bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused8BitRowwiseQuantized")
    .Input(
        1,
        "INDICES",
        "Integer vector containing indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        2,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsMeanFused8BitRowwise);
} // namespace caffe2
