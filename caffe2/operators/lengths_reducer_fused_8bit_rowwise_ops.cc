#include "caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.h"
#include "c10/util/Registry.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseOp<CPUContext>::DATA,
        SparseLengthsFused8BitRowwiseOp<CPUContext>::INDICES,
        SparseLengthsFused8BitRowwiseOp<CPUContext>::LENGTHS)
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
    .Output(0, "output", "output")
    .InheritOnnxSchema();
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(SparseLengthsSumFused8BitRowwise);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<CPUContext, /*with_weights=*/true>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwise)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseOp<CPUContext, true>::DATA,
        SparseLengthsFused8BitRowwiseOp<CPUContext, true>::INDICES,
        SparseLengthsFused8BitRowwiseOp<CPUContext, true>::LENGTHS,
        SparseLengthsFused8BitRowwiseOp<CPUContext, true>::WEIGHTS)
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
        "WEIGHTS",
        "Vector of weights to scale rows of DATA with before reduction")
    .Input(
        2,
        "INDICES",
        "Integer vector containing indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        3,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA")
    .Output(0, "output", "output");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwise);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SparseLengthsMeanFused8BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseOp<CPUContext, false, true>::DATA,
        SparseLengthsFused8BitRowwiseOp<CPUContext, false, true>::INDICES,
        SparseLengthsFused8BitRowwiseOp<CPUContext, false, true>::LENGTHS)
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(SparseLengthsMeanFused8BitRowwise);
} // namespace caffe2
