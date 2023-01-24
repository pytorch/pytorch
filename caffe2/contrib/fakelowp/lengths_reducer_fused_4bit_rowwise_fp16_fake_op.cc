#include "lengths_reducer_fused_4bit_rowwise_fp16_fake_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused4BitRowwiseFakeFP16NNPI,
    SparseLengthsFused4BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false>);
OPERATOR_SCHEMA(SparseLengthsSumFused4BitRowwiseFakeFP16NNPI)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, false>::DATA,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, false>::INDICES,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, false>::LENGTHS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsSum, but operating on
4-bit rowwise quantized matrices with fused storage (where each row
stores quantized values, and then 2-byte scale and 2-byte bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused4BitRowwiseQuantized")
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
NO_GRADIENT(SparseLengthsSumFused4BitRowwiseFakeFP16NNPI);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused4BitRowwiseFakeFP16EmbeddingOnly,
    SparseLengthsFused4BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false,
        /*use_fp16_for_embedding_only=*/true>);
OPERATOR_SCHEMA(SparseLengthsSumFused4BitRowwiseFakeFP16EmbeddingOnly)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, false, true>::DATA,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, false, true>::
            INDICES,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, false, true>::
            LENGTHS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsSum, but operating on
4-bit rowwise quantized matrices with fused storage (where each row
stores quantized values, and then 2-byte scale and 2-byte bias).
Convert only embedding entries using fake fp16.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused4BitRowwiseQuantized")
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
NO_GRADIENT(SparseLengthsSumFused4BitRowwiseFakeFP16EmbeddingOnly);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused4BitRowwiseFakeFP16NNPI,
    SparseLengthsFused4BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused4BitRowwiseFakeFP16NNPI)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true>::DATA,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true>::INDICES,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true>::LENGTHS,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true>::WEIGHTS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsWeightedSum,
but operating on 4-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 2-byte scale and 2-byte bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused4BitRowwiseQuantized")
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

NO_GRADIENT(SparseLengthsWeightedSumFused4BitRowwiseFakeFP16NNPI);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused4BitRowwiseFakeFP16EmbeddingOnly,
    SparseLengthsFused4BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/true,
        /*use_fp16_for_embedding_only=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused4BitRowwiseFakeFP16EmbeddingOnly)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true, true>::DATA,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true, true>::
            INDICES,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true, true>::
            LENGTHS,
        SparseLengthsFused4BitRowwiseFakeFP16Op<CPUContext, true, true>::
            WEIGHTS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsWeightedSum,
but operating on 4-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 2-byte scale and 2-byte bias).
Convert only embedding entries using fake fp16.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused4BitRowwiseQuantized")
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

NO_GRADIENT(SparseLengthsWeightedSumFused4BitRowwiseFakeFP16EmbeddingOnly);
} // namespace caffe2
