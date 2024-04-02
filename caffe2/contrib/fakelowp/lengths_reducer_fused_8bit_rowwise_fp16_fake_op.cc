#include "lengths_reducer_fused_8bit_rowwise_fp16_fake_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwiseFakeFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext>);
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwiseFakeFP16)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext>::LENGTHS)
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
NO_GRADIENT(SparseLengthsSumFused8BitRowwiseFakeFP16);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwiseFakeFP16EmbeddingOnly,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/false,
        /*use_acc_fp16=*/false,
        /*use_inv_scale=*/false,
        /*use_nnpi_fma=*/false,
        /*use_fp16_for_embedding_only=*/true>);
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwiseFakeFP16EmbeddingOnly)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            false,
            false,
            false,
            true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            false,
            false,
            false,
            true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            false,
            false,
            false,
            true>::LENGTHS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsSum, but operating on
8-bit rowwise quantized matrices with fused storage (where each row
stores quantized values, and then 4-byte scale and 4-byte bias).
Convert only embedding entries using fake fp16.
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
NO_GRADIENT(SparseLengthsSumFused8BitRowwiseFakeFP16EmbeddingOnly);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwiseFakeFP16NNPI,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/false,
        /*use_acc_fp16=*/true,
        /*use_inv_scale=*/false,
        /*use_nnpi_fma=*/true>);
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwiseFakeFP16NNPI)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::LENGTHS)
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
NO_GRADIENT(SparseLengthsSumFused8BitRowwiseFakeFP16NNPI);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwiseFakeFP32NNPI,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/false,
        /*use_acc_fp16=*/false,
        /*use_inv_scale=*/false,
        /*use_nnpi_fp16_fma=*/false,
        /*use_fp16_for_embedding_only*/ false,
        /*use_acc_fp32*/ true>);
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwiseFakeFP32NNPI)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            false,
            true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            false,
            true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            false,
            true>::LENGTHS)
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
NO_GRADIENT(SparseLengthsSumFused8BitRowwiseFakeFP32NNPI);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwiseFakeFP16AccFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/false,
        /*use_acc_fp16=*/true>);
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwiseFakeFP16AccFP16)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::LENGTHS)
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
NO_GRADIENT(SparseLengthsSumFused8BitRowwiseFakeFP16AccFP16);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused8BitRowwiseFakeFP16AccInvScaleFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights*/ false,
        /*is_mean*/ 0,
        /*use_acc_fp16*/ true,
        /*use_inv_scale*/ true>);
OPERATOR_SCHEMA(SparseLengthsSumFused8BitRowwiseFakeFP16AccInvScaleFP16)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            false,
            false,
            true>::LENGTHS)
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
NO_GRADIENT(SparseLengthsSumFused8BitRowwiseFakeFP16AccInvScaleFP16);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwiseFakeFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true>::LENGTHS,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true>::WEIGHTS)
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

NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwiseFakeFP16EmbeddingOnly,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/true,
        /*is_mean=*/false,
        /*use_acc_fp16=*/false,
        /*use_inv_scale=*/false,
        /*use_nnpi_fma=*/false,
        /*use_fp16_for_embedding_only=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16EmbeddingOnly)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            true>::LENGTHS,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            true>::WEIGHTS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsWeightedSum,
but operating on 8-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 4-byte scale and 4-byte bias).
Convert only embedding entries using fake fp16.
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

NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16EmbeddingOnly);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwiseFakeFP16AccFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/true,
        /*is_mean=*/false,
        /*use_acc_fp16=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16AccFP16)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            LENGTHS,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            WEIGHTS)
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

NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16AccFP16);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/true,
        /*is_mean=*/false,
        /*use_acc_fp16=*/true,
        /*use_inv_scale=*/false,
        /*use_fma=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            LENGTHS,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            WEIGHTS)
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

NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwiseFakeFP32NNPI,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/true,
        /*is_mean=*/false,
        /*use_acc_fp16=*/false,
        /*use_inv_scale=*/false,
        /*use_nnpi_fp16_fma=*/false,
        /*use_fp16_for_embedding_only*/ false,
        /*use_acc_fp32*/ true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwiseFakeFP32NNPI)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            false,
            true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            false,
            true>::INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            false,
            true>::LENGTHS,
        SparseLengthsFused8BitRowwiseFakeFP16Op<
            CPUContext,
            true,
            false,
            false,
            false,
            false,
            false,
            true>::WEIGHTS)
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

NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwiseFakeFP32NNPI);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused8BitRowwiseFakeFP16AccInvScaleFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/true,
        /*is_mean=*/false,
        /*use_acc_fp16=*/true,
        /*use_inv_scale=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16AccInvScaleFP16)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            LENGTHS,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, true, false, true>::
            WEIGHTS)
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

NO_GRADIENT(SparseLengthsWeightedSumFused8BitRowwiseFakeFP16AccInvScaleFP16);

REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFused8BitRowwiseFakeFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>);
OPERATOR_SCHEMA(SparseLengthsMeanFused8BitRowwiseFakeFP16)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, false, true>::DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, false, true>::
            INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, false, true>::
            LENGTHS)
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
NO_GRADIENT(SparseLengthsMeanFused8BitRowwiseFakeFP16);

REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFused8BitRowwiseFakeFP16AccFP16,
    SparseLengthsFused8BitRowwiseFakeFP16Op<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true,
        /*use_acc_fp16=*/true>);
OPERATOR_SCHEMA(SparseLengthsMeanFused8BitRowwiseFakeFP16AccFP16)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, false, true, true>::
            DATA,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, false, true, true>::
            INDICES,
        SparseLengthsFused8BitRowwiseFakeFP16Op<CPUContext, false, true, true>::
            LENGTHS)
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
NO_GRADIENT(SparseLengthsMeanFused8BitRowwiseFakeFP16AccFP16);

} // namespace caffe2
