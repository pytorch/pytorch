#include "caffe2/operators/lengths_reducer_fused_nbit_rowwise_ops.h"
#include "c10/util/Registry.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<4, CPUContext>);
OPERATOR_SCHEMA(SparseLengthsSumFused4BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::LENGTHS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsSum, but operating on
4-bit rowwise quantized matrices with fused storage (where each row
stores quantized values, and then 2-byte fp16 scale and bias).
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
NO_GRADIENT(SparseLengthsSumFused4BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<4, CPUContext, /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused4BitRowwise)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::LENGTHS,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::WEIGHTS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsWeightedSum,
but operating on 4-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 2-byte fp16 scale and bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused4BitRowwiseQuantized")
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
NO_GRADIENT(SparseLengthsWeightedSumFused4BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<
        4,
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>);
OPERATOR_SCHEMA(SparseLengthsMeanFused4BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::LENGTHS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsMean, but
operating on 4-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 2-byte fp16 scale and bias).
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
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsMeanFused4BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<2, CPUContext>);
OPERATOR_SCHEMA(SparseLengthsSumFused2BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::LENGTHS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsSum, but operating on
2-bit rowwise quantized matrices with fused storage (where each row
stores quantized values, and then 2-byte fp16 scale and bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused2BitRowwiseQuantized")
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
NO_GRADIENT(SparseLengthsSumFused2BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<2, CPUContext, /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSumFused2BitRowwise)
    .NumInputs(4)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::LENGTHS,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::WEIGHTS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsWeightedSum,
but operating on 2-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 2-byte fp16 scale and bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused2BitRowwiseQuantized")
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
NO_GRADIENT(SparseLengthsWeightedSumFused2BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<
        2,
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>);
OPERATOR_SCHEMA(SparseLengthsMeanFused2BitRowwise)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::LENGTHS)
    .SetDoc(R"DOC(
Performs the same operation as SparseLengthsMean, but
operating on 2-bit rowwise quantized matrices with fused storage
(where each row stores quantized values, and then 2-byte fp16 scale and bias).
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused2BitRowwiseQuantized")
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
NO_GRADIENT(SparseLengthsMeanFused2BitRowwise);

REGISTER_CPU_OPERATOR(
    SparseLengthsSumSparseLookup,
    SparseLengthsSumSparseLookupOp);
OPERATOR_SCHEMA(SparseLengthsSumSparseLookup)
    .NumInputs(3, 4)
    .NumOutputs(2, 3)
    .SetDoc(R"DOC(
This op converts compressed indices of SparseLengthsSum*Sparse to
uncompressed indices of SparseLengthsSum*. For compressed indices that maps
to -1. It means it will correspond to a zero row in the uncompressed data.
Therefore we will remove this indices and adjust the lengths.
)DOC")
    .Input(
        0,
        "INDICES",
        "Integer vector containing compressed indices of the first "
        "dimension of DATA for the slices that are being aggregated")
    .Input(
        1,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of INDICES")
    .Input(
        2,
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Input(
        3,
        "WEIGHTS",
        "Vector of weights to scale rows of DATA with before reduction. Same size as INDICES.")
    .Output(0, "output_indices", "Uncompressed indices")
    .Output(1, "output_lengths", "Adjusted lengths")
    .Output(2, "output_weights", "Adjusted weights")
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsSumSparseLookup);

REGISTER_CPU_OPERATOR(
    SparseLengthsSum4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<4>);
OPERATOR_SCHEMA(SparseLengthsSum4BitRowwiseSparse)
    .NumInputs(4)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<4>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4>::LENGTHS)
    .SetDoc(R"DOC(
Performs SparseLengthsSum, but operating on 4-bit rowwise quantized matrices
with fused storage (where each row stores quantized values, and then 2-byte
fp16 scale and 2-byte fp16 bias), and where rows are pruned.
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
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output")
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsSum4BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSum4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        4,
        /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSum4BitRowwiseSparse)
    .NumInputs(5)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<4, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<4, true>::WEIGHTS)
    .SetDoc(R"DOC(
Performs SparseLengthsWeightedSum, but operating on 4-bit rowwise quantized
matrices with fused storage (where each row stores quantized values, and then
2-byte fp16 scale and bias), and where rows are pruned.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused4BitRowwiseQuantized")
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
    .Input(
        4,
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsWeightedSum4BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsMean4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        4,
        /*with_weights=*/false,
        /*is_mean=*/true>);
OPERATOR_SCHEMA(SparseLengthsMean4BitRowwiseSparse)
    .NumInputs(4)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::
            COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::LENGTHS)
    .SetDoc(R"DOC(
Performs SparseLengthsMean, but operating on 4-bit rowwise quantized matrices
with fused storage (where each row stores quantized values, and then 2-byte
fp16 scale and bias), and where rows are pruned.
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
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsMean4BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsSum8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<8>);
OPERATOR_SCHEMA(SparseLengthsSum8BitRowwiseSparse)
    .NumInputs(4)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<8>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8>::LENGTHS)
    .SetDoc(R"DOC(
Performs SparseLengthsSum, but operating on 8-bit rowwise quantized matrices
with fused storage (where each row stores quantized values, and then 4-byte
fp32 scale and 4-byte fp32 bias), and where rows are pruned.
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
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output")
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsSum8BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSum8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        8,
        /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSum8BitRowwiseSparse)
    .NumInputs(5)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<8, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<8, true>::WEIGHTS)
    .SetDoc(R"DOC(
Performs SparseLengthsWeightedSum, but operating on 8-bit rowwise quantized
matrices with fused storage (where each row stores quantized values, and then
4-byte fp32 scale and bias), and where rows are pruned.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused4BitRowwiseQuantized")
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
    .Input(
        4,
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsWeightedSum8BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsMean8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        8,
        /*with_weights=*/false,
        /*is_mean=*/true>);
OPERATOR_SCHEMA(SparseLengthsMean8BitRowwiseSparse)
    .NumInputs(4)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::
            COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::LENGTHS)
    .SetDoc(R"DOC(
Performs SparseLengthsMean, but operating on 8-bit rowwise quantized matrices
with fused storage (where each row stores quantized values, and then 4-byte
fp32 scale and bias), and where rows are pruned.
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
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsMean8BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsSum2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<2>);
OPERATOR_SCHEMA(SparseLengthsSum2BitRowwiseSparse)
    .NumInputs(4)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<2>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2>::LENGTHS)
    .SetDoc(R"DOC(
Performs SparseLengthsSum, but operating on 2-bit rowwise quantized matrices
with fused storage (where each row stores quantized values, and then 2-byte
fp16 scale and 2-byte fp16 bias), and where rows are pruned.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused2BitRowwiseQuantized")
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
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output")
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsSum2BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSum2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        2,
        /*with_weights=*/true>);
OPERATOR_SCHEMA(SparseLengthsWeightedSum2BitRowwiseSparse)
    .NumInputs(5)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<2, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<2, true>::WEIGHTS)
    .SetDoc(R"DOC(
Performs SparseLengthsWeightedSum, but operating on 2-bit rowwise quantized
matrices with fused storage (where each row stores quantized values, and then
2-byte fp16 scale and bias), and where rows are pruned.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused2BitRowwiseQuantized")
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
    .Input(
        4,
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsWeightedSum2BitRowwiseSparse);

REGISTER_CPU_OPERATOR(
    SparseLengthsMean2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        2,
        /*with_weights=*/false,
        /*is_mean=*/true>);
OPERATOR_SCHEMA(SparseLengthsMean2BitRowwiseSparse)
    .NumInputs(4)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::
            COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::LENGTHS)
    .SetDoc(R"DOC(
Performs SparseLengthsMean, but operating on 2-bit rowwise quantized matrices
with fused storage (where each row stores quantized values, and then 2-byte
fp16 scale and bias), and where rows are pruned.
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToFused2BitRowwiseQuantized")
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
        "COMPRESSED_INDICES_MAPPING",
        "Integer vector mapping uncompressed indices to compressed indices")
    .Output(0, "output", "output");
NO_GRADIENT(SparseLengthsMean2BitRowwiseSparse);

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    SparseLengthsSum8BitRowwiseSparse,
    "_caffe2::SparseLengthsSum8BitRowwiseSparse("
    "Tensor data, "
    "Tensor indices, "
    "Tensor lengths, "
    "Tensor compressed_indices_mapping) -> Tensor output",
    caffe2::SparseLengthsNBitRowwiseSparseOp<8>);
