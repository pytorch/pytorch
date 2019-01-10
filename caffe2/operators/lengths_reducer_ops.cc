#include "caffe2/operators/lengths_reducer_ops.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Use _STR option because the schema is declared using _STR version too in
// generic fashion. Otherwise it'd break schema declaration check.
// TODO(dzhulgakov): remove _STR when all lengths ops are off generic version.

REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsSum",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 0>);
REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsWeightedSum",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 1, 0>);
REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsMean",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 1>);

OPERATOR_SCHEMA(SparseLengthsPositionalWeightedSum)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Variation of SparseLengthsWeightedSum operator, where, for each row,
weights are accessed by indices [0..L-1], where L is the length of given row.
This is basically a fused operator of LengthsRangeFill + Gather +
SparseWeightedSum
)DOC")
    .Input(
        0,
        "DATA",
        "uint8 tensor obtained with "
        "operator FloatToRowwiseQuantized8Bits")
    .Input(
        1,
        "WEIGHT",
        "Scalar multipliers for the input slices. Must "
        "be a vector with the length matching the length of DATA")
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

REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsPositionalWeightedSum",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 1, 0, 1>);

} // namespace caffe2
