#include "caffe2/operators/lengths_reducer_ops.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/segment_reduction_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Use _STR option because the schema is declared using _STR version too in
// generic fashion. Otherwise it'd break schema declaration check.
// TODO(dzhulgakov): remove _STR when all lengths ops are off generic version.

using SparseLengthsSumOp =
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 0>;
using SparseLengthsWeightedSumOp =
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 1, 0>;
using SparseLengthsMeanOp =
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 1>;
REGISTER_CPU_OPERATOR(SparseLengthsSum, SparseLengthsSumOp);
REGISTER_CPU_OPERATOR(SparseLengthsWeightedSum, SparseLengthsWeightedSumOp);
REGISTER_CPU_OPERATOR(SparseLengthsMean, SparseLengthsMeanOp);

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

template <typename Def>
string FormatDoc() {
  string doc = Def::doc;
  ReplaceAll(doc, "{op}", Def::OpDef::name);
  ReplaceAll(doc, "{op_doc}", Def::OpDef::doc);
  auto replaced = ReplaceAll(doc, "{extra}", "");
  CAFFE_ENFORCE_EQ(replaced, 0);
  return doc;
}

using SparseLengthsSumDef = AbstractSparseLengthsDef<
    float,
    int,
    CPUContext,
    SumReducerDef,
    true /*GradientNeedIndices*/>;
OPERATOR_SCHEMA(SparseLengthsSum)
    .NumInputs(SparseLengthsSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsSumOp::DATA,
        SparseLengthsSumOp::INDICES,
        SparseLengthsSumOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsSumDef::PopulateSchema);
REGISTER_CPU_OPERATOR(
    SparseLengthsSumGradient,
    SparseLengthsSumDef::BackwardOp);
OPERATOR_SCHEMA(SparseLengthsSumGradient)
    .NumInputs(SparseLengthsSumDef::BackwardOp::kNumInputs)
    .NumOutputs(1)
    .DisallowInputFillers();
REGISTER_GRADIENT(SparseLengthsSum, SparseLengthsSumDef::GetGradient)

using SparseLengthsWeightedSumDef = AbstractSparseLengthsDef<
    float,
    int,
    CPUContext,
    WeightedSumReducerDef,
    true /*GradientNeedIndices*/>;
OPERATOR_SCHEMA(SparseLengthsWeightedSum)
    .NumInputs(SparseLengthsWeightedSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable input fillers
    .SetDoc(FormatDoc<SparseLengthsWeightedSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsWeightedSumDef::PopulateSchema);
REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumGradient,
    SparseLengthsWeightedSumDef::BackwardOp);
OPERATOR_SCHEMA(SparseLengthsWeightedSumGradient)
    .NumInputs(SparseLengthsWeightedSumDef::BackwardOp::kNumInputs)
    .NumOutputs(1)
    .DisallowInputFillers();
REGISTER_GRADIENT(
    SparseLengthsWeightedSum,
    SparseLengthsWeightedSumDef::GetGradient)

using SparseLengthsMeanDef = AbstractSparseLengthsDef<
    float,
    int,
    CPUContext,
    MeanReducerDef,
    true /*GradientNeedIndices*/>;
OPERATOR_SCHEMA(SparseLengthsMean)
    .NumInputs(SparseLengthsMeanDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsMeanOp::DATA,
        SparseLengthsMeanOp::INDICES,
        SparseLengthsMeanOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsMeanDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsMeanDef::PopulateSchema);
REGISTER_CPU_OPERATOR(
    SparseLengthsMeanGradient,
    SparseLengthsMeanDef::BackwardOp);
OPERATOR_SCHEMA(SparseLengthsMeanGradient)
    .NumInputs(SparseLengthsMeanDef::BackwardOp::kNumInputs)
    .NumOutputs(1)
    .DisallowInputFillers();
REGISTER_GRADIENT(SparseLengthsMean, SparseLengthsMeanDef::GetGradient)
} // namespace caffe2
