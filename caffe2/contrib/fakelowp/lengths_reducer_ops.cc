#include "lengths_reducer_ops.h"

#include "caffe2/operators/segment_reduction_op.h"

namespace caffe2 {

// Use _STR option because the schema is declared using _STR version too in
// generic fashion. Otherwise it'd break schema declaration check.
// TODO(dzhulgakov): remove _STR when all lengths ops are off generic version.

using SparseLengthsSumOp =
    SparseLengthsReductionFakeFp16Op<TensorTypes<float, at::Half>, 0, 0>;
using SparseLengthsWeightedSumOp =
    SparseLengthsReductionFakeFp16Op<TensorTypes<float, at::Half>, 1, 0>;
using SparseLengthsMeanOp =
    SparseLengthsReductionFakeFp16Op<TensorTypes<float, at::Half>, 0, 1>;
using SparseLengthsSumAccFP16Op =
    SparseLengthsReductionFakeFp16Op<TensorTypes<float, at::Half>, 0, 0, 0, 1>;
using SparseLengthsWeightedSumAccFP16Op =
    SparseLengthsReductionFakeFp16Op<TensorTypes<float, at::Half>, 1, 0, 0, 1>;
using SparseLengthsMeanAccFP16Op =
    SparseLengthsReductionFakeFp16Op<TensorTypes<float, at::Half>, 0, 1, 0, 1>;
using SparseLengthsSumFakeFP16EmbeddingOnlyOp =
    SparseLengthsReductionFakeFp16Op<
        TensorTypes<float, at::Half>,
        0,
        0,
        0,
        0,
        1>;
using SparseLengthsWeightedSumFakeFP16EmbeddingOnlyOp =
    SparseLengthsReductionFakeFp16Op<
        TensorTypes<float, at::Half>,
        1,
        0,
        0,
        0,
        1>;
using SparseLengthsMeanFakeFP16EmbeddingOnlyOp =
    SparseLengthsReductionFakeFp16Op<
        TensorTypes<float, at::Half>,
        0,
        1,
        0,
        0,
        1>;

REGISTER_CPU_OPERATOR(SparseLengthsSumFakeFP16, SparseLengthsSumOp);
REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFakeFP16,
    SparseLengthsWeightedSumOp);
REGISTER_CPU_OPERATOR(SparseLengthsMeanFakeFP16, SparseLengthsMeanOp);
REGISTER_CPU_OPERATOR(
    SparseLengthsSumFakeFP16AccFP16,
    SparseLengthsSumAccFP16Op);
REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFakeFP16AccFP16,
    SparseLengthsWeightedSumAccFP16Op);
REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFakeFP16AccFP16,
    SparseLengthsMeanAccFP16Op);
REGISTER_CPU_OPERATOR(
    SparseLengthsSumFakeFP16EmbeddingOnly,
    SparseLengthsSumFakeFP16EmbeddingOnlyOp);
REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSumFakeFP16EmbeddingOnly,
    SparseLengthsWeightedSumFakeFP16EmbeddingOnlyOp);
REGISTER_CPU_OPERATOR(
    SparseLengthsMeanFakeFP16EmbeddingOnly,
    SparseLengthsMeanFakeFP16EmbeddingOnlyOp);

template <typename Def>
string FormatDoc() {
  string doc = Def::doc;
  c10::ReplaceAll(doc, "{op}", Def::OpDef::name);
  c10::ReplaceAll(doc, "{op_doc}", Def::OpDef::doc);
  auto replaced = c10::ReplaceAll(doc, "{extra}", "");
  CAFFE_ENFORCE_EQ(replaced, 0);
  return doc;
}

using SparseLengthsSumDef = AbstractSparseLengthsDef<
    float,
    int,
    CPUContext,
    SumReducerDef,
    true /*GradientNeedIndices*/>;
OPERATOR_SCHEMA(SparseLengthsSumFakeFP16)
    .NumInputs(SparseLengthsSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsSumOp::DATA,
        SparseLengthsSumOp::INDICES,
        SparseLengthsSumOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsSumDef::PopulateSchema)
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsSumFakeFP16);

using SparseLengthsWeightedSumDef = AbstractSparseLengthsDef<
    float,
    int,
    CPUContext,
    WeightedSumReducerDef,
    true /*GradientNeedIndices*/>;
OPERATOR_SCHEMA(SparseLengthsWeightedSumFakeFP16)
    .NumInputs(SparseLengthsWeightedSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsWeightedSumOp::DATA,
        SparseLengthsWeightedSumOp::INDICES,
        SparseLengthsWeightedSumOp::LENGTHS,
        SparseLengthsWeightedSumOp::WEIGHT)
    .SetDoc(FormatDoc<SparseLengthsWeightedSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsWeightedSumDef::PopulateSchema)
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsWeightedSumFakeFP16);

using SparseLengthsMeanDef = AbstractSparseLengthsDef<
    float,
    int,
    CPUContext,
    MeanReducerDef,
    true /*GradientNeedIndices*/>;
OPERATOR_SCHEMA(SparseLengthsMeanFakeFP16)
    .NumInputs(SparseLengthsMeanDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsMeanOp::DATA,
        SparseLengthsMeanOp::INDICES,
        SparseLengthsMeanOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsMeanDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsMeanDef::PopulateSchema);
NO_GRADIENT(SparseLengthsMeanFakeFP16);

OPERATOR_SCHEMA(SparseLengthsSumFakeFP16AccFP16)
    .NumInputs(SparseLengthsSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsSumOp::DATA,
        SparseLengthsSumOp::INDICES,
        SparseLengthsSumOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsSumDef::PopulateSchema)
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsSumFakeFP16AccFP16);

OPERATOR_SCHEMA(SparseLengthsWeightedSumFakeFP16AccFP16)
    .NumInputs(SparseLengthsWeightedSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsWeightedSumOp::DATA,
        SparseLengthsWeightedSumOp::INDICES,
        SparseLengthsWeightedSumOp::LENGTHS,
        SparseLengthsWeightedSumOp::WEIGHT)
    .SetDoc(FormatDoc<SparseLengthsWeightedSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsWeightedSumDef::PopulateSchema)
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsWeightedSumFakeFP16AccFP16);

OPERATOR_SCHEMA(SparseLengthsMeanFakeFP16AccFP16)
    .NumInputs(SparseLengthsMeanDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsMeanOp::DATA,
        SparseLengthsMeanOp::INDICES,
        SparseLengthsMeanOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsMeanDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsMeanDef::PopulateSchema);
NO_GRADIENT(SparseLengthsMeanFakeFP16AccFP16);

OPERATOR_SCHEMA(SparseLengthsSumFakeFP16EmbeddingOnly)
    .NumInputs(SparseLengthsSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsSumFakeFP16EmbeddingOnlyOp::DATA,
        SparseLengthsSumFakeFP16EmbeddingOnlyOp::INDICES,
        SparseLengthsSumFakeFP16EmbeddingOnlyOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsSumDef::PopulateSchema)
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsSumFakeFP16EmbeddingOnly);

OPERATOR_SCHEMA(SparseLengthsWeightedSumFakeFP16EmbeddingOnly)
    .NumInputs(SparseLengthsWeightedSumDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .WeightedValueKeyLengthInputFillers(
        SparseLengthsWeightedSumFakeFP16EmbeddingOnlyOp::DATA,
        SparseLengthsWeightedSumFakeFP16EmbeddingOnlyOp::INDICES,
        SparseLengthsWeightedSumFakeFP16EmbeddingOnlyOp::LENGTHS,
        SparseLengthsWeightedSumFakeFP16EmbeddingOnlyOp::WEIGHT)
    .SetDoc(FormatDoc<SparseLengthsWeightedSumDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsWeightedSumDef::PopulateSchema)
    .InheritOnnxSchema();
NO_GRADIENT(SparseLengthsWeightedSumFakeFP16EmbeddingOnly);

OPERATOR_SCHEMA(SparseLengthsMeanFakeFP16EmbeddingOnly)
    .NumInputs(SparseLengthsMeanDef::ForwardOp::kNumInputs)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        SparseLengthsMeanFakeFP16EmbeddingOnlyOp::DATA,
        SparseLengthsMeanFakeFP16EmbeddingOnlyOp::INDICES,
        SparseLengthsMeanFakeFP16EmbeddingOnlyOp::LENGTHS)
    .SetDoc(FormatDoc<SparseLengthsMeanDef>())
    .Output(0, "OUTPUT", "Aggregated tensor")
    .FillUsing(SparseLengthsMeanDef::PopulateSchema);
NO_GRADIENT(SparseLengthsMeanFakeFP16EmbeddingOnly);

} // namespace caffe2
