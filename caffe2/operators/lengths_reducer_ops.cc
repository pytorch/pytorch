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
    CPUSparseLengthsReductionOp<float, TensorTypes<float, at::Half>, 0, 0>;
using SparseLengthsWeightedSumOp =
    CPUSparseLengthsReductionOp<float, TensorTypes<float, at::Half>, 1, 0>;
using SparseLengthsMeanOp =
    CPUSparseLengthsReductionOp<float, TensorTypes<float, at::Half>, 0, 1>;

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
    CPUSparseLengthsReductionOp<float, TensorTypes<float, at::Half>, 1, 0, 1>);

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
OPERATOR_SCHEMA(SparseLengthsSum)
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
REGISTER_CPU_OPERATOR(
    SparseLengthsSumGradient,
    SparseLengthsSumDef::BackwardOp);
OPERATOR_SCHEMA(SparseLengthsSumGradient)
    .NumInputs(SparseLengthsSumDef::BackwardOp::kNumInputs)
    .NumOutputs(1)
    .DisallowInputFillers();
REGISTER_GRADIENT(SparseLengthsSum, SparseLengthsSumDef::GetGradient)

REGISTER_CPU_OPERATOR(
    TTSparseLengthsSum,
    TTSparseLengthsSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    TTSparseLengthsSumGradient,
    TTSparseLengthsSumGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(TTSparseLengthsSum)
    .NumInputs(5)
    .NumOutputs(4)
    .SetDoc(R"DOC(
This operator introduce a new, parameter efficient embedding layer, termed TT embedding, which
can be plugged in into any model and trained end-to-end. The benefits of our compressed TT layer
are twofold. Firstly, instead of storing huge embedding matrix, it stores a sequence of much smaller
2-dimensional and 3-dimensional tensors, necessary for reconstructing the required embeddings,
which allows compressing the model significantly at the cost of a negligible performance drop.
Secondly, the overall number of parameters can be relatively small (and constant) during the whole
training stage, which allows to use larger batches or train efficiently in a case of limited resources.
)DOC")
    .Arg("factor_i", "vector<int>: factorization of voc size")
    .Arg("factor_j", "vector<int>: factorization of emb size")
    .Arg("ranks", "int[] Ranks of cores")
    .Arg("emb_size", "int: the size of each embedding entry")
    .Input(0, "core0", "tensor core 0")
    .Input(1, "core1", "tensor core 1")
    .Input(2, "core2", "tensor core 2")
    .Input(3, "index", "index for embedding")
    .Input(4, "lengths", "segment lengths")
    .Output(0, "OUTPUT", "Aggregated tensor")
    .Output(
        1,
        "core0_output",
        "intermediate mm result from core0 for backward path")
    .Output(
        2,
        "core1_output",
        "intermediate mm result from core1 for backward path")
    .Output(3, "indices", "the index for each core");

using SparseLengthsWeightedSumDef = AbstractSparseLengthsDef<
    float,
    int,
    CPUContext,
    WeightedSumReducerDef,
    true /*GradientNeedIndices*/>;
OPERATOR_SCHEMA(SparseLengthsWeightedSum)
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

OPERATOR_SCHEMA(TTSparseLengthsSumGradient).NumInputs(8).NumOutputs(3);

class GetTTSparseLengthsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // set up the input and output
    return SingleGradientDef(
        "TTSparseLengthsSumGradient",
        "",
        // CORE0, CORE1, CORE2, LENGTHS, CORE0_output, CORE1_output,
        // indices, dY
        vector<string>{
            I(0), I(1), I(2), I(4), O(1), O(2), O(3), GO(0)},
        // dCore0, dCore1, dCore2
        vector<string>{GI(0), GI(1), GI(2)});
  }
};

REGISTER_GRADIENT(TTSparseLengthsSum, GetTTSparseLengthsGradient)

} // namespace caffe2
