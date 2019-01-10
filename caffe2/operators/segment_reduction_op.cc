#include "caffe2/operators/segment_reduction_op.h"

namespace caffe2 {

// registering 5 input gradient with main output
// gradient of SparseLengthsWeightedSum
OPERATOR_SCHEMA(SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient)
    .NumInputs(5)
    .NumOutputs(2);
REGISTER_CPU_OPERATOR(
    SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient,
    AbstractLengthsWithMainInputGradientOp<
        float,
        int,
        CPUContext,
        WeightedSumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*SparseFused*/,
        true /*GradientNeedIndices*/>);

// registering 4 input version
OPERATOR_SCHEMA(SparseLengthsIndicesInGradientWeightedSumGradient)
    .NumInputs(4)
    .NumOutputs(1);
REGISTER_CPU_OPERATOR(
    SparseLengthsIndicesInGradientWeightedSumGradient,
    AbstractLengthsGradientOp<
        float,
        int,
        CPUContext,
        WeightedSumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);

// registering 3 input version
// gradient of SparseLengthsSum
OPERATOR_SCHEMA(SparseLengthsIndicesInGradientSumGradient)
    .NumInputs(3)
    .NumOutputs(1);
REGISTER_CPU_OPERATOR(
    SparseLengthsIndicesInGradientSumGradient,
    AbstractLengthsGradientOp<
        float,
        int,
        CPUContext,
        SumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);
// gradient of LengthsSum
OPERATOR_SCHEMA(LengthsIndicesInGradientSumGradient).NumInputs(3).NumOutputs(1);
REGISTER_CPU_OPERATOR(
    LengthsIndicesInGradientSumGradient,
    AbstractLengthsGradientOp<
        float,
        int,
        CPUContext,
        SumReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);

namespace {

template <typename Def>
string FormatDoc() {
  string doc = Def::doc;
  ReplaceAll(doc, "{op}", Def::OpDef::name);
  ReplaceAll(doc, "{op_doc}", Def::OpDef::doc);
  return doc;
}

// Helper function to enforce naming conventions at compile time.
constexpr bool equal(
    char const* lhs,
    char const* rhs1,
    char const* rhs2,
    char const* rhs3 = "") {
  return (*lhs == 0 && *rhs1 == 0 && *rhs2 == 0 && *rhs3 == 0) ||
      (*rhs1 != 0 && *lhs == *rhs1 && equal(lhs + 1, rhs1 + 1, rhs2, rhs3)) ||
      (*rhs1 == 0 && *rhs2 != 0 && *lhs == *rhs2 &&
       equal(lhs + 1, rhs1, rhs2 + 1, rhs3)) ||
      (*rhs1 == 0 && *rhs2 == 0 && *rhs3 != 0 && *lhs == *rhs3 &&
       equal(lhs + 1, rhs1, rhs2, rhs3 + 1));
}

// Helper macro when the main op is defined elsewhere, and we only need to
// define the schema, and the gradient op.
#define REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(                            \
    segment_name, gradient_name, ...)                                         \
  static_assert(                                                              \
      equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name),  \
      #segment_name);                                                         \
  static_assert(                                                              \
      equal(                                                                  \
          #gradient_name,                                                     \
          __VA_ARGS__::basename,                                              \
          __VA_ARGS__::OpDef::name,                                           \
          "Gradient"),                                                        \
      #gradient_name);                                                        \
  OPERATOR_SCHEMA(segment_name)                                               \
      .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                          \
      .NumOutputs(1)                                                          \
      .SetDoc(FormatDoc<__VA_ARGS__>())                                       \
      .Output(0, "OUTPUT", "Aggregated tensor")                               \
      .FillUsing(__VA_ARGS__::PopulateSchema);                                \
  REGISTER_CPU_OPERATOR_STR(string(#gradient_name), __VA_ARGS__::BackwardOp); \
  OPERATOR_SCHEMA(gradient_name)                                              \
      .NumInputs(__VA_ARGS__::BackwardOp::kNumInputs)                         \
      .NumOutputs(1);                                                         \
  REGISTER_GRADIENT_STR(string(#segment_name), __VA_ARGS__::GetGradient)

#define REGISTER_SEGMENT_DEF(segment_name, gradient_name, ...)               \
  static_assert(                                                             \
      equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name), \
      #segment_name);                                                        \
  REGISTER_CPU_OPERATOR_STR(string(#segment_name), __VA_ARGS__::ForwardOp);  \
  REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(                                 \
      segment_name, gradient_name, __VA_ARGS__)

REGISTER_SEGMENT_DEF(
    SortedSegmentRangeSum,
    SortedSegmentRangeSumGradient,
    AbstractSortedSegmentRangeDef<float, int, CPUContext, SumRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    SortedSegmentRangeLogSumExp,
    SortedSegmentRangeLogSumExpGradient,
    AbstractSortedSegmentRangeDef<
        float,
        int,
        CPUContext,
        LogSumExpRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    SortedSegmentRangeLogMeanExp,
    SortedSegmentRangeLogMeanExpGradient,
    AbstractSortedSegmentRangeDef<
        float,
        int,
        CPUContext,
        LogMeanExpRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    SortedSegmentRangeMean,
    SortedSegmentRangeMeanGradient,
    AbstractSortedSegmentRangeDef<float, int, CPUContext, MeanRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    SortedSegmentRangeMax,
    SortedSegmentRangeMaxGradient,
    AbstractSortedSegmentRangeDef<float, int, CPUContext, MaxRangeReducerDef>);

REGISTER_SEGMENT_DEF(
    SortedSegmentSum,
    SortedSegmentSumGradient,
    AbstractSortedSegmentDef<float, int, CPUContext, SumReducerDef>);
REGISTER_SEGMENT_DEF(
    SparseSortedSegmentSum,
    SparseSortedSegmentSumGradient,
    AbstractSparseSortedSegmentDef<float, int, CPUContext, SumReducerDef>);
REGISTER_SEGMENT_DEF(
    UnsortedSegmentSum,
    UnsortedSegmentSumGradient,
    AbstractUnsortedSegmentDef<float, int, CPUContext, SumReducerDef>);
REGISTER_SEGMENT_DEF(
    SparseUnsortedSegmentSum,
    SparseUnsortedSegmentSumGradient,
    AbstractSparseUnsortedSegmentDef<float, int, CPUContext, SumReducerDef>);

REGISTER_SEGMENT_DEF(
    LengthsSum,
    LengthsSumGradient,
    AbstractLengthsDef<float, int, CPUContext, SumReducerDef, true>);

REGISTER_SEGMENT_DEF(
    SortedSegmentMean,
    SortedSegmentMeanGradient,
    AbstractSortedSegmentDef<float, int, CPUContext, MeanReducerDef>);
REGISTER_SEGMENT_DEF(
    SparseSortedSegmentMean,
    SparseSortedSegmentMeanGradient,
    AbstractSparseSortedSegmentDef<float, int, CPUContext, MeanReducerDef>);
REGISTER_SEGMENT_DEF(
    UnsortedSegmentMean,
    UnsortedSegmentMeanGradient,
    AbstractUnsortedSegmentDef<float, int, CPUContext, MeanReducerDef>);
REGISTER_SEGMENT_DEF(
    SparseUnsortedSegmentMean,
    SparseUnsortedSegmentMeanGradient,
    AbstractSparseUnsortedSegmentDef<float, int, CPUContext, MeanReducerDef>);

REGISTER_SEGMENT_DEF(
    LengthsMean,
    LengthsMeanGradient,
    AbstractLengthsDef<float, int, CPUContext, MeanReducerDef, false>);

REGISTER_SEGMENT_DEF(
    ReduceFrontWeightedSum,
    ReduceFrontWeightedSumGradient,
    AbstractReduceFrontDef<float, CPUContext, WeightedSumReducerDef>);
REGISTER_SEGMENT_DEF(
    SortedSegmentWeightedSum,
    SortedSegmentWeightedSumGradient,
    AbstractSortedSegmentDef<float, int, CPUContext, WeightedSumReducerDef>);
REGISTER_SEGMENT_DEF(
    SparseSortedSegmentWeightedSum,
    SparseSortedSegmentWeightedSumGradient,
    AbstractSparseSortedSegmentDef<
        float,
        int,
        CPUContext,
        WeightedSumReducerDef>);
REGISTER_SEGMENT_DEF(
    UnsortedSegmentWeightedSum,
    UnsortedSegmentWeightedSumGradient,
    AbstractUnsortedSegmentDef<float, int, CPUContext, WeightedSumReducerDef>);
REGISTER_SEGMENT_DEF(
    SparseUnsortedSegmentWeightedSum,
    SparseUnsortedSegmentWeightedSumGradient,
    AbstractSparseUnsortedSegmentDef<
        float,
        int,
        CPUContext,
        WeightedSumReducerDef>);
REGISTER_SEGMENT_DEF(
    LengthsWeightedSum,
    LengthsWeightedSumGradient,
    AbstractLengthsDef<float, int, CPUContext, WeightedSumReducerDef, false>);

// SparseLengths[Sum,WeightedSum,Mean] are now implemented separately,
// so we only rely to the historical implementation for the backward + schema.
REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(
    SparseLengthsSum,
    SparseLengthsSumGradient,
    AbstractSparseLengthsDef<
        float,
        int,
        CPUContext,
        SumReducerDef,
        true /*GradientNeedIndices*/>)
REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(
    SparseLengthsWeightedSum,
    SparseLengthsWeightedSumGradient,
    AbstractSparseLengthsDef<
        float,
        int,
        CPUContext,
        WeightedSumReducerDef,
        true /*GradientNeedIndices*/>)

REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(
    SparseLengthsMean,
    SparseLengthsMeanGradient,
    AbstractSparseLengthsDef<float, int, CPUContext, MeanReducerDef>);

// Auxiliary output gradients are currently implemented only for Lengths version
#define REGISTER_GRADIENT_WITH_MAIN_INPUT(gradient_name, ...)        \
  static_assert(                                                     \
      equal(                                                         \
          #gradient_name,                                            \
          __VA_ARGS__::basename,                                     \
          __VA_ARGS__::OpDef::name,                                  \
          "WithMainInputGradient"),                                  \
      #gradient_name);                                               \
  REGISTER_CPU_OPERATOR_STR(                                         \
      string(#gradient_name), __VA_ARGS__::WithMainInputBackwardOp); \
  OPERATOR_SCHEMA(gradient_name)                                     \
      .NumInputs(__VA_ARGS__::WithMainInputBackwardOp::kNumInputs)   \
      .NumOutputs(1, INT_MAX)

REGISTER_GRADIENT_WITH_MAIN_INPUT(
    LengthsWeightedSumWithMainInputGradient,
    AbstractLengthsDef<float, int, CPUContext, WeightedSumReducerDef>);
REGISTER_GRADIENT_WITH_MAIN_INPUT(
    SparseLengthsWeightedSumWithMainInputGradient,
    AbstractSparseLengthsDef<float, int, CPUContext, WeightedSumReducerDef>);
} // namespace

#define REGISTER_GRADIENT_WITH_MAIN_INPUT_AND_FORWARD_OUTPUT(               \
    gradient_name, ...)                                                     \
  static_assert(                                                            \
      equal(                                                                \
          #gradient_name,                                                   \
          __VA_ARGS__::basename,                                            \
          __VA_ARGS__::OpDef::name,                                         \
          "WithMainInputAndForwardOutputGradient"),                         \
      #gradient_name);                                                      \
  REGISTER_CPU_OPERATOR_STR(                                                \
      string(#gradient_name),                                               \
      __VA_ARGS__::WithMainInputAndForwardOutputBackwardOp);                \
  OPERATOR_SCHEMA(gradient_name)                                            \
      .NumInputs(                                                           \
          __VA_ARGS__::WithMainInputAndForwardOutputBackwardOp::kNumInputs) \
      .NumOutputs(1, INT_MAX)

#define REGISTER_SEGMENT_DEF_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(         \
    segment_name, gradient_name, ...)                                        \
  static_assert(                                                             \
      equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name), \
      #segment_name);                                                        \
  OPERATOR_SCHEMA(segment_name)                                              \
      .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                         \
      .NumOutputs(1)                                                         \
      .SetDoc(FormatDoc<__VA_ARGS__>())                                      \
      .Output(0, "OUTPUT", "Aggregated tensor")                              \
      .FillUsing(__VA_ARGS__::PopulateSchema);                               \
  REGISTER_GRADIENT_WITH_MAIN_INPUT_AND_FORWARD_OUTPUT(                      \
      gradient_name, __VA_ARGS__);                                           \
  REGISTER_GRADIENT_STR(string(#segment_name), __VA_ARGS__::GetGradient)

// This implements and registers a length op with a gradient which requires
// the main input as well as the output of the forward output.
#define REGISTER_LENGTHS_OPS_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(         \
    segment_name, gradient_name, ...)                                        \
  static_assert(                                                             \
      equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name), \
      #segment_name);                                                        \
  REGISTER_CPU_OPERATOR_STR(string(#segment_name), __VA_ARGS__::ForwardOp);  \
  REGISTER_SEGMENT_DEF_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(               \
      segment_name, gradient_name, __VA_ARGS__)

REGISTER_LENGTHS_OPS_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(
    LengthsMax,
    LengthsMaxWithMainInputAndForwardOutputGradient,
    AbstractLengthsDef<float, int, CPUContext, MaxReducerDef>);
} // namespace caffe2
