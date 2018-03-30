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

// Helper macro when the main op is defined elsewhere, and we only need to
// define the schema, and the gradient op.
#define REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(...)                         \
  OPERATOR_SCHEMA_STR(                                                         \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name))              \
      .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                           \
      .NumOutputs(1)                                                           \
      .SetDoc(FormatDoc<__VA_ARGS__>())                                        \
      .Output(0, "OUTPUT", "Aggregated tensor")                                \
      .FillUsing(__VA_ARGS__::PopulateSchema);                                 \
  REGISTER_CPU_OPERATOR_STR(                                                   \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) + "Gradient", \
      __VA_ARGS__::BackwardOp);                                                \
  OPERATOR_SCHEMA_STR(                                                         \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) + "Gradient") \
      .NumInputs(__VA_ARGS__::BackwardOp::kNumInputs)                          \
      .NumOutputs(1);                                                          \
  REGISTER_GRADIENT_STR(                                                       \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name),              \
      __VA_ARGS__::GetGradient)

#define REGISTER_SEGMENT_DEF(...)                                 \
  REGISTER_CPU_OPERATOR_STR(                                      \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name), \
      __VA_ARGS__::ForwardOp);                                    \
  REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(__VA_ARGS__)

REGISTER_SEGMENT_DEF(
    AbstractSortedSegmentRangeDef<float, int, CPUContext, SumRangeReducerDef>);
REGISTER_SEGMENT_DEF(AbstractSortedSegmentRangeDef<
                     float,
                     int,
                     CPUContext,
                     LogSumExpRangeReducerDef>);
REGISTER_SEGMENT_DEF(AbstractSortedSegmentRangeDef<
                     float,
                     int,
                     CPUContext,
                     LogMeanExpRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    AbstractSortedSegmentRangeDef<float, int, CPUContext, MeanRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    AbstractSortedSegmentRangeDef<float, int, CPUContext, MaxRangeReducerDef>);

#define REGISTER_REDUCER_WITH_OPS(reducer_def)                              \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractSortedSegmentDef<float, int, CPUContext, reducer_def>);       \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractSparseSortedSegmentDef<float, int, CPUContext, reducer_def>); \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractUnsortedSegmentDef<float, int, CPUContext, reducer_def>);     \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractSparseUnsortedSegmentDef<float, int, CPUContext, reducer_def>)

#define REGISTER_REDUCER_WITH_LENGTH_OPS(reducer_def, GradientNeedIndices) \
  REGISTER_SEGMENT_DEF(AbstractLengthsDef<                                 \
                       float,                                              \
                       int,                                                \
                       CPUContext,                                         \
                       reducer_def,                                        \
                       GradientNeedIndices>)

#define REGISTER_REDUCER_WITH_ALL_OPS(reducer_def)             \
  REGISTER_SEGMENT_DEF(                                        \
      AbstractReduceFrontDef<float, CPUContext, reducer_def>); \
  REGISTER_REDUCER_WITH_OPS(reducer_def)                       \
  REGISTER_REDUCER_WITH_LENGTH_OPS(reducer_def, false)

REGISTER_REDUCER_WITH_OPS(SumReducerDef);
REGISTER_REDUCER_WITH_LENGTH_OPS(SumReducerDef, true);

REGISTER_REDUCER_WITH_OPS(MeanReducerDef);
REGISTER_REDUCER_WITH_LENGTH_OPS(MeanReducerDef, false);

REGISTER_REDUCER_WITH_ALL_OPS(WeightedSumReducerDef);

// SparseLengths[Sum,WeightedSum,Mean] are now implemented separately,
// so we only rely to the historical implementation for the backward + schema.
REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(AbstractSparseLengthsDef<
                                          float,
                                          int,
                                          CPUContext,
                                          SumReducerDef,
                                          true /*GradientNeedIndices*/>)
REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(AbstractSparseLengthsDef<
                                          float,
                                          int,
                                          CPUContext,
                                          WeightedSumReducerDef,
                                          true /*GradientNeedIndices*/>)

REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(
    AbstractSparseLengthsDef<float, int, CPUContext, MeanReducerDef>)

// Auxiliary output gradients are currently implemented only for Lengths version
#define REGISTER_GRADIENT_WITH_MAIN_INPUT(...)                     \
  REGISTER_CPU_OPERATOR_STR(                                       \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) + \
          "WithMainInputGradient",                                 \
      __VA_ARGS__::WithMainInputBackwardOp);                       \
  OPERATOR_SCHEMA_STR(                                             \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) + \
      "WithMainInputGradient")                                     \
      .NumInputs(__VA_ARGS__::WithMainInputBackwardOp::kNumInputs) \
      .NumOutputs(1, INT_MAX)
REGISTER_GRADIENT_WITH_MAIN_INPUT(
    AbstractLengthsDef<float, int, CPUContext, WeightedSumReducerDef>);
REGISTER_GRADIENT_WITH_MAIN_INPUT(
    AbstractSparseLengthsDef<float, int, CPUContext, WeightedSumReducerDef>);

#define REGISTER_GRADIENT_WITH_MAIN_INPUT_AND_FORWARD_OUTPUT(...)           \
  REGISTER_CPU_OPERATOR_STR(                                                \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) +          \
          "WithMainInputAndForwardOutputGradient",                          \
      __VA_ARGS__::WithMainInputAndForwardOutputBackwardOp);                \
  OPERATOR_SCHEMA_STR(                                                      \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) +          \
      "WithMainInputAndForwardOutputGradient")                              \
      .NumInputs(                                                           \
          __VA_ARGS__::WithMainInputAndForwardOutputBackwardOp::kNumInputs) \
      .NumOutputs(1, INT_MAX)

#define REGISTER_SEGMENT_DEF_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(...) \
  OPERATOR_SCHEMA_STR(                                                   \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name))        \
      .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                     \
      .NumOutputs(1)                                                     \
      .SetDoc(FormatDoc<__VA_ARGS__>())                                  \
      .Output(0, "OUTPUT", "Aggregated tensor")                          \
      .FillUsing(__VA_ARGS__::PopulateSchema);                           \
  REGISTER_GRADIENT_WITH_MAIN_INPUT_AND_FORWARD_OUTPUT(__VA_ARGS__);     \
  REGISTER_GRADIENT_STR(                                                 \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name),        \
      __VA_ARGS__::GetGradient)

// This implements and registers a length op with a gradient which requires
// the main input as well as the output of the forward output.
#define REGISTER_LENGTHS_OPS_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(...) \
  REGISTER_CPU_OPERATOR_STR(                                             \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name),        \
      __VA_ARGS__::ForwardOp);                                           \
  REGISTER_SEGMENT_DEF_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(__VA_ARGS__)

REGISTER_LENGTHS_OPS_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(
    AbstractLengthsDef<float, int, CPUContext, MaxReducerDef>);
}
}
