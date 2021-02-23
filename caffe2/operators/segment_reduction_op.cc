#include "caffe2/operators/segment_reduction_op.h"

namespace caffe2 {

OpSchema::Cost CostInferenceForSparseLengths(
    const OperatorDef& def,
    const vector<TensorShape>& inputs,
    bool use_weight) {
  int min_num_of_inputs = 3 + use_weight;
  CAFFE_ENFORCE_GE(
      inputs.size(),
      min_num_of_inputs,
      def.type() + " requires at least " + c10::to_string(min_num_of_inputs));

  const TensorShape data = inputs[0];
  const TensorShape indices = inputs[1 + use_weight];
  const TensorShape lengths = inputs[2 + use_weight];

  OpSchema::Cost c;
  CAFFE_ENFORCE_GT(data.dims_size(), 0, "data requires at least 1 dimension");
  uint64_t N = data.dims(0);
  if (N == 0) {
    return c;
  }
  uint64_t D = nElemFromDim(data, 1);
  CAFFE_ENFORCE_GT(
      lengths.dims_size(), 0, "lengths requires at least 1 dimension");
  uint64_t M = lengths.dims(0);
  uint64_t indices_size = nElemFromDim(indices);

  c.flops = indices_size * D;
  c.bytes_read = indices_size *
          (D * sizeof(data.data_type()) + sizeof(indices.data_type())) +
      M * sizeof(lengths.data_type());
  c.params_bytes = N * D * sizeof(data.data_type());
  if (use_weight) {
    const TensorShape weights = inputs[1];
    c.flops += indices_size * D;
    c.bytes_read += indices_size * sizeof(weights.data_type());
  }

  return c;
}

// registering 5 input gradient with main output
// gradient of SparseLengthsWeightedSum
OPERATOR_SCHEMA(SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient)
    .NumInputs(5)
    .NumOutputs(2);
REGISTER_CPU_OPERATOR(
    SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient,
    AbstractLengthsWithMainInputGradientOp<
        float,
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

// registering 3 input version
// gradient of SparseLengthsMean
OPERATOR_SCHEMA(SparseLengthsIndicesInGradientMeanGradient)
    .NumInputs(3)
    .NumOutputs(1);
REGISTER_CPU_OPERATOR(
    SparseLengthsIndicesInGradientMeanGradient,
    AbstractLengthsGradientOp<
        float,
        int,
        CPUContext,
        MeanReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);
// gradient of LengthsMean
OPERATOR_SCHEMA(LengthsIndicesInGradientMeanGradient)
    .NumInputs(3)
    .NumOutputs(1);
REGISTER_CPU_OPERATOR(
    LengthsIndicesInGradientMeanGradient,
    AbstractLengthsGradientOp<
        float,
        int,
        CPUContext,
        MeanReducerDef::template ReducerGradient<float, CPUContext>,
        true /*GradientNeedIndices*/>);

namespace {

static const char* kLengthsMaxExtra = R"DOC(
The *LengthsMax* op takes two inputs *DATA* and *LENGTHS*, and produces a single output *OUTPUT*. The op finds the maximum value in each of the segments of *DATA*, where segments are defined by their lengths.
For example, if $DATA = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$ then $OUTPUT = [max([2,4]), max([3,1,2]), max([10])] = [4,3,10]$.

Github Link:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LengthsMax",
    ["DATA", "LENGTHS"],
    ["OUTPUT"],
)

workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
print("DATA:\n", workspace.FetchBlob("DATA"))

workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

workspace.RunOperatorOnce(op)
print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

```

**Result**

```

DATA:
 [ 2.  4.  3.  1.  2. 10.]
LENGTHS:
 [2 3 1]
OUTPUT:
 [ 4.  3. 10.]

```

</details>

)DOC";

static const char* kLengthsMeanExtra = R"DOC(
The *LengthsMean* op takes two inputs *DATA* and *LENGTHS*, and produces a single output *OUTPUT*. The op finds the mean value in each of the segments of *DATA*, where segments are defined by their lengths.
For example, if $DATA = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$ then $OUTPUT = [mean([2,4]), mean([3,1,2]), mean([10])] = [3,2,10]$.

Github Link:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LengthsMean",
    ["DATA", "LENGTHS"],
    ["OUTPUT"],
)

workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
print("DATA:\n", workspace.FetchBlob("DATA"))

workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

workspace.RunOperatorOnce(op)
print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

```

**Result**

```

DATA:
 [ 2.  4.  3.  1.  2. 10.]
LENGTHS:
 [2 3 1]
OUTPUT:
 [ 3.  2. 10.]

```

</details>

)DOC";

static const char* kLengthsSumExtra = R"DOC(
The *LengthsSum* op takes two inputs *DATA* and *LENGTHS*, and produces a single output *OUTPUT*. The op finds the sum in each of the segments of *DATA*, where segments are defined by their lengths.
For example, if $DATA = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$ then $OUTPUT = [sum([2,4]), sum([3,1,2]), sum([10])] = [6,6,10]$.

Github Link:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LengthsSum",
    ["DATA", "LENGTHS"],
    ["OUTPUT"],
)

workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
print("DATA:\n", workspace.FetchBlob("DATA"))

workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

workspace.RunOperatorOnce(op)
print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

```

**Result**

```

DATA:
 [ 2.  4.  3.  1.  2. 10.]
LENGTHS:
 [2 3 1]
OUTPUT:
 [ 6.  6. 10.]

```

</details>

)DOC";

static const char* kLengthsWeightedSumExtra = R"DOC(
The *LengthsWeightedSum* op takes three inputs *DATA*, *LENGTHS*, and *SCALARS*, and produces a single output *OUTPUT*. The op finds the weighted sum in each of the segments of *DATA*, where segments are defined by their lengths. Before calculating the sums, the input *DATA* is weighted by the contents of *SCALARS*.
For example, if $DATA = [2,4,3,1,2,10]$, $SCALARS = [8, 2, 1, 4, 1, 0.6]$, and $LENGTHS = [2,3,1]$, then $OUTPUT = [sum([8*2,2*4]), sum([1*3,4*1,1*2]), sum([0.6*10])] = [24,9,6]$.

Github Link:
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "LengthsWeightedSum",
    ["DATA", "SCALARS","LENGTHS"],
    ["OUTPUT"],
)

workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
print("DATA:\n", workspace.FetchBlob("DATA"))

workspace.FeedBlob("SCALARS", np.array([8, 2, 1, 4, 1, 0.6]).astype(np.float32))
print("SCALARS:\n", workspace.FetchBlob("SCALARS"))

workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

workspace.RunOperatorOnce(op)
print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

```

**Result**

```

DATA:
 [ 2.  4.  3.  1.  2. 10.]
SCALARS:
 [8.  2.  1.  4.  1.  0.6]
LENGTHS:
 [2 3 1]
OUTPUT:
 [24.  9.  6.]

```

</details>

)DOC";

template <typename Def>
string FormatDoc() {
  string doc = Def::doc;
  c10::ReplaceAll(doc, "{op}", Def::OpDef::name);
  c10::ReplaceAll(doc, "{op_doc}", Def::OpDef::doc);
  if (strcmp(Def::OpDef::name, "Max") == 0) {
    c10::ReplaceAll(doc, "{extra}", kLengthsMaxExtra);
  } else if (strcmp(Def::OpDef::name, "Mean") == 0) {
    c10::ReplaceAll(doc, "{extra}", kLengthsMeanExtra);
  } else if (strcmp(Def::OpDef::name, "Sum") == 0) {
    c10::ReplaceAll(doc, "{extra}", kLengthsSumExtra);
  } else if (strcmp(Def::OpDef::name, "WeightedSum") == 0) {
    c10::ReplaceAll(doc, "{extra}", kLengthsWeightedSumExtra);
  } else {
    c10::ReplaceAll(doc, "{extra}", " ");
  }
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
// TODO: enable input fillers
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
      .DisallowInputFillers()                                                 \
      .SetDoc(FormatDoc<__VA_ARGS__>())                                       \
      .Output(0, "OUTPUT", "Aggregated tensor")                               \
      .FillUsing(__VA_ARGS__::PopulateSchema);                                \
  REGISTER_CPU_OPERATOR_STR(string(#gradient_name), __VA_ARGS__::BackwardOp); \
  OPERATOR_SCHEMA(gradient_name)                                              \
      .NumInputs(__VA_ARGS__::BackwardOp::kNumInputs)                         \
      .NumOutputs(1)                                                          \
      .DisallowInputFillers();                                                \
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
    AbstractLengthsDef<float, int, CPUContext, MeanReducerDef, true>);

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

// Macro doesn't like comma
using LengthsSumCPUOp = caffe2::AbstractLengthsDef<
    float,
    int,
    caffe2::CPUContext,
    caffe2::SumReducerDef,
    true>::ForwardOp;
using LengthsMeanCPUOp = caffe2::AbstractLengthsDef<
    float,
    int,
    caffe2::CPUContext,
    caffe2::MeanReducerDef,
    true>::ForwardOp;
using LengthsMaxCPUOp = caffe2::AbstractLengthsDef<
    float,
    int,
    caffe2::CPUContext,
    caffe2::MaxReducerDef,
    true>::ForwardOp;

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    LengthsSum,
    "_caffe2::LengthsSum(Tensor data, Tensor lengths) -> Tensor",
    LengthsSumCPUOp);
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    LengthsMean,
    "_caffe2::LengthsMean(Tensor data, Tensor lengths) -> Tensor",
    LengthsMeanCPUOp);
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    LengthsMax,
    "_caffe2::LengthsMax(Tensor data, Tensor lengths) -> Tensor",
    LengthsMaxCPUOp);
