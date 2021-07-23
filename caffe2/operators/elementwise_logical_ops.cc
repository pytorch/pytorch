#include "caffe2/operators/elementwise_logical_ops.h"

namespace caffe2 {
namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Where, WhereOp<CPUContext>);

// Input: C, X, Y, output: Z
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Where)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{1, 2}})
    .IdenticalTypeAndShapeOfInput(1)
    .SetDoc(R"DOC(
Operator Where takes three input data (Tensor, Tensor, Tensor) and
produces one output data (Tensor) where z = c ? x : y is applied elementwise.
)DOC")
    .Input(0, "C", "input tensor containing booleans")
    .Input(1, "X", "input tensor")
    .Input(2, "Y", "input tensor")
    .Output(0, "Z", "output tensor");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Where);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(IsMemberOf, IsMemberOfOp<CPUContext>);

// Input: X, output: Y
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(IsMemberOf)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef&, const vector<TensorShape>& input_types) {
          vector<TensorShape> out(1);
          out[0] = input_types[0];
          out[0].set_data_type(TensorProto_DataType::TensorProto_DataType_BOOL);
          return out;
        })
    .Arg("value", "*(type: []; default: -)* List of values to check for membership.")
    .Arg("dtype", "*(type: TensorProto_DataType; default: -)* The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto.")
    .SetDoc(R"DOC(
The *IsMemberOf* op takes an input tensor *X* and a list of values as argument, and produces one output data tensor *Y*. The output tensor is the same shape as *X* and contains booleans. The output is calculated as the function *f(x) = x in value* and is applied to *X* elementwise.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "IsMemberOf",
    ["X"],
    ["Y"],
    value=[0,2,4,6,8],
)

// Use a not-empty tensor
workspace.FeedBlob("X", np.array([0,1,2,3,4,5,6,7,8]).astype(np.int32))
print("X:\n", workspace.FetchBlob("X"))

workspace.RunOperatorOnce(op)
print("Y: \n", workspace.FetchBlob("Y"))

```

**Result**

```
// value=[0,2,4,6,8]

X:
 [0 1 2 3 4 5 6 7 8]
Y:
 [ True False  True False  True False  True False  True]

```

</details>

)DOC")
    .Input(0, "X", "Input tensor of any shape")
    .Output(0, "Y", "Output tensor (same size as X containing booleans)");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(IsMemberOf);

} // namespace

template <>
std::unordered_set<int32_t>& IsMemberOfValueHolder::get<int32_t>() {
  return int32_values_;
}

template <>
std::unordered_set<int64_t>& IsMemberOfValueHolder::get<int64_t>() {
  return int64_values_;
}

template <>
std::unordered_set<bool>& IsMemberOfValueHolder::get<bool>() {
  return bool_values_;
}

template <>
std::unordered_set<string>& IsMemberOfValueHolder::get<string>() {
  return string_values_;
}

} // namespace caffe2
