#include "caffe2/operators/arg_ops.h"

#include <functional>

#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T, class Compare, class Context>
void ComputeArgImpl(
    const int prev_size,
    const int next_size,
    const int n,
    const Compare& comp,
    const T* X,
    int64_t* Y,
    Context* context) {
  math::Set<int64_t, Context>(prev_size * next_size, int64_t(0), Y, context);
  for (int i = 0; i < prev_size; ++i) {
    const T* cur_X = X + i * n * next_size + next_size;
    for (int k = 1; k < n; ++k) {
      for (int j = 0; j < next_size; ++j) {
        int64_t* cur_Y = Y + i * next_size + j;
        if (comp(*cur_X, X[i * n * next_size + *cur_Y * next_size + j])) {
          *cur_Y = k;
        }
        ++cur_X;
      }
    }
  }
}

} // namespace

template <>
template <typename T>
bool ArgMaxReducer<CPUContext>::operator()(
    const int prev_size,
    const int next_size,
    const int n,
    const T* X,
    int64_t* Y,
    CPUContext* context) const {
  ComputeArgImpl(prev_size, next_size, n, std::greater<T>(), X, Y, context);
  return true;
}

template <>
template <typename T>
bool ArgMinReducer<CPUContext>::operator()(
    const int prev_size,
    const int next_size,
    const int n,
    const T* X,
    int64_t* Y,
    CPUContext* context) const {
  ComputeArgImpl(prev_size, next_size, n, std::less<T>(), X, Y, context);
  return true;
}

REGISTER_CPU_OPERATOR(ArgMax, ArgOp<CPUContext, ArgMaxReducer<CPUContext>>);
REGISTER_CPU_OPERATOR(ArgMin, ArgOp<CPUContext, ArgMinReducer<CPUContext>>);

namespace {

std::vector<TensorShape> InferTensor(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  std::vector<TensorShape> out(1);
  ArgumentHelper helper(def);
  int axis = helper.GetSingleArgument("axis", -1);
  const bool keep_dims = helper.GetSingleArgument("keepdims", true);
  const auto& in_dims = in[0].dims();
  auto* out_dims = out[0].mutable_dims();
  if (axis == -1) {
    axis = in_dims.size() - 1;
  }
  for (int i = 0; i < axis; ++i) {
    out_dims->Add(in_dims.Get(i));
  }
  if (keep_dims) {
    out_dims->Add(1);
  }
  for (int i = axis + 1; i < in_dims.size(); ++i) {
    out_dims->Add(in_dims.Get(i));
  }
  out[0].set_data_type(TensorProto::INT64);
  return out;
}

} // namespace

OPERATOR_SCHEMA(ArgMax)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(InferTensor)
    .SetDoc(R"DOC(
Retrieve the argmax of an axis dimension specified by the `axis`
argument. Given an input tensor and two arguments (`axis` and
`keepdims`), returns a tensor containing the indices of the largest
element along the given axis. If the `keepdims` arg is *True* (default),
the shape of the output tensor matches the input tensor except the
`axis` dimension equals 1. Else, the `axis` dimension of the output
tensor is removed.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "ArgMax",
    ["X"],
    ["Indices"],
    axis=2,
    keepdims=False
)

workspace.FeedBlob("X", (np.random.randint(10, size=(3,3,3))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Indices:", workspace.FetchBlob("Indices"))

```

**Result**

```
X: [[[4. 9. 6.]
  [6. 6. 1.]
  [9. 5. 4.]]

 [[6. 7. 4.]
  [7. 9. 1.]
  [3. 2. 8.]]

 [[3. 4. 6.]
  [5. 2. 7.]
  [1. 5. 7.]]]
Indices: [[1 0 0]
 [1 1 2]
 [2 2 2]]

```

</details>

    )DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Indices",
        "*(type: Tensor`<float>`)* Tensor of indices for the largest values.")
    .Arg("axis", "*(type: int; default: -1)* The axis to get argmax.")
    .Arg(
        "keepdims",
        "*(type: bool; default: True)* If True (default), the output tensor "
        "shape will match the input tensor shape except the `axis` dimension "
        "equals 1. Else, the `axis` dimension of the output tensor is removed.");

OPERATOR_SCHEMA(ArgMin)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(InferTensor)
    .SetDoc(R"DOC(
Retrieve the argmin of an axis dimension specified by the `axis`
argument. Given an input tensor and two arguments (`axis` and
`keepdims`), returns a tensor containing the indices of the smallest
element along the given axis. If the `keepdims` arg is *True* (default),
the shape of the output tensor matches the input tensor except the
`axis` dimension equals 1. Else, the `axis` dimension of the output
tensor is removed.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "ArgMin",
    ["X"],
    ["Indices"],
    axis=1
)

workspace.FeedBlob("X", (np.random.randint(10, size=(5,5))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Indices:", workspace.FetchBlob("Indices"))

```

**Result**

```

X: [[9. 4. 6. 4. 1.]
  [5. 9. 8. 3. 4.]
  [6. 1. 0. 2. 9.]
  [7. 8. 2. 4. 9.]
  [3. 9. 4. 9. 4.]]
Indices: [[4]
  [3]
  [2]
  [2]
  [0]]

```

</details>

    )DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Indices",
        "*(type: Tensor`<float>`)* Tensor of indices for the smallest values.")
    .Arg("axis", "*(type: int; default: -1)* The axis to get argmin.")
    .Arg(
        "keepdims",
        "*(type: bool; default: True)* If True (default), the output tensor "
        "shape will match the input tensor shape except the `axis` dimension "
        "equals 1. Else, the `axis` dimension of the output tensor is removed.");

SHOULD_NOT_DO_GRADIENT(ArgMax);
SHOULD_NOT_DO_GRADIENT(ArgMin);

} // namespace caffe2
