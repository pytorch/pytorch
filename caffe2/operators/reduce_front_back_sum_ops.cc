#include "caffe2/core/operator_gradient.h"
#include "caffe2/operators/reduce_front_back_sum_mean_ops.h"

namespace caffe2 {

/***
  Sum Ops
***/

// ReduceFrontSum: columnwise sum
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, true, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int32_t* lengths_data,
    T* out_data) {
  for (int j = 0; j < cols; j++) {
    T sum = in_data[j];
    int length = lengths_data == nullptr ? rows : lengths_data[j];
    for (int i = 1; i < length; i++) {
      sum += in_data[i * cols + j];
    }
    out_data[j] = sum;
  }
}

// ReduceBackSum: rowwise sum
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, false, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int32_t* lengths_data,
    T* out_data) {
  for (int i = 0; i < rows; i++) {
    int offset = i * cols;
    T sum = in_data[offset];
    int length = lengths_data == nullptr ? cols : lengths_data[i];
    for (int j = 1; j < length; j++) {
      sum += in_data[offset + j];
    }
    out_data[i] = sum;
  }
}

// ReduceFrontSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, true, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr || row < lengths_data[col]) {
      dXdata[i] = dYdata[col];
    } else {
      dXdata[i] = 0;
    }
  }
}

// ReduceBackSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, false, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr || col < lengths_data[row]) {
      dXdata[i] = dYdata[row];
    } else {
      dXdata[i] = 0;
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ReduceFrontSum, SumReduceDimsOp<CPUContext, true, false>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    ReduceFrontSumGradient,
    SumReduceDimsGradientOp<CPUContext, true, false>);

class GetReduceFrontSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceFrontSumGradient", "", grad_in, vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(ReduceFrontSum, GetReduceFrontSumGradient);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ReduceBackSum, SumReduceDimsOp<CPUContext, false, false>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    ReduceBackSumGradient,
    SumReduceDimsGradientOp<CPUContext, false, false>);

class GetReduceBackSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceBackSumGradient", "", grad_in, vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(ReduceBackSum, GetReduceBackSumGradient);

#define REDUCTION_OP_SHAPE_INFERENCE(is_front_reducer)                      \
  CAFFE_ENFORCE_LE(1, in.size());                                           \
  CAFFE_ENFORCE_GE(2, in.size());                                           \
  ArgumentHelper helper(def);                                               \
  int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1); \
  int start_index = is_front_reducer ? num_reduce_dims : 0;                 \
  int end_index = is_front_reducer ? in[0].dims_size()                      \
                                   : in[0].dims_size() - num_reduce_dims;   \
  vector<int> output_shape;                                                 \
  for (int i = start_index; i < end_index; ++i) {                           \
    output_shape.push_back(in[0].dims(i));                                  \
  }                                                                         \
  return vector<TensorShape>{                                               \
      CreateTensorShape(output_shape, in[0].data_type())};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceFrontSum)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg(
        "num_reduce_dims",
        "(*int*): number of dimensions to reduce (default=1)")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the by applying **sum**.

Can reduce more than one of the "first" dimensions by setting `num_reduce_dim`.

A second (optional) input, `lengths`, can be passed, which enforces that only a subset of the elements are considered in the sum operation.
- If input tensor `X` has shape $(d_0, d_1, d_2, ..., d_n)$, `lengths` must have shape $(d_1 * d_2 * ... * d_{n})$.
- The values of the `lengths` tensor determine how many of the values to consider for each vector in the $d_{0}$ dimension.

For example, if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1,2]$, then $Y = [sum(1,4), sum(5,1,7), sum(2), sum(9,2)] = [2.5, 4.333, 2, 5.5]$

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_sum_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceFrontSum",
    ["X"],
    ["Y"],
    num_reduce_dim=2
)

workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[[4. 1. 1.]
  [0. 6. 7.]
  [7. 8. 6.]]

 [[5. 7. 7.]
  [0. 1. 6.]
  [2. 9. 0.]]]
Y: [18. 32. 27.]

```

</details>

)DOC")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Input(1, "lengths", "(*Tensor`<int>`*): number of elements in each sample")
    .Output(0, "Y", "(*Tensor`<float>`*): reduced tensor")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(true)
    });
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceFrontSumGradient).NumInputs(2, 3).NumOutputs(1);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceBackSum)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg(
        "num_reduce_dims",
        "(*int*): number of dimensions to reduce (default=1)")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the by applying **sum**.

Can reduce more than one of the "last" dimensions by setting `num_reduce_dim`.

A second (optional) input, `lengths`, can be passed, which enforces that only a subset of the elements are considered in the sum operation.
- If input tensor `X` has shape $(d_0, d_1, d_2, ..., d_n)$, `lengths` must have shape $(d_0 * d_1 * d_2 * ... * d_{n-1})$.
- The values of the `lengths` tensor determine how many of the values to consider for each vector in the $d_{n-1}$ dimension.

For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1]$, then $Y = [sum(1,5), sum(4,1,8), sum(2)] = [6, 13, 2]$


Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_sum_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceBackSum",
    ["X"],
    ["Y"],
    num_reduce_dim=2
)

workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[[[2. 7. 7.]
   [1. 1. 0.]
   [9. 7. 2.]]

  [[6. 6. 4.]
   [1. 2. 6.]
   [6. 6. 3.]]]]
Y: [[36. 40.]]

```

</details>

)DOC")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Input(1, "lengths", "(*Tensor`<int>`*): number of elements in each sample")
    .Output(0, "Y", "(*Tensor`<float>`*): reduced tensor")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(false)
    });
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceBackSumGradient).NumInputs(2, 3).NumOutputs(1);

#undef REDUCTION_OP_SHAPE_INFERENCE

} // namespace caffe2
