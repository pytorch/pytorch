#include "caffe2/core/operator_gradient.h"
#include "caffe2/operators/reduce_front_back_sum_mean_ops.h"

namespace caffe2 {

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

/***
  Mean Ops
***/

// ReduceFrontMean: columnwise mean
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, true, true>::Compute(
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
    out_data[j] = sum / length;
  }
}

// ReduceBackMean: rowwise mean
template <>
template <typename T>
void SumReduceDimsOp<CPUContext, false, true>::Compute(
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
    out_data[i] = sum / length;
  }
}

// ReduceFrontMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, true, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr) {
      dXdata[i] = dYdata[col] / rows;
    } else if (row < lengths_data[col]) {
      dXdata[i] = dYdata[col] / lengths_data[col];
    } else {
      dXdata[i] = 0;
    }
  }
}

// ReduceBackMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CPUContext, false, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  for (int i = 0; i < rows * cols; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr) {
      dXdata[i] = dYdata[row] / cols;
    } else if (col < lengths_data[row]) {
      dXdata[i] = dYdata[row] / lengths_data[row];
    } else {
      dXdata[i] = 0;
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ReduceFrontMean, SumReduceDimsOp<CPUContext, true, true>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    ReduceFrontMeanGradient,
    SumReduceDimsGradientOp<CPUContext, true, true>);

class GetReduceFrontMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceFrontMeanGradient", "", grad_in, vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(ReduceFrontMean, GetReduceFrontMeanGradient);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceFrontMean)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg(
        "num_reduce_dims",
        "(*int*): number of dimensions to reduce (default=1)")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the by applying **mean**.

Can reduce more than one of the "first" dimensions by setting `num_reduce_dim`.

A second (optional) input, `lengths`, can be passed, which enforces that only a subset of the elements are considered in the mean operation.
- If input tensor `X` has shape $(d_0, d_1, d_2, ..., d_n)$, `lengths` must have shape $(d_1 * d_2 * ... * d_{n})$.
- The values of the `lengths` tensor determine how many of the values to consider for each vector in the $d_{0}$ dimension.

For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1,2]$, then $Y = [mean(1,4), mean(5,1,7), mean(2), mean(9,2)] = [2.5, 4.333, 2, 5.5]$

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_mean_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceFrontMean",
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
[[[5. 0. 9.]
  [4. 1. 1.]
  [9. 0. 8.]]

 [[2. 6. 7.]
  [6. 2. 6.]
  [0. 4. 5.]]]
Y: [4.3333335    2.1666667     6.]

```

</details>

)DOC")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Input(1, "lengths", "(*Tensor`<int>`*): number of elements in each sample")
    .Output(0, "Y", "(*Tensor`<float>`*): reduced tensor")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(true)
    })
    .InheritOnnxSchema("ReduceMean");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceFrontMeanGradient).NumInputs(2, 3).NumOutputs(1);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ReduceBackMean, SumReduceDimsOp<CPUContext, false, true>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    ReduceBackMeanGradient,
    SumReduceDimsGradientOp<CPUContext, false, true>);

class GetReduceBackMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceBackMeanGradient", "", grad_in, vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(ReduceBackMean, GetReduceBackMeanGradient);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceBackMean)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg(
        "num_reduce_dims",
        "(*int*): number of dimensions to reduce (default=1)")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the by applying **mean**.

Can reduce more than one of the "last" dimensions by setting `num_reduce_dim`.

A second (optional) input, `lengths`, can be passed, which enforces that only a subset of the elements are considered in the mean operation.
- If input tensor `X` has shape $(d_0, d_1, d_2, ..., d_n)$, `lengths` must have shape $(d_0 * d_1 * d_2 * ... * d_{n-1})$.
- The values of the `lengths` tensor determine how many of the values to consider for each vector in the $d_{n-1}$ dimension.

For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1]$, then $Y = [mean(1,5), mean(4,1,8), mean(2)] = [3, 4.333, 2]$


Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_mean_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceBackMean",
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
[[[[5. 9. 0.]
   [8. 4. 0.]
   [2. 2. 4.]]

  [[9. 0. 9.]
   [7. 9. 7.]
   [1. 0. 2.]]]]
Y: [[3.7777777 4.888889 ]]

```

</details>

)DOC")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Input(1, "lengths", "(*Tensor`<int>`*): number of elements in each sample")
    .Output(0, "Y", "(*Tensor`<float>`*): reduced tensor")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      REDUCTION_OP_SHAPE_INFERENCE(false)
    })
    .InheritOnnxSchema("ReduceMean");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ReduceBackMeanGradient).NumInputs(2, 3).NumOutputs(1);

#undef REDUCTION_OP_SHAPE_INFERENCE

} // namespace caffe2
