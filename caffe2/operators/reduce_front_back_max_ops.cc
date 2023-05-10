#include "caffe2/operators/reduce_front_back_max_ops.h"
#include "caffe2/core/operator_gradient.h"

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
  Max Ops
***/

// ReduceFrontMax
template <>
void MaxReduceDimsOp<float, CPUContext, true>::Compute(
    int rows,
    int cols,
    const float* data,
    const int32_t* lengths_data,
    float* out_data) {
  for (int i = 0; i < cols; i++) {
    float mx = data[i];
    int length = lengths_data == nullptr ? rows : lengths_data[i];
    for (int j = 1; j < length; j++) {
      mx = std::max(mx, data[j * cols + i]);
    }
    out_data[i] = mx;
  }
}

// ReduceBackMax
template <>
void MaxReduceDimsOp<float, CPUContext, false>::Compute(
    int rows,
    int cols,
    const float* data,
    const int32_t* lengths_data,
    float* out_data) {
  for (int i = 0; i < rows; i++) {
    float mx = data[i * cols];
    int length = lengths_data == nullptr ? cols : lengths_data[i];
    for (int j = 1; j < length; j++) {
      mx = std::max(mx, data[i * cols + j]);
    }
    out_data[i] = mx;
  }
}

// ReduceFrontMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CPUContext, true>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int32_t* lengths_data,
    float* dXdata) {
  int len = cols * rows;
  for (int i = 0; i < len; i++) {
    int col = i % cols;
    int row = i / cols;
    if (lengths_data != nullptr && row >= lengths_data[col]) {
      dXdata[i] = 0.0f;
    } else {
      dXdata[i] = Xdata[i] == Ydata[col] ? dYdata[col] : 0.0f;
    }
  }
}

// ReduceBackMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CPUContext, false>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int32_t* lengths_data,
    float* dXdata) {
  int len = cols * rows;
  for (int i = 0; i < len; i++) {
    int row = i / cols;
    int col = i % cols;
    if (lengths_data == nullptr || col < lengths_data[row]) {
      dXdata[i] = Xdata[i] == Ydata[row] ? dYdata[row] : 0.0f;
    } else {
      dXdata[i] = 0.0f;
    }
  }
}

REGISTER_CPU_OPERATOR(ReduceFrontMax, MaxReduceDimsOp<float, CPUContext, true>);
REGISTER_CPU_OPERATOR(
    ReduceFrontMaxGradient,
    MaxReduceDimsGradientOp<float, CPUContext, true>);

REGISTER_CPU_OPERATOR(ReduceBackMax, MaxReduceDimsOp<float, CPUContext, false>);
REGISTER_CPU_OPERATOR(
    ReduceBackMaxGradient,
    MaxReduceDimsGradientOp<float, CPUContext, false>);

class GetReduceFrontMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0), O(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceFrontMaxGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceFrontMax, GetReduceFrontMaxGradient);

class GetReduceBackMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_in = {GO(0), I(0), O(0)};
    if (def_.input_size() == 2) {
      grad_in.push_back(I(1));
    }
    return SingleGradientDef(
        "ReduceBackMaxGradient", "", grad_in, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceBackMax, GetReduceBackMaxGradient);

OPERATOR_SCHEMA(ReduceFrontMax)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg(
        "num_reduce_dims",
        "(*int*): number of dimensions to reduce (default=1)")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the by applying **max**.

Can reduce more than one of the "first" dimensions by setting `num_reduce_dim`.

A second (optional) input, `lengths`, can be passed, which enforces that only a subset of the elements are considered in the max operation.
- If input tensor `X` has shape $(d_0, d_1, d_2, ..., d_n)$, `lengths` must have shape $(d_1 * d_2 * ... * d_{n})$.
- The values of the `lengths` tensor determine how many of the values to consider for each vector in the $d_{0}$ dimension.

For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1,2]$, then $Y = [max(1,4), max(5,1,7), max(2), max(9,2)] = [4, 7, 2, 9]$

Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reduce_front_back_max_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceFrontMax",
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
[[[2. 8. 1.]
  [9. 6. 6.]
  [7. 7. 0.]]

 [[4. 3. 9.]
  [9. 2. 7.]
  [6. 4. 7.]]]
Y: [9. 8. 9.]

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
OPERATOR_SCHEMA(ReduceFrontMaxGradient).NumInputs(3, 4).NumOutputs(1);

OPERATOR_SCHEMA(ReduceBackMax)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg(
        "num_reduce_dims",
        "(*int*): number of dimensions to reduce (default=1)")
    .SetDoc(R"DOC(
Reduces the input tensor along the last dimension of the by applying **max**.

Can reduce more than one of the "last" dimensions by setting `num_reduce_dim`.

A second (optional) input, `lengths`, can be passed, which enforces that only a subset of the elements are considered in the max operation.
- If input tensor `X` has shape $(d_0, d_1, d_2, ..., d_n)$, `lengths` must have shape $(d_0 * d_1 * d_2 * ... * d_{n-1})$.
- The values of the `lengths` tensor determine how many of the values to consider for each vector in the $d_{n-1}$ dimension.

For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1]$, then $Y = [max(1,5), max(4,1,8), max(2)] = [5, 8, 2]$


Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reduce_front_back_max_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceBackMax",
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
[[[[2. 5. 1.]
   [6. 1. 9.]
   [8. 5. 9.]]

  [[5. 7. 8.]
   [9. 9. 6.]
   [6. 5. 0.]]]]
Y: [[9. 9.]]

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
OPERATOR_SCHEMA(ReduceBackMaxGradient).NumInputs(3, 4).NumOutputs(1);

#undef REDUCTION_OP_SHAPE_INFERENCE

} // namespace caffe2
