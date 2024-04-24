#include "caffe2/operators/reduction_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SumElements, SumElementsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SumElementsInt, SumElementsIntOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(SumSqrElements, SumSqrElementsOp<CPUContext>);

REGISTER_CPU_OPERATOR(
    SumElementsGradient,
    SumElementsGradientOp<float, CPUContext>);

REGISTER_CPU_OPERATOR(RowwiseMax, MaxReductionOp<float, CPUContext, true>);
REGISTER_CPU_OPERATOR(
    RowwiseMaxGradient,
    MaxReductionGradientOp<float, CPUContext, true>);
REGISTER_CPU_OPERATOR(
    ColwiseMaxGradient,
    MaxReductionGradientOp<float, CPUContext, false>);
REGISTER_CPU_OPERATOR(ColwiseMax, MaxReductionOp<float, CPUContext, false>);

OPERATOR_SCHEMA(SumElements)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc(R"DOC(
Sums the elements of the input tensor. Tensor type must be float32.

Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reduction_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

sum_op = core.CreateOperator(
    "SumElements",
    ["X"],
    ["Y"]
)

avg_op = core.CreateOperator(
    "SumElements",
    ["X"],
    ["Y"],
    average=True
)

workspace.FeedBlob("X", np.random.randint(10, size=(3,3)).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(sum_op)
print("Y (sum_op):", workspace.FetchBlob("Y"))
workspace.RunOperatorOnce(avg_op)
print("Y (avg_op):", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[7. 2. 5.]
 [9. 4. 2.]
 [1. 2. 5.]]
Y (sum_op): 37.0
Y (avg_op): 4.111111

```

</details>

    )DOC")
    .Arg("average", "(*bool*): set to True to compute the average of the elements rather than the sum")
    .Input(0, "X", "(*Tensor`<float>`*): blob pointing to an instance of a counter")
    .Output(0, "sum", "(*Tensor`<float>`*): Scalar tensor containing the sum (or average)");

OPERATOR_SCHEMA(SumElementsInt)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::INT32)
    .SetDoc("Sums the integer elements of the input tensor.")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum");
SHOULD_NOT_DO_GRADIENT(SumElementsInt);

OPERATOR_SCHEMA(SumSqrElements)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Sums the squares elements of the input tensor.")
    .Arg("average", "whether to average or not")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum of squares");

OPERATOR_SCHEMA(SumElementsGradient).NumInputs(2).NumOutputs(1);

class GetSumElementsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SumElementsGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(SumElements, GetSumElementsGradient);

OPERATOR_SCHEMA(RowwiseMax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Compute row-wise max reduction of the input tensor. This op takes one input, $X$, of shape $BxMxN$, where $B$ is the batch size, $M$ is number of rows, and $N$ is number of columns. The output of this op, $Y$, is a matrix of shape $BxM$, with one row for each element of the batch, and the same number of columns as the number of rows of the input tensor.

Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reduction_ops.h
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reduction_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "RowwiseMax",
    ["X"],
    ["Y"]
)

// Create X, simulating a batch of 2, 4x4 matricies
X = np.random.randint(0,high=20,size=(2,4,4))
print("X:\n",X)

// Feed X into workspace
workspace.FeedBlob("X", X.astype(np.float32))

// Run op
workspace.RunOperatorOnce(op)

// Collect Output
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[[ 5 12 10  1]
  [ 4 16  2 15]
  [ 5 11 12 15]
  [15  4 17 19]]

 [[16  5  5 13]
  [17  2  1 17]
  [18  3 19  5]
  [14 16 10 16]]]
Y:
 [[12. 16. 15. 19.]
 [16. 17. 19. 16.]]

```

</details>

    )DOC")
    .Input(
        0,
        "X",
        "A tensor of dimensions $B x M x N$ to compute rowwise-max. Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of each element of the batch, respectively.")
    .Output(
        0,
        "Y",
        "The output tensor of shape $B x M$, where each row represents the row-wise maximums for that element of the input batch.");

OPERATOR_SCHEMA(RowwiseMaxGradient).NumInputs(3).NumOutputs(1);
class GetRowwiseMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RowwiseMaxGradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(RowwiseMax, GetRowwiseMaxGradient);

OPERATOR_SCHEMA(ColwiseMaxGradient);

OPERATOR_SCHEMA(ColwiseMax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Compute column-wise max reduction of the input tensor. This op takes one input, $X$, of shape $BxMxN$, where $B$ is the batch size, $M$ is number of rows, and $N$ is number of columns. The output of this op, $Y$, is a matrix of shape $BxN$, with one row for each element of the batch, and the same number of columns as the input tensor.

Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reduction_ops.h
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/reduction_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "ColwiseMax",
    ["X"],
    ["Y"]
)

// Create X, simulating a batch of 2, 4x4 matricies
X = np.random.randint(0,high=20,size=(2,4,4))
print("X:\n",X)

// Feed X into workspace
workspace.FeedBlob("X", X.astype(np.float32))

// Run op
workspace.RunOperatorOnce(op)

// Collect Output
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[[17 15  2  6]
  [ 8 12  6  0]
  [ 6  9  7  3]
  [ 4 13 16 13]]

 [[ 0  3  4 12]
  [18  1 17 12]
  [ 7 17 13 14]
  [12 17  2  1]]]
Y:
 [[17. 15. 16. 13.]
 [18. 17. 17. 14.]]

```

</details>

    )DOC")
    .TensorInferenceFunction([](const OperatorDef& /*unused*/,
                                const std::vector<TensorShape>& in) {
      vector<int64_t> output_dims = {in[0].dims()[0], in[0].dims()[2]};
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    })
    .Input(
        0,
        "X",
        "A tensor of dimensions $B x M x N$ to compute columnwise-max. Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of each element of the batch, respectively.")
    .Output(
        0,
        "Y",
        "The output tensor of shape $B x N$, where each row represents the column-wise maximums for that element of the input batch.");

OPERATOR_SCHEMA(ColumnMaxGradient).NumInputs(3).NumOutputs(1);
class GetColwiseMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ColwiseMaxGradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ColwiseMax, GetColwiseMaxGradient);

template <typename T, class Context>
bool SumElementsGradientOp<T, Context>::RunOnDevice()
// TODO: T21635077 fix float-divide-by-zero undefined behavior
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
    __attribute__((__no_sanitize__("float-divide-by-zero")))
#endif
#endif
{
  auto& X = Input(0);
  Tensor sum_grad(Input(1), CPU);

  auto* dX = Output(0, X.sizes(), at::dtype<T>());
  TORCH_DCHECK_EQ(sum_grad.numel(), 1);
  math::Set<T, Context>(
      dX->numel(),
      static_cast<T>(
          sum_grad.template data<T>()[0] * (average_ ? 1.0 / X.numel() : 1)),
      dX->template mutable_data<T>(),
      &context_);
  return true;
}

template <typename T, class Context, bool ROWWISE>
bool MaxReductionGradientOp<T, Context, ROWWISE>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);

  auto* dX = Output(0, X.sizes(), at::dtype<T>());

  CAFFE_ENFORCE_EQ(X.dim(), 3);

  const int batch_size = X.dim32(0);
  const int M = X.dim32(1);
  const int N = X.dim32(2);

  const T* Xdata = X.template data<T>();
  const T* Ydata = Y.template data<T>();
  const T* dYdata = dY.template data<T>();
  T* dXdata = dX->template mutable_data<T>();

  const int input_size = M * N;
  for (int i = 0; i < batch_size; ++i) {
    const T* Xdata_i = Xdata + i * input_size;
    T* dXdata_i = dXdata + i * input_size;
    if (ROWWISE) {
      const T* Ydata_i = Ydata + i * M;
      const T* dYdata_i = dYdata + i * M;
      for (int m = 0; m < M; ++m) {
        const T* Xdata_m = Xdata_i + m * N;
        T* dXdata_m = dXdata_i + m * N;
        for (int n = 0; n < N; ++n) {
          if (Xdata_m[n] == Ydata_i[m]) {
            dXdata_m[n] = dYdata_i[m];
          } else {
            dXdata_m[n] = static_cast<T>(0);
          }
        }
      }
    } else {
      const T* Ydata_i = Ydata + i * N;
      const T* dYdata_i = dYdata + i * N;
      for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
          const T* Xdata_m = Xdata_i + m * N;
          T* dXdata_m = dXdata_i + m * N;
          if (Xdata_m[n] == Ydata_i[n]) {
            dXdata_m[n] = dYdata_i[n];
          } else {
            dXdata_m[n] = static_cast<T>(0);
          }
        }
      }
    }
  }

  return true;
}

} // namespace caffe2
