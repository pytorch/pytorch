#include "caffe2/operators/softmax_op.h"

#include "caffe2/operators/softmax_utils.h"

namespace caffe2 {

// Implementation for the CPU context.
template <>
bool SoftmaxOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const int canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  const float* X_data = X.data<float>();
  float* Y_data = Y->mutable_data<float>();
  if (N == 0 || D == 0) {
    return true;
  }
  if (!scale_.defined()) {
    scale_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
  } else if (scale_.numel() != N) {
    scale_.Resize(N);
  }
  softmax_utils::SoftmaxCPU<float>(
      N, D, false, X_data, Y_data, scale_.mutable_data<float>(), &context_);
  return true;
}

// Implementation for the CPU context.
template <>
bool SoftmaxGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  const auto canonical_axis = Y.canonical_axis_index(axis_);
  const int64_t N = Y.size_to_dim(canonical_axis);
  const int64_t D = Y.size_from_dim(canonical_axis);
  // First, get scales
  if (!scale_.defined()) {
    scale_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
  } else if (scale_.numel() != N) {
    scale_.Resize(N);
  }

  if (!sum_multiplier_.defined()) {
    sum_multiplier_ = caffe2::empty({D}, at::dtype<float>().device(CPU));
    math::Set<float, CPUContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  } else if (sum_multiplier_.numel() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  auto* dX = Output(0, Y.sizes(), at::dtype<float>());
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  if (N == 0 || D == 0) {
    return true;
  }
  context_.CopySameDevice<float>(Y.numel(), dYdata, dXdata);
  float* scaledata = scale_.mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    math::Dot<float, CPUContext>(
        D, Ydata + i * D, dYdata + i * D, scaledata + i, &context_);
  }
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasNoTrans,
      N,
      D,
      1,
      -1,
      scaledata,
      sum_multiplier_.data<float>(),
      1,
      dXdata,
      &context_);
  math::Mul<float, CPUContext>(Y.numel(), dXdata, Ydata, dXdata, &context_);
  return true;
}

REGISTER_CPU_OPERATOR(Softmax, SoftmaxOp<float, CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(
    SoftmaxGradient,
    SoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Softmax)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(

Applies the Softmax function to an n-dimensional input Tensor rescaling them so
that the elements of the n-dimensional output Tensor lie in the range (0,1) and
sum to 1. The softmax operator is typically the last layer in a classifier network,
as its output can be interpreted as confidence probabilities of an input belonging
to each class. The input is a 2-D tensor (Tensor) of size (batch_size x
input_feature_dimensions). The output tensor has the same shape and contains the
softmax normalized values of the corresponding input. The softmax function is
defined as follows:

$$softmax(x_i) = \frac{\exp(x_i)}{\sum_{j} \exp(x_j)}$$

The input does not need to explicitly be a 2D vector; rather, it will be coerced
into one. For an arbitrary n-dimensional tensor `X` in
$[a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}]$, where k is the `axis` provided,
then `X` will be coerced into a 2-dimensional tensor with dimensions
$[(a_0 * ... * a_{k-1}), (a_k * ... * a_{n-1})]$. For the default case where
`axis`=1, the `X` tensor will be coerced into a 2D tensor of dimensions
$[a_0, (a_1 * ... * a_{n-1})]$, where $a_0$ is often the batch size. In this
situation, we must have $a_0 = N$ and $a_1 * ... * a_{n-1} = D$. Each of these
dimensions must be matched correctly, or else the operator will throw errors.

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/softmax_op.h
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/softmax_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Softmax",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.randn(1, 5).astype(np.float32))
print("input:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("softmax:", workspace.FetchBlob("Y"))

```

**Result**

```
input: [[ 0.0417839   0.61960053 -0.23150268 -0.64389366 -3.0000346 ]]
softmax: [[0.24422921 0.43525138 0.18582782 0.12303016 0.01166145]]

```

</details>



)DOC")
    .Arg(
        "axis",
        "*(type: int; default: 1)* Axis of the inputs when coerced to 2D matrix.")
    .Input(
        0,
        "X",
        "*(type: Tensor`<float>`)* Input tensor that's coerced into a 2D matrix of size (NxD) as described above.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* The softmax normalized output tensor with the same shape as input tensor.")
    .InheritOnnxSchema();

// Input: Y, dY. Output: dX
GRADIENT_OPERATOR_SCHEMA(SoftmaxGradient).NumInputs(2).NumOutputs(1);

class GetSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Softmax, GetSoftmaxGradient);
REGISTER_GRADIENT(SoftmaxFp16, GetSoftmaxGradient);

} // namespace caffe2
