#include "caffe2/operators/lpnorm_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
bool LpNormOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);

  auto* norm = Output(0, {1}, at::dtype<float>());
  const float* X_data = X.data<float>();
  const float size = average_ ? (float)X.numel() : 1.0f;
  CAFFE_ENFORCE_GT(size, 0);
  if (p_ == 1) {
    *(norm->template mutable_data<float>()) =
        (ConstEigenVectorMap<float>(X_data, X.numel()).array()).abs().sum() /
        size;
    // L1(x) = sum(|x|), L1_average(x) = sum(\x\) / x.size()
  } else if (p_ == 2) {
    *(norm->template mutable_data<float>()) =
        (ConstEigenVectorMap<float>(X_data, X.numel()).array()).square().sum() /
        size;
    // L2(x) = (sum(|x|^2)), L2_average(x) = sum(|x|^2) / x.size()
  }
  return true;
}

template <>
bool LpNormGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& dnorm = Input(1);

  CAFFE_ENFORCE_EQ(dnorm.dim(), 1);
  CAFFE_ENFORCE_EQ(dnorm.dim32(0), 1);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  const float size = average_ ? (float)X.numel() : 1.0f;
  if (p_ == 1) {
    EigenVectorMap<float>(dX->template mutable_data<float>(), X.numel())
        .array() = ConstEigenVectorMap<float>(X.data<float>(), X.numel())
                       .array()
                       .unaryExpr([](float x) {
                         const float kEps = 1e-12f;
                         if (x < -kEps) {
                           return -1.0f;
                         } else if (x > kEps) {
                           return 1.0f;
                         } else {
                           return 0.0f;
                         }
                       }) *
        ((dnorm.data<float>())[0] / size);
  } else if (p_ == 2) {
    EigenVectorMap<float>(dX->template mutable_data<float>(), X.numel())
        .array() =
        ConstEigenVectorMap<float>(X.data<float>(), X.numel()).array() * 2.0f *
        ((dnorm.data<float>())[0] / size);
  }

  return true;
}

// LpNorm
REGISTER_CPU_OPERATOR(LpNorm, LpNormOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LpNormGradient, LpNormGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LpNorm)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This op computes the $L_p$ norm of the one dimensional input tensor $X$, and outputs a one dimensional output tensor $Y$. Here, the $L_p$ norm is calculated as

$$L_p(\mathbf{x}) = \sum_i x_i^p$$

This op supports $p$ values of 1 or 2. If the average argument is set, the norm is calculated as Lp_averaged_norm(x) is defined as Lp_averaged_norm(x) = LpNorm(x) / size(x).

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "LpNorm",
    ["X"],
    ["Y"],
    p=2
)
X = np.array([5., 2.])
print("X:\n",X)

// Feed X into workspace
workspace.FeedBlob("X", X.astype(np.float32))

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [5. 2.]
Y:
 [29.]

```

</details>

)DOC")
    .Input(0, "X", "1D Input tensor of data to be operated on.")
    .Output(0, "Z", "1D output tensor")
    .Arg(
        "p",
        "*(type: int; default: 2, possible values: {1,2})* Order of the norm in p-norm.")
    .Arg(
        "average",
        "*(type: bool; default: False)* Whether we calculate norm or averaged_norm.The Lp_averaged_norm(x) is defined as Lp_averaged_norm(x) = LpNorm(x) / size(x)")
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      std::vector<int64_t> output_dims(1);
      output_dims[0] = 1; // 1
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    });

OPERATOR_SCHEMA(LpNormGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given one input float tensor X, derivative dout, and produces one output
float tensor dX. dX is the derivative of the Lp norm of tensor X, computed as
dx = d(sum over |x^p|)/dx, in which p is either 1 or 2(currently only
supports l1 and l2 norm) determined by the argument p.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Input(1, "dout", "1D input tensor")
    .Output(0, "dx", "1D output tensor")
    .Arg("p", "Order of the norm in p-norm")
    .Arg(
        "average",
        "whehther we calculate norm or averaged_norm."
        "The Lp_averaged_norm(x) is defined as"
        "Lp_averaged_normgradient(x) = LpNormGradient(x) / size(x)");

class GetLpNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LpNormGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(LpNorm, GetLpNormGradient);

} // namespace caffe2
