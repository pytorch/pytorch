#include "caffe2/operators/elu_op.h"

#include <algorithm>
#include <functional>
#include <string>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <typename T>
bool EluFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  ConstEigenVectorArrayMap<T> X_arr(X, N);
  EigenVectorMap<T>(Y, N) =
      (X_arr < 0).select(alpha * (X_arr.exp() - T(1)), X_arr);
  return true;
}

template <>
template <typename T>
bool EluGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> Y_arr(Y, size);
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  EigenVectorArrayMap<T>(dX, size) =
      (Y_arr < 0).select(dY_arr * (Y_arr + alpha), dY_arr);
  return true;
}

REGISTER_CPU_OPERATOR(
    Elu,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        EluFunctor<CPUContext>>);
REGISTER_CPU_GRADIENT_OPERATOR(
    EluGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        EluGradientFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Elu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(

This op implements the exponential linear unit (ELU) activation function as described in [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289). The op takes an input tensor $X$ of arbitrary shape, computes the elementwise elu operation, and returns a vector $Y$ of the same shape as output. The alpha parameter may be passed as an argument, but defaults to 1. The elu operation is defined as

$$y=f(x) =\begin{cases}\alpha(e^x-1) & x < 0 \\ x & otherwise\end{cases}$$

Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/elu_op.h
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/elu_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Elu",
    ["X"],
    ["Y"],
    alpha=1.1
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[ 0.35339102  1.1860217  -0.10710736]
 [-3.1173866  -0.1889988  -0.20330353]
 [ 1.8525308  -0.368949    0.506277  ]]

Y:
 [[ 0.35339102  1.1860217  -0.11172786]
 [-1.0513     -0.18943374 -0.20236646]
 [ 1.8525308  -0.33939326  0.506277  ]]

```

</details>

)DOC")
    .Input(0, "X", "1D input tensor of data to be operated on.")
    .Output(0, "Y", "1D input tensor, calculated as described above.")
    .Arg(
        "alpha",
        "*(type: float; default: 1.0)* Defines alpha parameter used in calculation.")
    .InheritOnnxSchema();

// Input: Y, dY, output: dX
GRADIENT_OPERATOR_SCHEMA(EluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
EluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

namespace {

class GetEluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        std::vector<std::string>{O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Elu, GetEluGradient);

} // namespace caffe2
