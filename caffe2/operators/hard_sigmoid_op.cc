#include "caffe2/operators/hard_sigmoid_op.h"

#include <algorithm>
#include <functional>
#include <string>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <typename T>
bool HardSigmoidFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  EigenVectorArrayMap<T>(Y, N) =
      (ConstEigenVectorArrayMap<T>(X, N) * T(alpha) + T(beta))
          .cwiseMin(T(1))
          .cwiseMax(T(0));
  return true;
}

template <>
template <typename T>
bool HardSigmoidGradientFunctor<CPUContext>::Forward(
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
  EigenVectorArrayMap<T>(dX, size) =
      (Y_arr > T(0) && Y_arr < T(1))
          .select(ConstEigenVectorArrayMap<T>(dY, size) * alpha, T(0));
  return true;
}

namespace {

OpSchema::Cost CostInferenceForHardSigmoid(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<4>(def, in);
  cost.params_bytes = 0;
  return cost;
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    HardSigmoid,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        HardSigmoidFunctor<CPUContext>>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    HardSigmoidGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        HardSigmoidGradientFunctor<CPUContext>>);

// Input: X, output: Y
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(HardSigmoid)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForHardSigmoid)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Applies hard sigmoid operation to the input data element-wise.
The HardSigmoid operation takes one input $X$, produces one output $Y$, and is defined as:

$$Y = max(0,min(1,x * alpha + beta))$$

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hard_sigmoid_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hard_sigmoid_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "HardSigmoid",
    ["X"],
    ["Y"],
    alpha = 0.2,
    beta = 0.5,
)

workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
print("input:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("sigmoid:", workspace.FetchBlob("Y"))

```

**Result**

```

input: [ 1.5744036   0.31632107  1.7842269   1.4450722  -2.1726978 ]
hard_sigmoid: [ 0.81488073,  0.56326419,  0.85684538,  0.78901446,  0.06546044]

```

</details>


)DOC")
    .Arg("alpha", "float: the slope of the function. Defaults to 0.2")
    .Arg("beta", "float: the bias value of the function. Defaults to 0.5")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor with same shape as input")
    .InheritOnnxSchema();

// Input: Y, dY, output: dX
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(HardSigmoidGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
HardSigmoidGradient takes both Y and dY as well as an argument alpha and uses
this to update dX according to the chain rule and derivatives of the hard
sigmoid function.
)DOC");

namespace {

class GetHardSigmoidGradient : public GradientMakerBase {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(HardSigmoid, GetHardSigmoidGradient);

} // namespace caffe2
