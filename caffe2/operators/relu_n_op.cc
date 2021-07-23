#include "caffe2/operators/relu_n_op.h"

#include <algorithm>
#include <functional>
#include <string>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <typename T>
bool ReluNFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  EigenVectorMap<T>(Y, N) =
      ConstEigenVectorMap<T>(X, N).cwiseMax(T(0)).cwiseMin(T(n));
  return true;
}

template <>
template <typename T>
bool ReluNGradientFunctor<CPUContext>::Forward(
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
      (Y_arr > T(0) && Y_arr < T(n))
          .select(ConstEigenVectorArrayMap<T>(dY, size), T(0));
  return true;
}

namespace {

OpSchema::Cost CostInferenceForReluN(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<2>(def, in);
  cost.params_bytes = 0;
  return cost;
}

} // namespace

REGISTER_CPU_OPERATOR(
    ReluN,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        ReluNFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    ReluNGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        ReluNGradientFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(ReluN)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("n", "the cap of output")
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForReluN)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Relu takes one input data (Tensor) and produces one output data
(Tensor) where the rectified linear function, y = min(max(0, x), n),
is applied to the tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(ReluNGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("n", "the cap of forward op output")
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
ReluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

namespace {

class GetReluNGradient : public GradientMakerBase {
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

REGISTER_GRADIENT(ReluN, GetReluNGradient);

} // namespace caffe2
