#include "caffe2/operators/gelu_op.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif // _MSC_VER

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
template <typename T>
bool GeluFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* context) const {
  if (fast_gelu) {
    // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
    constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;
    ConstEigenVectorArrayMap<T> X_arr(X, N);
    EigenVectorArrayMap<T> Y_arr(Y, N);
    Y_arr = X_arr *
        (((X_arr + X_arr.cube() * gelu_utils::kFastCoeff) * kAlpha).tanh() +
         T(1)) *
        static_cast<T>(0.5);
  } else {
    // y = x * P(X <= x) where X ~ N(0, 1)
    math::CdfNorm<T, CPUContext>(N, X, Y, context);
    math::Mul<T, CPUContext>(N, X, Y, Y, context);
  }
  return true;
}

template <>
template <typename T>
bool GeluGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* X_dims */,
    const T* dY,
    const T* X,
    T* dX,
    CPUContext* context) const {
  const int N = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, N);
  ConstEigenVectorArrayMap<T> X_arr(X, N);
  EigenVectorArrayMap<T> dX_arr(dX, N);
  if (fast_gelu) {
    constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;
    constexpr T kBeta = kAlpha * gelu_utils::kFastCoeff * T(3);
    dX_arr = ((X_arr + X_arr.cube() * gelu_utils::kFastCoeff) * kAlpha).tanh();
    dX_arr =
        (T(1) + dX_arr +
         X_arr * (T(1) - dX_arr.square()) * (kBeta * X_arr.square() + kAlpha)) *
        dY_arr * static_cast<T>(0.5);
  } else {
    constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2 * T(0.5);
    math::CdfNorm<T, CPUContext>(N, X, dX, context);
    dX_arr = (dX_arr +
              X_arr * (-X_arr.square() * static_cast<T>(0.5)).exp() * kAlpha) *
        dY_arr;
  }
  return true;
}

REGISTER_CPU_OPERATOR(Gelu, GeluOp<CPUContext>);
REGISTER_CPU_OPERATOR(GeluGradient, GeluGradientOp<CPUContext>);

namespace {

OpSchema::Cost CostInferenceForGelu(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<2>(def, in);
  cost.params_bytes = 0;
  return cost;
}

} // namespace

// Input: X, output: Y
OPERATOR_SCHEMA(Gelu)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg(
        "fast_gelu",
        "If true, use y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3))).")
    .CostInferenceFunction(CostInferenceForGelu)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Relu takes one input data (Tensor) and produces one output data
(Tensor) where the rectified linear function, y = xP(X <= x) where X ~ N(0, 1),
is applied to the tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

OPERATOR_SCHEMA(GeluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(1);

namespace {

class GetGeluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GeluGradient",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Gelu, GetGeluGradient);

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    Gelu,
    "_caffe2::Gelu(Tensor input, bool fast_gelu = False) -> (Tensor output)",
    caffe2::GeluOp<caffe2::CPUContext>);
