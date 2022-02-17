#include "caffe2/operators/moments_op.h"

#include <functional>
#include <string>

namespace caffe2 {

template <typename T, class Context>
bool MomentsGradientOp<T, Context>::Compute(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dmean_data,
    const T* dvariance_data,
    const T* X_data,
    const T* mean_data,
    T* dX_data) {
  const int dY_size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  const int dX_size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      dX_dims.cbegin(), dX_dims.cend(), 1, std::multiplies<int>());
  const int ndim = dX_dims.size();
  std::vector<int> index(ndim, 0);
  const T norm = static_cast<T>(dY_size) / static_cast<T>(dX_size);
  for (int dX_index = 0; dX_index < dX_size; ++dX_index) {
    const int dY_index =
        math::utils::GetIndexFromDims(ndim, dY_dims.data(), index.data());
    dX_data[dX_index] =
        (dmean_data[dY_index] +
         static_cast<T>(2) * (X_data[dX_index] - mean_data[dY_index]) *
             dvariance_data[dY_index]) *
        norm;
    math::utils::IncreaseIndexInDims(ndim, dX_dims.data(), index.data());
  }
  return true;
}

REGISTER_CPU_OPERATOR(Moments, MomentsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MomentsGradient, MomentsGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Moments)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
  Computes the mean and variance of the input tensor's element along the
  provided axes. The resulted tensor has the same rank as the input if keepdims
  equals True.
  If keepdims equals False, then the resulted tensor have the reduced dimension
  pruned.
)DOC")
    .Arg(
        "axes",
        "A list of integers, along which to reduce. If axes is not provided, "
        "the op computes the element-wise mean and variance.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default True keeps the reduced "
        "dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "mean", "Reduced mean tensor.")
    .Output(1, "variance", "Reduced variance tensor.");

OPERATOR_SCHEMA(MomentsGradient).NumInputs(4).NumOutputs(1);

namespace {

class GetMomentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MomentsGradient",
        "",
        std::vector<std::string>{GO(0), GO(1), I(0), O(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Moments, GetMomentsGradient);

} // namespace caffe2
