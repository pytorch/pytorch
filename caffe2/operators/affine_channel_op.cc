#include "caffe2/operators/affine_channel_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <vector>

namespace caffe2 {

namespace {

template <typename T>
void AffineChannelScaleBiasBackwardNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    T* dscale,
    T* dbias) {
  const T* dY_ptr = dY;
  const T* X_ptr = X;
  const int stride = C * HxW;
  EigenVectorArrayMap<T> dscale_arr(dscale, C);
  EigenVectorArrayMap<T> dbias_arr(dbias, C);
  dscale_arr.setZero();
  dbias_arr.setZero();
  for (int i = 0; i < N; ++i) {
    ConstEigenArrayMap<T> dY_arr(dY_ptr, HxW, C);
    ConstEigenArrayMap<T> X_arr(X_ptr, HxW, C);
    dscale_arr += (dY_arr * X_arr).colwise().sum();
    dbias_arr += dY_arr.colwise().sum();
    dY_ptr += stride;
    X_ptr += stride;
  }
}

template <typename T>
void AffineChannelScaleBiasBackwardNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    T* dscale,
    T* dbias) {
  ConstEigenArrayMap<T> dY_arr(dY, C, N * HxW);
  ConstEigenArrayMap<T> X_arr(X, C, N * HxW);
  EigenVectorMap<T>(dscale, C) = (dY_arr * X_arr).rowwise().sum();
  EigenVectorMap<T>(dbias, C) = dY_arr.rowwise().sum();
}

} // namespace

template <>
bool AffineChannelGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& scale = is_learnable_ ? Input(2) : Input(1);

  auto* dX = Output(0, dY.sizes(), at::dtype<float>());
  const int N = dY.dim32(0);
  const int C = dY.dim32(1);
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const int HxW = dY.numel() / (N * C);
  const float* dY_data = dY.data<float>();
  const float* scale_data = scale.data<float>();
  const std::array<int, 3> X_dims = {N, C, HxW};
  const std::array<int, 3> scale_dims = {1, C, 1};
  math::Mul<float, CPUContext>(
      3,
      X_dims.data(),
      3,
      scale_dims.data(),
      dY_data,
      scale_data,
      dX->template mutable_data<float>(),
      &context_);
  if (is_learnable_) {
    const auto& X = Input(1);
    const float* X_data = X.data<float>();

    auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
    auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
    AffineChannelScaleBiasBackwardNCHW<float>(
        N,
        C,
        HxW,
        dY_data,
        X_data,
        dscale->template mutable_data<float>(),
        dbias->template mutable_data<float>());
  }
  return true;
}

template <>
bool AffineChannelGradientOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  const auto& scale = is_learnable_ ? Input(2) : Input(1);

  auto* dX = Output(0, dY.sizes(), at::dtype<float>());
  const int ndim = dY.dim();
  const int C = dY.dim32(ndim - 1);
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const int rows = dY.numel() / C;
  const int cols = C;
  const float* dY_data = dY.data<float>();
  const float* scale_data = scale.data<float>();
  math::RowwiseMul<float, CPUContext>(
      rows,
      cols,
      dY_data,
      scale_data,
      dX->template mutable_data<float>(),
      &context_);
  if (is_learnable_) {
    const auto& X = Input(1);
    const float* X_data = X.data<float>();
    const int N = X.dim32(0);
    const int HxW = rows / N;

    auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
    auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
    AffineChannelScaleBiasBackwardNHWC<float>(
        N,
        C,
        HxW,
        dY_data,
        X_data,
        dscale->template mutable_data<float>(),
        dbias->template mutable_data<float>());
  }
  return true;
}

REGISTER_CPU_OPERATOR(AffineChannel, AffineChannelOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    AffineChannelGradient,
    AffineChannelGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(AffineChannel)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Applies a separate affine transformation to each channel of the input. Useful
for replacing spatial batch norm with its equivalent fixed transformation.
)DOC")
    .Input(0, "X", "Feature map input with order NCHW or NHWC.")
    .Input(
        1,
        "scale",
        "1D input of shape (C); the c-th element is the scale factor of the "
        "affine transformation for the c-th channel of the input.")
    .Input(
        2,
        "bias",
        "1D input of shape (C); the c-th element is the bias of the affine "
        "transformation for the c-th channel of the input.")
    .Output(0, "Y", "Output with the same order of Input.");

OPERATOR_SCHEMA(AffineChannelGradient)
    .NumInputs({2, 3})
    .NumOutputs({1, 3})
    .AllowInplace({{0, 0}});

namespace {

class GetAffineChannelGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper arg_helper(def_);
    const bool is_learnable =
        arg_helper.GetSingleArgument("is_learnable", false);
    if (is_learnable) {
      return SingleGradientDef(
          "AffineChannelGradient",
          "",
          std::vector<std::string>{GO(0), I(0), I(1)},
          std::vector<std::string>{GI(0), GI(1), GI(2)});
    } else {
      return SingleGradientDef(
          "AffineChannelGradient",
          "",
          std::vector<std::string>{GO(0), I(1)},
          std::vector<std::string>{GI(0)});
    }
  }
};

} // namespace

REGISTER_GRADIENT(AffineChannel, GetAffineChannelGradient);

} // namespace caffe2
