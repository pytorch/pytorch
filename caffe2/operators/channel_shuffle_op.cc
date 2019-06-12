#include "channel_shuffle_op.h"

#include <array>
#include <string>
#include <vector>

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
void RunChannelShuffleNCHW(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
    T* Y,
    CPUContext* context) {
  const int stride = G * K * HxW;
  for (int i = 0; i < N; ++i) {
    if (G < K) {
      for (int j = 0; j < G; ++j) {
        math::CopyMatrix<T, CPUContext>(
            K, HxW, X + j * K * HxW, HxW, Y + j * HxW, G * HxW, context);
      }
    } else {
      for (int j = 0; j < K; ++j) {
        math::CopyMatrix<T, CPUContext>(
            G, HxW, X + j * HxW, K * HxW, Y + j * G * HxW, HxW, context);
      }
    }
    X += stride;
    Y += stride;
  }
}

template <typename T>
void RunChannelShuffleNHWC(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
    T* Y,
    CPUContext* context) {
  const std::array<int, 2> dims = {G, K};
  const std::array<int, 2> axes = {1, 0};
  const int M = N * HxW;
  const int C = G * K;
  for (int i = 0; i < M; ++i) {
    math::Transpose<T, CPUContext>(2, dims.data(), axes.data(), X, Y, context);
    X += C;
    Y += C;
  }
}

} // namespace

template <>
bool ChannelShuffleOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  const int HxW = X.numel() / (N * C);
  const float* X_data = X.data<float>();
  float* Y_data = Y->mutable_data<float>();
  RunChannelShuffleNCHW<float>(N, G, K, HxW, X_data, Y_data, &context_);
  return true;
} // namespace caffe2

template <>
bool ChannelShuffleOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const int ndim = X.dim();
  const int N = X.dim32(0);
  const int C = X.dim32(ndim - 1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  const int HxW = X.numel() / (N * C);
  const float* X_data = X.data<float>();
  float* Y_data = Y->mutable_data<float>();
  RunChannelShuffleNHWC<float>(N, G, K, HxW, X_data, Y_data, &context_);
  return true;
}

template <>
bool ChannelShuffleGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const int N = dY.dim32(0);
  const int C = dY.dim32(1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  const int HxW = dY.numel() / (N * C);
  const float* dY_data = dY.data<float>();
  float* dX_data = dX->mutable_data<float>();
  RunChannelShuffleNCHW<float>(N, K, G, HxW, dY_data, dX_data, &context_);
  return true;
}

template <>
bool ChannelShuffleGradientOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const int ndim = dY.dim();
  const int N = dY.dim32(0);
  const int C = dY.dim32(ndim - 1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  const int HxW = dY.numel() / (N * C);
  const float* dY_data = dY.data<float>();
  float* dX_data = dX->mutable_data<float>();
  RunChannelShuffleNHWC<float>(N, K, G, HxW, dY_data, dX_data, &context_);
  return true;
}

REGISTER_CPU_OPERATOR(ChannelShuffle, ChannelShuffleOp<float, CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(
    ChannelShuffleGradient,
    ChannelShuffleGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ChannelShuffle)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1)
    .InheritOnnxSchema();
GRADIENT_OPERATOR_SCHEMA(ChannelShuffleGradient)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1);

namespace {

class GetChannelShuffleGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ChannelShuffleGradient",
        "",
        std::vector<std::string>{GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(ChannelShuffle, GetChannelShuffleGradient);

} // namespace caffe2
