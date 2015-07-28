#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

// Implementation for the CPU context.
template <>
bool SoftmaxOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  Y->ReshapeLike(X);
  // First, get scales
  if (scale_.size() != N) {
    scale_.Reshape(std::vector<int>{N});
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Reshape(std::vector<int>{D});
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data(),
                                 &device_context_);
  }
  math::RowwiseMax<float, CPUContext>(N, D, X.data(), scale_.mutable_data(),
                                      &device_context_);
  // Put the intermediate result X - max(X) into Y
  device_context_.template Copy<float, CPUContext, CPUContext>(
      X.size(), X.data(), Y->mutable_data());
  // Subtract the scale
  static const float kMinusOne = -1.;
  static const float kOne = 1.;
  static const float kZero = 0;
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, N, D, 1,
      &kMinusOne, scale_.data(), sum_multiplier_.data(), &kOne,
      Y->mutable_data(), &device_context_);
  // Exponentiation
  math::Exp<float, CPUContext>(Y->size(), Y->data(), Y->mutable_data(),
                               &device_context_);
  math::Gemv<float, CPUContext>(CblasNoTrans, N, D, &kOne, Y->data(),
                                sum_multiplier_.data(), &kZero,
                                scale_.mutable_data(), &device_context_);
  // Do division
  // TODO(Yangqing): maybe implement it more beautifully?
  float* output = Y->mutable_data();
  const float* scale = scale_.data();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      output[i * D + j] /= scale[i];
    }
  }
  return true;
}

// Implementation for the CPU context.
template <>
bool SoftmaxGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_EQ(Y.ndim(), 2);
  int N = Y.dim(0);
  int D = Y.dim(1);
  DCHECK_EQ(dY.dim(0), N);
  DCHECK_EQ(dY.dim(1), D);
  // First, get scales
  if (scale_.size() != N) {
    scale_.Reshape(std::vector<int>{N});
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Reshape(std::vector<int>{D});
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data(),
                                 &device_context_);
  }
  dX->Reshape(std::vector<int>{N, D});
  const float* Ydata = Y.data();
  const float* dYdata = dY.data();
  float* dXdata = dX->mutable_data();
  device_context_.Copy<float, CPUContext, CPUContext>(Y.size(), dYdata, dXdata);
  float* scaledata = scale_.mutable_data();
  for (int i = 0; i < N; ++i) {
    math::Dot<float, CPUContext>(D, Ydata + i * D, dYdata + i * D,
                                 scaledata + i, &device_context_);
  }
  const float kMinusOne = -1.;
  const float kOne = 1.;
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, N, D, 1, &kMinusOne,
                                scaledata, sum_multiplier_.data(), &kOne,
                                dXdata, &device_context_);
  math::Mul<float, CPUContext>(Y.size(), dXdata, Ydata, dXdata,
                               &device_context_);
  return true;
}

REGISTER_CPU_OPERATOR(Softmax, SoftmaxOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(SoftmaxGradient, SoftmaxGradientOp<float, CPUContext>)
}  // namespace caffe2
