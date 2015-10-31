#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

// Implementation for the CPU context.
template <>
bool SoftmaxOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  Y->ReshapeLike(X);
  float* Ydata = Y->mutable_data<float>();
  // First, get scales
  if (scale_.size() != N) {
    scale_.Reshape(std::vector<int>{N});
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Reshape(std::vector<int>{D});
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data<float>(),
                                 &device_context_);
  }
  math::RowwiseMax<float, CPUContext>(N, D, X.data<float>(), scale_.mutable_data<float>(),
                                      &device_context_);
  // Put the intermediate result X - max(X) into Y
  device_context_.template Copy<float, CPUContext, CPUContext>(
      X.size(), X.data<float>(), Ydata);
  // Subtract the scale
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, N, D, 1,
      -1, scale_.data<float>(), sum_multiplier_.data<float>(), 1,
      Ydata, &device_context_);
  // Exponentiation
  math::Exp<float, CPUContext>(Y->size(), Ydata, Ydata,
                               &device_context_);
  math::Gemv<float, CPUContext>(CblasNoTrans, N, D, 1, Ydata,
                                sum_multiplier_.data<float>(), 0,
                                scale_.mutable_data<float>(), &device_context_);
  // Do division
  // TODO(Yangqing): maybe implement it more beautifully?
  const float* scale = scale_.data<float>();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      Ydata[i * D + j] /= scale[i];
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
  CAFFE_DCHECK_EQ(Y.ndim(), 2);
  int N = Y.dim(0);
  int D = Y.dim(1);
  CAFFE_DCHECK_EQ(dY.dim(0), N);
  CAFFE_DCHECK_EQ(dY.dim(1), D);
  // First, get scales
  if (scale_.size() != N) {
    scale_.Reshape(std::vector<int>{N});
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Reshape(std::vector<int>{D});
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data<float>(),
                                 &device_context_);
  }
  dX->Reshape(std::vector<int>{N, D});
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  device_context_.Copy<float, CPUContext, CPUContext>(Y.size(), dYdata, dXdata);
  float* scaledata = scale_.mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    math::Dot<float, CPUContext>(D, Ydata + i * D, dYdata + i * D,
                                 scaledata + i, &device_context_);
  }
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, N, D, 1, -1,
                                scaledata, sum_multiplier_.data<float>(), 1,
                                dXdata, &device_context_);
  math::Mul<float, CPUContext>(Y.size(), dXdata, Ydata, dXdata,
                               &device_context_);
  return true;
}

REGISTER_CPU_OPERATOR(Softmax, SoftmaxOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(SoftmaxGradient, SoftmaxGradientOp<float, CPUContext>)
}  // namespace caffe2
