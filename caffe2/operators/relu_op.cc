#include "caffe2/operators/relu_op.h"

namespace caffe2 {

template <>
bool ReluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = std::max(Xdata[i], 0.f);
  }
  return true;
}

template <>
bool ReluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_GT(Y.size(), 0);
  CAFFE_DCHECK_EQ(dY.size(), Y.size());
  dX->ReshapeLike(Y);
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = dYdata[i] * (Ydata[i] > 0);
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Relu, ReluOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(ReluGradient, ReluGradientOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2
