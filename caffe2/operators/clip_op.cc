#include "caffe2/operators/clip_op.h"

namespace caffe2 {

template <>
bool ClipOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = std::min(std::max(Xdata[i], min_), max_);
  }
  return true;
}

template <>
bool ClipGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_GT(X.size(), 0);
  CAFFE_DCHECK_EQ(dY.size(), X.size());
  dX->ReshapeLike(X);
  const float* Xdata = X.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    dXdata[i] = dYdata[i] * (Xdata[i] > min_ && Xdata[i] < max_);
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Clip, ClipOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(ClipGradient, ClipGradientOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2
