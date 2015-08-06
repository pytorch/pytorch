#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

template <>
bool DropoutOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Tensor<bool, CPUContext>* mask =
      OperatorBase::Output<Tensor<bool, CPUContext> >(1);
  Y->Reshape(X.dims());
  mask->Reshape(X.dims());
  DCHECK_GT(X.size(), 0);
  float scale = 1. / (1. - ratio_);
  // mask=true means keep, and mask=false means not keep, so we will
  // generate probability depending on 1-ratio.
  std::bernoulli_distribution dist(1. - ratio_);
  const float* Xdata = X.data();
  float* Ydata = Y->mutable_data();
  bool* mask_data = mask->mutable_data();
  auto& gen = device_context_.RandGenerator();
  for (int i = 0; i < X.size(); ++i) {
    mask_data[i] = dist(gen);
    Ydata[i] = Xdata[i] * scale * mask_data[i];
  }
  return true;
}

template <>
bool DropoutGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);
  const Tensor<bool, CPUContext>& mask =
      OperatorBase::Input<Tensor<bool, CPUContext> >(1);
  auto* dX = Output(0);
  DCHECK_GT(dY.size(), 0);
  DCHECK_EQ(dY.size(), mask.size());
  dX->Reshape(dY.dims());
  const float* dYdata = dY.data();
  const bool* mask_data = mask.data();
  float* dXdata = dX->mutable_data();
  float scale = 1. / (1. - ratio_);
  for (int i = 0; i < dY.size(); ++i) {
    dXdata[i] = dYdata[i] * mask_data[i] * scale;
  }
  return true;
}


namespace {
REGISTER_CPU_OPERATOR(Dropout, DropoutOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(DropoutGrad, DropoutGradientOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2
