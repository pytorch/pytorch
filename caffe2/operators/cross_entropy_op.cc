#include "caffe2/operators/cross_entropy_op.h"

namespace caffe2 {

template <>
bool LabelCrossEntropyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = OperatorBase::Input<Tensor<int, CPUContext> >(1);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim(0), N);
  Y->Reshape(std::vector<int>{N});
  const auto* Xdata = X.data();
  const auto* labeldata = label.data();
  auto* Ydata = Y->mutable_data();
  for (int i = 0; i < N; ++i) {
    DCHECK_LT(labeldata[i], D);
    Ydata[i] = -log(std::max(Xdata[i * D + labeldata[i]], kLOG_THRESHOLD()));
  }
  return true;
}

template <>
bool LabelCrossEntropyGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = OperatorBase::Input<Tensor<int, CPUContext> >(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim(0), N);
  DCHECK_EQ(dY.ndim(), 1);
  DCHECK_EQ(dY.dim(0), N);
  dX->ReshapeLike(X);
  math::Set<float, CPUContext>(dX->size(), 0.f, dX->mutable_data(),
                               &device_context_);
  const float* Xdata = X.data();
  const float* dYdata = dY.data();
  const int* labeldata = label.data();
  float* dXdata = dX->mutable_data();
  for (int i = 0; i < N; ++i) {
    DCHECK_LT(labeldata[i], D);
    dXdata[i * D + labeldata[i]] =
        - dYdata[i] / std::max(Xdata[i * D + labeldata[i]], kLOG_THRESHOLD());
  }
  return true;
}

REGISTER_CPU_OPERATOR(LabelCrossEntropy,
                      LabelCrossEntropyOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(LabelCrossEntropyGradient,
                      LabelCrossEntropyGradientOp<float, CPUContext>)
}  // namespace caffe2
