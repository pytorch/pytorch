#include "caffe2/operators/cross_entropy_op.h"

namespace caffe2 {

template <>
bool LabelCrossEntropyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto* Y = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  CAFFE_DCHECK_EQ(label.ndim(), 1);
  CAFFE_DCHECK_EQ(label.dim(0), N);
  Y->Reshape(std::vector<int>{N});
  const auto* Xdata = X.data<float>();
  const auto* labeldata = label.data<int>();
  auto* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    CAFFE_DCHECK_LT(labeldata[i], D);
    Ydata[i] = -log(std::max(Xdata[i * D + labeldata[i]], kLOG_THRESHOLD()));
  }
  return true;
}

template <>
bool LabelCrossEntropyGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  CAFFE_DCHECK_EQ(label.ndim(), 1);
  CAFFE_DCHECK_EQ(label.dim(0), N);
  CAFFE_DCHECK_EQ(dY.ndim(), 1);
  CAFFE_DCHECK_EQ(dY.dim(0), N);
  dX->ReshapeLike(X);
  math::Set<float, CPUContext>(dX->size(), 0.f, dX->mutable_data<float>(),
                               &device_context_);
  const float* Xdata = X.data<float>();
  const float* dYdata = dY.data<float>();
  const int* labeldata = label.data<int>();
  float* dXdata = dX->mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    CAFFE_DCHECK_LT(labeldata[i], D);
    dXdata[i * D + labeldata[i]] =
        - dYdata[i] / std::max(Xdata[i * D + labeldata[i]], kLOG_THRESHOLD());
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(LabelCrossEntropy,
                      LabelCrossEntropyOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LabelCrossEntropyGradient,
                      LabelCrossEntropyGradientOp<float, CPUContext>);

struct GetLabelCrossEntropyGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    return SingleGradientDef(
        "LabelCrossEntropyGradient", "",
        vector<string>{I(def, 0), I(def, 1), GO(def, 0)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(LabelCrossEntropy, GetLabelCrossEntropyGradient);
}  // namespace
}  // namespace caffe2
