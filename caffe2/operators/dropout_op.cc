#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

template <>
bool DropoutOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto* mask = Output(1);
  Y->Reshape(X.dims());
  mask->Reshape(X.dims());
  CAFFE_DCHECK_GT(X.size(), 0);
  if (is_test_) {
    device_context_.Copy<float, CPUContext, CPUContext>(
      X.size(), X.data<float>(), Y->mutable_data<float>());
    return true;
  } else {
    float scale = 1. / (1. - ratio_);
    // mask=true means keep, and mask=false means not keep, so we will
    // generate probability depending on 1-ratio.
    std::bernoulli_distribution dist(1. - ratio_);
    const float* Xdata = X.data<float>();
    float* Ydata = Y->mutable_data<float>();
    bool* mask_data = mask->mutable_data<bool>();
    auto& gen = device_context_.RandGenerator();
    for (int i = 0; i < X.size(); ++i) {
      mask_data[i] = dist(gen);
      Ydata[i] = Xdata[i] * scale * mask_data[i];
    }
    return true;
  }
}

template <>
bool DropoutGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto& mask = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_GT(dY.size(), 0);
  CAFFE_DCHECK_EQ(dY.size(), mask.size());
  dX->Reshape(dY.dims());
  if (is_test_) {
    device_context_.Copy<float, CPUContext, CPUContext>(
      dY.size(), dY.data<float>(), dX->mutable_data<float>());
    return true;
  } else {
    const float* dYdata = dY.data<float>();
    const bool* mask_data = mask.data<bool>();
    float* dXdata = dX->mutable_data<float>();
    float scale = 1. / (1. - ratio_);
    for (int i = 0; i < dY.size(); ++i) {
      dXdata[i] = dYdata[i] * mask_data[i] * scale;
    }
    return true;
  }
}


namespace {
REGISTER_CPU_OPERATOR(Dropout, DropoutOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(DropoutGrad, DropoutGradientOp<float, CPUContext>);

struct GetDropoutGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    return new vector<OperatorDef>{
        CreateOperatorDef(
            "DropoutGrad", "",
            std::vector<string>{GradientName(def.output(0)), def.output(1)},
            std::vector<string>{GradientName(def.input(0))})};
  }
};
REGISTER_GRADIENT(Dropout, GetDropoutGradient);
}  // namespace
}  // namespace caffe2
