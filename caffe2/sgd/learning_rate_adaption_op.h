#pragma once

#include <cfloat>
#include <cmath>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename Context>
void lr_update(
    int n,
    const float* grad,
    const float* effgrad,
    const float* lr,
    float* nlr,
    float lr_alpha,
    bool normalized_lr_adaption,
    Context* /*context*/) {
  float x = 0;
  float y = 0, z = 0;
  const float kEps = 1e-12f;
  for (const auto i : c10::irange(n)) {
    x += grad[i] * effgrad[i];
    if (normalized_lr_adaption) {
      y += grad[i] * grad[i];
      z += effgrad[i] * effgrad[i];
    }
  }
  if (normalized_lr_adaption) {
    y = fmax(std::sqrt(y), kEps);
    z = fmax(std::sqrt(z), kEps);
    nlr[0] = lr[0] * (1 - lr_alpha * x / (y * z));
  } else {
    nlr[0] = lr[0] - lr_alpha * x;
  }
}

template <typename T, class Context>
class LearningRateAdaptionOp final : public Operator<Context> {
 public:
  LearningRateAdaptionOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        lr_alpha_(this->template GetSingleArgument<float>("lr_alpha", 0.01f)),
        normalized_lr_adaption_(this->template GetSingleArgument<bool>(
            "normalized_lr_adaption",
            true)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(LR).numel() == 1);
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(EFFGRAD).numel());
    Output(OUTPUT_LR)->ResizeLike(Input(LR));
    lr_update<Context>(
        Input(GRAD).numel(),
        Input(GRAD).template data<T>(),
        Input(EFFGRAD).template data<T>(),
        Input(LR).template data<T>(),
        Output(OUTPUT_LR)->template mutable_data<T>(),
        lr_alpha_,
        normalized_lr_adaption_,
        &context_);
    return true;
  }

 protected:
  T lr_alpha_{1e-2};
  bool normalized_lr_adaption_{true};
  INPUT_TAGS(LR, GRAD, EFFGRAD);
  OUTPUT_TAGS(OUTPUT_LR);
};

} // namespace caffe2
