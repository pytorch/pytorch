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
    Context* /*context*/) {
  float x = 0, y = 0, z = 0;
  const float kEps = 1e-12f;
  for (auto i = 0; i < n; i++) {
    x += grad[i] * effgrad[i];
    y += grad[i] * grad[i];
    z += effgrad[i] * effgrad[i];
  }
  y = fmax(std::sqrt(y), kEps);
  z = fmax(std::sqrt(z), kEps);
  nlr[0] = lr[0] * (1 - lr_alpha * x / (y * z));
}

template <typename T, class Context>
class LearningRateAdaptionOp final : public Operator<Context> {
 public:
  LearningRateAdaptionOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        lr_alpha_(OperatorBase::GetSingleArgument<float>("lr_alpha", 0.01f)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(LR).size() == 1);
    CAFFE_ENFORCE(Input(GRAD).size() == Input(EFFGRAD).size());
    Output(OUTPUT_LR)->ResizeLike(Input(LR));
    lr_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(EFFGRAD).template data<T>(),
        Input(LR).template data<T>(),
        Output(OUTPUT_LR)->template mutable_data<T>(),
        lr_alpha_,
        &context_);
    return true;
  }

 protected:
  T lr_alpha_{1e-2};
  INPUT_TAGS(LR, GRAD, EFFGRAD);
  OUTPUT_TAGS(OUTPUT_LR);
};

} // namespace caffe2
