#pragma once

#include "caffe2/core/common_omp.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void rmsprop_update(
    int N,
    const float* g,
    const float* ms,
    const float* mom,
    float* ng,
    float* nms,
    float* nmom,
    float decay,
    float momentum,
    float epsilon,
    const float* lr,
    Context* context);

template <typename T, class Context>
class RmsPropOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RmsPropOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        decay_(OperatorBase::GetSingleArgument<float>("decay", 0.9f)),
        momentum_(OperatorBase::GetSingleArgument<float>("momentum", 0.0f)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}
  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(LR).size() == 1);
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MEAN_SQUARES).size());
    CAFFE_ENFORCE(Input(GRAD).size() == Input(OUTPUT_MOMENTUM).size());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MEAN_SQUARES)->ResizeLike(Input(MEAN_SQUARES));
    Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));
    rmsprop_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(MEAN_SQUARES).template data<T>(),
        Input(MOMENTUM).template data<T>(),
        Output(OUTPUT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MEAN_SQUARES)->template mutable_data<T>(),
        Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
        decay_,
        momentum_,
        epsilon_,
        Input(LR).template data<T>(),
        &context_);
    return true;
  }

 protected:
  T decay_{0.9};
  T momentum_{0.0};
  T epsilon_{1e-8};
  INPUT_TAGS(GRAD, MEAN_SQUARES, MOMENTUM, LR);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MEAN_SQUARES, OUTPUT_MOMENTUM);
};
}
