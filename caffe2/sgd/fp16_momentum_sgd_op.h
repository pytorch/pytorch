#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

template <class Context>
void fp16_momentum_sgd_update(
    int N,
    const at::Half* g,
    const at::Half* m,
    at::Half* ng,
    at::Half* nm,
    const float* lr,
    float momentum,
    bool nesterov,
    float weight_decay,
    bool fp32_update,
    at::Half* param,
    Context* /*context*/) {}

template <typename T, class Context>
class FP16MomentumSGDUpdateOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FP16MomentumSGDUpdateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        momentum_(this->template GetSingleArgument<float>("momentum", 0.0)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.0)),
        nesterov_(this->template GetSingleArgument<int>("nesterov", 0)),
        // when set, fp32_update will read in the fp16 data but
        // perform all the compute in fp32 precision.
        fp32_update_(this->template GetSingleArgument<int>("fp32_update", 0)) {}

  bool RunOnDevice() override {
    auto device_type = Context::GetDeviceType();
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(GRAD, device_type));
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(MOMENTUM, device_type));
    CAFFE_ENFORCE(Input(LR).size() == 1);
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENTUM).size());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

    fp16_momentum_sgd_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(MOMENTUM).template data<T>(),
        Output(OUTPUT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
        Input(LR).template data<float>(),
        momentum_,
        nesterov_,
        weight_decay_,
        fp32_update_,
        Output(OUTPUT_PARAM)->template mutable_data<T>(),
        &context_);

    return true;
  }

 protected:
  float momentum_{0.9};
  float weight_decay_{0.0};
  bool nesterov_;
  bool fp32_update_;
  INPUT_TAGS(GRAD, MOMENTUM, LR, PARAM);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENTUM, OUTPUT_PARAM);
};
}
