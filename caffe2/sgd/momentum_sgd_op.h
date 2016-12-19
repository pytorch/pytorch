#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void momentum_sgd_update(
    int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    float momentum,
    bool nesterov,
    float* param,
    Context* context) {
  const float LR = lr[0];
  for (auto i = 0; i < N; ++i) {
    if (!nesterov) {
      const float adjusted_gradient = LR * g[i] + momentum * m[i];
      nm[i] = adjusted_gradient;
      ng[i] = adjusted_gradient;
    } else {
      const float mi = m[i];
      const float mi_new = momentum * mi + LR * g[i];
      nm[i] = mi_new;
      ng[i] = (1 + momentum) * mi_new - momentum * mi;
    }

    if (param) {
      param[i] -= ng[i];
    }
  }
}

template <typename T, class Context>
class MomentumSGDOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MomentumSGDOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        momentum_(OperatorBase::GetSingleArgument<T>("momentum", 0.0)),
        nesterov_(OperatorBase::GetSingleArgument<int>("nesterov", 0)) {}

  bool RunOnDevice() override {
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsType<Tensor<Context>>(GRAD));
    CAFFE_ENFORCE(OperatorBase::InputIsType<Tensor<Context>>(MOMENTUM));
    CAFFE_ENFORCE(Input(LR).size() == 1);
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENTUM).size());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

    momentum_sgd_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(MOMENTUM).template data<T>(),
        Output(OUTPUT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
        Input(LR).template data<T>(),
        momentum_,
        nesterov_,
        NULL,
        &context_);
    return true;
  }

 protected:
  T momentum_{0.9};
  bool nesterov_;
  INPUT_TAGS(GRAD, MOMENTUM, LR);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENTUM);
};

template <typename T, class Context>
class MomentumSGDUpdateOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MomentumSGDUpdateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        momentum_(OperatorBase::GetSingleArgument<T>("momentum", 0.0)),
        nesterov_(OperatorBase::GetSingleArgument<int>("nesterov", 0)) {}

  bool RunOnDevice() override {
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsType<Tensor<Context>>(GRAD));
    CAFFE_ENFORCE(OperatorBase::InputIsType<Tensor<Context>>(MOMENTUM));
    CAFFE_ENFORCE(Input(LR).size() == 1);
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENTUM).size());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

    momentum_sgd_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(MOMENTUM).template data<T>(),
        Output(OUTPUT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
        Input(LR).template data<T>(),
        momentum_,
        nesterov_,
        Output(OUTPUT_PARAM)->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  T momentum_{0.9};
  bool nesterov_;
  INPUT_TAGS(GRAD, MOMENTUM, LR, PARAM);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENTUM, OUTPUT_PARAM);
};
}
