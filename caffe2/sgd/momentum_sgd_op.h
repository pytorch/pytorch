#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void momentum_sgd_update(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    const bool nesterov,
    float* param,
    Context* /*context*/) {
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
        momentum_(this->template GetSingleArgument<T>("momentum", 0.0)),
        nesterov_(this->template GetSingleArgument<int>("nesterov", 0)) {}

  bool RunOnDevice() override {
    auto device_type = Context::GetDeviceType();
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(GRAD, device_type));
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(MOMENTUM, device_type));
    CAFFE_ENFORCE(Input(LR).numel() == 1);
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENTUM).numel());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

    momentum_sgd_update<Context>(
        Input(GRAD).numel(),
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
        momentum_(this->template GetSingleArgument<T>("momentum", 0.0)),
        nesterov_(this->template GetSingleArgument<int>("nesterov", 0)) {}

  bool RunOnDevice() override {
    auto device_type = Context::GetDeviceType();
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(GRAD, device_type));
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(MOMENTUM, device_type));
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(MOMENTUM).numel());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

    momentum_sgd_update<Context>(
        Input(GRAD).numel(),
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

template <typename T, class Context>
class SparseMomentumSGDUpdateOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseMomentumSGDUpdateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        momentum_(this->template GetSingleArgument<T>("momentum", 0.0)),
        nesterov_(this->template GetSingleArgument<int>("nesterov", 0)) {}

  bool RunOnDevice() override {
    // Resize [potentially] out-of-place blobs
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));

    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENTUM).numel());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    auto block_size = Input(PARAM).numel() / Input(PARAM).size(0);
    auto n = Input(GRAD).numel() / block_size;

    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* momentumIn = Input(MOMENTUM).template data<T>();
    const auto* lr = Input(LR).template data<T>();
    // const auto* paramIn = Input(PARAM).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();

    auto* gradOut = Output(OUTPUT_GRAD)->template mutable_data<T>();
    auto* momentumOut = Output(OUTPUT_MOMENTUM)->template mutable_data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();

    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      auto offsetI = i * block_size;
      auto offsetIdx = idx * block_size;

      CAFFE_ENFORCE(offsetIdx + block_size <= Input(PARAM).numel());
      CAFFE_ENFORCE(offsetI + block_size <= Input(GRAD).numel());

      momentum_sgd_update<Context>(
          block_size,
          gradIn + offsetI,
          momentumIn + offsetIdx,
          gradOut + offsetI,
          momentumOut + offsetIdx,
          lr,
          momentum_,
          nesterov_,
          paramOut + offsetIdx,
          &context_);
    }
    return true;
  }

 protected:
  T momentum_;
  bool nesterov_;
  INPUT_TAGS(GRAD, MOMENTUM, LR, PARAM, INDICES);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENTUM, OUTPUT_PARAM);
};
}
