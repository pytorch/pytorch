#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void adagrad_update(
    int N,
    const float* g,
    const float* h,
    float* ng,
    float* nh,
    float epsilon,
    const float* lr,
    Context* context) {
  // TODO(cxj): use OMP when it is reliable
  // #pragma omp parallel for
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = h[i] + gi * gi;
    ng[i] = lr[0] * gi / (std::sqrt(hi) + epsilon);
  }
}

template <typename Context>
void adagrad_compute(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float lr,
    Context* context) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = h[i] + gi * gi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

template <typename T, class Context>
class AdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}
  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENT_1).size());
    CAFFE_ENFORCE(Input(GRAD).size() == Input(PARAM).size());
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    adagrad_compute<Context>(
        Input(GRAD).size(),
        Input(PARAM).template data<T>(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_1).template data<T>(),
        Output(OUTPUT_PARAM)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
        epsilon_,
        Input(LR).template data<T>()[0],
        &context_);
    return true;
  }

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename T, class Context>
class SparseAdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto n = Input(GRAD).dim(0);

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* momentIn = Input(MOMENT_1).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).size_from_dim(1);
    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        float hi = momentOut[idx] = momentIn[idx] + gi * gi;
        paramOut[idx] = paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;
        adagrad_compute(
            block_size,
            paramIn + offsetIdx,
            gradIn + offsetI,
            momentIn + offsetIdx,
            paramOut + offsetIdx,
            momentOut + offsetIdx,
            epsilon_,
            lr[0],
            &context_);
      }
    }
    return true;
  }

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};
}
