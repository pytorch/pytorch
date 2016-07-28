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
    ng[i] = lr[0] * gi / (sqrt(hi) + epsilon);
  }
}

template <typename T, class Context>
class AdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)) {}
  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENT_1).size());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    adagrad_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_1).template data<T>(),
        Output(OUTPUT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
        epsilon_,
        Input(LR).template data<T>(),
        &context_);
    return true;
  }

 protected:
  T epsilon_{1e-8};
  INPUT_TAGS(GRAD, MOMENT_1, LR);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENT_1);
};

template <typename T, class Context>
class SparseAdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto n = Input(GRAD).dim(0);
    auto block_size = Input(GRAD).size() / n;

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* momentIn = Input(MOMENT_1).template data<T>();
    auto* gradOut = Output(OUTPUT_GRAD)->template mutable_data<T>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    // TODO(cxj): use OMP when it is reliable
    // #pragma omp parallel for
    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        float hi = momentOut[idx] = momentIn[idx] + gi * gi;
        gradOut[i] = lr[0] * gi / (sqrt(hi) + epsilon_);
      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;
        adagrad_update(
            block_size,
            gradIn + offsetI,
            momentIn + offsetIdx,
            gradOut + offsetI,
            momentOut + offsetIdx,
            epsilon_,
            lr,
            &context_);
      }
    }
    return true;
  }

 protected:
  T epsilon_{1e-8};
  INPUT_TAGS(INDICES, GRAD, MOMENT_1, LR);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENT_1);
};
}
