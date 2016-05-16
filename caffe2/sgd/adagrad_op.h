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
    float lr,
    Context* context) {
#pragma omp parallel for
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = h[i] + gi*gi;
    ng[i] = lr * gi / (sqrt(hi) + epsilon);
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
    // LR and iter live on the CPU
    CAFFE_CHECK(OperatorBase::InputIsType<TensorCPU>(LR)) << "LR wrong type";
    const auto lr = OperatorBase::Input<TensorCPU>(LR).template data<T>()[0];
    CAFFE_CHECK(OperatorBase::InputIsType<Tensor<Context>>(GRAD))
        << "Grad wrong type";
    CAFFE_CHECK_EQ(Input(GRAD).size(), Input(MOMENT_1).size());
    Output(OUTPUT_GRAD)->ReshapeLike(Input(GRAD));
    Output(OUTPUT_MOMENT_1)->ReshapeLike(Input(MOMENT_1));
    adagrad_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_1).template data<T>(),
        Output(OUTPUT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
        epsilon_,
        lr,
        &context_);
    return true;
  }

 protected:
  T epsilon_{1e-8};
  INPUT_TAGS(GRAD, MOMENT_1, LR);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENT_1);
  DISABLE_COPY_AND_ASSIGN(AdagradOp);
};
}
