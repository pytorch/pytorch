#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void adam_update(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float corrected_local_rate,
    Context* context) {
#pragma omp parallel for
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    ng[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}

template <typename T, class Context>
class AdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(OperatorBase::GetSingleArgument<float>("beta1", 0.9)),
        beta2_(OperatorBase::GetSingleArgument<float>("beta2", 0.999)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)) {}
  bool RunOnDevice() override {
    // LR and iter live on the CPU
    CAFFE_CHECK(OperatorBase::InputIsType<TensorCPU>(LR)) << "LR wrong type";
    CAFFE_CHECK(OperatorBase::InputIsType<TensorCPU>(ITER)) << "Iter wrong type";
    const auto lr = OperatorBase::Input<TensorCPU>(LR).template data<T>()[0];
    const auto iter =
        OperatorBase::Input<TensorCPU>(ITER).template data<int>()[0];
    CAFFE_CHECK(OperatorBase::InputIsType<Tensor<Context>>(GRAD))
        << "Grad wrong type";
    CAFFE_CHECK_EQ(Input(GRAD).size(), Input(MOMENT_1).size());
    CAFFE_CHECK_EQ(Input(GRAD).size(), Input(MOMENT_2).size());
    Output(OUTPUT_GRAD)->ReshapeLike(Input(GRAD));
    Output(OUTPUT_MOMENT_1)->ReshapeLike(Input(MOMENT_1));
    Output(OUTPUT_MOMENT_2)->ReshapeLike(Input(MOMENT_2));
    const auto t = iter + 1;
    const auto correction =
        std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));
    adam_update<Context>(
        Input(GRAD).size(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_1).template data<T>(),
        Input(MOMENT_2).template data<T>(),
        Output(OUTPUT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_2)->template mutable_data<T>(),
        beta1_,
        beta2_,
        epsilon_,
        correction * lr,
        &context_);
    return true;
  }

 protected:
  T beta1_{0.9};
  T beta2_{0.999};
  T epsilon_{1e-8};
  INPUT_TAGS(GRAD, MOMENT_1, MOMENT_2, LR, ITER);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2);
  DISABLE_COPY_AND_ASSIGN(AdamOp);
};
}
