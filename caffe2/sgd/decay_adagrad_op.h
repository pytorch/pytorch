#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <typename Context>
void decay_adagrad_compute(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float weight_decay,
    float c,
    const float* lr,
    Context* /*context*/) {
    ConstEigenVectorArrayMap<float> w_arr(w, N);
    ConstEigenVectorArrayMap<float> g_arr(g, N);
    ConstEigenVectorArrayMap<float> m_arr(m, N);
    ConstEigenVectorArrayMap<float> v_arr(v, N);
    EigenVectorArrayMap<float> nw_arr(nw, N);
    EigenVectorArrayMap<float> nm_arr(nm, N);
    EigenVectorArrayMap<float> nv_arr(nv, N);
    nm_arr = m_arr * beta1 + g_arr * (1.0f - beta1);
    nv_arr = v_arr + g_arr.square();
    nw_arr = w_arr + *lr * (nm_arr / c / (nv_arr.sqrt() + eps_hat) + weight_decay * w_arr);
}

template <typename T, class Context>
class DecayAdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  DecayAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        weight_decay_(this->template GetSingleArgument<float>("weight_decay", 0.0f)),
        bias_correction_first_(this->template GetSingleArgument<bool>("bias_correction_first", true)) {}

  bool RunOnDevice() override {
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(ITER, CPU));
    CAFFE_ENFORCE(Input(LR).numel() == 1);
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(PARAM).numel());
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_1).numel());
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_2).numel());
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    Output(OUTPUT_MOMENT_2)->ResizeLike(Input(MOMENT_2));

    const auto iter =
        OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

    const auto t = iter + 1;
    const auto c = (bias_correction_first_)? (T(1.) - std::pow(beta1_, t)) : 1.0;
    decay_adagrad_compute<Context>(
        Input(GRAD).numel(),
        Input(PARAM).template data<T>(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_1).template data<T>(),
        Input(MOMENT_2).template data<T>(),
        Output(OUTPUT_PARAM)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_2)->template mutable_data<T>(),
        beta1_,
        beta2_,
        epsilon_,
        weight_decay_,
        c,
        Input(LR).template data<T>(),
        &context_);

    return true;
  }

 protected:
  T beta1_{0.9};
  T beta2_{0.999};
  T epsilon_{1e-8};
  T weight_decay_{0.0};
  bool bias_correction_first_{true};
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2);
};

} // namespace caffe2
