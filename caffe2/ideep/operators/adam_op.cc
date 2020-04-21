#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

void adam_ideep_update(
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
    float correction,
    const float* lr) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    ng[i] = lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
  }
}

void adam_ideep_compute(
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
    float correction,
    const float* lr) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    nw[i] = w[i] + lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
  }
}

void adam_ideep_compute_output_grad(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float* ng,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr) {

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ngi = ng[i] = correction * mi / (std::sqrt(vi) + eps_hat);
    nw[i] = w[i] + lr[0] * ngi;
  }
}

template <typename T>
class IDEEPAdamOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPAdamOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        beta1_(OperatorBase::GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(OperatorBase::GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}
  bool RunOnDevice() override {
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(ITER, CPU));
    const auto& params = Input(PARAM);
    const auto& moment_1 = Input(MOMENT_1);
    const auto& moment_2 = Input(MOMENT_2);
    const auto& grad = Input(GRAD);
    // TODO: Use itensor after 0-dim is supported. Now use CPU tensor.
    const auto& lr = OperatorBase::Input<TensorCPU>(LR, CPU);
    auto* out_params = Output(OUTPUT_PARAM);
    auto* out_moment1 = Output(OUTPUT_MOMENT_1);
    auto* out_moment2 = Output(OUTPUT_MOMENT_2);

    CAFFE_ENFORCE(lr.size() == 1);
    CAFFE_ENFORCE(grad.get_nelems() == params.get_nelems());
    CAFFE_ENFORCE(grad.get_nelems() == moment_1.get_nelems());
    CAFFE_ENFORCE(grad.get_nelems() == moment_2.get_nelems());
    if (params != *out_params)
        out_params->init(params.get_descriptor());
    if (moment_1 != *out_moment1)
        out_moment1->init(moment_1.get_descriptor());
    if (moment_2 != *out_moment2)
        out_moment2->init(moment_2.get_descriptor());
    const auto w = static_cast<float *>(params.get_data_handle());
    const auto g = static_cast<float *>(grad.get_data_handle());
    const auto m = static_cast<float *>(moment_1.get_data_handle());
    const auto v = static_cast<float *>(moment_2.get_data_handle());
    auto nw = static_cast<float *>(out_params->get_data_handle());
    auto nm = static_cast<float *>(out_moment1->get_data_handle());
    auto nv = static_cast<float *>(out_moment2->get_data_handle());
    const auto nlr = lr.template data<T>();
    const auto iter =
        OperatorBase::Input<TensorCPU>(ITER, CPU).template data<int64_t>()[0];
    const auto t = iter + 1;
    const auto correction =
        std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));
    if (OutputSize() == 3) {
      adam_ideep_compute(
          grad.get_nelems(),
          w,
          g,
          m,
          v,
          nw,
          nm,
          nv,
          beta1_,
          beta2_,
          epsilon_,
          correction,
          nlr);
    } else {
      auto* out_grad = Output(OUTPUT_GRAD);
      if (grad != *out_grad)
        out_grad->init(grad.get_descriptor());
      auto ng = static_cast<float *>(out_grad->get_data_handle());
      adam_ideep_compute_output_grad(
          grad.get_nelems(),
          w,
          g,
          m,
          v,
          nw,
          nm,
          nv,
          ng,
          beta1_,
          beta2_,
          epsilon_,
          correction,
          nlr);
    }

    return true;
  }

 protected:
  T beta1_{0.9};
  T beta2_{0.999};
  T epsilon_{1e-8};
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2, OUTPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(Adam, IDEEPAdamOp<float>);

} // namespace
