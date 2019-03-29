#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

void momentum_sgd_update(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    const bool nesterov,
    float* param) {
  const float LR = lr[0];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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

class IDEEPMomentumSGDOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPMomentumSGDOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        momentum_(OperatorBase::GetSingleArgument<float>("momentum", 0.0)),
        nesterov_(OperatorBase::GetSingleArgument<int>("nesterov", 0)) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(GRAD).get_nelems() == Input(MOMENTUM).get_nelems());
    if (Input(GRAD) != *Output(OUTPUT_GRAD)) {
      Output(OUTPUT_GRAD)->reinit(Input(GRAD).get_descriptor());
    }
    if (Input(MOMENTUM) != *Output(OUTPUT_MOMENTUM)) {
      Output(OUTPUT_MOMENTUM)->reinit(Input(MOMENTUM).get_descriptor());
    }

    // TODO: Use itensor after 0-dim is supported. Now use CPU tensor.
    const auto& lr = OperatorBase::Input<TensorCPU>(LR, CPU);
    CAFFE_ENFORCE(lr.numel() == 1);

    momentum_sgd_update(
        Input(GRAD).get_nelems(),
        static_cast<float*>(Input(GRAD).get_data_handle()),
        static_cast<float*>(Input(MOMENTUM).get_data_handle()),
        static_cast<float*>(Output(OUTPUT_GRAD)->get_data_handle()),
        static_cast<float*>(Output(OUTPUT_MOMENTUM)->get_data_handle()),
        lr.template data<float>(),
        momentum_,
        nesterov_,
        nullptr);
    return true;
  }

 protected:
  float momentum_{0.9};
  bool nesterov_;
  INPUT_TAGS(GRAD, MOMENTUM, LR);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENTUM);
};

class IDEEPMomentumSGDUpdateOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();
  IDEEPMomentumSGDUpdateOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        momentum_(OperatorBase::GetSingleArgument<float>("momentum", 0.0)),
        nesterov_(OperatorBase::GetSingleArgument<int>("nesterov", 0)) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(GRAD).get_nelems() == Input(MOMENTUM).get_nelems());
    if (Input(GRAD) != *Output(OUTPUT_GRAD)) {
      Output(OUTPUT_GRAD)->reinit(Input(GRAD).get_descriptor());
    }
    if (Input(MOMENTUM) != *Output(OUTPUT_MOMENTUM)) {
      Output(OUTPUT_MOMENTUM)->reinit(Input(MOMENTUM).get_descriptor());
    }

    // TODO: Use itensor after 0-dim is supported. Now use CPU tensor.
    const auto& lr = OperatorBase::Input<TensorCPU>(LR, CPU);
    CAFFE_ENFORCE(lr.numel() == 1);

    momentum_sgd_update(
        Input(GRAD).get_nelems(),
        static_cast<float*>(Input(GRAD).get_data_handle()),
        static_cast<float*>(Input(MOMENTUM).get_data_handle()),
        static_cast<float*>(Output(OUTPUT_GRAD)->get_data_handle()),
        static_cast<float*>(Output(OUTPUT_MOMENTUM)->get_data_handle()),
        lr.template data<float>(),
        momentum_,
        nesterov_,
        static_cast<float*>(Output(OUTPUT_PARAM)->get_data_handle()));
    return true;
  }

 protected:
  float momentum_{0.9};
  bool nesterov_;
  INPUT_TAGS(GRAD, MOMENTUM, LR, PARAM);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENTUM, OUTPUT_PARAM);
};

REGISTER_IDEEP_OPERATOR(MomentumSGD, IDEEPMomentumSGDOp);
REGISTER_IDEEP_OPERATOR(MomentumSGDUpdate, IDEEPMomentumSGDUpdateOp);

} // namespace caffe2
