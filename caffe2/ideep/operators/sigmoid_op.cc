#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPSigmoidOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSigmoidOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {
  }
  ~IDEEPSigmoidOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    ideep::eltwise_forward::compute(
        X, *Y, ialgo::eltwise_logistic, iprop::forward_training);

    return true;
  }

 private:
  
  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPSigmoidGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSigmoidGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {
  }
  ~IDEEPSigmoidGradientOp() override {}

  bool RunOnDevice() override {
    const auto& Y = Input(OUTPUT);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    ideep::eltwise_backward::compute(Y, dY, *dX, ialgo::eltwise_logistic);

    return true;
  }

 private:

  INPUT_TAGS(OUTPUT, OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(Sigmoid, IDEEPSigmoidOp);
REGISTER_IDEEP_OPERATOR(SigmoidGradient, IDEEPSigmoidGradientOp);

} // namespace caffe2
