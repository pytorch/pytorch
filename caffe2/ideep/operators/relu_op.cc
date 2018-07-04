#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPReluOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {}
  virtual ~IDEEPReluOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    ideep::eltwise_forward::compute(X, *Y);

    return true;
  }

 private:

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPReluGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {}
  virtual ~IDEEPReluGradientOp() {}

  bool RunOnDevice() override {
    const auto& Y = Input(OUTPUT);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    ideep::eltwise_backward::compute(Y, dY, *dX);

    return true;
  }

 private:

  INPUT_TAGS(OUTPUT, OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(Relu, IDEEPReluOp);
REGISTER_IDEEP_OPERATOR(ReluGradient, IDEEPReluGradientOp);

} // namespace caffe2
