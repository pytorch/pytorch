#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPReluOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws), alpha_(0.0) {
    // Figure out the Relu descriptor.
    if (operator_def.type().substr(0, 4) == "Relu") {
      alpha_ = 0.0;
    } else if (operator_def.type().substr(0, 9) == "LeakyRelu") {
      if (HasArgument("alpha")) {
        alpha_ = static_cast<float>(
            OperatorBase::GetSingleArgument<float>("alpha", 0.01));
      }
    } else {
      LOG(FATAL) << "Unsupported Relu method: " << operator_def.type();
    }
  }
  ~IDEEPReluOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    ideep::eltwise_forward::compute(
        X, *Y, ialgo::eltwise_relu, iprop::forward_training, alpha_);

    return true;
  }

 private:
  float alpha_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPReluGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws), alpha_(0.0) {
    // Figure out the Relu descriptor.
    if (operator_def.type().substr(0, 12) == "ReluGradient") {
      alpha_ = 0.0;
    } else if (operator_def.type().substr(0, 17) == "LeakyReluGradient") {
      if (HasArgument("alpha")) {
        alpha_ = static_cast<float>(
            OperatorBase::GetSingleArgument<float>("alpha", 0.01));
      }
    } else {
      LOG(FATAL) << "Unsupported Relu method: " << operator_def.type();
    }
  }
  ~IDEEPReluGradientOp() override {}

  bool RunOnDevice() override {
    const auto& Y = Input(OUTPUT);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    ideep::eltwise_backward::compute(Y, dY, *dX, ialgo::eltwise_relu, alpha_);

    return true;
  }

 private:
  float alpha_;

  INPUT_TAGS(OUTPUT, OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(Relu, IDEEPReluOp);
REGISTER_IDEEP_OPERATOR(ReluGradient, IDEEPReluGradientOp);

REGISTER_IDEEP_OPERATOR(LeakyRelu, IDEEPReluOp);
REGISTER_IDEEP_OPERATOR(LeakyReluGradient, IDEEPReluGradientOp);

} // namespace caffe2
