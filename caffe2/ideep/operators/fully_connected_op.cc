#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPFullyConnectedOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPFullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        float16_compute_(
            OperatorBase::GetSingleArgument<bool>("float16_compute", false)) {}
  virtual ~IDEEPFullyConnectedOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);

    if (InputSize() > BIAS) {
      ideep::inner_product_forward::compute(X, filter, Input(BIAS), *Y);
    } else {
      ideep::inner_product_forward::compute(X, filter, *Y);
    }

    return true;
  }

 private:
  bool float16_compute_;

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPFullyConnectedGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPFullyConnectedGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        float16_compute_(
            OperatorBase::GetSingleArgument<bool>("float16_compute", false)) {}
  virtual ~IDEEPFullyConnectedGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dfilter = Output(FILTER_GRAD);
    auto* dbias = Output(BIAS_GRAD);

    ideep::inner_product_backward_weights::compute(X, dY, *dfilter, *dbias);

    if (OutputSize() > INPUT_GRAD) {
      ideep::inner_product_backward_data::compute(
          dY, filter, X.get_dims(), *Output(INPUT_GRAD));
    }

    return true;
  }

 private:
  bool float16_compute_;

  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_GRAD, INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(FC, IDEEPFullyConnectedOp);
REGISTER_IDEEP_OPERATOR(FCGradient, IDEEPFullyConnectedGradientOp);

} // namespace caffe2
