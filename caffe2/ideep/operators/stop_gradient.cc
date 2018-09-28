#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPStopGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPStopGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {}
  virtual ~IDEEPStopGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    if (Y != &X) {
        ideep::direct_copy::compute(X, *Y);
    }
    return true;
  }

private:
  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(StopGradient, IDEEPStopGradientOp);

} // namespace caffe2
