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

REGISTER_IDEEP_OPERATOR(Relu, IDEEPReluOp);

} // namespace caffe2
