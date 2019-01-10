#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPSumOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSumOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {}
  virtual ~IDEEPSumOp() {}

  bool RunOnDevice() override {
    const auto &X = Input(INPUT0);
    auto* Y = Output(OUTPUT);

    if (InputSize() == 1) {
      ideep::direct_copy::compute(X, *Y);

    } else {
      vector<itensor> inputs;
      vector<float> scales (InputSize(), 1.0);
      for (int i = 0; i < InputSize(); ++i) {
        inputs.emplace_back(Input(i));
      }

      ideep::sum::compute(scales, inputs, *Y);
    }

    return true;
  }

 private:

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(Sum, IDEEPSumOp);

} // namespace caffe2
