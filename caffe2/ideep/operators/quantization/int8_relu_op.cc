#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

class IDEEPInt8ReluOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPInt8ReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws), alpha_(0.0) {
    // Figure out the Relu descriptor.
    if (operator_def.type().substr(0, 8) == "Int8Relu") {
      alpha_ = 0.0;
    } else {
      LOG(FATAL) << "Unsupported Relu method: " << operator_def.type();
    }
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPInt8ReluOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    ideep::eltwise_forward::compute(
        X, *Y, ialgo::eltwise_relu, iprop::forward_inference, alpha_);

    return true;
  }

 private:
  float alpha_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Relu, DNNLOWP, IDEEPInt8ReluOp);

} // namespace
