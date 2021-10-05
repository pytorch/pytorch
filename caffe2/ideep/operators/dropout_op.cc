#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

class IDEEPDropoutOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPDropoutOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)),
        is_test_(
            OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPDropoutOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    if (is_test_) {
      if (Y != &X) {
        ideep::direct_copy::compute(X, *Y);
      }
      return true;
    }

    auto* mask = Output(MASK);
    ideep::dropout_forward::compute(X, ratio_, *Y, *mask);

    return true;
  }

 private:
  float ratio_;
  bool is_test_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT, MASK);
};

class IDEEPDropoutGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPDropoutGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        ratio_(OperatorBase::GetSingleArgument<float>("ratio", 0.5)),
        is_test_(
            OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPDropoutGradientOp() override {}

  bool RunOnDevice() override {
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    if (is_test_) {
      if (dX != &dY) {
        ideep::direct_copy::compute(dY, *dX);
      }
      return true;
    }

    const auto& mask = Input(MASK);
    ideep::dropout_backward::compute(mask, dY, *dX);

    return true;
  }

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  float ratio_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool is_test_;

  INPUT_TAGS(OUTPUT_GRAD , MASK);
  OUTPUT_TAGS(INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(Dropout, IDEEPDropoutOp);
REGISTER_IDEEP_OPERATOR(DropoutGrad, IDEEPDropoutGradientOp);

} // namespace
