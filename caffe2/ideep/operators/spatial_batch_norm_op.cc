#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPSpatialBNOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSpatialBNOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        is_test_(OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        momentum_(OperatorBase::GetSingleArgument<float>("momentum", 0.9)) {
    CAFFE_ENFORCE(
        (is_test_ && OutputSize() > OUTPUT)
          || (!is_test_ && OutputSize() > SAVED_VAR));
    CAFFE_ENFORCE_GT(epsilon_, 0);
    CAFFE_ENFORCE_GE(momentum_, 0);
    CAFFE_ENFORCE_LE(momentum_, 1);
  }
  ~IDEEPSpatialBNOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& scale = Input(SCALE);
    const auto& bias = Input(BIAS);
    auto* Y = Output(OUTPUT);

    DCHECK_EQ(scale.ndims(), 1);
    DCHECK_EQ(bias.ndims(), 1);
    DCHECK_EQ(scale.get_dim(0), X.get_dim(1));
    DCHECK_EQ(bias.get_dim(0), X.get_dim(1));

    if (is_test_) {
      const auto& est_mean = Input(EST_MEAN);
      const auto& est_var = Input(EST_VAR);
      ideep::batch_normalization_forward_inference::compute(
          X, est_mean, est_var, scale, bias, *Y, epsilon_);
    } else {
      auto* saved_mean = Output(SAVED_MEAN);
      auto* saved_var = Output(SAVED_VAR);
      auto* running_mean = Output(RUNNING_MEAN);
      auto* running_var = Output(RUNNING_VAR);
      ideep::batch_normalization_forward_training::compute(
          X, scale, bias, *Y, *saved_mean, *saved_var,
          *running_mean, *running_var, momentum_, epsilon_);
    }

    return true;
  }

 private:
  bool is_test_;
  double epsilon_;
  double momentum_;

  INPUT_TAGS(INPUT, SCALE, BIAS, EST_MEAN, EST_VAR);
  OUTPUT_TAGS(OUTPUT, RUNNING_MEAN, RUNNING_VAR, SAVED_MEAN, SAVED_VAR);
};

class IDEEPSpatialBNGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSpatialBNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)) {
    CAFFE_ENFORCE(InputSize() > SAVED_VAR);
    CAFFE_ENFORCE(OutputSize() > BIAS_GRAD);
  }
  ~IDEEPSpatialBNGradientOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& scale = Input(SCALE);
    const auto& dY = Input(OUTPUT_GRAD);
    const auto& saved_mean = Input(SAVED_MEAN);
    const auto& saved_var = Input(SAVED_VAR);
    auto* dX = Output(INPUT_GRAD);
    auto* dscale = Output(SCALE_GRAD);
    auto* dbias = Output(BIAS_GRAD);

    ideep::batch_normalization_backward::compute(
        X, saved_mean, saved_var, dY, scale,
        *dX, *dscale, *dbias, epsilon_);

    return true;
  }

 private:
  double epsilon_;

  INPUT_TAGS(INPUT, SCALE, OUTPUT_GRAD, SAVED_MEAN, SAVED_VAR);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
};

REGISTER_IDEEP_OPERATOR(SpatialBN, IDEEPSpatialBNOp);
REGISTER_IDEEP_OPERATOR(SpatialBNGradient, IDEEPSpatialBNGradientOp)

}  // namespace caffe2
