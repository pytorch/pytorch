#include <caffe2/ideep/operators/conv_pool_base_op.h>

using namespace caffe2;

namespace {

class IDEEPPoolOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {
    CAFFE_ENFORCE(
        (dilation_h() == 1) && (dilation_w() == 1),
        "Pooling op does not support dilation right now.");
    if (!global_pooling_) {
      CAFFE_ENFORCE(
          pad_t() < kernel_h() && pad_b() < kernel_h() &&
              pad_l() < kernel_w() && pad_r() < kernel_w(),
          "Pad should be smaller than kernel.");
    }

    bool training_mode = OperatorBase::GetSingleArgument<int>("training_mode", 1);
    pk_ = training_mode ? iprop::forward_training : iprop::forward_inference;

    // Figure out the pooling descriptor.
    if (operator_def.type().substr(0, 7) == "MaxPool") {
      algo_ = ialgo::pooling_max;
    } else if (operator_def.type().substr(0, 11) == "AveragePool") {
      algo_ = ialgo::pooling_avg;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
    }
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPPoolOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);
    auto Y_dims = CalcOutputDims(X, X.get_dim(1));

    if (cached_X_descriptor_ != X.get_descriptor()) {
      cached_X_descriptor_ = X.dup_descriptor();
    }

    ideep::pooling_forward::compute(X, Y_dims, *Y,
                                    {stride_.begin(), stride_.end()},
                                    {kernel_.begin(), kernel_.end()},
                                    pad_tl(), pad_br(), algo_, pk_);

    return true;
  }

 private:
  iprop pk_;
  ialgo algo_;
  itensor::descriptor cached_X_descriptor_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPPoolGradientOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPPoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {
    CAFFE_ENFORCE(
        (dilation_h() == 1) && (dilation_w() == 1),
        "Pooling op does not support dilation right now.");
    if (!global_pooling_) {
      CAFFE_ENFORCE(
          pad_t() < kernel_h() && pad_b() < kernel_h() &&
              pad_l() < kernel_w() && pad_r() < kernel_w(),
          "Pad should be smaller than kernel.");
    }
    // Figure out the pooling descriptor.
    if (operator_def.type().substr(0, 15) == "MaxPoolGradient") {
      algo_ = ialgo::pooling_max;
    } else if (operator_def.type().substr(0, 19) == "AveragePoolGradient") {
      algo_ = ialgo::pooling_avg;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
    }
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPPoolGradientOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& Y = Input(OUTPUT);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    ideep::pooling_backward::compute(dY, Y, X, *dX,
                                     {stride_.begin(), stride_.end()},
                                     {kernel_.begin(), kernel_.end()},
                                     pad_tl(), pad_br(), algo_);

    return true;
  }

 private:
  ialgo algo_;

  INPUT_TAGS(INPUT, OUTPUT, OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(MaxPool, IDEEPPoolOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(MaxPoolGradient, IDEEPPoolGradientOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(AveragePool, IDEEPPoolOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(AveragePoolGradient, IDEEPPoolGradientOp);

} // namespace
