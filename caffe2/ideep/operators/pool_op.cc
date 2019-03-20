#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

class IDEEPPoolOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws),
        training_mode_(
            OperatorBase::GetSingleArgument<int>("training_mode", 1)) {
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
    if (operator_def.type().substr(0, 7) == "MaxPool") {
      algo_ = ialgo::pooling_max;
    } else if (operator_def.type().substr(0, 11) == "AveragePool") {
      algo_ = ialgo::pooling_avg;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
    }
  }
  ~IDEEPPoolOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);
    auto Y_dims = CalcOutputDims(X, X.get_dim(1));
    mkldnn::prop_kind pk = training_mode_ ?
      mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_inference;

    ideep::pooling_forward::compute(X, Y_dims, *Y,
        stride_, kernel_, pad_tl(), pad_br(), algo_, pk);

    return true;
  }

 private:
  ialgo algo_;
  bool training_mode_;

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
  ~IDEEPPoolGradientOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& Y = Input(OUTPUT);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    ideep::pooling_backward::compute(dY, Y, X, *dX,
        stride_, kernel_, pad_tl(), pad_br(), algo_);

    return true;
  }

 private:
  ialgo algo_;

  INPUT_TAGS(INPUT, OUTPUT, OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(MaxPool, IDEEPPoolOp);
REGISTER_IDEEP_OPERATOR(MaxPoolGradient, IDEEPPoolGradientOp);

REGISTER_IDEEP_OPERATOR(AveragePool, IDEEPPoolOp);
REGISTER_IDEEP_OPERATOR(AveragePoolGradient, IDEEPPoolGradientOp);

} // namespace caffe2
