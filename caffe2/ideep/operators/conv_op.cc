#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

class IDEEPConvOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
  }
  virtual ~IDEEPConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);
    auto Y_dims = CalcOutputDims(X, filter.get_dim(0));

    CAFFE_ENFORCE(4 == X.ndims());
    CAFFE_ENFORCE(4 == filter.ndims());
    CAFFE_ENFORCE(filter.get_dim(2) == kernel_h());
    CAFFE_ENFORCE(filter.get_dim(3) == kernel_w());
    CAFFE_ENFORCE(
        X.get_dim(1) == filter.get_dim(1) * group_,
        "Convolution op: input channels does not match: # of input channels ",
        X.get_dim(1),
        " is not equal to kernel channels * group:",
        filter.get_dim(1),
        "*",
        group_);

    if (InputSize() > BIAS) {
      ideep::convolution_forward::compute(
          X, filter, Input(BIAS), Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_);
    } else {
      ideep::convolution_forward::compute(
          X, filter, Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_);
    }

    return true;
  }

 private:

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPConvGradientOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvGradientOp(const OperatorDef& operator_def, Workspace *ws)
    : IDEEPConvPoolOpBase(operator_def, ws),
      no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
  }
  virtual ~IDEEPConvGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dfilter = Output(FILTER_GRAD);

    if (no_bias_) {
      ideep::convolution_backward_weights::compute(
          X, dY, filter.get_dims(), *dfilter,
          stride_, dilation_, pad_tl(), pad_br(), group_);
    } else {
      auto* dbias = Output(BIAS_OR_INPUT_GRAD);
      ideep::convolution_backward_weights::compute(
          X, dY, filter.get_dims(), *dfilter, *dbias,
          stride_, dilation_, pad_tl(), pad_br(), group_);
    }

    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
      ideep::convolution_backward_data::compute(
          dY, filter, X.get_dims(), *dX,
          stride_, dilation_, pad_tl(), pad_br(), group_);
    }

    return true;
  }

 private:
  bool no_bias_;

  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};


REGISTER_IDEEP_OPERATOR(Conv, IDEEPConvOp);
REGISTER_IDEEP_OPERATOR(ConvGradient, IDEEPConvGradientOp);

}  // namespace caffe2
