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

REGISTER_IDEEP_OPERATOR(Conv, IDEEPConvOp);

}  // namespace caffe2
