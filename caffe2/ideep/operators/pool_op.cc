#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

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
    // Figure out the pooling descriptor.
    if (operator_def.type().substr(0, 7) == "MaxPool") {
      algo_ = ialgo::pooling_max;
    } else if (operator_def.type().substr(0, 11) == "AveragePool") {
      algo_ = ialgo::pooling_avg;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
    }
  }
  virtual ~IDEEPPoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);
    auto Y_dims = CalcOutputDims(X, X.get_dim(1));

    ideep::pooling_forward::compute(X, Y_dims, *Y,
        stride_, kernel_, pad_tl(), pad_br(), algo_);

    return true;
  }

 private:
  ialgo algo_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(MaxPool, IDEEPPoolOp);
REGISTER_IDEEP_OPERATOR(AveragePool, IDEEPPoolOp);

} // namespace caffe2
