#include <caffe2/ideep/operators/conv_pool_base_op.h>

using namespace caffe2;

namespace {

class IDEEPInt8PoolOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8PoolOp(const OperatorDef& operator_def, Workspace* ws)
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
    if (operator_def.type().substr(0, 11) == "Int8MaxPool") {
      algo_ = ialgo::pooling_max;
    } else if (operator_def.type().substr(0, 15) == "Int8AveragePool") {
      algo_ = ialgo::pooling_avg_exclude_padding;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
    }
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPInt8PoolOp() override {}

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
                                    pad_tl(), pad_br(), algo_,
                                    iprop::forward_inference);

    return true;
  }

 private:
  ialgo algo_;
  itensor::descriptor cached_X_descriptor_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8MaxPool, DNNLOWP, IDEEPInt8PoolOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8AveragePool, DNNLOWP, IDEEPInt8PoolOp);

} // namespace
