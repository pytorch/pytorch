#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

USE_IDEEP_DEF_ALIASES();

class IDEEPInt8FullyConnectedOp final : public IDEEPOperator {
public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPInt8FullyConnectedOp(const OperatorDef &operator_def, Workspace *ws)
      : IDEEPOperator(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)),
        scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) {
    CAFFE_ENFORCE(zero_point_ == 128 || zero_point_ == 0);
    if (zero_point_ == 0) {
      Y_data_type_ = idtype::u8;
    } else {
      Y_data_type_ = idtype::s8;
    }
    Y_scales_ = ConvertScales({scale_});
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPInt8FullyConnectedOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);

    itensor X_in = X;
    auto X_dims = CanonicalDims(X_in.get_dims(), axis_);
    if (X_in.get_dims() != X_dims) {
      X_in.reshape(X_dims);
    }

    if (cached_X_descriptor_ != X.get_descriptor()) {
      cached_X_descriptor_ = X.dup_descriptor();
      Y_.init({{X.get_dim(0), filter.get_dim(0)}, idtype::f32});
    }

    if (cached_weights_descriptor_ != filter.get_descriptor()) {
      cached_weights_descriptor_ = filter.dup_descriptor();
      CAFFE_ENFORCE(filter.get_data_type() == idtype::s8 && filter.has_scale());

      // INT8 FC is not supported so far.
      filter_ = filter.to_public();
      auto filter_dims = CanonicalDims(filter_.get_dims(), axis_w_);
      if (filter_.get_dims() != filter_dims) {
        filter_.reshape(filter_dims);
      }

      if (InputSize() > BIAS) {
        bias_ = Input(BIAS).to_public();
      }

      Y_.init({{X.get_dim(0), filter.get_dim(0)}, idtype::f32});
    }

    X_in = X_in.to_public();
    if (InputSize() > BIAS) {
      ideep::inner_product_forward::compute(
          X_in, filter_, bias_, Y_);
    } else {
      ideep::inner_product_forward::compute(X_in, filter_, Y_);
    }
    Y->init({Y_.get_dims(), Y_data_type_});
    Y->set_scale(Y_scales_);
    Y->feed_from(Y_);
    return true;
  }

private:
  size_t axis_{1};
  size_t axis_w_{1};
  float scale_;
  int32_t zero_point_;

  idtype Y_data_type_;
  itensor filter_, bias_, Y_;
  iscale  Y_scales_;
  itensor::descriptor cached_X_descriptor_, cached_weights_descriptor_;

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8FC, DNNLOWP, IDEEPInt8FullyConnectedOp);

} // namespace
