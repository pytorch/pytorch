#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

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
  virtual ~IDEEPInt8FullyConnectedOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);
    bool with_bias=(InputSize() > BIAS);

    CAFFE_ENFORCE(X.has_scale());
    CAFFE_ENFORCE(X.get_data_type() == idtype::s8 || X.get_data_type() == idtype::u8);

    //itensor X_in = X;
    if (X.get_internal_format() != ideep::format::nhwc) {
      X_in.init({X.get_dims(), X.get_data_type(), ideep::format::nhwc});
      X_in.set_scale(X.get_scale());
      X_in.feed_from_src(X);
    } else {
      X_in = X;
    }

    auto X_dims = CanonicalDims(X_in.get_dims(), axis_);
    if (X_in.get_dims() != X_dims) {
      X_in.reshape(X_dims);
    }

    if (cached_X_descriptor_ != X.get_descriptor()) {
      op_key_.clear();
      cached_X_descriptor_ = X.dup_descriptor();
    }

    if (cached_weights_descriptor_ != filter.get_descriptor()) {
      op_key_.clear();
      cached_weights_descriptor_ = filter.dup_descriptor();
      CAFFE_ENFORCE(filter.get_data_type() == idtype::s8 && filter.has_scale());

      itensor filter_in = filter;
      auto filter_dims = CanonicalDims(filter_in.get_dims(), axis_w_);
      if (filter_in.get_dims() != filter_dims) {
        filter_in.reshape(filter_dims);
      }

      auto X_dt = X.get_data_type();
      lowp_kind_ = ilowp_kind::LOWP_U8S8;
      auto filter_scale = filter_in.get_scale();
      auto filter_mask =
        IDEEP_TENSOR_SCALE_MASK(filter_scale.size(), 0);
      if (X_dt == idtype::s8) {
        lowp_kind_ = ilowp_kind::LOWP_S8S8;
        filter_ = filter_in.as_weights().to_public();
      } else {
        filter_ = filter_in.as_weights();
      }

      auto expected_descriptor =
          ideep::inner_product_forward::expected_weights_descriptor(filter_in.get_dims(), idtype::s8, X_dt);
      if (filter_in.get_descriptor() != expected_descriptor) {
        filter_.init(expected_descriptor);
        filter_.set_scale(filter_scale);
        filter_.feed_from(filter_in);
      }

      if (with_bias) {
        auto bias = Input(BIAS);
        bias_.init({bias.get_dims(), idtype::s32});
        iscale bias_scales (filter_.get_scale());
        for (auto &scale : bias_scales) { scale *= X.get_scale()[0]; }
        bias_.set_scale(bias_scales);
        bias_.feed_from(bias);
      }
    }

    if (with_bias) {
      ideep::inner_product_forward::compute(
          op_key_, comp, X_in, filter_, bias_, *Y,
          iscale(), iscale(), Y_scales_, attr_, iprop::forward_inference, lowp_kind_);
    } else {
      ideep::inner_product_forward::compute(
          op_key_, comp, X_in, filter_, *Y,
          iscale(), iscale(), Y_scales_, attr_, iprop::forward_inference, lowp_kind_);
    }

    return true;
  }

private:
  iattr attr_;
  size_t axis_{1};
  size_t axis_w_{1};
  float scale_;
  int32_t zero_point_;
  ilowp_kind lowp_kind_;

  ikey op_key_;
  idtype Y_data_type_;
  itensor X_in, filter_, bias_;
  iscale  Y_scales_;
  itensor::descriptor cached_X_descriptor_, cached_weights_descriptor_;
  ideep::inner_product_forward comp;

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8FC, DNNLOWP, IDEEPInt8FullyConnectedOp);

} // namespace caffe2
