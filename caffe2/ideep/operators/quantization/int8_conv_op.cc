#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

class IDEEPInt8ConvOp : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) {
    OPERATOR_NEEDS_FEATURE(pad_l() == pad_r() && pad_t() == pad_b(),
                           "Uneven padding not supported.");
    algo_ = ialgo::convolution_direct;
    CAFFE_ENFORCE(zero_point_ == 128 || zero_point_ == 0);
    Y_scales_ = ConvertScales({scale_});
  }
  virtual ~IDEEPInt8ConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto &X = Input(INPUT_X);
    const auto &filter = Input(FILTER);
    auto *Y = Output(OUTPUT);
    auto Y_dims = CalcOutputDims(X, filter.get_dim(0));
    auto last_input =
      (fusion_type_ == FUSION_CONV_RELU || fusion_type_ == FUSION_UNKNOWN)
      ? BIAS_OR_INPUT_S : INPUT_S;

    CAFFE_ENFORCE(4 == X.ndims() && 4 == filter.ndims());
    CAFFE_ENFORCE(X.get_data_type() == idtype::s8
        || X.get_data_type() == idtype::u8);
    CAFFE_ENFORCE(filter.get_dim(2) == kernel_h());
    CAFFE_ENFORCE(filter.get_dim(3) == kernel_w());
    CAFFE_ENFORCE(
        X.get_dim(1) == filter.get_dim(1) * group_,
        "Convolution op: input channels does not match: # of input channels ",
        X.get_dim(1), " is not equal to kernel channels * group:",
        filter.get_dim(1), "*", group_);

    if (cached_weights_descriptor_ != filter.get_descriptor()) {
      op_key_.clear();
      cached_weights_descriptor_ = filter.dup_descriptor();
      CAFFE_ENFORCE(filter.get_data_type() == idtype::s8 && filter.has_scale());

      itensor filter_in;
      auto X_dt = X.get_data_type();
      lowp_kind_ = ilowp_kind::LOWP_U8S8;
      auto filter_scale = filter.get_scale();
      auto filter_mask =
        IDEEP_TENSOR_SCALE_MASK(filter_scale.size(), group_ > 1);
      if (X_dt == idtype::s8) {
        lowp_kind_ = ilowp_kind::LOWP_S8S8;
        filter_in = filter.as_weights().to_public();
      } else {
        filter_in = filter.as_weights();
      }
      filter_in.make_group(group_);

      auto expected_descriptor =
          ideep::convolution_forward::expected_weights_descriptor(
              filter_in.get_dims(), idtype::s8, stride_, pad_tl(), pad_br(),
              dilation_, group_, algo_, iprop::forward_inference, X_dt);
      if (filter_in.get_descriptor() != expected_descriptor) {
        filter_.init(expected_descriptor);
        if (filter_in.get_data_type() == filter_.get_data_type()) {
          ideep::reorder::compute(filter_in, filter_);
        } else {
          ideep::reorder::compute(
              filter_in, filter_, {filter_mask, filter_scale});
        }
        filter_.set_scale(filter_scale);
      } else {
        filter_ = filter_in;
      }

      if (InputSize() > last_input) {
        bias_ = Input(BIAS_OR_INPUT_S);
        CAFFE_ENFORCE(bias_.get_data_type() == idtype::s32);
      }
    }

    if (cached_X_descriptor_ != X.get_descriptor()) {
      op_key_.clear();
      cached_X_descriptor_ = X.dup_descriptor();
    }

    if (InputSize() > last_input) {
      ideep::convolution_forward::compute(
          op_key_, X, filter_, bias_, Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_,
          iscale(), iscale(), Y_scales_, attr_, algo_,
          iprop::forward_inference, ipadding::zero, lowp_kind_);
    } else {
      ideep::convolution_forward::compute(
          op_key_, X, filter_, Y_dims, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_,
          iscale(), iscale(), Y_scales_, attr_, algo_,
          iprop::forward_inference, ipadding::zero, lowp_kind_);
    }

    if (fusion_type_ != FUSION_CONV_RELU && fusion_type_ != FUSION_UNKNOWN) {
      CAFFE_ENFORCE(
          Y == &(Input(InputSize() - 1)),
          "Convolution fusion op: InPlace is enforced for sum fusion.");
    }

    return true;
  }

 protected:
  iattr attr_;
  ialgo algo_;
  ikey op_key_;
  float scale_;
  int32_t zero_point_;
  ilowp_kind lowp_kind_;
  FusionType fusion_type_ {FUSION_UNKNOWN};

  itensor filter_, bias_;
  iscale  Y_scales_;
  itensor::descriptor cached_X_descriptor_, cached_weights_descriptor_;

  INPUT_TAGS(INPUT_X, FILTER, BIAS_OR_INPUT_S, INPUT_S);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPInt8ConvReluOp final : public IDEEPInt8ConvOp {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPInt8ConvOp(operator_def, ws) {
    CAFFE_ENFORCE(zero_point_ == 0);
    attr_ = iattr::fuse_relu();
    fusion_type_ = FUSION_CONV_RELU;
  }
  virtual ~IDEEPInt8ConvReluOp() {}
};

class IDEEPInt8ConvSumOp final : public IDEEPInt8ConvOp {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvSumOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPInt8ConvOp(operator_def, ws) {
    attr_ = iattr::fuse_sum();
    fusion_type_ = FUSION_CONV_SUM;
  }
  virtual ~IDEEPInt8ConvSumOp() {}
};

class IDEEPInt8ConvSumReluOp final : public IDEEPInt8ConvOp {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPInt8ConvSumReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPInt8ConvOp(operator_def, ws) {
    attr_ = iattr::residual();
    fusion_type_ = FUSION_CONV_SUM_RELU;
  }
  virtual ~IDEEPInt8ConvSumReluOp() {}
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Conv, DNNLOWP, IDEEPInt8ConvOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8ConvRelu, DNNLOWP, IDEEPInt8ConvReluOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8ConvSum, DNNLOWP, IDEEPInt8ConvSumOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8ConvSumRelu, DNNLOWP, IDEEPInt8ConvSumReluOp);

OPERATOR_SCHEMA(Int8ConvSum)
    .NumInputs(2, 4)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .AllowInplace({{2, 0}, {3, 0}});

OPERATOR_SCHEMA(Int8ConvSumRelu)
    .NumInputs(2, 4)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .AllowInplace({{2, 0}, {3, 0}});

} // namespace caffe2
