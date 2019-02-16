#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

class IDEEPConvOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws),
        training_mode_(
            OperatorBase::GetSingleArgument<int>("training_mode", 0)),
        conv_algorithm_(
            OperatorBase::GetSingleArgument<int>("conv_algorithm", CONV_ALGORITHM_AUTO)) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
  }
  ~IDEEPConvOp() override {}

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

    ideep::algorithm aalgorithm = ideep::algorithm::convolution_direct;
    if (conv_algorithm_ == CONV_ALGORITHM_WINOGRAD) {
      aalgorithm = ideep::algorithm::convolution_winograd;
    }

    bool weights_changed =
        (cached_weights_descriptor_ != filter.get_descriptor());
    if (weights_changed && !training_mode_) {
      cached_weights_descriptor_ = filter.get_descriptor();
      auto filter_in = filter;
      filter_in.make_group(group_);
      auto expected_descriptor =
          ideep::convolution_forward::expected_weights_descriptor(
              filter_in.get_dims(),
              filter_in.get_data_type(),
              stride_,
              pad_tl(),
              pad_br(),
              dilation_,
              group_,
              aalgorithm);
      filter_.init<ideep::utils::allocator, ideep::convolution_forward>(
          expected_descriptor);
      ideep::reorder::compute(filter_in, filter_);
    }

    // NB: actually, in the case when `group_ > 1`, IDEEP will create
    // an itermediate tensor for each run below. However, this tensor is merely
    // a view of of the weights and there is no actual data copy, so I'll let it
    // go now. If we encounter performance surprise when convoluting with group
    // > 1, this is the first place to check and we need to do the same cache
    // trick as above
    if (InputSize() > BIAS) {
      ideep::convolution_forward::compute(
          X,
          training_mode_ ? filter : filter_,
          Input(BIAS),
          Y_dims,
          *Y,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_,
          ideep::descriptor_group::attr_t(),
          aalgorithm);
    } else {
      ideep::convolution_forward::compute(
          X,
          training_mode_ ? filter : filter_,
          Y_dims,
          *Y,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_,
          ideep::descriptor_group::attr_t(),
          aalgorithm);
    }

    return true;
  }

 private:
  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);

  bool training_mode_;
  int conv_algorithm_;
  ideep::tensor filter_;
  ideep::tensor::descriptor cached_weights_descriptor_;
};

class IDEEPConvGradientOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    CAFFE_ENFORCE(
        OperatorBase::GetSingleArgument<int>("training_mode", 0),
        "In order to backward propagate weights correctly, "
        "please set training_mode=1");
  }
  ~IDEEPConvGradientOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dfilter = Output(FILTER_GRAD);

    if (no_bias_) {
      ideep::convolution_backward_weights::compute(
          X,
          dY,
          filter.get_dims(),
          *dfilter,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_);
    } else {
      auto* dbias = Output(BIAS_OR_INPUT_GRAD);
      ideep::convolution_backward_weights::compute(
          X,
          dY,
          filter.get_dims(),
          *dfilter,
          *dbias,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_);
    }

    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
      ideep::convolution_backward_data::compute(
          dY,
          filter,
          X.get_dims(),
          *dX,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_);
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

} // namespace caffe2
