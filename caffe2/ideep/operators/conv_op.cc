#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

class IDEEPConvOp : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW, "Unsupported storage order.");
    //OPERATOR_NEEDS_FEATURE(
    //    pad_l() == pad_r() && pad_t() == pad_b(),
    //    "Uneven padding not supported.");

    fusion_type_ = FUSION_UNKNOWN;
    last_input_ = BIAS_OR_INPUT_S;

    training_mode_ = OperatorBase::GetSingleArgument<int>("training_mode", 0);
    pk_ = training_mode_ ? iprop::forward_training : iprop::forward_inference;

    algo_ = ialgo::convolution_direct;
    auto conv_algorithm = OperatorBase::GetSingleArgument<int>(
        "conv_algorithm", CONV_ALGORITHM_AUTO);
    if (conv_algorithm == CONV_ALGORITHM_WINOGRAD) {
      algo_ = ialgo::convolution_winograd;
    }
#ifdef USE_EULER
    need_euler_ = OperatorBase::GetSingleArgument<int>("need_euler", 0);
    if (need_euler_) {
      eparam.nthreads = OperatorBase::GetSingleArgument<int>("nthreads", 0);
      eparam.tile_size = OperatorBase::GetSingleArgument<int>("tile_size", 5);
      eparam.conv_algorithm = OperatorBase::GetSingleArgument<int>("conv_algorithm", 0);
      eparam.execution_mode = execution_mode_trans(
              OperatorBase::GetSingleArgument<string>("execution_mode", "0x0000"));
      eparam.scratch_pad = NULL;
      eparam.flatting = OperatorBase::GetRepeatedArgument<int>("flatting", {1, 1});
      eparam.blocking = OperatorBase::GetRepeatedArgument<int>("blocking", {1, 1});
      eparam.partition = OperatorBase::GetRepeatedArgument<int>("partition", {1, 1});
    }
#endif
  }
  virtual ~IDEEPConvOp() {}

#ifdef USE_EULER
  int execution_mode_trans(string execution_mode_) {
    int execution_mode_tr = 0x0000;
    if (execution_mode_ == "0xa060") {
      execution_mode_tr = 0xa060;
    } else if (execution_mode_ == "0xc060") {
      execution_mode_tr = 0xc060;
    } else if (execution_mode_ == "0xa060") {
      execution_mode_tr = 0xa060;
    } else if (execution_mode_ == "0xb061") {
      execution_mode_tr = 0xb061;
    } else if (execution_mode_ == "0xa033") {
      execution_mode_tr = 0xa033;
    } else if (execution_mode_ == "0xb060") {
      execution_mode_tr = 0xb060;
    } else {
      printf("No support execution_mode");
    }
    return execution_mode_tr;
  }
#endif

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT_X);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);
    auto grouped = filter.is_grouped() ? 1 : 0;
    auto Y_dims_conv = CalcOutputDims(
        X,
        grouped ? (filter.get_dim(0) * filter.get_dim(1)) : filter.get_dim(0));

    CAFFE_ENFORCE(4 == X.ndims());
    CAFFE_ENFORCE(4 == filter.ndims() || (grouped && (group_ > 1)));
    CAFFE_ENFORCE(filter.get_dim(2 + grouped) == kernel_h());
    CAFFE_ENFORCE(filter.get_dim(3 + grouped) == kernel_w());
    CAFFE_ENFORCE(
        X.get_dim(1) == filter.get_dim(1 + grouped) * group_,
        "Convolution op: input channels does not match: # of input channels ",
        X.get_dim(1),
        " is not equal to kernel channels * group:",
        filter.get_dim(1 + grouped),
        "*",
        group_);

    bool input_changed = (cached_X_descriptor_ != X.get_descriptor());
    if (input_changed) {
      op_key_.clear();
      cached_X_descriptor_ = X.dup_descriptor();
    }

    if (need_euler_ && X.get_internal_format() != ideep::format::nChw16c) {
      X_in.init({X.get_dims(), X.get_data_type(), ideep::format::nChw16c});
      X_in.feed_from(X);
    } else {
      X_in = X;
    }

    bool weights_changed = (cached_weights_descriptor_ != filter.get_descriptor());
#ifdef USE_EULER
    if (need_euler_) {
      if (!training_mode_ && weights_changed) {
        op_key_.clear();

        cached_weights_descriptor_ = filter.dup_descriptor();
        auto filter_in = filter.as_weights();
        filter_in.make_group(group_);
        ideep::format filter_format = ideep::format::OIhw16i16o;
        ideep::tensor::descriptor expected_descriptor = {filter_in.get_dims(), filter_in.get_data_type(), filter_format};

        if (filter_in.get_descriptor() != expected_descriptor) {
          filter_.init(expected_descriptor);
          filter_.feed_from(filter_in);
        } else {
          filter_ = filter_in;
        }
      }

      if (InputSize() > last_input_) {
        ideep::convolution_forward::compute(
            op_key_,
	    comp,
            X_in,
            training_mode_ ? filter : filter_,
            Input(BIAS_OR_INPUT_S),
            Y_dims_conv,
            *Y,
            stride_,
            dilation_,
            pad_tl(),
            pad_br(),
            group_,
            eparam,
            dummy_scale_,
            dummy_scale_,
            dummy_scale_,
            attr_,
            algo_,
            pk_,
            ipadding::zero);
      } else {
        ideep::convolution_forward::compute(
            op_key_,
	    comp,
            X_in,
            training_mode_ ? filter : filter_,
            Y_dims_conv,
            *Y,
            stride_,
            dilation_,
            pad_tl(),
            pad_br(),
            group_,
            eparam,
            dummy_scale_,
            dummy_scale_,
            dummy_scale_,
            attr_,
            algo_,
            pk_,
            ipadding::zero);
      }

      if (fusion_type_ == FUSION_CONV_SUM
          && fusion_type_ == FUSION_CONV_SUM_RELU) {
        CAFFE_ENFORCE(Y == &(Input(InputSize() - 1)),
            "Convolution fusion op: InPlace is enforced for sum fusion.");
      }
      return true;
    }
#endif
    if (!training_mode_ && weights_changed) {
      op_key_.clear();
      cached_weights_descriptor_ = filter.dup_descriptor();
      auto filter_in = filter.as_weights();
      filter_in.make_group(group_);

      auto expected_descriptor =
          ideep::convolution_forward::expected_weights_descriptor(
              filter_in.get_dims(),
              idtype::f32,
              stride_,
              pad_tl(),
              pad_br(),
              dilation_,
              group_,
              algo_,
              pk_,
              idtype::f32,
              X.get_dims());
      if (filter_in.get_descriptor() != expected_descriptor) {
        filter_.init(expected_descriptor);
        filter_.feed_from(filter_in);
      } else {
        filter_ = filter_in;
      }
    }

    if (InputSize() > last_input_) {
      ideep::convolution_forward::compute(
          op_key_,
	  comp,
          X,
          training_mode_ ? filter : filter_,
          Input(BIAS_OR_INPUT_S),
          Y_dims_conv,
          *Y,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_,
          dummy_scale_,
          dummy_scale_,
          dummy_scale_,
          attr_,
          algo_,
          pk_);
    } else {
      ideep::convolution_forward::compute(
          op_key_,
	  comp,
          X,
          training_mode_ ? filter : filter_,
          Y_dims_conv,
          *Y,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_,
          dummy_scale_,
          dummy_scale_,
          dummy_scale_,
          attr_,
          algo_,
          pk_);
    }

    if (fusion_type_ == FUSION_CONV_SUM
        && fusion_type_ == FUSION_CONV_SUM_RELU) {
      CAFFE_ENFORCE(Y == &(Input(InputSize() - 1)),
          "Convolution fusion op: InPlace is enforced for sum fusion.");
    }

    return true;
  }

 protected:
  iprop pk_;
  ialgo algo_;
  iattr attr_;
  ikey op_key_;
  int last_input_;
  bool training_mode_;
  FusionType fusion_type_;
  ideep::convolution_forward::euler_params eparam;
  ideep::convolution_forward comp;
  int need_euler_;
  itensor X_in, filter_;
  iscale dummy_scale_;
  itensor::descriptor cached_X_descriptor_, cached_weights_descriptor_;

  INPUT_TAGS(INPUT_X, FILTER, BIAS_OR_INPUT_S, INPUT_S);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPConvFusionOp final : public IDEEPConvOp {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvFusionOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvOp(operator_def, ws) {
    CAFFE_ENFORCE(OperatorBase::HasArgument("fusion_type"),
          "You should specify the fusion type");
    fusion_type_ = static_cast<FusionType>(
        OperatorBase::GetSingleArgument<int>("fusion_type", FUSION_UNKNOWN));
    OPERATOR_NEEDS_FEATURE(
        fusion_type_ > FUSION_UNKNOWN && fusion_type_ < FUSION_MAX,
        "Undefined Conv fusion type.",
        fusion_type_);

    if (fusion_type_ == FUSION_CONV_RELU) {
      attr_ = iattr::fuse_relu();
      last_input_ = BIAS_OR_INPUT_S;
    } else if (fusion_type_ == FUSION_CONV_SUM) {
      attr_ = iattr::fuse_sum();
      last_input_ = INPUT_S;
    } else if (fusion_type_ == FUSION_CONV_SUM_RELU) {
      attr_ = iattr::residual();
      last_input_ = INPUT_S;
    } else if (fusion_type_ == FUSION_CONV_BRELU) {
      float bound_max = OperatorBase::GetSingleArgument<float>("bound_max", 0.0);
      float bound_min = OperatorBase::GetSingleArgument<float>("bound_min", 0.0);
      attr_ = iattr::fuse_relu(1.0, bound_max, bound_min, ialgo::eltwise_bounded_relu);
      last_input_ = BIAS_OR_INPUT_S;
    } else if (fusion_type_ == FUSION_CONV_SUM_BRELU) {
      float sum_scale=1.0, relu_scale=1.0;
      float bound_max = OperatorBase::GetSingleArgument<float>("bound_max", 0.0);
      float bound_min = OperatorBase::GetSingleArgument<float>("bound_min", 0.0);
      attr_ = iattr::residual(sum_scale, relu_scale, bound_max, bound_min, ialgo::eltwise_bounded_relu);
      last_input_ = INPUT_S;
    }
  }
  virtual ~IDEEPConvFusionOp() {}
};

const char* kConvFusionDoc = R"DOC(
Note that other parameters, such as the stride and
kernel size, or the pads' sizes in each direction are not necessary for input
because they are provided by the ConvPoolOpBase operator. Various dimension
checks are done implicitly, and the sizes are specified in the Input docs for
this operator. As is expected, the filter is convolved with a subset of the
image and the bias is added; this is done throughout the image data and the
output is computed. As a side note on the implementation layout:
conv_op_impl.h is the templated implementation of the conv_op.h file, which is
why they are separate files.
)DOC";

std::function<void(OpSchema&)> ConvFusionDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
The convolution fusion operator consumes an input vector, a {dim}filter blob,
a bias blob and another input vector and computes the output. This operator
gives the chance to fuse the ReLU or element-wise Sum with a convolution
operator. {conv_fusion_doc})DOC";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{conv_fusion_doc}", kConvFusionDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data blob from previous layer; has size (N x C x H x W), "
        "where N is the batch size, C is the number of channels, "
        "and H and W are the height and width. Note that this is for the NCHW "
        "usage. On the other hand, the NHWC Op has a different set of "
        "dimension constraints. ");
    schema.Input(
        1,
        "filter",
        "The filter blob that will be used in the "
        "convolutions; has size (M x C x kH x kW), where C is the number of "
        "channels, and kH and kW are the height and width of the kernel.");
    schema.Input(
        2,
        "bias",
        "The 1D bias blob that is added through the "
        "convolution; has size (M).");
    schema.Input(
        3,
        "S",
        "Input data blob for element-wise Sum fusion from previous layer; "
        "has the same size of convolution output. Its input index should "
        "be 2 if no bias for this convolution, and it MUST be inplace with "
        "output Y.");
    schema.Output(
        0,
        "Y",
        "Output data blob that contains the result of the "
        "convolution fusion. The output dimensions are functions of the kernel "
        "size, stride size, and pad lengths."
        "");
  };
}

OPERATOR_SCHEMA(ConvFusion)
    .NumInputs(2, 4)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .Arg("fusion_type", "Which fusion type is used")
    .AllowInplace({{2, 0}, {3, 0}})
    .FillUsing(ConvFusionDocGenerator(""));

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
  virtual ~IDEEPConvGradientOp() {}

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
REGISTER_IDEEP_OPERATOR(ConvFusion, IDEEPConvFusionOp);
REGISTER_IDEEP_OPERATOR(ConvGradient, IDEEPConvGradientOp);

} // namespace caffe2
