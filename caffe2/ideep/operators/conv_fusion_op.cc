#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

class IDEEPConvFusionOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  IDEEPConvFusionOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");

    fusion_type_ = static_cast<FusionType>(
        OperatorBase::GetSingleArgument<int>("fusion_type", 0));
    OPERATOR_NEEDS_FEATURE(
        fusion_type_ > FUSION_UNKNOWN && fusion_type_ < FUSION_MAX,
        "Undefined Conv fusion type.",
        fusion_type_);

    // Check kernel only if we are doing conv. The reason is that a
    // few other ops, like PadImage, are also using this base class. We really
    // need to clean this up.
    for (int dim = 0; dim < kernel_.size(); ++dim) {
      CAFFE_ENFORCE_GE(pads_[dim], 0);
      CAFFE_ENFORCE_GE(pads_[kernel_.size() + dim], 0);
      CAFFE_ENFORCE(
          kernel_[dim],
          "If you are doing convolution, you will need to set "
          "explicitly the kernel size.");
    }

    need_quantize_ = OperatorBase::GetSingleArgument<int>("need_quantize", 0);
    if (need_quantize_) {
      FillScales(X_scales_, IDEEP_ABSMAX_I(INPUT_X), idtype::u8);
      FillScales(Y_absmax_, IDEEP_ABSMAX_O(OUTPUT), idtype::f32);
    }

    training_mode_ = OperatorBase::GetSingleArgument<int>("training_mode", 0);
    pk_ = training_mode_ ? iprop::forward_training : iprop::forward_inference;

    algo_ = ialgo::convolution_direct;
    auto conv_algorithm = OperatorBase::GetSingleArgument<int>(
        "conv_algorithm", CONV_ALGORITHM_AUTO);
    if (conv_algorithm == CONV_ALGORITHM_WINOGRAD) {
      algo_ = ialgo::convolution_winograd;
    }
  }
  virtual ~IDEEPConvFusionOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT_X);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);
    auto grouped = filter.is_grouped() ? 1 : 0;
    auto Y_dims_conv = CalcOutputDims(
        X, grouped ? (filter.get_dim(0) * filter.get_dim(1))
        : filter.get_dim(0));
    auto last_input = [this]() {
      return (fusion_type_ == FUSION_CONV_RELU) ? BIAS_OR_INPUT_S : INPUT_S;
    };
    auto Y_dt = [this, Y]() {
      return (fusion_type_ == FUSION_CONV_RELU) ? idtype::u8
        : (((fusion_type_ == FUSION_CONV_SUM)
            || (fusion_type_ == FUSION_CONV_SUM_RELU))
            ? Y->get_data_type() : idtype::f32);
    };
    auto attr = [this]() {
      return (fusion_type_ == FUSION_CONV_RELU)
          ? iattr::fuse_relu()
          : ((fusion_type_ == FUSION_CONV_SUM)
                 ? iattr::fuse_sum()
                 : ((fusion_type_ == FUSION_CONV_SUM_RELU) ? iattr::residual()
                                                           : iattr()));
    };

    CAFFE_ENFORCE(4 == X.ndims());
    CAFFE_ENFORCE(4 == filter.ndims() || (grouped && (group_ > 1)));
    CAFFE_ENFORCE(filter.get_dim(2 + grouped) == kernel_h());
    CAFFE_ENFORCE(filter.get_dim(3 + grouped) == kernel_w());
    CAFFE_ENFORCE(
        X.get_dim(1) == filter.get_dim(1 + grouped) * group_,
        "Convolution op: input channels does not match: # of input channels ",
        X.get_dim(1), " is not equal to kernel channels * group:",
        filter.get_dim(1 + grouped), "*", group_);

    bool weights_changed =
        (cached_weights_descriptor_ != filter.get_descriptor());
    if (weights_changed && !training_mode_) {
      op_key_.clear();
      cached_weights_descriptor_ = filter.dup_descriptor();
      if (need_quantize_)
        filter_scales_ = filter.calculate_scale(idtype::s8, 0 + grouped);
      auto filter_in = filter;
      filter_in.make_group(group_);
      auto filter_dtype = filter_scales_.empty()
        ? filter_in.get_data_type() : idtype::s8;
      auto filter_mask = (filter_scales_.size() > 1)
        ? ((filter_in.is_grouped()) ? 3 : 1) : 0;
      auto filter_attr = (filter_in.get_data_type() != filter_dtype)
        ? iattr(filter_mask, filter_scales_) : iattr();

      auto expected_descriptor =
          ideep::convolution_forward::expected_weights_descriptor(
              filter_in.get_dims(), filter_dtype, stride_,
              pad_tl(), pad_br(), dilation_, group_, algo_);
      if (filter_in.get_descriptor() != expected_descriptor) {
        filter_.init(expected_descriptor);
        ideep::reorder::compute(filter_in, filter_, filter_attr);
      } else {
        filter_ = filter_in;
      }

      if (InputSize() > last_input()) {
        bias_ = Input(BIAS_OR_INPUT_S);
        if (!filter_scales_.empty()) {
          auto bias_mask = (filter_scales_.size() > 1) ? 1 : 0;
          auto bias_scales (filter_scales_);
          auto &X_scales = X.has_scale() ? X.get_scale() : X_scales_;
          for (int i = 0; i < bias_scales.size(); i++) {
            bias_scales[i] *= X_scales[0];
          }
          bias_.init({bias_.get_dims(), idtype::s32, iformat::x});
          ideep::reorder::compute(
              Input(BIAS_OR_INPUT_S), bias_, {bias_mask, bias_scales});
        }
      }
    }

    if (cached_X_descriptor_ != X.get_descriptor()) {
      op_key_.clear();
      cached_X_descriptor_ = X.dup_descriptor();
      if (!Y_absmax_.empty() && Y_dt() != idtype::f32) {
        Y_scales_ = Y_absmax_;
        for (auto it = Y_scales_.begin() ; it != Y_scales_.end(); it++) {
          *it = ideep::dt_max_map.at(Y_dt()) / (*it);
        }
      }
    }

    if (InputSize() > last_input()) {
      ideep::convolution_forward::compute(
          op_key_, X, training_mode_ ? filter : filter_,
          training_mode_ ? Input(BIAS_OR_INPUT_S) : bias_, Y_dims_conv, *Y,
          stride_, dilation_, pad_tl(), pad_br(), group_, X_scales_,
          filter_scales_, Y_scales_, attr(), algo_, pk_);
    } else {
      ideep::convolution_forward::compute(
          op_key_, X, training_mode_ ? filter : filter_,
          Y_dims_conv, *Y, stride_, dilation_, pad_tl(), pad_br(),
          group_, X_scales_, filter_scales_, Y_scales_, attr(), algo_, pk_);
    }

    if (fusion_type_ != FUSION_CONV_RELU) {
      CAFFE_ENFORCE(
          Y == &(Input(InputSize() - 1)),
          "Convolution fusion op: InPlace is enforced for sum fusion.");
    }

    return true;
  }

 private:
  iprop pk_;
  ialgo algo_;
  ikey op_key_;
  bool training_mode_;
  bool need_quantize_;
  FusionType fusion_type_;
  itensor filter_, bias_;
  iscale X_scales_, filter_scales_, Y_scales_, Y_absmax_;
  itensor::descriptor cached_X_descriptor_, cached_weights_descriptor_;

  INPUT_TAGS(INPUT_X, FILTER, BIAS_OR_INPUT_S, INPUT_S);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(ConvFusion, IDEEPConvFusionOp);

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

} // namespace caffe2
