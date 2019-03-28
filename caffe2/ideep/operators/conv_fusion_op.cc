#include <caffe2/ideep/operators/conv_pool_base_op.h>

namespace caffe2 {

class IDEEPConvFusionOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  enum FusionType {
    FUSION_UNKNOWN = 0,
    FUSION_CONV_RELU = 1,
    FUSION_CONV_SUM = 2,
    FUSION_CONV_SUM_RELU = 3,
    FUSION_MAX = FUSION_CONV_SUM_RELU + 1,
  };

  IDEEPConvFusionOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws),
        fusion_type_(static_cast<FusionType>(
            OperatorBase::GetSingleArgument<int>("fusion_type", 0))),
        training_mode_(
            OperatorBase::GetSingleArgument<int>("training_mode", 0)),
        conv_algorithm_(
            OperatorBase::GetSingleArgument<int>("conv_algorithm", CONV_ALGORITHM_AUTO)) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
    OPERATOR_NEEDS_FEATURE(group_ == 1, "Group not supported.");
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
  }
  ~IDEEPConvFusionOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT_X);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);
    auto Y_dims_conv = CalcOutputDims(X, filter.get_dim(0));
    auto attr = [this]() {
      return (fusion_type_ == FUSION_CONV_RELU)
          ? iattr::fuse_relu()
          : ((fusion_type_ == FUSION_CONV_SUM)
                 ? iattr::fuse_sum()
                 : ((fusion_type_ == FUSION_CONV_SUM_RELU) ? iattr::residual()
                                                           : iattr()));
    };
    auto last_input = [this]() {
      return (fusion_type_ == FUSION_CONV_RELU) ? BIAS_OR_INPUT_S : INPUT_S;
    };

    CAFFE_ENFORCE(4 == X.ndims());
    CAFFE_ENFORCE(4 == filter.ndims());
    CAFFE_ENFORCE(filter.get_dim(2) == kernel_h());
    CAFFE_ENFORCE(filter.get_dim(3) == kernel_w());
    CAFFE_ENFORCE(
        X.get_dim(1) == filter.get_dim(1) * group_,
        "Convolution fusion op: input channels does not match: "
        "# of input channels ",
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
      filter_ = filter;
      auto expected_descriptor =
          ideep::convolution_forward::expected_weights_descriptor(
              filter.get_dims());
      if (filter_.get_descriptor() != expected_descriptor) {
        filter_.init<ideep::utils::allocator, ideep::convolution_forward>(
            expected_descriptor);
        ideep::reorder::compute(filter, filter_);
      }
    }

    if (InputSize() > last_input()) {
      ideep::convolution_forward::compute(
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
          attr(),
          aalgorithm);
    } else {
      ideep::convolution_forward::compute(
          X,
          training_mode_ ? filter : filter_,
          Y_dims_conv,
          *Y,
          stride_,
          dilation_,
          pad_tl(),
          pad_br(),
          group_,
          attr(),
          aalgorithm);
    }

    if (fusion_type_ != FUSION_CONV_RELU) {
      CAFFE_ENFORCE(
          Y == &(Input(InputSize() - 1)),
          "Convolution fusion op: InPlace is enforced for sum fusion.");
    }

    return true;
  }

 private:
  FusionType fusion_type_;
  bool training_mode_;
  int conv_algorithm_;
  ideep::tensor filter_;
  ideep::tensor::descriptor cached_weights_descriptor_;

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
