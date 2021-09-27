#ifndef CAFFE2_OPERATORS_CONV_POOL_OP_BASE_H_
#define CAFFE2_OPERATORS_CONV_POOL_OP_BASE_H_

#include <algorithm>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_legacy.pb.h"
#include "caffe2/utils/math.h"

// This macro is here just to allow us to experiment with padding values that
// determines, when we have an odd number of pads, which side gets the one
// additional pad value, the head side, or the tail side. Setting it to false
// will enable the TensorFlow behavior, and setting it to true will enable
// a behavior more consistent with Caffe and CuDNN.
// This only affects the case when you set legacy pad to VALID or SAME. The
// behavior inherits from the early designs of Google's CNN implementation,
// where padding values are implicitly calculated instead of explicitly
// specified. This is still the case with TensorFlow. Many frameworks have
// followed a slightly different approach of explicitly giving padding values,
// in which case the value of this constant value does not matter.
const bool CAFFE2_PAD_HEAD_MORE = false;

namespace caffe2 {

template <class Context>
class ConvPoolOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit ConvPoolOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        legacy_pad_(
            static_cast<LegacyPadding>(this->template GetSingleArgument<int>(
                "legacy_pad",
                LegacyPadding::NOTSET))),
        global_pooling_(
            this->template GetSingleArgument<int>("global_pooling", 0)),
        kernel_(this->template GetRepeatedArgument<int>("kernels")),
        dilation_(this->template GetRepeatedArgument<int>("dilations")),
        stride_(this->template GetRepeatedArgument<int>("strides")),
        pads_(this->template GetRepeatedArgument<int>("pads")),
        float16_compute_(
            this->template GetSingleArgument<bool>("float16_compute", false)),
        group_(this->template GetSingleArgument<int>("group", 1)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        shared_buffer_(
            this->template GetSingleArgument<int>("shared_buffer", 0)),
        ws_(ws) {
    // For the padding, they should either be the legacy padding strategy
    // (VALID or SAME), or an explicit, non-negative value.
    if (legacy_pad_ == LegacyPadding::VALID ||
        legacy_pad_ == LegacyPadding::SAME) {
      CAFFE_ENFORCE(
          !OperatorBase::HasArgument("pads"),
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
    }

    // Get old arguments values.
    if (OperatorBase::HasArgument("kernel")) {
      kernel_.resize(2, this->template GetSingleArgument<int>("kernel", 0));
    } else if (
        OperatorBase::HasArgument("kernel_h") &&
        OperatorBase::HasArgument("kernel_w")) {
      kernel_.push_back(this->template GetSingleArgument<int>("kernel_h", 0));
      kernel_.push_back(this->template GetSingleArgument<int>("kernel_w", 0));
    }

    if (OperatorBase::HasArgument("stride")) {
      stride_.resize(2, this->template GetSingleArgument<int>("stride", 0));
    } else if (
        OperatorBase::HasArgument("stride_h") &&
        OperatorBase::HasArgument("stride_w")) {
      stride_.push_back(this->template GetSingleArgument<int>("stride_h", 0));
      stride_.push_back(this->template GetSingleArgument<int>("stride_w", 0));
    }

    if (OperatorBase::HasArgument("dilation")) {
      dilation_.resize(2, this->template GetSingleArgument<int>("dilation", 0));
    } else if (
        OperatorBase::HasArgument("dilation_h") &&
        OperatorBase::HasArgument("dilation_w")) {
      dilation_.push_back(
          this->template GetSingleArgument<int>("dilation_h", 0));
      dilation_.push_back(
          this->template GetSingleArgument<int>("dilation_w", 0));
    }

    if (OperatorBase::HasArgument("pad")) {
      CAFFE_ENFORCE(
          legacy_pad_ != LegacyPadding::VALID &&
              legacy_pad_ != LegacyPadding::SAME,
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
      pads_.resize(4, this->template GetSingleArgument<int>("pad", 0));
    } else if (
        OperatorBase::HasArgument("pad_t") &&
        OperatorBase::HasArgument("pad_l") &&
        OperatorBase::HasArgument("pad_b") &&
        OperatorBase::HasArgument("pad_r")) {
      CAFFE_ENFORCE(
          legacy_pad_ != LegacyPadding::VALID &&
              legacy_pad_ != LegacyPadding::SAME,
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
      pads_.push_back(this->template GetSingleArgument<int>("pad_t", 0));
      pads_.push_back(this->template GetSingleArgument<int>("pad_l", 0));
      pads_.push_back(this->template GetSingleArgument<int>("pad_b", 0));
      pads_.push_back(this->template GetSingleArgument<int>("pad_r", 0));
    }

    // Fill default values.
    if (kernel_.size() == 0) {
      kernel_.assign({0, 0});
    }

    if (stride_.size() == 0) {
      stride_.resize(kernel_.size(), 1);
    }

    if (pads_.size() == 0) {
      pads_.resize(kernel_.size() * 2, 0);
    }

    if (dilation_.size() == 0) {
      dilation_.resize(kernel_.size(), 1);
    }

    CAFFE_ENFORCE_EQ(stride_.size(), kernel_.size());
    CAFFE_ENFORCE_EQ(dilation_.size(), kernel_.size());

    if (legacy_pad_ != LegacyPadding::VALID &&
        legacy_pad_ != LegacyPadding::SAME) {
      CAFFE_ENFORCE_EQ(pads_.size(), 2 * kernel_.size());
    }

    if (global_pooling_) {
      for (size_t dim = 0; dim < kernel_.size(); ++dim) {
        CAFFE_ENFORCE(
            pads_[2 * dim] == 0 && pads_[2 * dim + 1] == 0 &&
                dilation_[dim] == 1 && stride_[dim] == 1,
            "If global_pooling is set pad, dilation and stride shouldn't be set.");
      }
    }

    // Check kernel only if we are doing conv or pooling. The reason is that a
    // few other ops, like PadImage, are also using this base class. We really
    // need to clean this up.
    if (operator_def.name().find("Conv") == 0 ||
        operator_def.name().find("Pool") != std::string::npos) {
      for (size_t dim = 0; dim < kernel_.size(); ++dim) {
        CAFFE_ENFORCE_GE(pads_[dim], 0);
        CAFFE_ENFORCE_GE(pads_[kernel_.size() + dim], 0);
        CAFFE_ENFORCE(
            kernel_[dim],
            "If you are doing convolution or pooling, you will need to set "
            "explicitly the kernel size.");
      }
    }

    for (size_t dim = 0; dim < kernel_.size(); ++dim) {
      CAFFE_ENFORCE_GE(kernel_[dim], 0);
      CAFFE_ENFORCE_GE(dilation_[dim], 0);
      CAFFE_ENFORCE_GE(stride_[dim], 0);
    }
  }

  // Returns the input image dimensions for the current storage order type.
  vector<int> GetDims(const Tensor& input) {
    vector<int> dims;
    switch (order_) {
      case StorageOrder::NCHW:
        dims.assign(input.sizes().begin() + 2, input.sizes().end());
        break;
      case StorageOrder::NHWC:
        dims.assign(input.sizes().begin() + 1, input.sizes().end() - 1);
        break;
      default:
        CAFFE_THROW("Unknown storage order : ", order_);
    }
    return dims;
  }

  // Returns the size of the input image for the current storage type.
  int GetDimsSize(const Tensor& input) {
    int size = 0;
    switch (order_) {
      case StorageOrder::NCHW:
        size = std::accumulate(
            input.sizes().begin() + 2,
            input.sizes().end(),
            1,
            std::multiplies<int>());
        break;
      case StorageOrder::NHWC:
        size = std::accumulate(
            input.sizes().begin() + 1,
            input.sizes().end() - 1,
            1,
            std::multiplies<int>());
        break;
      default:
        CAFFE_THROW("Unknown storage order : ", order_);
    }
    return size;
  }

  // Gets the output size. The output channel is manually provided since
  // it may not be identical to the input channels.
  // This function can be used in the forward functions to obtain the output
  // sizes.
  // Note(jiayq): the templatization of this function is mainly to help
  // implementations that do not use first-class Tensor objects, such as the
  // MKL operator. One can still call this function with dummy
  // Tensor objects in order to obtain the sizes.
  std::vector<int64_t> GetOutputSize(const Tensor& input, int output_channel) {
    CAFFE_ENFORCE_GE(input.dim(), 2);
    const int inner_size = input.size_from_dim(1);
    CAFFE_ENFORCE_GT(inner_size, 0);
    std::vector<int64_t> output_dims;
    InferOutputSize64(
        input.sizes(),
        output_channel,
        order_,
        global_pooling_,
        legacy_pad_,
        dilation_,
        stride_,
        &kernel_,
        &pads_,
        &output_dims);
    return output_dims;
  }

  void SetOutputSize(const Tensor& input, Tensor* output, int output_channel) {
    const int inner_size = input.size_from_dim(1);
    CAFFE_ENFORCE_GT(inner_size, 0);
    std::vector<int> output_dims;
    InferOutputSize(
        input.sizes(),
        output_channel,
        order_,
        global_pooling_,
        legacy_pad_,
        dilation_,
        stride_,
        &kernel_,
        &pads_,
        &output_dims);
    output->Resize(output_dims);
  }

  // Helper function that is also called from OperatorSchema. Modified
  // kernel parameters and output output_dims and channel_first.
  static void InferOutputSize(
      const at::IntArrayRef& input_dims,
      const int output_channel,
      const StorageOrder order,
      const bool global_pooling,
      const LegacyPadding legacy_pad,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      std::vector<int>* kernel,
      std::vector<int>* pads,
      std::vector<int>* output_dims) {
    CAFFE_ENFORCE_NE(order, StorageOrder::UNKNOWN);
    const int ndim = input_dims.size() - 2;
    output_dims->resize(ndim + 2);
    output_dims->front() = input_dims.front();
    if (order == StorageOrder::NCHW) {
      output_dims->at(1) = output_channel;
    } else {
      output_dims->back() = output_channel;
    }
    const int offset = order == StorageOrder::NCHW ? 2 : 1;
    if (global_pooling) {
      std::copy_n(input_dims.cbegin() + offset, ndim, kernel->begin());
      std::fill_n(output_dims->begin() + offset, ndim, 1LL);
    } else {
      for (int i = 0; i < ndim; ++i) {
        ComputeSizeAndPad(
            input_dims[i + offset],
            stride[i],
            kernel->at(i),
            dilation[i],
            legacy_pad,
            &pads->at(i),
            &pads->at(i + ndim),
            &output_dims->at(i + offset));
      }
    }
  }

  static void InferOutputSize64(
      const at::IntArrayRef& input_dims,
      const int output_channel,
      const StorageOrder order,
      const bool global_pooling,
      const LegacyPadding legacy_pad,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      std::vector<int>* kernel,
      std::vector<int>* pads,
      std::vector<int64_t>* output_dims) {
    CAFFE_ENFORCE_NE(order, StorageOrder::UNKNOWN);
    const int ndim = input_dims.size() - 2;
    output_dims->resize(ndim + 2);
    output_dims->front() = input_dims.front();
    if (order == StorageOrder::NCHW) {
      output_dims->at(1) = output_channel;
    } else {
      output_dims->back() = output_channel;
    }
    const int offset = order == StorageOrder::NCHW ? 2 : 1;
    if (global_pooling) {
      std::copy_n(input_dims.cbegin() + offset, ndim, kernel->begin());
      std::fill_n(output_dims->begin() + offset, ndim, 1LL);
    } else {
      for (int i = 0; i < ndim; ++i) {
        ComputeSizeAndPad64(
            input_dims[i + offset],
            stride[i],
            kernel->at(i),
            dilation[i],
            legacy_pad,
            &pads->at(i),
            &pads->at(i + ndim),
            &output_dims->at(i + offset));
      }
    }
  }

  // ComputePads could be used in backward functions to figure out the padding
  // values for the given input.
  void ComputePads(const vector<int>& dims) {
    if (global_pooling_) {
      kernel_ = dims;
    } else if (legacy_pad_ != LegacyPadding::NOTSET) {
      int output_unused;
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int dim = 0; dim < dims.size(); ++dim) {
        ComputeSizeAndPad(
            dims[dim],
            stride_[dim],
            kernel_[dim],
            dilation_[dim],
            legacy_pad_,
            &pads_[dim],
            &pads_[dims.size() + dim],
            &output_unused);
      }
    }
  }

  bool HasPad() const {
    if (kernel_.size() == 2) {
      return pad_t() > 0 || pad_b() > 0 || pad_l() > 0 || pad_r() > 0;
    }
    return std::any_of(
        pads_.cbegin(), pads_.cend(), [](const int x) { return x > 0; });
  }

  bool HasStride() const {
    if (kernel_.size() == 2) {
      return stride_h() > 1 || stride_w() > 1;
    }
    return std::any_of(
        stride_.cbegin(), stride_.cend(), [](const int x) { return x > 1; });
  }

  void SetDeviceTensor(const std::vector<int>& data, Tensor* tensor) {
    bool reset_tensor_device_ = false;

    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    if (tensor->numel() != data.size()) {
      tensor->Resize(data.size());
      reset_tensor_device_ = true;
    } else {
      const int* tensor_data = tensor->template data<int>();
      for (int d_i = 0; d_i < data.size(); ++d_i) {
        if (tensor_data[d_i] != data[d_i]) {
          reset_tensor_device_ = true;
          break;
        }
      }
    }

    if (reset_tensor_device_) {
      context_.template Copy<int, CPUContext, Context>(
          data.size(), data.data(), tensor->template mutable_data<int>());
    }
  }

  template <typename T>
  void SetBiasMultiplier(const int size, Tensor* bias_multiplier_) {
    if (bias_multiplier_->numel() != size) {
      // If the helper bias multiplier is not image size, reshape and fill it
      // with one.
      bias_multiplier_->Resize(std::vector<int64_t>{size});
      math::Set<T, Context>(
          size,
          static_cast<T>(1),
          bias_multiplier_->template mutable_data<T>(),
          &context_);
    }
  }

  bool RunOnDevice() override {
    if (!global_pooling_) {
      for (size_t dim = 0; dim < kernel_.size(); ++dim) {
        CAFFE_ENFORCE_GT(kernel_[dim], 0);
      }
    }
    switch (order_) {
      case StorageOrder::NHWC:
        // VLOG(2) << "Running NHWC";
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        // VLOG(2) << "Running NCHW";
        return RunOnDeviceWithOrderNCHW();
      default:
        CAFFE_THROW("Unknown Storage order: ", order_);
    }
  }

  // The actual function that does the computation, if the different
  // storage order leads to different implementations.
  virtual bool RunOnDeviceWithOrderNHWC() {
    CAFFE_NOT_IMPLEMENTED;
  }
  virtual bool RunOnDeviceWithOrderNCHW() {
    CAFFE_NOT_IMPLEMENTED;
  }

  static struct OpSchema::Cost CostInferenceForConv(
      const OperatorDef& def,
      const vector<TensorShape>& inputs) {
    CAFFE_ENFORCE_GE(inputs.size(), 2, "Conv requires at least 2 inputs");
    struct OpSchema::Cost c;
    const TensorShape X = inputs[0];
    const TensorShape W = inputs[1];
    const TensorShape Y = TensorInferenceForConv(def, inputs)[0];
    ArgumentHelper helper(def);
    const auto order =
        StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
    uint64_t N;
    uint64_t Y_h;
    uint64_t Y_w = 1;
    uint64_t Y_t = 1;
    uint64_t kernel_h;
    uint64_t kernel_w = 1;
    uint64_t kernel_t = 1;
    uint64_t in_channels;
    uint64_t out_channels;

    if (X.dims_size() == 0 || W.dims_size() == 0) {
      return c;
    }
    N = X.dims(0);
    if (X.dims_size() == 5) {
      // 3D convolution
      if (order == StorageOrder::NHWC) {
        Y_t = Y.dims(1);
        Y_h = Y.dims(2);
        Y_w = Y.dims(3);
        kernel_t = W.dims(1);
        kernel_h = W.dims(2);
        kernel_w = W.dims(3);
        in_channels = W.dims(4);
        out_channels = W.dims(0);
      } else {
        Y_t = Y.dims(2);
        Y_h = Y.dims(3);
        Y_w = Y.dims(4);
        kernel_t = W.dims(2);
        kernel_h = W.dims(3);
        kernel_w = W.dims(4);
        in_channels = W.dims(1);
        out_channels = W.dims(0);
      }
    } else if (X.dims_size() == 4) {
      // 2D convolution
      CAFFE_ENFORCE_EQ(W.dims_size(), 4, "Conv2D should have 4D filter tensor");
      if (order == StorageOrder::NHWC) {
        Y_h = Y.dims(1);
        Y_w = Y.dims(2);
        kernel_h = W.dims(1);
        kernel_w = W.dims(2);
        in_channels = W.dims(3);
        out_channels = W.dims(0);
      } else {
        Y_h = Y.dims(2);
        Y_w = Y.dims(3);
        kernel_h = W.dims(2);
        kernel_w = W.dims(3);
        in_channels = W.dims(1);
        out_channels = W.dims(0);
      }
    } else {
      // 1D convolution
      CAFFE_ENFORCE_EQ(W.dims_size(), 3, "Conv1D should have 3D filter tensor");
      if (order == StorageOrder::NHWC) {
        Y_h = Y.dims(1);
        kernel_h = W.dims(1);
        in_channels = W.dims(2);
        out_channels = W.dims(0);
      } else {
        Y_h = Y.dims(2);
        kernel_h = W.dims(2);
        in_channels = W.dims(1);
        out_channels = W.dims(0);
      }
    }

    uint64_t nElemX = nElemFromDim(X);
    uint64_t nElemW = nElemFromDim(W);
    uint64_t nElemBias = inputs.size() > 2 ? nElemFromDim(inputs[2]) : 0;

    auto const& X_elemenet_size_byte =
        DataTypeToTypeMeta(X.data_type()).itemsize();
    auto const& Y_element_size_byte =
        DataTypeToTypeMeta(Y.data_type()).itemsize();
    auto const& W_element_size_byte =
        DataTypeToTypeMeta(W.data_type()).itemsize();

    // grouping is NOT properly handled yet
    c.flops = N * Y_t * Y_h * Y_w * kernel_t * kernel_w * kernel_h *
        in_channels * out_channels * 2;
    c.bytes_read = (nElemX + nElemW + nElemBias) * X_elemenet_size_byte;
    c.bytes_written = N * out_channels * Y_t * Y_h * Y_w * Y_element_size_byte;
    c.params_bytes = out_channels * in_channels * kernel_t * kernel_h *
        kernel_w * W_element_size_byte;
    return c;
  }

  static vector<TensorShape> TensorInferenceForSchema(
      const OperatorDef& def,
      const vector<TensorShape>& in,
      int output_channel) {
    ArgumentHelper helper(def);
    CAFFE_ENFORCE_GT(in.size(), 0U);
    CAFFE_ENFORCE_GT(in[0].dims_size(), 0);
    vector<int> pads = helper.GetRepeatedArgument<int>("pads");
    vector<int> kernel = helper.GetRepeatedArgument<int>("kernels");
    vector<int> strides = helper.GetRepeatedArgument<int>("strides");
    vector<int> dilations = helper.GetRepeatedArgument<int>("dilation");
    if (helper.HasArgument("pad")) {
      pads.resize(4, helper.GetSingleArgument<int>("pad", 0));
    } else if (
        helper.HasArgument("pad_t") && helper.HasArgument("pad_l") &&
        helper.HasArgument("pad_b") && helper.HasArgument("pad_r")) {
      pads.push_back(helper.GetSingleArgument<int>("pad_t", 0));
      pads.push_back(helper.GetSingleArgument<int>("pad_l", 0));
      pads.push_back(helper.GetSingleArgument<int>("pad_b", 0));
      pads.push_back(helper.GetSingleArgument<int>("pad_r", 0));
    }

    if (helper.HasArgument("kernel")) {
      kernel.resize(2, helper.GetSingleArgument<int>("kernel", 1));
    } else if (
        helper.HasArgument("kernel_h") && helper.HasArgument("kernel_w")) {
      kernel.push_back(helper.GetSingleArgument<int>("kernel_h", 1));
      kernel.push_back(helper.GetSingleArgument<int>("kernel_w", 1));
    }

    if (helper.HasArgument("stride")) {
      strides.resize(2, helper.GetSingleArgument<int>("stride", 1));
    } else if (
        helper.HasArgument("stride_h") && helper.HasArgument("stride_w")) {
      strides.push_back(helper.GetSingleArgument<int>("stride_h", 1));
      strides.push_back(helper.GetSingleArgument<int>("stride_w", 1));
    }

    if (helper.HasArgument("dilation")) {
      strides.resize(2, helper.GetSingleArgument<int>("dilation", 1));
    } else if (
        helper.HasArgument("dilation_h") && helper.HasArgument("dilation_w")) {
      strides.push_back(helper.GetSingleArgument<int>("dilation_h", 1));
      strides.push_back(helper.GetSingleArgument<int>("dilation_w", 1));
    }

    auto check_and_set_default_value =
        [](vector<int>& vec, int size, int value) {
          if (vec.size() == 0) {
            vec.resize(size, value);
          }
        };

    check_and_set_default_value(kernel, 2, 1);
    check_and_set_default_value(strides, kernel.size(), 1);
    check_and_set_default_value(pads, kernel.size() * 2, 0);
    check_and_set_default_value(dilations, kernel.size(), 1);

    std::vector<int> output_dims;
    ConvPoolOpBase<CPUContext>::InferOutputSize(
        GetDimsVector(in[0]),
        output_channel,
        StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW")),
        helper.GetSingleArgument<int>("global_pooling", 0),
        static_cast<LegacyPadding>(
            helper.GetSingleArgument<int>("legacy_pad", LegacyPadding::NOTSET)),
        dilations,
        strides,
        &kernel,
        &pads,
        &output_dims);
    return {CreateTensorShape(output_dims, TensorProto::FLOAT)};
  }

  static std::vector<TensorShape> TensorInferenceForConv(
      const OperatorDef& def,
      const std::vector<TensorShape>& in) {
    if (in[0].unknown_shape()) {
      std::vector<TensorShape> out(1);
      out[0].set_unknown_shape(true);
      return out;
    }
    return TensorInferenceForSchema(def, in, in[1].dims(0));
  }

  static std::vector<TensorShape> TensorInferenceForPool(
      const OperatorDef& def,
      const std::vector<TensorShape>& in) {
    if (in[0].unknown_shape()) {
      std::vector<TensorShape> out(1);
      out[0].set_unknown_shape(true);
      return out;
    }
    ArgumentHelper helper(def);
    auto order =
        StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
    int num_channels =
        (order == StorageOrder::NCHW ? in[0].dims(1) : in[0].dims(3));
    return TensorInferenceForSchema(def, in, num_channels);
  }

  static std::vector<TensorShape> TensorInferenceForLC(
      const OperatorDef& def,
      const std::vector<TensorShape>& in) {
    if (in[0].unknown_shape()) {
      std::vector<TensorShape> out(1);
      out[0].set_unknown_shape(true);
      return out;
    }
    const int img_ndim = in[0].dims_size() - 2;
    return TensorInferenceForSchema(def, in, in[1].dims(img_ndim));
  }

  virtual ~ConvPoolOpBase() {}

 protected:
  LegacyPadding legacy_pad_;
  bool global_pooling_;
  vector<int> kernel_;
  vector<int> dilation_;
  vector<int> stride_;
  vector<int> pads_;

  bool float16_compute_;

  int group_;
  StorageOrder order_;
  bool shared_buffer_;
  Workspace* ws_;

  static inline void ComputeSizeAndPad(
      const int in_size,
      const int stride,
      const int kernel,
      const int dilation,
      LegacyPadding legacy_pad,
      int* pad_head,
      int* pad_tail,
      int* out_size) {
    const int dkernel = dilation * (kernel - 1) + 1;
    switch (legacy_pad) {
      case LegacyPadding::NOTSET:
        // We will just use the direct padding head and tail values, but we
        // will verify that they are non-negative.
        CAFFE_ENFORCE_GE(in_size + *pad_head + *pad_tail, dkernel);
        *out_size = static_cast<int>(
            static_cast<float>(in_size + *pad_head + *pad_tail - dkernel) /
                stride +
            1);
        break;
      case LegacyPadding::VALID:
        *pad_head = 0;
        *pad_tail = 0;
        *out_size = (in_size - dkernel) / stride + 1;
        break;
      case LegacyPadding::SAME: {
        CAFFE_ENFORCE(
            1 == dilation, "Dilation not supported for legacy padding.");
        int legacy_target_size = (in_size + stride - 1) / stride;
        int pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
        if (CAFFE2_PAD_HEAD_MORE) {
          *pad_head = (pad_needed + 1) / 2;
        } else {
          *pad_head = pad_needed / 2;
        }
        *pad_tail = pad_needed - *pad_head;
        *out_size = (in_size + pad_needed - dkernel) / stride + 1;
        break;
      }
      case LegacyPadding::CAFFE_LEGACY_POOLING:
        // This is in order to adapt Caffe's pooling padding case. In this case,
        // we will only use pad_head and will compute pad_tail to match the
        // old caffe pooling strategy. Also see caffe2_legacy.proto for more
        // details.
        CAFFE_ENFORCE_GE(*pad_head, 0);
        // Here, notice that caffe casts UP while caffe2 casts DOWN for the
        // output size computation.
        *out_size = std::ceil(
            static_cast<float>(in_size + *pad_head * 2 - kernel) / stride + 1);
        // If we have padding, caffe also ensures that the last pooling starts
        // strictly inside the image (instead of at the padding); otherwise clip
        // the last.
        if (*pad_head > 0 && (*out_size - 1) * stride >= in_size + *pad_head) {
          --*out_size;
        }
        // Now, compare the output size with the standard Caffe2 output size.
        // The
        // caffe2 standard output size should always be no larger than the
        // output
        // size of caffe.
        int standard_out_size = static_cast<int>(
            static_cast<float>(in_size + *pad_head * 2 - kernel) / stride + 1);
        CAFFE_ENFORCE_GE(
            *out_size,
            standard_out_size,
            "This should never happen. If this happens, double check the logic "
            "above.");
        if (*out_size > standard_out_size) {
          LOG(WARNING)
              << "You are hitting a case where Caffe's legacy padding calculation "
                 "is hit. This leads to inefficient and sometimes incorrect "
                 "results. We are keeping this behavior for backward compatibility"
                 ", but you are strongly recommended to move away from it.";
        }
        *pad_tail = *pad_head + stride * (*out_size - standard_out_size);
        break;
    }
  }

  static inline void ComputeSizeAndPad64(
      const int in_size,
      const int stride,
      const int kernel,
      const int dilation,
      LegacyPadding legacy_pad,
      int* pad_head,
      int* pad_tail,
      int64_t* out_size) {
    const int dkernel = dilation * (kernel - 1) + 1;
    switch (legacy_pad) {
      case LegacyPadding::NOTSET:
        // We will just use the direct padding head and tail values, but we
        // will verify that they are non-negative.
        CAFFE_ENFORCE_GE(in_size + *pad_head + *pad_tail, dkernel);
        *out_size = static_cast<int>(
            static_cast<float>(in_size + *pad_head + *pad_tail - dkernel) /
                stride +
            1);
        break;
      case LegacyPadding::VALID:
        *pad_head = 0;
        *pad_tail = 0;
        *out_size = (in_size - dkernel) / stride + 1;
        break;
      case LegacyPadding::SAME: {
        CAFFE_ENFORCE(
            1 == dilation, "Dilation not supported for legacy padding.");
        int legacy_target_size = (in_size + stride - 1) / stride;
        int pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
        if (CAFFE2_PAD_HEAD_MORE) {
          *pad_head = (pad_needed + 1) / 2;
        } else {
          *pad_head = pad_needed / 2;
        }
        *pad_tail = pad_needed - *pad_head;
        *out_size = (in_size + pad_needed - dkernel) / stride + 1;
        break;
      }
      case LegacyPadding::CAFFE_LEGACY_POOLING:
        // This is in order to adapt Caffe's pooling padding case. In this case,
        // we will only use pad_head and will compute pad_tail to match the
        // old caffe pooling strategy. Also see caffe2_legacy.proto for more
        // details.
        CAFFE_ENFORCE_GE(*pad_head, 0);
        // Here, notice that caffe casts UP while caffe2 casts DOWN for the
        // output size computation.
        *out_size = std::ceil(
            static_cast<float>(in_size + *pad_head * 2 - kernel) / stride + 1);
        // If we have padding, caffe also ensures that the last pooling starts
        // strictly inside the image (instead of at the padding); otherwise clip
        // the last.
        if (*pad_head > 0 && (*out_size - 1) * stride >= in_size + *pad_head) {
          --*out_size;
        }
        // Now, compare the output size with the standard Caffe2 output size.
        // The
        // caffe2 standard output size should always be no larger than the
        // output
        // size of caffe.
        int standard_out_size = static_cast<int>(
            static_cast<float>(in_size + *pad_head * 2 - kernel) / stride + 1);
        CAFFE_ENFORCE_GE(
            *out_size,
            standard_out_size,
            "This should never happen. If this happens, double check the logic "
            "above.");
        if (*out_size > standard_out_size) {
          LOG(WARNING)
              << "You are hitting a case where Caffe's legacy padding calculation "
                 "is hit. This leads to inefficient and sometimes incorrect "
                 "results. We are keeping this behavior for backward compatibility"
                 ", but you are strongly recommended to move away from it.";
        }
        *pad_tail = *pad_head + stride * (*out_size - standard_out_size);
        break;
    }
  }

  // Accessors for 2D conv params.

  inline int pad_t() const {
    return pads_[0];
  }

  inline int pad_l() const {
    return pads_[1];
  }

  inline int pad_b() const {
    return pads_[2];
  }

  inline int pad_r() const {
    return pads_[3];
  }

  inline int kernel_h() const {
    return kernel_[0];
  }

  inline int kernel_w() const {
    return kernel_[1];
  }

  inline int stride_h() const {
    return stride_[0];
  }

  inline int stride_w() const {
    return stride_[1];
  }

  inline int dilation_h() const {
    return dilation_[0];
  }

  inline int dilation_w() const {
    return dilation_[1];
  }

 private:
  inline void AllocateAndCopy(const vector<int>& vec, Tensor& tensor) {
    tensor.Resize(vec.size());
    context_.template CopyFromCPU<int>(
        vec.size(), vec.data(), tensor.template mutable_data<int>());
  }

#define USE_CONV_POOL_BASE_FUNCTIONS(Context)     \
  USE_OPERATOR_FUNCTIONS(Context);                \
  using ConvPoolOpBase<Context>::pads_;           \
  using ConvPoolOpBase<Context>::pad_t;           \
  using ConvPoolOpBase<Context>::pad_l;           \
  using ConvPoolOpBase<Context>::pad_b;           \
  using ConvPoolOpBase<Context>::pad_r;           \
  using ConvPoolOpBase<Context>::legacy_pad_;     \
  using ConvPoolOpBase<Context>::global_pooling_; \
  using ConvPoolOpBase<Context>::kernel_;         \
  using ConvPoolOpBase<Context>::kernel_h;        \
  using ConvPoolOpBase<Context>::kernel_w;        \
  using ConvPoolOpBase<Context>::dilation_;       \
  using ConvPoolOpBase<Context>::dilation_h;      \
  using ConvPoolOpBase<Context>::dilation_w;      \
  using ConvPoolOpBase<Context>::stride_;         \
  using ConvPoolOpBase<Context>::stride_h;        \
  using ConvPoolOpBase<Context>::stride_w;        \
  using ConvPoolOpBase<Context>::group_;          \
  using ConvPoolOpBase<Context>::order_;          \
  using ConvPoolOpBase<Context>::shared_buffer_;  \
  using ConvPoolOpBase<Context>::GetDims;         \
  using ConvPoolOpBase<Context>::GetDimsSize;     \
  using ConvPoolOpBase<Context>::SetDeviceTensor; \
  using ConvPoolOpBase<Context>::HasPad;          \
  using ConvPoolOpBase<Context>::HasStride;       \
  using ConvPoolOpBase<Context>::ws_
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_POOL_OP_BASE_H_
