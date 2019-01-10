#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"
#include "caffe2/operators/conv_pool_op_base.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class MKLConvOp final : public ConvPoolOpBase<MKLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(MKLContext);
  MKLConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<MKLContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        dilation_h() == 1 && dilation_w() == 1, "Dilation not supported.");
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW, "Only NCHW order supported.");
  }
  ~MKLConvOp() {}

  // TODO(jiayq): support double if needed.
  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = OperatorBase::Input<MKLMemory<float>>(INPUT);
    const auto& filter = OperatorBase::Input<MKLMemory<float>>(FILTER);

    const int M = filter.dim32(0);
    if (InputSize() == 2 && !zero_bias_) {
      TensorCPU cpu_zero_bias;
      cpu_zero_bias.Resize(M);
      CPUContext ctx;
      math::Set<T, CPUContext>(
          M, 0.0, cpu_zero_bias.template mutable_data<float>(), &ctx);

      zero_bias_.reset(new MKLMemory<T>(std::vector<TIndex>{M}));
      zero_bias_->CopyFrom(cpu_zero_bias);
    }
    const auto& bias = InputSize() == 2
        ? *zero_bias_
        : OperatorBase::Input<MKLMemory<float>>(BIAS);

    MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);
    CAFFE_ENFORCE(4 == X.ndim());
    const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
    CAFFE_ENFORCE(4 == filter.ndim());

    bool dims_changed;
    CHECK_INPUT_FILTER_DIMS(X, filter, dims_changed);
    if (dims_changed || FLAGS_caffe2_mkl_memonger_in_use) {
      CAFFE_ENFORCE(
          C == filter.dim32(1) * group_,
          "Convolution op: input channels does not match: # of input channels ",
          C,
          " is not equal to kernel channels * group:",
          filter.dim32(1),
          "*",
          group_);
      CAFFE_ENFORCE(
          M % group_ == 0,
          "The number of output channels is not divisible by group.");
      CAFFE_ENFORCE(filter.dim32(2) == kernel_h());
      CAFFE_ENFORCE(filter.dim32(3) == kernel_w());
      CAFFE_ENFORCE(bias.ndim() == 1);
      CAFFE_ENFORCE(bias.dim32(0) == M);

      size_t dimension = 4;
      size_t bdata_sizes[4] = {W, H, C, N};
      // We will utilize the SetOutputSize() function int he base class
      // with dummy TensorCPU input and output to calculate the sizes.
      TensorCPU dummy_input(X.dims());
      TensorCPU dummy_output;
      ConvPoolOpBase<MKLContext>::SetOutputSize(
          dummy_input, &dummy_output, M);
      size_t tdata_sizes[4] = {
          dummy_output.dim(3), dummy_output.dim(2),
          dummy_output.dim(1), dummy_output.dim(0)};
      size_t fdata_sizes[5] = {
          kernel_w(), kernel_h(), C / group_, M / group_, group_};
      size_t strides[2] = {stride_w(), stride_h()};
      int pads[2] = {-pad_l(), -pad_t()};

      if (group_ > 1) {
        primitive_.Reset(
            dnnGroupsConvolutionCreateForwardBias<float>,
            nullptr,
            dnnAlgorithmConvolutionDirect,
            group_,
            dimension,
            bdata_sizes,
            tdata_sizes,
            fdata_sizes,
            strides,
            pads,
            dnnBorderZeros);
      } else {
        primitive_.Reset(
            dnnConvolutionCreateForwardBias<float>,
            nullptr,
            dnnAlgorithmConvolutionDirect,
            dimension,
            bdata_sizes,
            tdata_sizes,
            fdata_sizes,
            strides,
            pads,
            dnnBorderZeros);
      }
      Y->Reset(dummy_output.dims(), primitive_, dnnResourceDst);
      buffer_.Reset(dummy_output.dims(), primitive_, dnnResourceDst, true);
      input_layout_.Reset(primitive_, dnnResourceSrc);
      filter_layout_.Reset(primitive_, dnnResourceFilter);
      bias_layout_.Reset(primitive_, dnnResourceBias);
    }

    // Try to share from the output: this allows us to avoid unnecessary copy
    // operations, if the output is already allocated and is having the same
    // layout as the buffer has.
    bool shared = buffer_.ShareFrom(*Y);

    std::shared_ptr<void> X_view = X.View(
        input_layout_, primitive_, dnnResourceSrc);
    std::shared_ptr<void> bias_view =
        bias.View(bias_layout_, primitive_, dnnResourceBias);
    std::shared_ptr<void> filter_view;
    if (group_ > 1) {
      // Explicitly reformat the buffer.
      MKLMemory<float> group_filter(
          std::vector<TIndex>{TIndex(group_),
                              TIndex(filter.dim32(0) / group_),
                              TIndex(filter.dim32(1)),
                              TIndex(filter.dim32(2)),
                              TIndex(filter.dim32(3))},
          nullptr,
          dnnResourceFilter,
          /*share_memory_if_possible=*/true);
      group_filter.CopyFrom(filter.buffer());
      filter_view =
          group_filter.View(filter_layout_, primitive_, dnnResourceFilter);
    } else {
      filter_view = filter.View(filter_layout_, primitive_, dnnResourceFilter);
    }

    resources_[dnnResourceSrc] = X_view.get(); // X.buffer();
    resources_[dnnResourceFilter] = filter_view.get();
    resources_[dnnResourceBias] = bias_view.get();
    resources_[dnnResourceDst] = buffer_.buffer();

    MKLDNN_SAFE_CALL(mkl::dnnExecute<T>(primitive_, resources_));
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    if (FLAGS_caffe2_mkl_memonger_in_use && !shared) {
      // buffer_ is not shared with Y. Free memory since it'll
      // be re-allocated in the next run anyway due to memonger in use.
      buffer_.Reset();
    }
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 private:
  // Input: X, W, b
  // Output: Y
  std::unique_ptr<MKLMemory<T>> zero_bias_;
  vector<TIndex> cached_input_dims_;
  vector<TIndex> cached_filter_dims_;
  PrimitiveWrapper<T> primitive_;
  LayoutWrapper<T> input_layout_;
  LayoutWrapper<T> filter_layout_;
  LayoutWrapper<T> bias_layout_;
  MKLMemory<T> buffer_;
  void* resources_[dnnResourceNumber] = {0};
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace mkl


REGISTER_MKL_OPERATOR(Conv, mkl::MKLConvOp<float>);

}  // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
