#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class ConvMKLDNNOp final : public ConvPoolOpBase<CPUContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  ConvMKLDNNOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        dilation_h_ == 1 && dilation_w_ == 1, "Dilation not supported.");
    OPERATOR_NEEDS_FEATURE(
        pad_l_ == pad_r_ && pad_t_ == pad_b_, "Uneven padding not supported.");
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW, "Only NCHW order supported.");
  }
  ~ConvMKLDNNOp() {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto& filter = Input(FILTER);
    auto& bias = Input(BIAS);
    TensorCPU* Y = Output(0);
    CAFFE_ENFORCE(4 == X.ndim());
    const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
    CAFFE_ENFORCE(4 == filter.ndim());
    const int M = filter.dim32(0);
    CAFFE_ENFORCE(
        C == filter.dim32(1),
        "Convolution op: # of input channels ",
        C,
        " is not equal to kernel channels:",
        filter.dim32(1));
    CAFFE_ENFORCE(filter.dim32(2) == kernel_h_);
    CAFFE_ENFORCE(filter.dim32(3) == kernel_w_);
    CAFFE_ENFORCE(bias.ndim() == 1);
    CAFFE_ENFORCE(bias.dim32(0) == M);
    ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));

    if (cached_input_dims_ != X.dims() ||
        cached_filter_dims_ != filter.dims()) {
      cached_input_dims_ = X.dims();
      cached_filter_dims_ = filter.dims();
      // In order to create an internal layout, let's use convolution as
      // primitive.
      size_t dimension = 4;
      size_t bdata_sizes[4] = {W, H, C, N};
      size_t bdata_offsets[4] = {1, W, W * H, W * H * C};
      size_t tdata_sizes[4] = {Y->dim(3), Y->dim(2), Y->dim(1), Y->dim(0)};
      size_t fdata_sizes[4] = {kernel_w_, kernel_h_, C, M};
      size_t strides[2] = {stride_w_, stride_h_};
      int pads[2] = {-pad_l_, -pad_t_};

      MKLDNN_SAFE_CALL(dnnConvolutionCreateForwardBias<float>(
          primitive_.ptr(),
          nullptr,
          dnnAlgorithmConvolutionDirect,
          dimension,
          bdata_sizes,
          tdata_sizes,
          fdata_sizes,
          strides,
          pads,
          dnnBorderZeros));
      X_wrapper_.reset(
          new InternalResourceWrapper<T>(X, primitive_.ref(), dnnResourceSrc));
      filter_wrapper_.reset(new InternalResourceWrapper<T>(
          filter, primitive_.ref(), dnnResourceFilter));
      bias_wrapper_.reset(new InternalResourceWrapper<T>(
          bias, primitive_.ref(), dnnResourceBias));
      Y_wrapper_.reset(
          new InternalResourceWrapper<T>(*Y, primitive_.ref(), dnnResourceDst));
      resources_[dnnResourceSrc] = X_wrapper_->buffer();
      resources_[dnnResourceFilter] = filter_wrapper_->buffer();
      resources_[dnnResourceBias] = bias_wrapper_->buffer();
      resources_[dnnResourceDst] = Y_wrapper_->buffer();
    }
    X_wrapper_->CopyIn(X);
    filter_wrapper_->CopyIn(filter);
    bias_wrapper_->CopyIn(bias);

    MKLDNN_SAFE_CALL(dnnExecute<float>(primitive_.ref(), resources_));
    Y_wrapper_->CopyOut(Y);
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 private:
  // Input: X, W, b
  // Output: Y
  vector<TIndex> cached_input_dims_;
  vector<TIndex> cached_filter_dims_;
  PrimitiveWrapper<T> primitive_;
  unique_ptr<InternalResourceWrapper<T>> X_wrapper_ = nullptr;
  unique_ptr<InternalResourceWrapper<T>> filter_wrapper_ = nullptr;
  unique_ptr<InternalResourceWrapper<T>> bias_wrapper_ = nullptr;
  unique_ptr<InternalResourceWrapper<T>> Y_wrapper_ = nullptr;
  void* resources_[dnnResourceNumber] = {0};
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace mkl

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, MKLDNN, mkl::ConvMKLDNNOp<float>);

}  // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
