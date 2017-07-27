#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"
#include "caffe2/operators/conv_pool_op_base.h"

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
        dilation_h() == 1 && dilation_w() == 1, "Dilation not supported.");
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
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
    CAFFE_ENFORCE(filter.dim32(2) == kernel_h());
    CAFFE_ENFORCE(filter.dim32(3) == kernel_w());
    CAFFE_ENFORCE(bias.ndim() == 1);
    CAFFE_ENFORCE(bias.dim32(0) == M);
    ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
    // Pre-allocate Y so we can potentially share memory if applicable.
    Y->mutable_data<T>();

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
      size_t fdata_sizes[4] = {kernel_w(), kernel_h(), C, M};
      size_t strides[2] = {stride_w(), stride_h()};
      int pads[2] = {-pad_l(), -pad_t()};

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
      X_wrapper_.reset(
          new MKLMemory<T>(X.dims(), primitive_, dnnResourceSrc, true));
      filter_wrapper_.reset(
          new MKLMemory<T>(filter.dims(), primitive_, dnnResourceFilter, true));
      bias_wrapper_.reset(
          new MKLMemory<T>(bias.dims(), primitive_, dnnResourceBias, true));
      Y_wrapper_.reset(
          new MKLMemory<T>(Y->dims(), primitive_, dnnResourceDst, true));
      X_wrapper_->CopyFrom(X);
      filter_wrapper_->CopyFrom(filter);
      bias_wrapper_->CopyFrom(bias);
      Y_wrapper_->ShareFromTensor(*Y);
      resources_[dnnResourceSrc] = X_wrapper_->buffer();
      resources_[dnnResourceFilter] = filter_wrapper_->buffer();
      resources_[dnnResourceBias] = bias_wrapper_->buffer();
      resources_[dnnResourceDst] = Y_wrapper_->buffer();
    } else {
      X_wrapper_->CopyFrom(X);
      filter_wrapper_->CopyFrom(filter);
      bias_wrapper_->CopyFrom(bias);
      Y_wrapper_->ShareFromTensor(*Y);
    }
    MKLDNN_SAFE_CALL(dnnExecute<float>(primitive_, resources_));
    Y_wrapper_->CopyTo(Y);
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
  unique_ptr<MKLMemory<T>> X_wrapper_ = nullptr;
  unique_ptr<MKLMemory<T>> filter_wrapper_ = nullptr;
  unique_ptr<MKLMemory<T>> bias_wrapper_ = nullptr;
  unique_ptr<MKLMemory<T>> Y_wrapper_ = nullptr;
  void* resources_[dnnResourceNumber] = {0};
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace mkl

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, MKLDNN, mkl::ConvMKLDNNOp<float>);

}  // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
