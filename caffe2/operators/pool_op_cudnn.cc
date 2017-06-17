#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

namespace {

template <typename T>
void setTensorDescriptor(
    const int size,
    const StorageOrder order,
    const int N,
    const int C,
    const int H,
    const int W,
    const int D,
    cudnnTensorDescriptor_t& desc) {
  if (size == 4) {
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        desc,
        GetCudnnTensorFormat(order),
        cudnnTypeWrapper<T>::type,
        N,
        C,
        H,
        W));
  } else {
    vector<int> dims = {N, C, H, W, D};
    vector<int> strides;
    order == NCHW
        ? strides.insert(strides.end(), {C * H * W * D, H * W * D, W * D, D, 1})
        : strides.insert(
              strides.end(), {H * W * D * C, 1, W * D * C, D * C, C});
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        desc,
        cudnnTypeWrapper<T>::type,
        size > 3 ? size : 4,
        dims.data(),
        strides.data()));
  }
}

} // namespace

class CuDNNPoolOp : public ConvPoolOpBase<CUDAContext> {
 public:
  CuDNNPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_ENFORCE(cudnnCreatePoolingDescriptor(&pooling_desc_));
    // Figure out the pooling descriptor.
    if (def().type().substr(0, 7) == "MaxPool") {
#if CUDNN_VERSION_MIN(6,0,0)
      mode_ = CUDNN_POOLING_MAX_DETERMINISTIC;
#else
      mode_ = CUDNN_POOLING_MAX;
#endif
    } else if (def().type().substr(0, 11) == "AveragePool") {
      mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << def().type();
    }
  }

  ~CuDNNPoolOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_ENFORCE(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType() {
    auto& X = Input(0);
    auto* Y = Output(0);
    int N = 0, C = 0, H = 0, W = 0, D = 0;
    int H_out = 0, W_out = 0, D_out = 0;

    // cuDNN pooling support only 2 and 3 spatial dimensions.
    CAFFE_ENFORCE(X.ndim() >= 4 && X.ndim() <= 5);

    switch (order_) {
      case StorageOrder::NHWC:
        N = X.dim32(0);
        H = X.dim32(1);
        W = X.ndim() > 3 ? X.dim32(2) : 1;
        D = X.ndim() > 4 ? X.dim32(3) : 1;
        C = X.dim32(X.ndim() - 1);
        ConvPoolOpBase::SetOutputSize(X, Y, C);
        H_out = Y->dim32(1);
        W_out = Y->ndim() > 3 ? Y->dim32(2) : 1;
        D_out = Y->ndim() > 4 ? Y->dim32(3) : 1;
        break;
      case StorageOrder::NCHW:
        N = X.dim32(0);
        C = X.dim32(1);
        H = X.dim32(2);
        W = X.ndim() > 3 ? X.dim32(3) : 1;
        D = X.ndim() > 4 ? X.dim32(4) : 1;
        ConvPoolOpBase::SetOutputSize(X, Y, C);
        H_out = Y->dim32(2);
        W_out = Y->ndim() > 3 ? Y->dim32(3) : 1;
        D_out = Y->ndim() > 4 ? Y->dim32(4) : 1;
        break;
      default:
        LOG(FATAL) << "Unknown storage order: " << order_;
    }

    if (cudnn_input_dims_ != X.dims()) {
      // Dimensions changed; we will need to re-initialize things.
      VLOG(1) << "Changing the cudnn descriptor configurations.";
      cudnn_input_dims_ = X.dims();
      setTensorDescriptor<T>(X.ndim(), order_, N, C, H, W, D, bottom_desc_);
      setTensorDescriptor<T>(
          Y->ndim(), order_, N, C, H_out, W_out, D_out, top_desc_);
      for (int i = 0; i < kernel_.size(); ++i) {
        if (pads_[i] != pads_[kernel_.size() + i]) {
          CAFFE_ENFORCE(
              legacy_pad_ == LegacyPadding::CAFFE_LEGACY_POOLING,
              "Cudnn pooling only supports even padding on both sides, with "
              "the only exception of the caffe legacy pooling case where we "
              "try to preserve backward compatibility with Caffe.");
        }
      }
      if (kernel_.size() == 2) {
        CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(
            pooling_desc_,
            mode_,
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_h(),
            kernel_w(),
            pad_t(),
            pad_l(),
            stride_h(),
            stride_w()));
      } else {
        CUDNN_ENFORCE(cudnnSetPoolingNdDescriptor(
            pooling_desc_,
            mode_,
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_.size(),
            kernel_.data(),
            pads_.data(),
            stride_.data()));
      }
    }
    // Carry out the pooling computation.
    CUDNN_ENFORCE(cudnnPoolingForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        pooling_desc_,
        cudnnTypeWrapper<T>::kOne(),
        bottom_desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        top_desc_,
        Y->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() final {
    auto& X = Input(0);
    auto* Y = Output(0);

    if (X.IsType<float>()) {
      return DoRunWithType<float,float>();
    } else if (X.IsType<float16>()) {
      return DoRunWithType<float16,float>();
    } else {
      LOG(FATAL) << "Unsupported input types";
    }
    return true;
  }

 protected:
  vector<TIndex> cudnn_input_dims_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  cudnnPoolingMode_t mode_;
 private:
};

class CuDNNPoolGradientOp : public ConvPoolOpBase<CUDAContext> {
 public:
  CuDNNPoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_ENFORCE(cudnnCreatePoolingDescriptor(&pooling_desc_));
    // Figure out the pooling descriptor.
    if (def().type() == "MaxPoolGradient") {
      mode_ = CUDNN_POOLING_MAX;
    } else if (def().type() == "AveragePoolGradient") {
      mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << def().type();
    }
  }

  ~CuDNNPoolGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_ENFORCE(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType() {
    auto& X = Input(0);
    auto& Y = Input(1);
    auto& dY = Input(2);
    auto* dX = Output(0);

    // cuDNN pooling support only 2 and 3 spatial dimensions.
    CAFFE_ENFORCE(X.ndim() >= 4 && X.ndim() <= 5);

    dX->ResizeLike(X);
    int N = 0, C = 0, H = 0, W = 0, D = 0;
    int H_out = 0, W_out = 0, D_out = 0;
    switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.ndim() > 3 ? X.dim32(2) : 1;
      D = X.ndim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.ndim() - 1);
      H_out = Y.dim32(1);
      W_out = Y.ndim() > 3 ? Y.dim32(2) : 1;
      D_out = Y.ndim() > 4 ? Y.dim32(3) : 1;
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.ndim() > 3 ? X.dim32(3) : 1;
      D = X.ndim() > 4 ? X.dim32(4) : 1;
      H_out = Y.dim32(2);
      W_out = Y.ndim() > 3 ? Y.dim32(3) : 1;
      D_out = Y.ndim() > 4 ? Y.dim32(4) : 1;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
    }

    if (kernel_.size() == 1) {
      ConvPoolOpBase<CUDAContext>::ComputePads({H});
    } else if (kernel_.size() == 2) {
      ConvPoolOpBase<CUDAContext>::ComputePads({H, W});
    } else if (kernel_.size() == 3) {
      ConvPoolOpBase<CUDAContext>::ComputePads({H, W, D});
    } else {
      CAFFE_THROW("Unsupported kernel size :", kernel_.size());
    }

    if (cudnn_input_dims_ != X.dims()) {
      // Dimensions changed; we will need to re-initialize things.
      VLOG(1) << "Changing the cudnn descriptor configurations.";
      cudnn_input_dims_ = X.dims();
      setTensorDescriptor<T>(X.ndim(), order_, N, C, H, W, D, bottom_desc_);
      setTensorDescriptor<T>(
          Y.ndim(), order_, N, C, H_out, W_out, D_out, top_desc_);
      if (pad_t() != pad_l() || pad_l() != pad_r()) {
        CAFFE_ENFORCE(
            legacy_pad_ == LegacyPadding::CAFFE_LEGACY_POOLING,
            "Cudnn pooling only supports even padding on both sides, with "
            "the only exception of the caffe legacy pooling case where we "
            "try to preserve backward compatibility with Caffe.");
      }
      if (kernel_.size() == 2) {
        CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(
            pooling_desc_,
            mode_,
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_h(),
            kernel_w(),
            pad_t(),
            pad_l(),
            stride_h(),
            stride_w()));
      } else {
        CUDNN_ENFORCE(cudnnSetPoolingNdDescriptor(
            pooling_desc_,
            mode_,
            CUDNN_NOT_PROPAGATE_NAN,
            kernel_.size(),
            kernel_.data(),
            pads_.data(),
            stride_.data()));
      }
    }
    // Carry out the pooling computation.
    CUDNN_ENFORCE(cudnnPoolingBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        pooling_desc_,
        cudnnTypeWrapper<T>::kOne(),
        top_desc_,
        Y.template data<T>(),
        top_desc_,
        dY.template data<T>(),
        bottom_desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        bottom_desc_,
        dX->template mutable_data<T>()));
    return true;
  }

  bool RunOnDevice() final {
    auto& X = Input(0);
    auto& Y = Input(1);
    auto& dY = Input(2);
    auto* dX = Output(0);
    dX->ResizeLike(X);

    if (X.IsType<float>()) {
      return DoRunWithType<float,float>();
    } else if (X.IsType<float16>()) {
      return DoRunWithType<float16,float>();
    } else {
      LOG(FATAL) << "Unsupported input types";
    }
    return true;
  }

 protected:
  vector<TIndex> cudnn_input_dims_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  cudnnPoolingMode_t mode_;

  // Input: X, Y, dY
  // Output: dX
  INPUT_TAGS(IN, OUT, OUT_GRAD);
};

namespace {
REGISTER_CUDNN_OPERATOR(AveragePool, CuDNNPoolOp);
REGISTER_CUDNN_OPERATOR(AveragePoolGradient, CuDNNPoolGradientOp);
REGISTER_CUDNN_OPERATOR(MaxPool, CuDNNPoolOp);
REGISTER_CUDNN_OPERATOR(MaxPoolGradient, CuDNNPoolGradientOp);

}  // namespace
}  // namespace caffe2
