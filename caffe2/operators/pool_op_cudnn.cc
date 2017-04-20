#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

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
    int N = 0, C = 0, H = 0, W = 0;
    switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0); H = X.dim32(1); W = X.dim32(2); C = X.dim32(3);
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0); C = X.dim32(1); H = X.dim32(2); W = X.dim32(3);
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
    }
    ConvPoolOpBase::SetOutputSize(X, Y, C);

    if (cudnn_input_dims_ != X.dims()) {
      // Dimensions changed; we will need to re-initialize things.
      VLOG(1) << "Changing the cudnn descriptor configurations.";
      cudnn_input_dims_ = X.dims();
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          bottom_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          N,
          C,
          H,
          W));
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          N,
          C,
          order_ == StorageOrder::NCHW ? Y->dim32(2) : Y->dim32(1),
          order_ == StorageOrder::NCHW ? Y->dim32(3) : Y->dim32(2)));
      if (pad_t() != pad_l() || pad_l() != pad_r()) {
        CAFFE_ENFORCE(
            legacy_pad_ == LegacyPadding::CAFFE_LEGACY_POOLING,
            "Cudnn pooling only supports even padding on both sides, with "
            "the only exception of the caffe legacy pooling case where we "
            "try to preserve backward compatibility with Caffe.");
      }
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
    dX->ResizeLike(X);
    int N = 0, C = 0, H = 0, W = 0;
    switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0); H = X.dim32(1); W = X.dim32(2); C = X.dim32(3);
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0); C = X.dim32(1); H = X.dim32(2); W = X.dim32(3);
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
    }
    ConvPoolOpBase<CUDAContext>::ComputePads({H, W});

    if (cudnn_input_dims_ != X.dims()) {
      // Dimensions changed; we will need to re-initialize things.
      VLOG(1) << "Changing the cudnn descriptor configurations.";
      cudnn_input_dims_ = X.dims();
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          bottom_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          N,
          C,
          H,
          W));
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type,
          N,
          C,
          order_ == StorageOrder::NCHW ? Y.dim32(2) : Y.dim32(1),
          order_ == StorageOrder::NCHW ? Y.dim32(3) : Y.dim32(2)));
      if (pad_t() != pad_l() || pad_l() != pad_r()) {
        CAFFE_ENFORCE(
            legacy_pad_ == LegacyPadding::CAFFE_LEGACY_POOLING,
            "Cudnn pooling only supports even padding on both sides, with "
            "the only exception of the caffe legacy pooling case where we "
            "try to preserve backward compatibility with Caffe.");
      }
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
