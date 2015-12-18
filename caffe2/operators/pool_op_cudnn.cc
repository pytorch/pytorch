#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename T>
class CuDNNPoolOp : public ConvPoolOpBase<CUDAContext> {
 public:
  CuDNNPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&device_context_) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));
    // Figure out the pooling descriptor.
    if (def().type() == "MaxPool") {
      mode_ = CUDNN_POOLING_MAX;
    } else if (def().type() == "AveragePool") {
      mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else {
      CAFFE_LOG_FATAL << "Unsupported pooling method: " << def().type();
    }
  }

  ~CuDNNPoolOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }

  bool RunOnDevice() final {
    auto& X = Input(0);
    auto* Y = Output(0);
    int N, C, H, W;
    switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim(0); H = X.dim(1); W = X.dim(2); C = X.dim(3);
      break;
    case StorageOrder::NCHW:
      N = X.dim(0); C = X.dim(1); H = X.dim(2); W = X.dim(3);
      break;
    default:
      CAFFE_LOG_FATAL << "Unknown storage order: " << order_;
    }
    ConvPoolOpBase::SetOutputSize(X, Y, C);

    if (cudnn_input_dims_ != X.dims()) {
      // Dimensions changed; we will need to re-initialize things.
      CAFFE_LOG_INFO << "Changing the cudnn descriptor configurations.";
      cudnn_input_dims_ = X.dims();
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          bottom_desc_, GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type, N, C, H, W));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          top_desc_, GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type, N, C,
          order_ == StorageOrder::NCHW ? Y->dim(2) : Y->dim(1),
          order_ == StorageOrder::NCHW ? Y->dim(3) : Y->dim(2)));
      if (pad_t_ != pad_l_ || pad_l_ != pad_r_) {
        CAFFE_LOG_FATAL << "Cudnn pooling only supports even padding on both sides.";
      }
      CUDNN_CHECK(cudnnSetPooling2dDescriptor(
          pooling_desc_, mode_, kernel_h_, kernel_w_, pad_t_, pad_l_,
          stride_h_, stride_w_));
    }
    // Carry out the pooling computation.
    const typename cudnnTypeWrapper<T>::ScalingParamType kOne = 1;
    const typename cudnnTypeWrapper<T>::ScalingParamType kZero = 0;
    CUDNN_CHECK(cudnnPoolingForward(
        cudnn_wrapper_.cudnn_handle(), pooling_desc_, &kOne,
        bottom_desc_, X.template data<T>(), &kZero, 
        top_desc_, Y->template mutable_data<T>()));
    return true;
  }

 protected:
  vector<int> cudnn_input_dims_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  cudnnPoolingMode_t mode_;

  DISABLE_COPY_AND_ASSIGN(CuDNNPoolOp);
};

namespace {
REGISTER_CUDNN_OPERATOR(AveragePool, CuDNNPoolOp<float>);
REGISTER_CUDNN_OPERATOR(MaxPool, CuDNNPoolOp<float>);
}
}  // namespace caffe2
