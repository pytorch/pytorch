#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

template <typename T>
class CuDNNReluOp final : public Operator<CUDAContext> {
 public:
  CuDNNReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&device_context_),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc_));
  }

  ~CuDNNReluOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc_));
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ReshapeLike(X);
    // See if we need to reshape.
    if (X.dims() != cudnn_input_dims_) {
      CAFFE_VLOG(1) << "Setting descriptors.";
      cudnn_input_dims_ = X.dims();
      int C = (order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(3));
      int H = 1;
      int W = 1;
      if (X.ndim() == 4) {
        H = (order_ == StorageOrder::NCHW ? X.dim(2) : X.dim(1));
        W = (order_ == StorageOrder::NCHW ? X.dim(3) : X.dim(2));
      }
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          data_desc_, GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type, X.dim(0), C, H, W));
    }
    CUDNN_CHECK(cudnnActivationForward(cudnn_wrapper_.cudnn_handle(),
        CUDNN_ACTIVATION_RELU, cudnnTypeWrapper<T>::kOne(), data_desc_,
        X.template data<T>(), cudnnTypeWrapper<T>::kZero(),
        data_desc_, Y->template mutable_data<T>()));
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  vector<int> cudnn_input_dims_;
  StorageOrder order_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  IN_PLACE_ALLOWED({0, 0});
  DISABLE_COPY_AND_ASSIGN(CuDNNReluOp);
};


// Note: You can see that in CuDNNReluGradientOp, we abused the cudnn interface
// by passing in the output tensor for both bottom and top. This is dependent on
// the assumption that the Relu gradient actually does not rely on the bottom
// data, or it treats input=0 the same way as input<0. This is of course not
// very safe, but we have been running in this way in Caffe for a while so it
// *might* be safe to assume so.
template <typename T>
class CuDNNReluGradientOp final : public Operator<CUDAContext> {
 public:
  CuDNNReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&device_context_),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc_));
  }

  ~CuDNNReluGradientOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc_));
  }

  bool RunOnDevice() override {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ReshapeLike(Y);
    // See if we need to reshape.
    if (Y.dims() != cudnn_input_dims_) {
      CAFFE_VLOG(1) << "Setting descriptors.";
      cudnn_input_dims_ = Y.dims();
      int C = (order_ == StorageOrder::NCHW ? Y.dim(1) : Y.dim(3));
      int H = 1;
      int W = 1;
      if (Y.ndim() == 4) {
        H = (order_ == StorageOrder::NCHW ? Y.dim(2) : Y.dim(1));
        W = (order_ == StorageOrder::NCHW ? Y.dim(3) : Y.dim(2));
      }
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          data_desc_, GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T>::type, Y.dim(0), C, H, W));
    }
    const typename cudnnTypeWrapper<T>::ScalingParamType kOne = 1;
    const typename cudnnTypeWrapper<T>::ScalingParamType kZero = 0;
    CUDNN_CHECK(cudnnActivationBackward(cudnn_wrapper_.cudnn_handle(),
        CUDNN_ACTIVATION_RELU, &kOne, data_desc_, Y.template data<T>(),
        data_desc_, dY.template data<T>(), data_desc_, Y.template data<T>(),
        &kZero, data_desc_, dX->template mutable_data<T>()));
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  vector<int> cudnn_input_dims_;
  StorageOrder order_;
  // Input: Y, dY; Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  IN_PLACE_ALLOWED({1, 0});
  DISABLE_COPY_AND_ASSIGN(CuDNNReluGradientOp);
};

namespace {
REGISTER_CUDNN_OPERATOR(Relu, CuDNNReluOp<float>);
REGISTER_CUDNN_OPERATOR(ReluGradient, CuDNNReluGradientOp<float>);
//REGISTER_CUDNN_OPERATOR(ReluFp16, CuDNNReluOp<float16>);
//REGISTER_CUDNN_OPERATOR(ReluFp16Gradient, CuDNNReluGradientOp<float16>);
}  // namespace
}  // namespace caffe2
