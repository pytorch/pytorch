#include "caffe2/operators/relu_n_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"

namespace caffe2 {

namespace {

class CuDNNReluNOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNReluNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        OP_SINGLE_ARG(float, "n", n_, 6.0f) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        act_desc_,
        CUDNN_ACTIVATION_CLIPPED_RELU,
        CUDNN_PROPAGATE_NAN,
        static_cast<double>(n_)));
  }

  ~CuDNNReluNOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(act_desc_));
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    if (X.size() == 0) {
      Y->mutable_data<float>();
      return true;
    }
    if (input_size_ != X.size()) {
      input_size_ = X.size();
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<float>::type,
          1,
          1,
          1,
          input_size_));
    }
    CUDNN_ENFORCE(cudnnActivationForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        act_desc_,
        cudnnTypeWrapper<float>::kOne(),
        data_desc_,
        X.data<float>(),
        cudnnTypeWrapper<float>::kZero(),
        data_desc_,
        Y->mutable_data<float>()));
    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnActivationDescriptor_t act_desc_;

  int input_size_ = 0;

  const float n_;
};

class CuDNNReluNGradientOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNReluNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        OP_SINGLE_ARG(float, "n", n_, 6.0f) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        act_desc_,
        CUDNN_ACTIVATION_CLIPPED_RELU,
        CUDNN_PROPAGATE_NAN,
        static_cast<double>(n_)));
  }

  ~CuDNNReluNGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(act_desc_));
  }

  bool RunOnDevice() override {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(Y);
    if (Y.size() == 0) {
      dX->mutable_data<float>();
      return true;
    }
    if (input_size_ != Y.size()) {
      input_size_ = Y.size();
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<float>::type,
          1,
          1,
          1,
          input_size_));
    }
    CUDNN_ENFORCE(cudnnActivationBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        act_desc_,
        cudnnTypeWrapper<float>::kOne(),
        data_desc_,
        Y.data<float>(),
        data_desc_,
        dY.data<float>(),
        data_desc_,
        Y.data<float>(),
        cudnnTypeWrapper<float>::kZero(),
        data_desc_,
        dX->mutable_data<float>()));
    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnActivationDescriptor_t act_desc_;

  int input_size_ = 0;

  const float n_;
};

} // namespace

REGISTER_CUDNN_OPERATOR(ReluN, CuDNNReluNOp);
REGISTER_CUDNN_OPERATOR(ReluNGradient, CuDNNReluNGradientOp);

} // namespace caffe2
