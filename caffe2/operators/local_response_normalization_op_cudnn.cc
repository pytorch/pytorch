#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

class CuDNNLRNOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNLRNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        size_(OperatorBase::GetSingleArgument<int>("size", 0)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0)),
        bias_(OperatorBase::GetSingleArgument<float>("bias", 1)) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

    CUDNN_ENFORCE(cudnnCreateLRNDescriptor(&norm_desc_));
    CUDNN_ENFORCE(
        cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, bias_));
  }

  ~CuDNNLRNOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyLRNDescriptor(norm_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType();

  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnLRNDescriptor_t norm_desc_;

  vector<int64_t> cudnn_input_dims_;

  const int size_;
  const float alpha_;
  const float beta_;
  const float bias_;

  // Input: X, Output: Y
};

class CuDNNLRNGradientOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  CuDNNLRNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        size_(OperatorBase::GetSingleArgument<int>("size", 0)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0)),
        bias_(OperatorBase::GetSingleArgument<float>("bias", 1)) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

    CUDNN_ENFORCE(cudnnCreateLRNDescriptor(&norm_desc_));
    CUDNN_ENFORCE(
        cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, bias_));
  }

  ~CuDNNLRNGradientOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyLRNDescriptor(norm_desc_));
  }

  template <typename T, typename M>
  bool DoRunWithType();

  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnLRNDescriptor_t norm_desc_;

  vector<int64_t> cudnn_input_dims_;

  const int size_;
  const float alpha_;
  const float beta_;
  const float bias_;

  // Input: X, Y, dY
  // Output: dX
};

template <typename T, typename M>
bool CuDNNLRNOp::DoRunWithType() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  // Reshape tensor descriptors if necessary
  if (X.sizes() != cudnn_input_dims_) {
    VLOG(1) << "Setting descriptors";
    cudnn_input_dims_ = X.sizes().vec();
    int C = 1, H = 1, W = 1;
    // Normal 4-dimensional tensors for images.
    C = X.dim32(1);
    H = X.dim32(2);
    W = X.dim32(3);
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        data_desc_,
        GetCudnnTensorFormat(StorageOrder::NCHW),
        cudnnTypeWrapper<T>::type,
        X.dim32(0),
        C,
        H,
        W));
  }

  // now actually run the computation
  CUDNN_ENFORCE(cudnnLRNCrossChannelForward(
      cudnn_wrapper_.inline_cudnn_handle(),
      norm_desc_,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      cudnnTypeWrapper<T>::kOne(),
      data_desc_,
      X.template data<T>(),
      cudnnTypeWrapper<T>::kZero(),
      data_desc_,
      Y->template mutable_data<T>()));

  return true;
}

bool CuDNNLRNOp::RunOnDevice() {
  // dispatch based on contents of tensor(s)
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  if (X.IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (X.IsType<at::Half>()) {
    return DoRunWithType<at::Half, float>();
  } else {
    CAFFE_THROW("Unsupported input type");
  }
  return false;
}

template <typename T, typename M>
bool CuDNNLRNGradientOp::DoRunWithType() {
  const auto& X = Input(0);
  const auto& Y = Input(1);
  const auto& dY = Input(2);
  auto* dX = Output(0);

  if (dY.sizes() != cudnn_input_dims_) {
    VLOG(1) << "Setting descriptors";
    cudnn_input_dims_ = dY.sizes().vec();
    int C = 1, H = 1, W = 1;
    // Normal 4-dimensional tensors for images.
    C = dY.dim32(1);
    H = dY.dim32(2);
    W = dY.dim32(3);
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        data_desc_,
        GetCudnnTensorFormat(StorageOrder::NCHW),
        cudnnTypeWrapper<T>::type,
        dY.dim32(0),
        C,
        H,
        W));
  }

  // run the computation
  CUDNN_ENFORCE(cudnnLRNCrossChannelBackward(
      cudnn_wrapper_.inline_cudnn_handle(),
      norm_desc_,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      cudnnTypeWrapper<T>::kOne(),
      data_desc_,
      Y.template data<T>(),
      data_desc_,
      dY.template data<T>(),
      data_desc_,
      X.template data<T>(),
      cudnnTypeWrapper<T>::kZero(),
      data_desc_,
      dX->template mutable_data<T>()));
  return true;
}

bool CuDNNLRNGradientOp::RunOnDevice() {
  // dispatch based on contents of tensor(s)
  const auto& X = Input(0);
  const auto& Y = Input(1);
  const auto& dY = Input(2);
  auto* dX = Output(0);

  dX->ResizeLike(dY);

  if (dY.IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (dY.IsType<at::Half>()) {
    return DoRunWithType<at::Half, float>();
  } else {
    CAFFE_THROW("Unsupported input type");
  }

  return false;
}

namespace {
REGISTER_CUDNN_OPERATOR(LRN, CuDNNLRNOp);
REGISTER_CUDNN_OPERATOR(LRNGradient, CuDNNLRNGradientOp);
}

}; // namespace caffe2
