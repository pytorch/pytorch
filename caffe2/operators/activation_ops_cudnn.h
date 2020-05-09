#ifndef CAFFE2_OPERATORS_ACTIVATION_OPS_CUDNN_H_
#define CAFFE2_OPERATORS_ACTIVATION_OPS_CUDNN_H_

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"

namespace caffe2 {

class CuDNNActivationOpBase : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  template <class... Args>
  explicit CuDNNActivationOpBase(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...),
        cudnn_wrapper_(&context_) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateActivationDescriptor(&act_desc_));
  }

  virtual ~CuDNNActivationOpBase() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(act_desc_));
  }

 protected:
  void SetTensorDescriptor(
      const cudnnDataType_t data_type,
      const int data_size) {
    if (data_size != input_size_) {
      // Since the best performance is obtained when the tensor is HW-packed, we
      // put X.size() to W.
      input_size_ = data_size;
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          data_desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          data_type,
          1,
          1,
          1,
          input_size_));
    }
  }

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnActivationDescriptor_t act_desc_;

  int input_size_ = 0;
};

template <cudnnActivationMode_t kCuDNNActivationMode>
class CuDNNActivationOp final : public CuDNNActivationOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  template <class... Args>
  explicit CuDNNActivationOp(Args&&... args)
      : CuDNNActivationOpBase(std::forward<Args>(args)...) {
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        act_desc_, kCuDNNActivationMode, CUDNN_PROPAGATE_NAN, 0.0));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);

    auto* Y = Output(0, X.sizes(), at::dtype<T>());
    if (X.numel() == 0) {
      Y->template mutable_data<T>();
      return true;
    }
    this->SetTensorDescriptor(cudnnTypeWrapper<T>::type, X.numel());
    CUDNN_ENFORCE(cudnnActivationForward(
        this->cudnn_wrapper_.inline_cudnn_handle(),
        this->act_desc_,
        cudnnTypeWrapper<T>::kOne(),
        this->data_desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        this->data_desc_,
        Y->template mutable_data<T>()));
    return true;
  }
};

template <cudnnActivationMode_t kCuDNNActivationMode>
class CuDNNActivationGradientOp final : public CuDNNActivationOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  template <class... Args>
  explicit CuDNNActivationGradientOp(Args&&... args)
      : CuDNNActivationOpBase(std::forward<Args>(args)...) {
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        act_desc_, kCuDNNActivationMode, CUDNN_PROPAGATE_NAN, 0.0));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& Y = Input(0);
    const auto& dY = Input(1);

    auto* dX = Output(0, Y.sizes(), at::dtype<T>());
    if (Y.numel() == 0) {
      dX->template mutable_data<T>();
      return true;
    }
    this->SetTensorDescriptor(cudnnTypeWrapper<T>::type, Y.numel());
    CUDNN_ENFORCE(cudnnActivationBackward(
        this->cudnn_wrapper_.inline_cudnn_handle(),
        this->act_desc_,
        cudnnTypeWrapper<T>::kOne(),
        this->data_desc_,
        Y.template data<T>(),
        this->data_desc_,
        dY.template data<T>(),
        this->data_desc_,
        Y.template data<T>(), // Use Y_data as placeholder here.
        cudnnTypeWrapper<T>::kZero(),
        this->data_desc_,
        dX->template mutable_data<T>()));
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ACTIVATION_OPS_CUDNN_H_
