#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

class CuDNNSoftmaxOp final : public Operator<CUDAContext> {
 public:
  template <class... Args>
  explicit CuDNNSoftmaxOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...),
        cudnn_wrapper_(&context_),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&desc_));
  }

  ~CuDNNSoftmaxOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& X = Input(0);

    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int N = X.size_to_dim(canonical_axis);
    const int D = X.size_from_dim(canonical_axis);

    auto* Y = Output(0, X.sizes(), at::dtype<T>());
    auto* Y_data = Y->template mutable_data<T>();
    if (N == 0 || D == 0) {
      return true;
    }
    if (dims_ != X.sizes()) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<T>::type,
          N,
          D,
          1,
          1));
      dims_ = X.sizes().vec();
    }
    CUDNN_ENFORCE(cudnnSoftmaxForward(
        cudnn_wrapper_.inline_cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        cudnnTypeWrapper<T>::kOne(),
        desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        desc_,
        Y_data));
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  int axis_;
  cudnnTensorDescriptor_t desc_;
  vector<int64_t> dims_;
};

class CuDNNSoftmaxGradientOp final : public Operator<CUDAContext> {
 public:
  template <class... Args>
  explicit CuDNNSoftmaxGradientOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...),
        cudnn_wrapper_(&context_),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&desc_));
  }

  ~CuDNNSoftmaxGradientOp() override {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(desc_));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& Y = Input(0);
    auto& dY = Input(1);

    const auto canonical_axis = Y.canonical_axis_index(axis_);
    const int N = Y.size_to_dim(canonical_axis);
    const int D = Y.size_from_dim(canonical_axis);

    TORCH_CHECK_EQ(Y.sizes(), dY.sizes());
    auto* dX = Output(0, Y.sizes(), at::dtype<T>());
    auto* dX_data = dX->template mutable_data<T>();
    if (N == 0 || D == 0) {
      return true;
    }
    if (dims_ != Y.sizes()) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<T>::type,
          N,
          D,
          1,
          1));
      dims_ = Y.sizes().vec();
    }
    CUDNN_ENFORCE(cudnnSoftmaxBackward(
        cudnn_wrapper_.inline_cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        cudnnTypeWrapper<T>::kOne(),
        desc_,
        Y.template data<T>(),
        desc_,
        dY.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        desc_,
        dX_data));
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  int axis_;
  cudnnTensorDescriptor_t desc_;
  vector<int64_t> dims_;
};

namespace {
REGISTER_CUDNN_OPERATOR(Softmax, CuDNNSoftmaxOp);
REGISTER_CUDNN_OPERATOR(SoftmaxGradient, CuDNNSoftmaxGradientOp);
} // namespace
} // namespace caffe2
