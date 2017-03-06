#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

namespace {
constexpr int NUM_DESCRIPTORS = 2;
constexpr int GRADIENT_NUM_DESCRIPTORS = 3;
constexpr int BOTTOM_DESC_ID = 0;
constexpr int TOP_DESC_ID = 1;
constexpr int TOP_GRADIENT_DESC_ID = 2;
}  // namespace

template <typename T>
class CuDNNSoftmaxOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        cudnn_wrapper_(&context_) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&desc_));
  }

  ~CuDNNSoftmaxOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(desc_));
  }

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    DCHECK_EQ(X.ndim(), 2);
    Y->ResizeLike(X);
    if (dims_ != X.dims()) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<T>::type,
          X.dim32(0),
          X.dim32(1),
          1,
          1));
      dims_ = X.dims();
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
        Y->template mutable_data<T>()));
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t desc_;
  vector<TIndex> dims_;
};


template <typename T>
class CuDNNSoftmaxGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        cudnn_wrapper_(&context_) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&desc_));
  }

  ~CuDNNSoftmaxGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(desc_));
  }

  bool RunOnDevice() override {
    auto& Y = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    DCHECK_EQ(Y.ndim(), 2);
    DCHECK(Y.dims() == dY.dims());
    int N = Y.dim32(0);
    int D = Y.dim32(1);
    DCHECK_EQ(dY.dim32(0), N);
    DCHECK_EQ(dY.dim32(1), D);
    dX->ResizeLike(Y);
    if (dims_ != Y.dims()) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          desc_,
          GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<T>::type,
          Y.dim32(0),
          Y.dim32(1),
          1,
          1));
      dims_ = Y.dims();
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
        dX->template mutable_data<T>()));
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t desc_;
  vector<TIndex> dims_;
};

namespace {
REGISTER_CUDNN_OPERATOR(Softmax, CuDNNSoftmaxOp<float>);
REGISTER_CUDNN_OPERATOR(SoftmaxGradient, CuDNNSoftmaxGradientOp<float>);
REGISTER_CUDNN_OPERATOR(SoftmaxFp16, CuDNNSoftmaxOp<float16>);
REGISTER_CUDNN_OPERATOR(SoftmaxFp16Gradient, CuDNNSoftmaxGradientOp<float16>);
}  // namespace
}  // namespace caffe2
