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
        cudnn_wrapper_(&device_context_) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
  }

  ~CuDNNSoftmaxOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
  }

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_DCHECK_EQ(X.ndim(), 2);
    Y->ReshapeLike(X);
    if (dims_ != X.dims()) {
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          desc_, GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<T>::type, X.dim(0), X.dim(1), 1, 1));
      dims_ = X.dims();
    }
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn_wrapper_.cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        cudnnTypeWrapper<T>::kOne(), desc_, X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(), desc_, Y->template mutable_data<T>()));
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t desc_;
  vector<int> dims_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CuDNNSoftmaxOp);
};


template <typename T>
class CuDNNSoftmaxGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        cudnn_wrapper_(&device_context_) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
  }

  ~CuDNNSoftmaxGradientOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
  }

  bool RunOnDevice() override {
    auto& Y = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    CAFFE_DCHECK_EQ(Y.ndim(), 2);
    CAFFE_DCHECK_EQ(Y.dims(), dY.dims());
    int N = Y.dim(0);
    int D = Y.dim(1);
    CAFFE_DCHECK_EQ(dY.dim(0), N);
    CAFFE_DCHECK_EQ(dY.dim(1), D);
    dX->ReshapeLike(Y);
    if (dims_ != Y.dims()) {
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          desc_, GetCudnnTensorFormat(StorageOrder::NCHW),
          cudnnTypeWrapper<T>::type, Y.dim(0), Y.dim(1), 1, 1));
      dims_ = Y.dims();
    }
    CUDNN_CHECK(cudnnSoftmaxBackward(cudnn_wrapper_.cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        cudnnTypeWrapper<T>::kOne(), desc_, Y.template data<T>(),
        desc_, dY.template data<T>(), cudnnTypeWrapper<T>::kZero(), desc_,
        dX->template mutable_data<T>()));
    return true;
  }

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t desc_;
  vector<int> dims_;
  // Input: Y, dY. Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CuDNNSoftmaxGradientOp);
};

namespace {
REGISTER_CUDNN_OPERATOR(Softmax, CuDNNSoftmaxOp<float>);
REGISTER_CUDNN_OPERATOR(SoftmaxGradient, CuDNNSoftmaxGradientOp<float>);
REGISTER_CUDNN_OPERATOR(SoftmaxFp16, CuDNNSoftmaxOp<float16>);
REGISTER_CUDNN_OPERATOR(SoftmaxFp16Gradient, CuDNNSoftmaxGradientOp<float16>);
}  // namespace
}  // namespace caffe2
