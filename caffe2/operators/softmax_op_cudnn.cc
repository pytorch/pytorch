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


class CuDNNSoftmaxOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        cudnn_wrapper_(&device_context_) {}
  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescWrapper descriptors_[NUM_DESCRIPTORS];
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CuDNNSoftmaxOp);
};


class CuDNNSoftmaxGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        cudnn_wrapper_(&device_context_) {}
  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescWrapper descriptors_[GRADIENT_NUM_DESCRIPTORS];
  // Input: Y, dY. Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CuDNNSoftmaxGradientOp);
};

bool CuDNNSoftmaxOp::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  Y->ReshapeLike(X);
  vector<int> dims{N, D, 1, 1};
  CUDNN_CHECK(cudnnSoftmaxForward(cudnn_wrapper_.cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
      cudnnTypeWrapper<float>::kOne(),
      descriptors_[BOTTOM_DESC_ID].Descriptor<float>(StorageOrder::NCHW, dims),
      X.data<float>(), cudnnTypeWrapper<float>::kZero(),
      descriptors_[TOP_DESC_ID].Descriptor<float>(StorageOrder::NCHW, dims),
      Y->mutable_data<float>()));
  return true;
}

bool CuDNNSoftmaxGradientOp::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_EQ(Y.ndim(), 2);
  int N = Y.dim(0);
  int D = Y.dim(1);
  CAFFE_DCHECK_EQ(dY.dim(0), N);
  CAFFE_DCHECK_EQ(dY.dim(1), D);
  dX->ReshapeLike(Y);
  vector<int> dims{N, D, 1, 1};
  CUDNN_CHECK(cudnnSoftmaxBackward(cudnn_wrapper_.cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
      cudnnTypeWrapper<float>::kOne(),
      descriptors_[TOP_DESC_ID].Descriptor<float>(StorageOrder::NCHW, dims),
      Y.data<float>(),
      descriptors_[TOP_GRADIENT_DESC_ID].Descriptor<float>(
          StorageOrder::NCHW, dims),
      dY.data<float>(), cudnnTypeWrapper<float>::kZero(),
      descriptors_[BOTTOM_DESC_ID].Descriptor<float>(StorageOrder::NCHW, dims),
      dX->mutable_data<float>()));
  return true;
}

namespace {
REGISTER_CUDNN_OPERATOR(Softmax, CuDNNSoftmaxOp);
REGISTER_CUDNN_OPERATOR(SoftmaxGradient, CuDNNSoftmaxGradientOp);
}  // namespace
}  // namespace caffe2
