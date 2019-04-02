#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

namespace {
const int NUM_DESCRIPTORS = 2;
const int GRADIENT_NUM_DESCRIPTORS = 3;
const int BOTTOM_DESC_ID = 0;
const int TOP_DESC_ID = 1;
const int TOP_GRADIENT_DESC_ID = 2;
}  // namespace


class CuDNNSoftmaxOp final : public Operator<float, CUDAContext> {
 public:
  explicit CuDNNSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<float, CUDAContext>(def, ws),
        cudnn_wrapper_(&device_context_) {}
  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CuDNNSoftmaxOp);
};


class CuDNNSoftmaxGradientOp final : public Operator<float, CUDAContext> {
 public:
  explicit CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<float, CUDAContext>(def, ws),
        cudnn_wrapper_(&device_context_) {}
  bool RunOnDevice() override;

 protected:
  CuDNNWrapper cudnn_wrapper_;
  // Input: Y, dY. Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(CuDNNSoftmaxGradientOp);
};

bool CuDNNSoftmaxOp::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  Y->ReshapeLike(X);
  const float alpha = 1.0;
  const float beta = 0.0;
  vector<int> dims{N, D, 1, 1};
  cudnn_wrapper_.cudnnSetNumTensorDescriptors(NUM_DESCRIPTORS);
  CUDNN_CHECK(cudnnSoftmaxForward(cudnn_wrapper_.cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
      cudnn_wrapper_.cudnnGetTensor4dDesc<float>(
          BOTTOM_DESC_ID, CUDNN_TENSOR_NCHW, dims, nullptr),
      X.data(), &beta,
      cudnn_wrapper_.cudnnGetTensor4dDesc<float>(
          TOP_DESC_ID, CUDNN_TENSOR_NCHW, dims, nullptr),
      Y->mutable_data()));
  return true;
}

bool CuDNNSoftmaxGradientOp::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_EQ(Y.ndim(), 2);
  int N = Y.dim(0);
  int D = Y.dim(1);
  DCHECK_EQ(dY.dim(0), N);
  DCHECK_EQ(dY.dim(1), D);
  dX->ReshapeLike(Y);
  const float alpha = 1.0;
  const float beta = 0.0;
  cudnn_wrapper_.cudnnSetNumTensorDescriptors(GRADIENT_NUM_DESCRIPTORS);
  vector<int> dims{N, D, 1, 1};
  CUDNN_CHECK(cudnnSoftmaxBackward(cudnn_wrapper_.cudnn_handle(),
      CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
      cudnn_wrapper_.cudnnGetTensor4dDesc<float>(
          TOP_DESC_ID, CUDNN_TENSOR_NCHW, dims, nullptr),
      Y.data(),
      cudnn_wrapper_.cudnnGetTensor4dDesc<float>(
          TOP_GRADIENT_DESC_ID, CUDNN_TENSOR_NCHW, dims, nullptr),
      dY.data(), &beta,
      cudnn_wrapper_.cudnnGetTensor4dDesc<float>(
          BOTTOM_DESC_ID, CUDNN_TENSOR_NCHW, dims, nullptr),
      dX->mutable_data()));
  return true;
}

namespace {
REGISTER_CUDNN_OPERATOR(Softmax, CuDNNSoftmaxOp)
REGISTER_CUDNN_OPERATOR(SoftmaxGradient, CuDNNSoftmaxGradientOp)
}  // namespace
}  // namespace caffe2
