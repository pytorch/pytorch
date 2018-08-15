#include "caffe2/operators/order_switch_ops.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

__global__ void NHWC2NCHWKernel(
    const int N,
    const int HW,
    const int C,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N * HW * C) {
    const int c = i % C;
    const int hw = i / C % HW;
    const int n = i / C / HW;
    Y[(n * C + c) * HW + hw] = X[i];
  }
}

__global__ void NCHW2NHWCKernel(
    const int N,
    const int C,
    const int HW,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N * C * HW) {
    const int hw = i % HW;
    const int c = i / HW % C;
    const int n = i / C / HW;
    Y[(n * HW + hw) * C + c] = X[i];
  }
}

template <>
bool NHWC2NCHWOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  auto ndim = X.ndim();
  DCHECK_GE(ndim, 3);
  const int N = X.dim32(0), C = X.dim32(ndim - 1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  Y_dims[1] = C;
  size_t image_size = 1;
  for (auto i = 2; i < ndim; ++i) {
    Y_dims[i] = X.dim32(i - 1);
    image_size *= Y_dims[i];
  }
  Y->Resize(Y_dims);

  NHWC2NCHWKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, image_size, C, X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool NCHW2NHWCOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  auto ndim = X.ndim();
  DCHECK_GE(X.ndim(), 3);
  const int N = X.dim32(0), C = X.dim32(1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  size_t image_size = 1;
  for (auto i = 1; i < ndim - 1; ++i) {
    Y_dims[i] = X.dim32(i + 1);
    image_size *= Y_dims[i];
  }
  Y_dims[ndim - 1] = C;
  Y->Resize(Y_dims);

  NCHW2NHWCKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, C, image_size, X.data<float>(), Y->template mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(NHWC2NCHW, NHWC2NCHWOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(NCHW2NHWC, NCHW2NHWCOp<float, CUDAContext>);
}  // namespace caffe2
