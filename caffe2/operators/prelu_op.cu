#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/prelu_op.h"

#include <cub/block/block_reduce.cuh>

namespace caffe2 {
namespace {
template <typename T>
__global__ void PReluKernel(const int N, const T* X, const T* W, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = (X[i] > 0) * X[i] + (X[i] < 0) * X[i] * W[0];
  }
}

template <typename T>
__global__ void PReluKernelNCHW(
    const int N,
    const int C,
    const int dim,
    const T* X,
    const T* W,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N * C * dim) {
    int c = (i / dim) % C;
    Y[i] = (X[i] > 0) * X[i] + (X[i] < 0) * X[i] * W[c];
  }
}

template <typename T>
__global__ void
PReluKernelNHWC(const int nitems, const int C, const T* X, const T* W, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, nitems) {
    int c = i % C;
    Y[i] = (X[i] > 0) * X[i] + (X[i] < 0) * X[i] * W[c];
  }
}

template <typename T>
__global__ void
PReluGradientKernel(const int N, const T* X, const T* W, const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = (X[i] > 0) * dY[i] + (X[i] <= 0) * dY[i] * W[0];
  }
}

template <typename T>
__global__ void PReluGradientKernelNCHW(
    const int N,
    const int C,
    const int dim,
    const T* X,
    const T* W,
    const T* dY,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N * C * dim) {
    int c = (i / dim) % C;
    dX[i] = (X[i] > 0) * dY[i] + (X[i] <= 0) * dY[i] * W[c];
  }
}

template <typename T>
__global__ void PReluGradientKernelNHWC(
    const int nitems,
    const int C,
    const T* X,
    const T* W,
    const T* dY,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, nitems) {
    int c = i % C;
    dX[i] = (X[i] > 0) * dY[i] + (X[i] <= 0) * dY[i] * W[c];
  }
}

template <typename T>
__global__ void PReluSharedWGradientKernelNCHW(
    const int num_items,
    const T* Xdata,
    const T* dYdata,
    T* dW) {
  T wsum = 0.0;
  for (int i = threadIdx.x; i < num_items; i += blockDim.x) {
    wsum += (Xdata[i] <= 0) * dYdata[i] * Xdata[i];
  }

  typedef cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum = BlockReduce(temp_storage).Sum(wsum);
  if (threadIdx.x == 0) {
    *dW = sum;
  }
}

template <typename T>
__global__ void PReluWGradientKernelNCHW(
    const int C,
    const int N,
    const int num_items,
    const T* Xdata,
    const T* dYdata,
    T* dW) {
  int c = blockIdx.x;

  T wsum = 0.0;
  int items_per_channel = num_items / C;
  int items_per_sample_channel = items_per_channel / N;
  for (int i = threadIdx.x; i < items_per_channel; i += blockDim.x) {
    // TODO: simplify
    int n = i / items_per_sample_channel;
    int ii = n * items_per_sample_channel * C + c * items_per_sample_channel +
        i % items_per_sample_channel;
    wsum += (Xdata[ii] <= 0) * dYdata[ii] * Xdata[ii];
  }

  typedef cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum = BlockReduce(temp_storage).Sum(wsum);
  if (threadIdx.x == 0) {
    dW[c] = sum;
  }
}

template <typename T>
__global__ void PReluWGradientKernelNHWC(
    const int C,
    const int N,
    const int num_items,
    const T* Xdata,
    const T* dYdata,
    T* dW) {
  int c = blockIdx.x;
  T wsum = 0.0;
  int items_per_channel = num_items / C;
  for (int i = threadIdx.x; i < items_per_channel; i += blockDim.x) {
    int ii = i * C + c;
    wsum += (Xdata[ii] <= 0) * dYdata[ii] * Xdata[ii];
  }

  typedef cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum = BlockReduce(temp_storage).Sum(wsum);
  if (threadIdx.x == 0) {
    dW[c] = sum;
  }
}

} // namespace

template <>
bool PReluOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& W = Input(1);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const auto* Xdata = X.data<float>();
  const auto* Wdata = W.data<float>();
  auto* Ydata = Y->mutable_data<float>();

  const auto C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(X.ndim() - 1);
  const auto C_shared = (W.size() == 1);

  if (!C_shared) {
    CAFFE_ENFORCE_EQ(C, W.size());
  }
  if (C_shared) {
    PReluKernel<<<
        CAFFE_GET_BLOCKS(X.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(X.size(), Xdata, Wdata, Ydata);
    return true;
  }
  // non-shared case.
  switch (order_) {
    case StorageOrder::NCHW: {
      const auto N = X.dim(0);
      const auto dim = X.size_from_dim(2);
      CHECK(N * C * dim == X.size());
      PReluKernelNCHW<<<
          CAFFE_GET_BLOCKS(X.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(N, C, dim, Xdata, Wdata, Ydata);

      break;
    }
    case StorageOrder::NHWC: {
      PReluKernelNHWC<<<
          CAFFE_GET_BLOCKS(X.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(X.size(), C, Xdata, Wdata, Ydata);

      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
  return true;
}

template <>
bool PReluGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto& X = Input(2);
  auto& W = Input(3);

  CAFFE_ENFORCE(&Y != &X, "Cannot backpropagate through an in-place PReLU");
  auto* dX = Output(0);
  auto* dW = Output(1);

  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  dW->ResizeLike(W);

  const auto C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(X.ndim() - 1);
  const auto C_shared = (W.size() == 1);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  const float* Xdata = X.data<float>();
  const float* Wdata = W.data<float>();
  float* dXdata = dX->mutable_data<float>();
  float* dWdata = dW->mutable_data<float>();
  int N = Y.dim(0);

  if (C_shared) {
    PReluSharedWGradientKernelNCHW<<<
        1,
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(X.size(), Xdata, dYdata, dWdata);
    PReluGradientKernel<<<
        CAFFE_GET_BLOCKS(X.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(X.size(), Xdata, Wdata, dYdata, dXdata);

    return true;
  }
  // non-shared case.
  switch (order_) {
    case StorageOrder::NCHW: {
      const auto dim = Y.size_from_dim(2);
      PReluWGradientKernelNCHW<<<
          C,
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(C, N, X.size(), Xdata, dYdata, dWdata);
      PReluGradientKernelNCHW<<<
          CAFFE_GET_BLOCKS(X.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(N, C, dim, Xdata, Wdata, dYdata, dXdata);

      break;
    }
    case StorageOrder::NHWC: {
      PReluWGradientKernelNHWC<<<
          C,
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(C, N, X.size(), Xdata, dYdata, dWdata);
      PReluGradientKernelNHWC<<<
          CAFFE_GET_BLOCKS(Y.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(X.size(), C, Xdata, Wdata, dYdata, dXdata);

      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(PRelu, PReluOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PReluGradient, PReluGradientOp<float, CUDAContext>);
} // namespace caffe2
