#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/distance_op.h"
#include "caffe2/utils/conversions.h"

#include <cub/block/block_reduce.cuh>

namespace caffe2 {

namespace {

template <typename T>
__global__ void SquaredL2DistanceKernel(
    const int N, const int D, const T* X, const T* Y, T* distance) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    float dist = 0.0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      T diff = X[i * D + j] - Y[i * D + j];
      dist += diff * diff;
    }

    float total_dist = BlockReduce(temp_storage).Sum(dist);
    __syncthreads();
    if (threadIdx.x == 0) {
      distance[i] = total_dist / 2.0;
    }
  }
}
}  // namespace

template<>
bool SquaredL2DistanceOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  DCHECK_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    DCHECK_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = X.size() / N;
  distance->Resize(vector<TIndex>(size_t(1), N));
  SquaredL2DistanceKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, X.data<float>(), Y.data<float>(), distance->mutable_data<float>());
  return true;
}

namespace {
template <typename T>
__global__ void
StripedScaleKernel(const int N, const int D, const T* alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    int k = i / D;
    y[i] = x[i] * alpha[k];
  }
}
}

template <>
bool SquaredL2DistanceGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dDistance = Input(2);
  auto* dX = Output(0);
  auto* dY = Output(1);
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  CAFFE_ENFORCE(X.ndim() == Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dDistance.ndim() == 1);
  CAFFE_ENFORCE(dDistance.dim32(0) == N);
  dX->ResizeLike(X);
  dY->ResizeLike(Y);
  math::Sub<float, CUDAContext>(
      X.size(),
      X.data<float>(),
      Y.data<float>(),
      dX->mutable_data<float>(),
      &context_);

  StripedScaleKernel<float><<<
      CAFFE_GET_BLOCKS(N * D),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      dDistance.data<float>(),
      dX->data<float>(),
      dX->mutable_data<float>());

  // The gradient of the other side is basically the negative.
  math::Scale<float, CUDAContext>(
      X.size(), -1, dX->data<float>(), dY->mutable_data<float>(), &context_);
  return true;
}

namespace {
template <typename T>
__global__ void L1DistanceKernel(
    const int N,
    const int D,
    const T* X,
    const T* Y,
    T* distance) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    float sum = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      sum +=
          abs(convert::To<T, float>(X[i * D + j]) -
              convert::To<T, float>(Y[i * D + j]));
    }

    float aggregate = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    if (threadIdx.x == 0) {
      distance[i] = aggregate;
    }
  }
}
} // namespace

template <>
bool L1DistanceOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  const int N = X.ndim() > 0 ? X.dim32(0) : 1;
  const int D = N > 0 ? X.size() / N : 0;

  distance->Resize(N);

  L1DistanceKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, X.data<float>(), Y.data<float>(), distance->mutable_data<float>());
  math::Sum<float, CUDAContext>(
      N, distance->data<float>(), distance->mutable_data<float>(), &context_);

  distance->Resize(1);

  return true;
}

namespace {
template <typename T>
__global__ void L1DistanceGradientKernel(
    const int N,
    const int D,
    const T* X,
    const T* Y,
    const T* dDistance,
    T* dX,
    T* dY) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    constexpr float kEps = 1e-12;
    if (X[i] - Y[i] < -kEps) {
      dX[i] = -dDistance[0];
      dY[i] = dDistance[0];
    } else if (X[i] - Y[i] > kEps) {
      dX[i] = dDistance[0];
      dY[i] = -dDistance[0];
    } else {
      dX[i] = 0;
      dY[i] = 0;
    }
  }
}
} // namespace

template <>
bool L1DistanceGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dDistance = Input(2);
  auto* dX = Output(0);
  auto* dY = Output(1);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  CAFFE_ENFORCE(dDistance.ndim() == 1);
  CAFFE_ENFORCE(dDistance.dim32(0) == 1);
  dX->ResizeLike(X);
  dY->ResizeLike(Y);
  const int N = X.ndim() > 0 ? X.dim32(0) : 1;
  const int D = N > 0 ? X.size() / N : 0;

  L1DistanceGradientKernel<<<
      CAFFE_GET_BLOCKS(N * D),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      Y.data<float>(),
      dDistance.data<float>(),
      dX->mutable_data<float>(),
      dY->mutable_data<float>());

  return true;
}

namespace {
template <typename T>
__global__ void
DotProductKernel(const int N, const int D, const T* X, const T* Y, T* result) {
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T partialSum = 0;
    int offset = i * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      partialSum += X[offset + j] * Y[offset + j];
    }

    typedef cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T sum = BlockReduce(temp_storage).Sum(partialSum);
    __syncthreads();
    if (threadIdx.x == 0) {
      result[i] = sum;
    }
  }
}
} // namespace

template <>
bool DotProductOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto* result = Output(DOT_OUT);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N, D;
  if (X.size() > 0) {
    N = X.ndim() > 0 ? X.dim32(0) : 1;
    D = X.size() / N;
  } else {
    N = 0;
    D = 0;
  }
  result->Resize(N);

  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, X.data<float>(), Y.data<float>(), result->mutable_data<float>());

  return true;
}

namespace {
template <typename T>
__global__ void DotProductGradientKernel(
    const int N,
    const int D,
    const T* X,
    const T* Y,
    const T* dDot,
    T* dX,
    T* dY) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    T scale = dDot[i / D];
    dX[i] = Y[i] * scale;
    dY[i] = X[i] * scale;
  }
}
} // namespace

template <>
bool DotProductGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto& dDot = Input(DER_DOT_IN);
  auto* dX = Output(DER_X_OUT);
  auto* dY = Output(DER_Y_OUT);
  int N, D;
  if (X.size() > 0) {
    N = X.ndim() > 0 ? X.dim32(0) : 1;
    D = X.size() / N;
  } else {
    N = 0;
    D = 0;
  }
  CAFFE_ENFORCE(X.ndim() == Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dDot.ndim() == 1);
  CAFFE_ENFORCE(dDot.dim32(0) == N);
  dX->ResizeLike(X);
  dY->ResizeLike(Y);
  DotProductGradientKernel<<<
      CAFFE_GET_BLOCKS(N * D),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      Y.data<float>(),
      dDot.data<float>(),
      dX->mutable_data<float>(),
      dY->mutable_data<float>());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(SquaredL2Distance,
                       SquaredL2DistanceOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SquaredL2DistanceGradient,
                       SquaredL2DistanceGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(L1Distance, L1DistanceOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    L1DistanceGradient,
    L1DistanceGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(DotProduct, DotProductOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    DotProductGradient,
    DotProductGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
