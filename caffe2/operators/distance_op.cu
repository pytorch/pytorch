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

template <>
bool SquaredL2DistanceOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch in dimensions",
        X.dims(),
        " / ",
        Y.dims());
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
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch on dimensions: ",
        X.dims(),
        " / ",
        Y.dims());
  }
  CAFFE_ENFORCE_EQ(dDistance.ndim(), 1);
  CAFFE_ENFORCE_EQ(dDistance.dim32(0), N);
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
  distance->Resize(vector<TIndex>(size_t(1), N));
  L1DistanceKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, X.data<float>(), Y.data<float>(), distance->mutable_data<float>());

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
    int k = i / D;
    if (X[i] - Y[i] < -kEps) {
      dX[i] = -dDistance[k];
      dY[i] = dDistance[k];
    } else if (X[i] - Y[i] > kEps) {
      dX[i] = dDistance[k];
      dY[i] = -dDistance[k];
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
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  CAFFE_ENFORCE(X.ndim() == Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch on dimensions: ",
        X.dims(),
        " / ",
        Y.dims());
  }
  CAFFE_ENFORCE_EQ(dDistance.ndim(), 1);
  CAFFE_ENFORCE_EQ(dDistance.dim32(0), N);
  dX->ResizeLike(X);
  dY->ResizeLike(Y);

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

// X.size() = N*D, Y.size() = N
template <typename T>
__global__ void
BatchedMul(const int N, const int D, const T* X, const T* Y, T* result) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    result[i] = X[i] * Y[i / D];
  }
}

// X.size() = N*D, Y.size() = N
template <typename T>
__global__ void Scale2AxpyScale(
    const int N,
    const T* scale,
    const T* XY,
    const T* XN,
    T* result) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    result[i] = -scale[i] * XY[i] / (XN[i] * XN[i]);
  }
}

// X.size() = X*N, alpha.size() = N, Y.size() = X*N
template <typename T>
__global__ void
BatchedAxpy(const int N, const int D, const T* alpha, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    Y[i] += X[i] * alpha[i / D];
  }
}

} // namespace

template <>
bool CosineSimilarityOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto* result = Output(COS_OUT);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  const int N = X.ndim() > 0 ? X.dim32(0) : 1;
  const int D = X.size_from_dim(1);
  result->Resize(N);
  float* result_data = result->mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  // Auxiliary arrays, one allocation of memory
  aux_.Resize(2 * N);
  float* aux_data = aux_.mutable_data<float>();
  float* x2 = aux_data;
  float* y2 = aux_data + N;
  float* scale = x2;
  const float kEps = 1e-12f;

  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, X_data, x2);
  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, Y_data, Y_data, y2);
  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, Y_data, result_data);
  math::Maximum<float, CUDAContext>(N, kEps, x2, x2, &context_);
  math::Maximum<float, CUDAContext>(N, kEps, y2, y2, &context_);
  math::Mul(N, x2, y2, scale, &context_);
  math::InvSqrt(N, scale, scale, &context_);
  math::Mul(N, result_data, scale, result_data, &context_);
  return true;
}

template <>
bool CosineSimilarityGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto& dCos = Input(DER_COS_IN);
  auto* dX = Output(DER_X_OUT);
  auto* dY = Output(DER_Y_OUT);
  const int N = X.ndim() > 0 ? X.dim32(0) : 1;
  const int D = X.size_from_dim(1);
  CAFFE_ENFORCE(X.ndim() == Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dCos.ndim() == 1);
  CAFFE_ENFORCE(dCos.dim32(0) == N);
  dX->ResizeLike(X);
  dY->ResizeLike(Y);

  const auto* X_data = X.data<float>();
  const auto* Y_data = Y.data<float>();
  const auto* dCos_data = dCos.data<float>();
  auto* dX_data = dX->mutable_data<float>();
  auto* dY_data = dY->mutable_data<float>();

  // one memory allocation, a few arrays
  aux_.Resize(6 * N);
  float* aux_data = aux_.mutable_data<float>();
  float* xn = aux_data;
  float* yn = aux_data + N;
  float* xy = aux_data + 2 * N;
  float* xyn = aux_data + 3 * N;
  float* scale = aux_data + 4 * N;
  float* axpy_scale = aux_data + 5 * N;
  float kEps = 1e-12f;

  // ||x||
  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, X_data, xn);
  math::Maximum<float, CUDAContext>(N, kEps, xn, xn, &context_);
  math::Sqrt<float, CUDAContext>(N, xn, xn, &context_);
  // ||y||
  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, Y_data, Y_data, yn);
  math::Maximum<float, CUDAContext>(N, kEps, yn, yn, &context_);
  math::Sqrt<float, CUDAContext>(N, yn, yn, &context_);
  // ||x|| * || y ||
  math::Mul<float, CUDAContext>(N, xn, yn, xyn, &context_);

  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, Y_data, xy);
  math::Div<float, CUDAContext>(N, dCos_data, xyn, scale, &context_);
  // dX
  BatchedMul<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, Y_data, scale, dX_data);
  Scale2AxpyScale<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, scale, xy, xn, axpy_scale);
  BatchedAxpy<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, axpy_scale, X_data, dX_data);
  // dY
  BatchedMul<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, scale, dY_data);
  Scale2AxpyScale<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, scale, xy, yn, axpy_scale);
  BatchedAxpy<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, axpy_scale, Y_data, dY_data);

  return true;
}

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

REGISTER_CUDA_OPERATOR(
    CosineSimilarity,
    CosineSimilarityOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    CosineSimilarityGradient,
    CosineSimilarityGradientOp<float, CUDAContext>);
}  // namespace caffe2
