#include <cub/block/block_reduce.cuh>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/distance_op.h"
#include "caffe2/utils/conversions.h"

#include <cub/block/block_reduce.cuh>

namespace caffe2 {

namespace {

template <typename T>
__global__ void SquaredL2DistanceKernel(
    const int N, const int D, const T* X, const T* Y, T* distance) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
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
bool SquaredL2DistanceOp<float, HIPContext>::RunOnDevice() {
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
  hipLaunchKernelGGL(SquaredL2DistanceKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(N), static_cast<const int>(D), X.data<float>(), Y.data<float>(), distance->mutable_data<float>());
  return true;
}

namespace {
template <typename T>
__global__ void
StripedScaleKernel(const int N, const int D, const T* alpha, const T* x, T* y) {
  HIP_1D_KERNEL_LOOP(i, N * D) {
    int k = i / D;
    y[i] = x[i] * alpha[k];
  }
}
}

template <>
bool SquaredL2DistanceGradientOp<float, HIPContext>::RunOnDevice() {
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
  math::Sub<float, HIPContext>(
      X.size(),
      X.data<float>(),
      Y.data<float>(),
      dX->mutable_data<float>(),
      &context_);

  hipLaunchKernelGGL(StripedScaleKernel<float>, dim3(CAFFE_GET_BLOCKS(static_cast<const int>(N) * static_cast<const int>(D))), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(N),
      static_cast<const int>(D),
      dDistance.data<float>(),
      dX->data<float>(),
      dX->mutable_data<float>());

  // The gradient of the other side is basically the negative.
  math::Scale<float, HIPContext>(
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
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    float sum = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      sum +=
          fabsf(convert::To<T, float>(X[i * D + j]) -
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
bool L1DistanceOp<float, HIPContext>::RunOnDevice() {
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
  hipLaunchKernelGGL(L1DistanceKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(N), static_cast<const int>(D), X.data<float>(), Y.data<float>(), distance->mutable_data<float>());

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
  HIP_1D_KERNEL_LOOP(i, N * D) {
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
bool L1DistanceGradientOp<float, HIPContext>::RunOnDevice() {
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

  hipLaunchKernelGGL(L1DistanceGradientKernel, dim3(CAFFE_GET_BLOCKS(static_cast<const int>(N) * static_cast<const int>(D))), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(N),
      static_cast<const int>(D),
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

    typedef cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS> BlockReduce;
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
  HIP_1D_KERNEL_LOOP(i, N * D) {
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
  HIP_1D_KERNEL_LOOP(i, N) {
    result[i] = -scale[i] * XY[i] / (XN[i] * XN[i]);
  }
}

// X.size() = X*N, alpha.size() = N, Y.size() = X*N
template <typename T>
__global__ void
BatchedAxpy(const int N, const int D, const T* alpha, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N * D) {
    Y[i] += X[i] * alpha[i / D];
  }
}

} // namespace

template <>
bool CosineSimilarityOp<float, HIPContext>::RunOnDevice() {
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

  hipLaunchKernelGGL(DotProductKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), X_data, X_data, x2);
  hipLaunchKernelGGL(DotProductKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), Y_data, Y_data, y2);
  hipLaunchKernelGGL(DotProductKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), X_data, Y_data, result_data);
  math::Maximum<float, HIPContext>(N, kEps, x2, x2, &context_);
  math::Maximum<float, HIPContext>(N, kEps, y2, y2, &context_);
  math::Mul(N, x2, y2, scale, &context_);
  math::InvSqrt(N, scale, scale, &context_);
  math::Mul(N, result_data, scale, result_data, &context_);
  return true;
}

template <>
bool CosineSimilarityGradientOp<float, HIPContext>::RunOnDevice() {
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
  hipLaunchKernelGGL(DotProductKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), X_data, X_data, xn);
  math::Maximum<float, HIPContext>(N, kEps, xn, xn, &context_);
  math::Sqrt<float, HIPContext>(N, xn, xn, &context_);
  // ||y||
  hipLaunchKernelGGL(DotProductKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), Y_data, Y_data, yn);
  math::Maximum<float, HIPContext>(N, kEps, yn, yn, &context_);
  math::Sqrt<float, HIPContext>(N, yn, yn, &context_);
  // ||x|| * || y ||
  math::Mul<float, HIPContext>(N, xn, yn, xyn, &context_);

  hipLaunchKernelGGL(DotProductKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), X_data, Y_data, xy);
  math::Div<float, HIPContext>(N, dCos_data, xyn, scale, &context_);
  // dX
  hipLaunchKernelGGL(BatchedMul, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), Y_data, static_cast<const float*>(scale), dX_data);
  hipLaunchKernelGGL(Scale2AxpyScale, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const float*>(scale), static_cast<const float*>(xy), static_cast<const float*>(xn), axpy_scale);
  hipLaunchKernelGGL(BatchedAxpy, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), static_cast<const float*>(axpy_scale), X_data, dX_data);
  // dY
  hipLaunchKernelGGL(BatchedMul, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), X_data, static_cast<const float*>(scale), dY_data);
  hipLaunchKernelGGL(Scale2AxpyScale, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const float*>(scale), static_cast<const float*>(xy), static_cast<const float*>(yn), axpy_scale);
  hipLaunchKernelGGL(BatchedAxpy, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), static_cast<const int>(N), static_cast<const int>(D), static_cast<const float*>(axpy_scale), Y_data, dY_data);

  return true;
}

template <>
bool DotProductOp<float, HIPContext>::RunOnDevice() {
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

  hipLaunchKernelGGL(DotProductKernel, dim3(std::min(static_cast<const int>(N), CAFFE_MAXIMUM_NUM_BLOCKS)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(N), static_cast<const int>(D), X.data<float>(), Y.data<float>(), result->mutable_data<float>());

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
  HIP_1D_KERNEL_LOOP(i, N * D) {
    T scale = dDot[i / D];
    dX[i] = Y[i] * scale;
    dY[i] = X[i] * scale;
  }
}
} // namespace

template <>
bool DotProductGradientOp<float, HIPContext>::RunOnDevice() {
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
  hipLaunchKernelGGL(DotProductGradientKernel, dim3(CAFFE_GET_BLOCKS(static_cast<const int>(N) * static_cast<const int>(D))), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(N),
      static_cast<const int>(D),
      X.data<float>(),
      Y.data<float>(),
      dDot.data<float>(),
      dX->mutable_data<float>(),
      dY->mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(SquaredL2Distance,
                       SquaredL2DistanceOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(SquaredL2DistanceGradient,
                       SquaredL2DistanceGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(L1Distance, L1DistanceOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    L1DistanceGradient,
    L1DistanceGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(DotProduct, DotProductOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    DotProductGradient,
    DotProductGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(
    CosineSimilarity,
    CosineSimilarityOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    CosineSimilarityGradient,
    CosineSimilarityGradientOp<float, HIPContext>);
}  // namespace caffe2
