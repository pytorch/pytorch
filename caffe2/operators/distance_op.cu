#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/distance_op.h"
#include "caffe2/utils/conversions.h"

#include "caffe2/utils/cub_namespace.cuh"
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
  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch in dimensions",
        X.sizes(),
        " / ",
        Y.sizes());
  }
  int N = X.dim() > 0 ? X.dim32(0) : 1;
  int D = X.size() / N;
  auto* distance = Output(0, vector<int64_t>(size_t(1), N), at::dtype<float>());
  SquaredL2DistanceKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      Y.data<float>(),
      distance->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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


  int N = X.dim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  CAFFE_ENFORCE(X.dim() == Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch on dimensions: ",
        X.sizes(),
        " / ",
        Y.sizes());
  }
  CAFFE_ENFORCE_EQ(dDistance.dim(), 1);
  CAFFE_ENFORCE_EQ(dDistance.dim32(0), N);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  auto* dY = Output(1, Y.sizes(), at::dtype<float>());
  math::Sub<float, CUDAContext>(
      X.size(),
      X.data<float>(),
      Y.data<float>(),
      dX->template mutable_data<float>(),
      &context_);

  StripedScaleKernel<float>
      <<<CAFFE_GET_BLOCKS(N * D),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          N,
          D,
          dDistance.data<float>(),
          dX->data<float>(),
          dX->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // The gradient of the other side is basically the negative.
  math::Scale<float, float, CUDAContext>(
      X.size(),
      -1,
      dX->data<float>(),
      dY->template mutable_data<float>(),
      &context_);
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
      sum += fabsf(
          convert::To<T, float>(X[i * D + j]) -
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
  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  const int N = X.dim() > 0 ? X.dim32(0) : 1;
  const int D = N > 0 ? X.size() / N : 0;
  auto* distance = Output(0, vector<int64_t>(size_t(1), N), at::dtype<float>());
  L1DistanceKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      Y.data<float>(),
      distance->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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


  int N = X.dim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  CAFFE_ENFORCE(X.dim() == Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch on dimensions: ",
        X.sizes(),
        " / ",
        Y.sizes());
  }
  CAFFE_ENFORCE_EQ(dDistance.dim(), 1);
  CAFFE_ENFORCE_EQ(dDistance.dim32(0), N);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  auto* dY = Output(1, Y.sizes(), at::dtype<float>());

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
      dX->template mutable_data<float>(),
      dY->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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
  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  const int N = X.dim() > 0 ? X.dim32(0) : 1;
  const int D = X.size_from_dim(1);
  auto* result = Output(COS_OUT, {N}, at::dtype<float>());
  float* result_data = result->template mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  // Auxiliary arrays, one allocation of memory
  ReinitializeTensor(&aux_, {2 * N}, at::dtype<float>().device(CUDA));
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, Y_data, Y_data, y2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, Y_data, result_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  math::Maximum<float, CUDAContext>(N, kEps, x2, x2, &context_);
  math::Maximum<float, CUDAContext>(N, kEps, y2, y2, &context_);
  math::Mul(N, x2, y2, scale, &context_);
  math::Rsqrt(N, scale, scale, &context_);
  math::Mul(N, result_data, scale, result_data, &context_);
  return true;
}

template <>
bool CosineSimilarityGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto& dCos = Input(DER_COS_IN);


  const int N = X.dim() > 0 ? X.dim32(0) : 1;
  const int D = X.size_from_dim(1);
  CAFFE_ENFORCE(X.dim() == Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dCos.dim() == 1);
  CAFFE_ENFORCE(dCos.dim32(0) == N);
  auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<float>());
  auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<float>());

  const auto* X_data = X.data<float>();
  const auto* Y_data = Y.data<float>();
  const auto* dCos_data = dCos.data<float>();
  auto* dX_data = dX->template mutable_data<float>();
  auto* dY_data = dY->template mutable_data<float>();

  // one memory allocation, a few arrays
  ReinitializeTensor(&aux_, {6 * N}, at::dtype<float>().device(CUDA));
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  math::Maximum<float, CUDAContext>(N, kEps, xn, xn, &context_);
  math::Sqrt<float, CUDAContext>(N, xn, xn, &context_);
  // ||y||
  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, Y_data, Y_data, yn);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  math::Maximum<float, CUDAContext>(N, kEps, yn, yn, &context_);
  math::Sqrt<float, CUDAContext>(N, yn, yn, &context_);
  // ||x|| * || y ||
  math::Mul<float, CUDAContext>(N, xn, yn, xyn, &context_);

  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, Y_data, xy);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  math::Div<float, CUDAContext>(N, dCos_data, xyn, scale, &context_);
  // dX
  BatchedMul<float><<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, Y_data, scale, dX_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  Scale2AxpyScale<float><<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, scale, xy, xn, axpy_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  BatchedAxpy<float><<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, axpy_scale, X_data, dX_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // dY
  BatchedMul<float><<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, X_data, scale, dY_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  Scale2AxpyScale<float><<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, scale, xy, yn, axpy_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  BatchedAxpy<float><<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, D, axpy_scale, Y_data, dY_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool DotProductOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N, D;
  if (X.size() > 0) {
    N = X.dim() > 0 ? X.dim32(0) : 1;
    D = X.size() / N;
  } else {
    N = 0;
    D = 0;
  }
  auto* result = Output(DOT_OUT, {N}, at::dtype<float>());

  DotProductKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      Y.data<float>(),
      result->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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


  int N, D;
  if (X.size() > 0) {
    N = X.dim() > 0 ? X.dim32(0) : 1;
    D = X.size() / N;
  } else {
    N = 0;
    D = 0;
  }
  CAFFE_ENFORCE(X.dim() == Y.dim());
  for (int i = 0; i < X.dim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dDot.dim() == 1);
  CAFFE_ENFORCE(dDot.dim32(0) == N);
  auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<float>());
  auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<float>());
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
      dX->template mutable_data<float>(),
      dY->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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
