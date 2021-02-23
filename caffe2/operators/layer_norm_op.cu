#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/operators/layer_norm_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/reduce.cuh"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ComputeSigmaAndFusedParamsCUDAKernel(
    const int N,
    const T eps,
    const T* mean,
    const T* var,
    T* sigma,
    T* scale,
    T* bias);

#define DELEGATE_COMPUTE_SIGMA_AND_FUSED_PARAMS_CUDA_KERNEL(T, RsqrtFunc) \
  template <>                                                             \
  __global__ void ComputeSigmaAndFusedParamsCUDAKernel<T>(                \
      const int N,                                                        \
      const T eps,                                                        \
      const T* mean,                                                      \
      const T* var,                                                       \
      T* sigma,                                                           \
      T* scale,                                                           \
      T* bias) {                                                          \
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;  \
    if (index < N) {                                                      \
      const T rstd = RsqrtFunc(var[index] + eps);                         \
      sigma[index] = rstd * (var[index] + eps);                           \
      scale[index] = rstd;                                                \
      bias[index] = -rstd * mean[index];                                  \
    }                                                                     \
  }
DELEGATE_COMPUTE_SIGMA_AND_FUSED_PARAMS_CUDA_KERNEL(float, rsqrtf)
DELEGATE_COMPUTE_SIGMA_AND_FUSED_PARAMS_CUDA_KERNEL(double, rsqrt)
#undef DELEGATE_COMPUTE_SIGMA_AND_FUSED_PARAMS_CUDA_KERNEL

template <typename T>
__global__ void LayerNormForwardCUDAKernel(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M * N) {
    const int i = index / N;
    Y[index] = X[index] * scale[i] + bias[i];
  }
}

template <typename T>
__global__ void LayerNormForwardCUDAKernel(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    const T* gamma,
    const T* beta,
    T* Y) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M * N) {
    const int i = index / N;
    const int j = index % N;
    Y[index] = (X[index] * scale[i] + bias[i]) * gamma[j] + beta[j];
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    const int M,
    const int N,
    const T* dYxX,
    const T* dY,
    T* ds,
    T* db) {
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  const int i = blockIdx.x;
  T ds_val = 0;
  T db_val = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
#if __CUDA_ARCH__ >= 350
    ds_val += __ldg(dYxX + index);
    db_val += __ldg(dY + index);
#else
    ds_val += dYxX[index];
    db_val += dY[index];
#endif
  }
  ds_val = BlockReduce<T>(ds_storage).Sum(ds_val);
  db_val = BlockReduce<T>(db_storage).Sum(db_val);
  if (threadIdx.x == 0) {
    ds[i] = ds_val;
    db[i] = db_val;
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    const int M,
    const int N,
    const T* dYxX,
    const T* dY,
    const T* gamma,
    T* ds,
    T* db) {
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  const int i = blockIdx.x;
  T ds_val = 0;
  T db_val = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
#if __CUDA_ARCH__ >= 350
    ds_val += __ldg(dYxX + index) * __ldg(gamma + j);
    db_val += __ldg(dY + index) * __ldg(gamma + j);
#else
    ds_val += dYxX[index] * gamma[j];
    db_val += dY[index] * gamma[j];
#endif
  }
  ds_val = BlockReduce<T>(ds_storage).Sum(ds_val);
  db_val = BlockReduce<T>(db_storage).Sum(db_val);
  if (threadIdx.x == 0) {
    ds[i] = ds_val;
    db[i] = db_val;
  }
}

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    const int M,
    const int N,
    const T* mean,
    const T* sigma,
    const T* ds,
    const T* db,
    T* rstd,
    T* X_scale,
    T* bias,
    T* g_scale) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M) {
    const T scale = T(1) / static_cast<T>(N);
    const T rstd_val = T(1) / sigma[index];
    const T X_scale_val = (db[index] * mean[index] - ds[index]) *
        math::utils::Cube<T>(rstd_val) * scale;
    rstd[index] = rstd_val;
    X_scale[index] = X_scale_val;
    bias[index] = -(X_scale_val * mean[index] + db[index] * rstd_val * scale);
    if (g_scale != nullptr) {
      g_scale[index] = -rstd_val * mean[index];
    }
  }
}

template <typename T>
__global__ void LayerNormBackwardCUDAKenrel(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    const T* dY_scale,
    const T* X_scale,
    const T* bias,
    T* dX) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M * N) {
    const int i = index / N;
    dX[index] = dY[index] * dY_scale[i] + X[index] * X_scale[i] + bias[i];
  }
}

template <typename T>
__global__ void LayerNormBackwardCUDAKenrel(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* dY_scale,
    const T* X_scale,
    const T* bias,
    T* dX) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M * N) {
    const int i = index / N;
    const int j = index % N;
    dX[index] =
        dY[index] * dY_scale[i] * gamma[j] + X[index] * X_scale[i] + bias[i];
  }
}

} //  namespace

template <>
template <typename T>
void LayerNormOp<CUDAContext>::ComputeSigmaAndFusedParams(
    const int N,
    const float eps,
    const T* mean,
    const T* var,
    T* sigma,
    T* scale,
    T* bias) {
  if (N > 0) {
    const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
    ComputeSigmaAndFusedParamsCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            N, static_cast<T>(eps), mean, var, sigma, scale, bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <>
template <typename T>
void LayerNormOp<CUDAContext>::LayerNormForward(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    const T* gamma,
    const T* beta,
    T* Y) {
  if (M * N > 0) {
    const int K = math::DivUp(M * N, CAFFE_CUDA_NUM_THREADS);
    if (gamma != nullptr && beta != nullptr) {
      LayerNormForwardCUDAKernel<T>
          <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
              M, N, X, scale, bias, gamma, beta, Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      CAFFE_ENFORCE(gamma == nullptr);
      CAFFE_ENFORCE(beta == nullptr);
      LayerNormForwardCUDAKernel<T>
          <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
              M, N, X, scale, bias, Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

REGISTER_CUDA_OPERATOR(LayerNorm, LayerNormOp<CUDAContext>);

template <>
template <typename T>
void LayerNormGradientOp<CUDAContext>::ComputeInternalGradients(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    const T* gamma,
    T* dYxX,
    T* ds,
    T* db) {
  math::Mul<T, CUDAContext>(M * N, dY, X, dYxX, &context_);
  if (gamma != nullptr) {
    ComputeInternalGradientsCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            M, N, dYxX, dY, gamma, ds, db);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    ComputeInternalGradientsCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            M, N, dYxX, dY, ds, db);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <>
template <typename T>
void LayerNormGradientOp<CUDAContext>::ComputeFusedParams(
    const int M,
    const int N,
    const T* mean,
    const T* sigma,
    const T* ds,
    const T* db,
    T* rstd,
    T* X_scale,
    T* bias,
    T* g_scale) {
  if (M > 0) {
    const int K = math::DivUp(M, CAFFE_CUDA_NUM_THREADS);
    ComputeFusedParamsCUDAKernel<T>
        <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            M, N, mean, sigma, ds, db, rstd, X_scale, bias, g_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <>
template <typename T>
void LayerNormGradientOp<CUDAContext>::LayerNormBackward(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* dY_scale,
    const T* X_scale,
    const T* bias,
    T* dX) {
  if (M * N > 0) {
    const int K = math::DivUp(M * N, CAFFE_CUDA_NUM_THREADS);
    if (gamma != nullptr) {
      LayerNormBackwardCUDAKenrel<T>
          <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
              M, N, dY, X, gamma, dY_scale, X_scale, bias, dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      LayerNormBackwardCUDAKenrel<T>
          <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
              M, N, dY, X, dY_scale, X_scale, bias, dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

template <>
template <typename T>
void LayerNormGradientOp<CUDAContext>::GammaBetaBackward(
    const int M,
    const int N,
    const T* dYxX,
    const T* dY,
    const T* rstd,
    const T* g_scale,
    T* dgamma,
    T* dbeta) {
  if (M == 0) {
    math::Set<T, CUDAContext>(N, T(0), dgamma, &context_);
    math::Set<T, CUDAContext>(N, T(0), dbeta, &context_);
  } else {
    if (ones_.numel() != M) {
      ReinitializeTensor(&ones_, {M}, at::dtype<T>().device(CUDA));
      math::Set<T, CUDAContext>(
          M, T(1), ones_.template mutable_data<T>(), &context_);
    }
    math::Gemv<T, CUDAContext>(
        CblasTrans, M, N, 1.0f, dYxX, rstd, 0.0f, dgamma, &context_);
    math::Gemv<T, CUDAContext>(
        CblasTrans, M, N, 1.0f, dY, g_scale, 1.0f, dgamma, &context_);
    const T* ones_data = ones_.template data<T>();
    math::Gemv<T, CUDAContext>(
        CblasTrans, M, N, 1.0f, dY, ones_data, 0.0f, dbeta, &context_);
  }
}

REGISTER_CUDA_OPERATOR(LayerNormGradient, LayerNormGradientOp<CUDAContext>);

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(
    LayerNorm,
    caffe2::LayerNormOp<caffe2::CUDAContext>)

namespace caffe2 {

C10_EXPORT_C10_OP_TO_CAFFE2_CUDA(
    "_caffe2::LayerNorm",
    C10LayerNorm_DontUseThisOpYet);

} // namespace caffe2
