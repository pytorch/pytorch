#include "caffe2/operators/instance_norm_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/reduce.cuh"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    const int64_t N,
    const int64_t C,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C) {
    const int64_t c = index % C;
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    const T scale_val = __ldg(gamma + c) * __ldg(rstd + index);
    scale[index] = scale_val;
    bias[index] = __ldg(beta + c) - scale_val * __ldg(mean + index);
#else
    const T scale_val = gamma[c] * rstd[index];
    scale[index] = scale_val;
    bias[index] = beta[c] - scale_val * mean[index];
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void InstanceNormForwardCUDAKernel(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * HxW) {
    const int64_t nc = kOrder == StorageOrder::NCHW
        ? (index / HxW)
        : (index / (HxW * C) * C + index % C);
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    Y[index] = __ldg(scale + nc) * __ldg(X + index) + __ldg(bias + nc);
#else
    Y[index] = scale[nc] * X[index] + bias[nc];
#endif
  }
}

template <typename T>
__global__ void ComputeInternalGradientsNCHWCUDAKernel(
    const int64_t HxW,
    const T* dY,
    const T* X,
    T* ds,
    T* db) {
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  const int64_t i = blockIdx.x;
  T ds_sum = 0;
  T db_sum = 0;
  for (int64_t j = threadIdx.x; j < HxW; j += blockDim.x) {
    const int64_t index = i * HxW + j;
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    ds_sum += __ldg(dY + index) * __ldg(X + index);
    db_sum += __ldg(dY + index);
#else
    ds_sum += dY[index] * X[index];
    db_sum += dY[index];
#endif
  }
  ds_sum = BlockReduce<T>(ds_storage).Sum(ds_sum);
  db_sum = BlockReduce<T>(db_storage).Sum(db_sum);
  if (threadIdx.x == 0) {
    ds[i] = ds_sum;
    db[i] = db_sum;
  }
}

template <typename T>
__global__ void ComputeFusedParams(
    const int64_t N,
    const int64_t C,
    const T scale,
    const T* ds,
    const T* db,
    const T* mean,
    const T* rstd,
    const T* gamma,
    T* c1,
    T* c2,
    T* c3) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C) {
    const int64_t c = index % C;
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    T x = __ldg(ds + index) * __ldg(gamma + c);
    T y = __ldg(db + index) * __ldg(gamma + c);
    x = (y * __ldg(mean + index) - x) *
        math::utils::Cube<T>(__ldg(rstd + index)) * scale;
    y = -x * __ldg(mean + index) - y * __ldg(rstd + index) * scale;
    c1[index] = __ldg(rstd + index) * __ldg(gamma + c);
#else
    T x = ds[index] * gamma[c];
    T y = db[index] * gamma[c];
    x = (y * mean[index] - x) * math::utils::Cube<T>(rstd[index]) * scale;
    y = -x * mean[index] - y * rstd[index] * scale;
    c1[index] = rstd[index] * gamma[c];
#endif
    c2[index] = x;
    c3[index] = y;
  }
}

template <typename T, StorageOrder kOrder>
__global__ void InstanceNormBackwardCUDAKernel(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const T* dY,
    const T* X,
    const T* c1,
    const T* c2,
    const T* c3,
    T* dX) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * HxW) {
    const int64_t c = kOrder == StorageOrder::NCHW
        ? (index / HxW)
        : (index / (HxW * C) * C + index % C);
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    dX[index] = __ldg(c1 + c) * __ldg(dY + index) +
        __ldg(c2 + c) * __ldg(X + index) + __ldg(c3 + c);
#else
    dX[index] = c1[c] * dY[index] + c2[c] * X[index] + c3[c];
#endif
  }
}

template <typename T>
__global__ void GammaBetaBackwardCUDAKernel(
    const int64_t N,
    const int64_t C,
    const T* ds,
    const T* db,
    const T* mean,
    const T* rstd,
    T* dgamma,
    T* dbeta) {
  __shared__ typename BlockReduce<T>::TempStorage s1_storage;
  __shared__ typename BlockReduce<T>::TempStorage s2_storage;
  const int64_t c = blockIdx.x;
  T sum1 = 0;
  T sum2 = 0;
  for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
    const int64_t index = i * C + c;
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    sum1 += (__ldg(ds + index) - __ldg(db + index) * __ldg(mean + index)) *
        __ldg(rstd + index);
    sum2 += __ldg(db + index);
#else
    sum1 += (ds[index] - db[index] * mean[index]) * rstd[index];
    sum2 += db[index];
#endif
  }
  sum1 = BlockReduce<T>(s1_storage).Sum(sum1);
  sum2 = BlockReduce<T>(s2_storage).Sum(sum2);
  if (threadIdx.x == 0) {
    dgamma[c] = sum1;
    dbeta[c] = sum2;
  }
}

} // namespace

template <>
bool InstanceNormOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    float* mean,
    float* rstd) {
  ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CUDA));
  float* scale_data = scale_.template mutable_data<float>();
  float* bias_data = bias_.template mutable_data<float>();
  const std::array<int, 2> X_dims = {static_cast<int>(N * C),
                                     static_cast<int>(HxW)};
  const std::array<int, 2> Y_dims = {static_cast<int>(N * C), 1};
  math::Moments<float, CUDAContext>(
      2, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
  math::InvStd<float, CUDAContext>(
      static_cast<int>(N * C), epsilon_, rstd, rstd, &context_);
  int64_t B = math::DivUp<int64_t>(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamsCUDAKernel<float>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, mean, rstd, gamma, beta, scale_data, bias_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  B = math::DivUp<int64_t>(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  InstanceNormForwardCUDAKernel<float, StorageOrder::NCHW>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, X, scale_data, bias_data, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool InstanceNormOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    float* mean,
    float* rstd) {
  ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CUDA));
  float* scale_data = scale_.template mutable_data<float>();
  float* bias_data = bias_.template mutable_data<float>();
  const std::array<int, 3> X_dims = {
      static_cast<int>(N), static_cast<int>(HxW), static_cast<int>(C)};
  const std::array<int, 3> Y_dims = {
      static_cast<int>(N), 1, static_cast<int>(C)};
  math::Moments<float, CUDAContext>(
      3, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
  math::InvStd<float, CUDAContext>(
      static_cast<int>(N * C), epsilon_, rstd, rstd, &context_);
  int64_t B = math::DivUp<int64_t>(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamsCUDAKernel<float>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, mean, rstd, gamma, beta, scale_data, bias_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  B = math::DivUp<int64_t>(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  InstanceNormForwardCUDAKernel<float, StorageOrder::NHWC>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, X, scale_data, bias_data, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
void InstanceNormGradientOp<float, CUDAContext>::ComputeMoments(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    float* mean,
    float* rstd) {
  if (order_ == StorageOrder::NCHW) {
    const std::array<int, 2> X_dims = {static_cast<int>(N * C),
                                       static_cast<int>(HxW)};
    const std::array<int, 2> Y_dims = {static_cast<int>(N * C), 1};
    math::Moments<float, CUDAContext>(
        2, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
  } else {
    const std::array<int, 3> X_dims = {
        static_cast<int>(N), static_cast<int>(HxW), static_cast<int>(C)};
    const std::array<int, 3> Y_dims = {
        static_cast<int>(N), 1, static_cast<int>(C)};
    math::Moments<float, CUDAContext>(
        3, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
  }
  math::InvStd<float, CUDAContext>(
      static_cast<int>(N * C), epsilon_, rstd, rstd, &context_);
}

template <>
bool InstanceNormGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* dX,
    float* dgamma,
    float* dbeta) {
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CUDA));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  ComputeInternalGradientsNCHWCUDAKernel<float>
      <<<N * C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          HxW, dY, X, ds_data, db_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ReinitializeTensor(&c1_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&c2_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&c3_, {N, C}, at::dtype<float>().device(CUDA));
  float* c1_data = c1_.mutable_data<float>();
  float* c2_data = c2_.mutable_data<float>();
  float* c3_data = c3_.mutable_data<float>();
  int64_t B = math::DivUp<int64_t>(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParams<float>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          C,
          1.0f / static_cast<float>(HxW),
          ds_data,
          db_data,
          mean,
          rstd,
          gamma,
          c1_data,
          c2_data,
          c3_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  B = math::DivUp<int64_t>(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  InstanceNormBackwardCUDAKernel<float, StorageOrder::NCHW>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, dY, X, c1_data, c2_data, c3_data, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  GammaBetaBackwardCUDAKernel<float>
      <<<C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool InstanceNormGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* dX,
    float* dgamma,
    float* dbeta) {
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CUDA));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  ReinitializeTensor(&ones_, {HxW}, at::dtype<float>().device(CUDA));
  math::Set<float, CUDAContext>(
      ones_.numel(), 1.0f, ones_.mutable_data<float>(), &context_);
  const float* ones_data = ones_.data<float>();
  math::Mul<float, CUDAContext>(N * C * HxW, dY, X, dX, &context_);
  math::GemmStridedBatched<float, CUDAContext>(
      CblasTrans,
      CblasNoTrans,
      N,
      C,
      1,
      HxW,
      1.0f,
      dX,
      C * HxW,
      ones_data,
      0,
      0.0f,
      ds_data,
      C,
      &context_);
  math::GemmStridedBatched<float, CUDAContext>(
      CblasTrans,
      CblasNoTrans,
      N,
      C,
      1,
      HxW,
      1.0f,
      dY,
      C * HxW,
      ones_data,
      0,
      0.0f,
      db_data,
      C,
      &context_);
  ReinitializeTensor(&c1_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&c2_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&c3_, {N, C}, at::dtype<float>().device(CUDA));
  float* c1_data = c1_.mutable_data<float>();
  float* c2_data = c2_.mutable_data<float>();
  float* c3_data = c3_.mutable_data<float>();
  int64_t B = math::DivUp<int64_t>(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParams<float>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          C,
          1.0f / static_cast<float>(HxW),
          ds_data,
          db_data,
          mean,
          rstd,
          gamma,
          c1_data,
          c2_data,
          c3_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  B = math::DivUp<int64_t>(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  InstanceNormBackwardCUDAKernel<float, StorageOrder::NHWC>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, dY, X, c1_data, c2_data, c3_data, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  GammaBetaBackwardCUDAKernel<float>
      <<<C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(InstanceNorm, InstanceNormOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    InstanceNormGradient,
    InstanceNormGradientOp<float, CUDAContext>);

} // namespace caffe2
