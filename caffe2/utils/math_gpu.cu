// Implementes the math functions for CPU.

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/version.h>

#include "caffe2/utils/math.h"
#include "caffe2/core/context_gpu.h"

#if THRUST_VERSION >= 100800
#define THRUST_SUPPORTS_PER_THREAD
#endif  // THRUST_VERSION >= 100800

namespace caffe2 {
namespace math {

// TODO(Yangqing): Yuck again. Maybe change it to templated functors?
#define DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(T, Funcname, function)             \
__global__                                                                     \
void _Kernel_##T##_##Funcname(const int N, const T* x, T* y) {                 \
  CUDA_1D_KERNEL_LOOP(i, N) {                                                  \
    y[i] = function(x[i]);                                                     \
  }                                                                            \
}                                                                              \
template <>                                                                    \
void Funcname<T, CUDAContext>(                                                 \
    const int N, const T* x, T* y,                                             \
    CUDAContext* context) {                                                    \
  _Kernel_##T##_##Funcname<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,      \
                                 0, context->cuda_stream()>>>(                 \
      N, x, y);                                                                \
}

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Exp, expf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Exp, exp)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Log, logf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Log, log)

__device__ float cuda_sqrf(const float x) { return x * x; }
__device__ double cuda_sqr(const double x) { return x * x; }

DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqr, cuda_sqrf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Sqr, cuda_sqr)

#define DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(T, Funcname, function)        \
__global__                                                                     \
void _Kernel_##T##_##Funcname(                                             \
    const int N, const T* a, const T* b, T* y) {                   \
  CUDA_1D_KERNEL_LOOP(i, N) {                                                  \
    y[i] = function(a[i], b[i]);                                               \
  }                                                                            \
}                                                                              \
template <>                                                                    \
void Funcname<T, CUDAContext>(                                             \
    const int N, const T* a, const T* b, T* y,                     \
    CUDAContext* context) {                                                    \
  _Kernel_##T##_##Funcname<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,  \
                                 0, context->cuda_stream()>>>(                 \
      N, a, b, y);                                                             \
}


#define CAFFE_MATH_CUDA_ADD(x, y) (x + y)
#define CAFFE_MATH_CUDA_SUB(x, y) (x - y)
#define CAFFE_MATH_CUDA_MUL(x, y) (x * y)
#define CAFFE_MATH_CUDA_DIV(x, y) (x / y)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float,  Add, CAFFE_MATH_CUDA_ADD)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Add, CAFFE_MATH_CUDA_ADD)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float,  Sub, CAFFE_MATH_CUDA_SUB)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Sub, CAFFE_MATH_CUDA_SUB)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float,  Mul, CAFFE_MATH_CUDA_MUL)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Mul, CAFFE_MATH_CUDA_MUL)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float,  Div, CAFFE_MATH_CUDA_DIV)
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Div, CAFFE_MATH_CUDA_DIV)


/*
#define CAFFE2_SPECIALIZED_ROWWISEMAX(T)                                       \
template <>                                                                    \
void RowwiseMax<T, CPUContext>(                                                \
    const int N, const int D, const T* x, T* y, CPUContext* context) {         \
  for (int i = 0; i < N; ++i) {                                                \
    y[i] = x[i*D];                                                             \
    for (int j = 1; j < D; ++j) {                                              \
      y[i] = std::max(y[i], x[i * D + j]);                                     \
    }                                                                          \
  }                                                                            \
}
CAFFE2_SPECIALIZED_ROWWISEMAX(float)

#define CAFFE2_SPECIALIZED_COLWISEMAX(T)                                       \
template <>                                                                    \
void ColwiseMax<T, CPUContext>(                                                \
    const int N, const int D, const T* x, T* y, CPUContext* context) {         \
  memcpy(y, x, sizeof(T) * D);                                                 \
  for (int i = 1; i < N; ++i) {                                                \
    for (int j = 0; j < D; ++j) {                                              \
      y[j] = std::max(y[j], x[i * D + j]);                                     \
    }                                                                          \
  }                                                                            \
}
CAFFE2_SPECIALIZED_COLWISEMAX(float)
*/

namespace {
template<typename T>
__global__ void AddToRowKernel(const int M, const int N, const T* x,
                               T* y) {
  CUDA_1D_KERNEL_LOOP(i, M * N) {
    y[i] += x[i % N];
  }
}
template<typename T>
__global__ void AddToColKernel(const int M, const int N, const T* x,
                               T* y) {
  CUDA_1D_KERNEL_LOOP(i, M * N) {
    y[i] += x[i % M];
  }
}
}  // namespace

template <>
void AddToRow<float, CUDAContext>(
    const int M, const int N, const float* x, float* y, CUDAContext* context) {
  AddToRowKernel<float><<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS,
                          0, context->cuda_stream()>>>(M, N, x, y);
}
template <>
void AddToCol<float, CUDAContext>(
    const int M, const int N, const float* x, float* y, CUDAContext* context) {
  AddToColKernel<float><<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS,
                          0, context->cuda_stream()>>>(M, N, x, y);
}

// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <>
void Gemm<float, CUDAContext>(
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const float alpha, const float* A,
    const float* B, const float beta, float* C, CUDAContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(context->cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}


template <>
void Gemv<float, CUDAContext>(
    const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha,
    const float* A, const float* x, const float beta, float* y,
    CUDAContext* context) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(context->cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}


namespace {
template <typename T>
__global__ void SetKernel(const int N, const T alpha, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = alpha;
  }
}
}  // namespace

#define CAFFE2_SPECIALIZED_CUDA_SET(T)                                     \
  template <>                                                                  \
  void Set<T, CUDAContext>(const int N, const T alpha, T *Y,       \
                              CUDAContext* context) {                          \
    SetKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,                   \
                0, context->cuda_stream()>>>(N, alpha, Y);                     \
  }

CAFFE2_SPECIALIZED_CUDA_SET(float);
CAFFE2_SPECIALIZED_CUDA_SET(double);
CAFFE2_SPECIALIZED_CUDA_SET(int);
#undef CAFFE2_SPECIALIZED_CUDA_SET

namespace {
template <typename T>
__global__ void UniformShift(const int N, const T min, const T max,
                             T* x) {
  T scale = max - min;
  CUDA_1D_KERNEL_LOOP(i, N) {
    x[i] = x[i] * scale + min;
  }
}
}  // namespace

template <>
void RandUniform<float, CUDAContext>(
    const int n, const float min, const float max, float* r,
    CUDAContext* context) {
  CURAND_CHECK(curandGenerateUniform(context->curand_generator(), r, n));
  UniformShift<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                        0, context->cuda_stream()>>>(n, min, max, r);
}

template <>
void RandUniform<double, CUDAContext>(
    const int n, const double min, const double max, double* r,
    CUDAContext* context) {
  CURAND_CHECK(curandGenerateUniformDouble(context->curand_generator(), r, n));
  UniformShift<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                         0, context->cuda_stream()>>>(n, min, max, r);
}

template <>
void RandGaussian<float, CUDAContext>(
    const int n, const float mean, const float std, float* r,
    CUDAContext* context) {
  CURAND_CHECK(curandGenerateNormal(
      context->curand_generator(), r, n, mean, std));
}

template <>
void RandGaussian<double, CUDAContext>(
    const int n, const double mean, const double std, double* r,
    CUDAContext* context) {
  CURAND_CHECK(curandGenerateNormalDouble(
      context->curand_generator(), r, n, mean, std));
}


template<>
void Dot<float, CUDAContext>(
    const int n, const float* a, const float* b, float* y,
    CUDAContext* context) {
  float result;
  CUBLAS_CHECK(cublasSdot(context->cublas_handle(), n, a, 1, b, 1, &result));
  context->Copy<float, CPUContext, CUDAContext>(1, &result, y);
}

template<>
void Dot<double, CUDAContext>(
    const int n, const double* a, const double* b, double* y,
    CUDAContext* context) {
  double result;
  CUBLAS_CHECK(cublasDdot(context->cublas_handle(), n, a, 1, b, 1, y));
  context->Copy<double, CPUContext, CUDAContext>(1, &result, y);
}

#ifdef THRUST_SUPPORTS_PER_THREAD

#define CAFFE2_MATH_SUM_FUNC(T)                                                \
template<>                                                                     \
void Sum<T, CUDAContext>(const int N, const T* x, T* y, CUDAContext* context) {\
  thrust::device_ptr<const T> dev_ptr(x);                                      \
  T result = thrust::reduce(                                                   \
      thrust::cuda::par.on(context->cuda_stream()),                            \
      dev_ptr, dev_ptr + N, static_cast<T>(0), thrust::plus<T>());             \
  context->Copy<T, CPUContext, CUDAContext>(1, &result, y);                    \
}

#else  // THRUST_SUPPORTS_PER_THREAD

// Really, for any real use you should not be invoking this but should use the
// thrust version, so I was not very careful in tuning the performance of the
// sum kernel.
#define SUM_KERNEL_NTHREADS 128
template <typename T>
__global__ void SumKernel(const int N, const T* X, T* Y) {
  const int idx = threadIdx.x;
  __shared__ T reduction_buffer[SUM_KERNEL_NTHREADS];

  reduction_buffer[idx] = 0;

  // A two-level reduction to get the sum.
  for (int i = idx; i < N; i += SUM_KERNEL_NTHREADS) {
    reduction_buffer[idx] += X[i];
  }
  __syncthreads();
  if (idx == 0) {
    float tmp = 0;
    for (int i = 0; i < SUM_KERNEL_NTHREADS; ++i) {
      tmp += reduction_buffer[i];
    }
    *Y = tmp;
  }
}

#define CAFFE2_MATH_SUM_FUNC(T)                                                \
template<>                                                                     \
void Sum<T, CUDAContext>(const int N, const T* x, T* y, CUDAContext* context) {\
  SumKernel<<<1, SUM_KERNEL_NTHREADS, 0, context->cuda_stream()>>>(N, x, y);   \
}

#endif  // THRUST_SUPPORTS_PER_THREAD

CAFFE2_MATH_SUM_FUNC(float)
CAFFE2_MATH_SUM_FUNC(double)
#undef CAFFE2_MATH_SUM_FUNC

namespace {
template <typename T>
__global__ void SelectKernel(
    const int N, const int D, const T* x, const int* idx, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i * D + idx[i]];
  }
}
}  // namespace

template <>
void Select<float, CUDAContext>(
      const int N, const int D, const float* x, const int* idx, float* y,
      CUDAContext* context) {
  SelectKernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                        0, context->cuda_stream()>>>(N, D, x, idx, y);
}

namespace {
template <typename T>
__global__ void ScaleKernel(
    const int n, const T alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = x[i] * alpha;
  }
}

template <typename T>
__global__ void ScaleKernelDeviceAlpha(
    const int n, const T* alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = x[i] * (*alpha);
  }
}
}  // namespace

template <>
void Scale<float, CUDAContext>(
    const int n, const float alpha, const float *x, float* y,
    CUDAContext* context) {
  ScaleKernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                       0, context->cuda_stream()>>>(n, alpha, x, y);
}

template <>
void Scale<double, CUDAContext>(
    const int n, const double alpha, const double *x, double* y,
    CUDAContext* context) {
  ScaleKernel<double><<<
      CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          n, alpha, x, y);
}

template <>
void Scale<float, CUDAContext>(
    const int n, const float* alpha, const float *x, float* y,
    CUDAContext* context) {
  ScaleKernelDeviceAlpha<float><<<
      CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          n, alpha, x, y);
}

template <>
void Scale<double, CUDAContext>(
    const int n, const double* alpha, const double *x, double* y,
    CUDAContext* context) {
  ScaleKernelDeviceAlpha<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                       0, context->cuda_stream()>>>(n, alpha, x, y);
}

template <>
void Axpy<float, CUDAContext>(const int N, const float alpha, const float* X,
                              float* Y, CUDAContext* context) {
  CUBLAS_CHECK(cublasSaxpy(context->cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void Axpy<double, CUDAContext>(
    const int N, const double alpha, const double* X,
    double* Y, CUDAContext* context) {
  CUBLAS_CHECK(cublasDaxpy(context->cublas_handle(), N, &alpha, X, 1, Y, 1));
}

namespace {
template <typename T>
__global__ void AxpyKernel(const int n, const T* a, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    y[index] += x[index] * (*a);
  }
}
}  // namespace

template <>
void Axpy<float, CUDAContext>(
    const int n, const float* alpha, const float* X,
    float* Y, CUDAContext* context) {
  AxpyKernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                       0, context->cuda_stream()>>>(n, alpha, X, Y);
}

template <>
void Axpy<double, CUDAContext>(
    const int n, const double* alpha, const double* X,
    double* Y, CUDAContext* context) {
  AxpyKernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                       0, context->cuda_stream()>>>(n, alpha, X, Y);
}


namespace {
template <typename T>
__global__ void AxpbyKernel(const int n, const T a, const T* x,
                             const T b, T* y) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    y[index] = x[index] * a + y[index] * b;
  }
}
}  // namespace

template <>
void Axpby<float, CUDAContext>(
    const int n, const float a, const float* x, const float b, float* y,
    CUDAContext* context) {
  AxpbyKernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                       0, context->cuda_stream()>>>(n, a, x, b, y);
}

template <>
void Axpby<double, CUDAContext>(
    const int n, const double a, const double* x, const double b, double* y,
    CUDAContext* context) {
  AxpbyKernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                        0, context->cuda_stream()>>>(n, a, x, b, y);
}

namespace {

template <typename T>
__global__ void im2col_gpu_kernel_nchw(const int n, const T* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    T* data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_t;
    int w_in = w_out * stride_w - pad_l;
    T* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const T* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename T>
__global__ void im2col_gpu_kernel_nhwc(const int n, const T* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l,
    const int stride_h, const int stride_w,
    const int width_col, const int channels,
    T* data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int channel_in = index % channels;
    int w_out = index / channels % width_col;
    int h_out = index / channels / width_col;
    int h_in = h_out * stride_h - pad_t;
    int w_in = w_out * stride_w - pad_l;
    T* local_data_col = data_col +
        ((h_out * width_col) + w_out) * channels * kernel_h * kernel_w
        + channel_in;
    for (int i = 0; i < kernel_h; ++i) {
      int h = h_in + i;
      for (int j = 0; j < kernel_w; ++j) {
        int w = w_in + j;
        *local_data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im[(h * width + w) * channels + channel_in] : 0;
        local_data_col += channels;
      }
    }
  }
}

template <typename T>
__global__ void col2im_gpu_kernel_nchw(const int n, const T* data_col,
    const int height, const int width,
    const int patch_h, const int patch_w,
    const int pad_t, const int pad_l,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    T* data_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    int w = index % width + pad_l;
    int h = (index / width) % height + pad_t;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    int offset =
        (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename T>
__global__ void col2im_gpu_kernel_nhwc(const int n, const T* data_col,
    const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_t, const int pad_l,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    T* data_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0;
    int c = index % channels;
    int w = index / channels % width + pad_l;
    int h = index / channels / width + pad_t;
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    int channels_col = patch_h * patch_w * channels;
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int c_col = ((h - h_col * stride_h) * patch_w + w - w_col * stride_w) * channels + c;
        val += data_col[(h_col * width_col + w_col) * channels_col + c_col];
      }
    }
    */
    // Equivalent of above
    int offset = (h * patch_w + w) * channels + c;
    int coeff_h_col = width_col * channels_col - stride_h * patch_w * channels;
    int coeff_w_col = channels_col - stride_w * channels;
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

}  // namespace

template <>
void Im2col<float, CUDAContext, StorageOrder::NCHW>(
    const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h,
    const int stride_w, float* data_col, CUDAContext* context) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel_nchw<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                  CAFFE_CUDA_NUM_THREADS, 0,
                                  context->cuda_stream()>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_t,
      pad_l, stride_h, stride_w, height_col, width_col, data_col);
}

template <>
void Im2col<float, CUDAContext, StorageOrder::NHWC>(
    const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h,
    const int stride_w, float* data_col, CUDAContext* context) {
  // We are going to launch height_col * width_col * channels kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;
  int num_kernels = height_col * width_col * channels;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel_nhwc<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                  CAFFE_CUDA_NUM_THREADS, 0,
                                  context->cuda_stream()>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_t,
      pad_l, stride_h, stride_w, width_col, channels, data_col);
}


template <>
void Col2im<float, CUDAContext, StorageOrder::NCHW>(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h,
    const int stride_w, float* data_im, CUDAContext* context) {
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_gpu_kernel_nchw<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                  CAFFE_CUDA_NUM_THREADS, 0,
                                  context->cuda_stream()>>>(
      num_kernels, data_col, height, width, kernel_h, kernel_w,
      pad_t, pad_l, stride_h, stride_w,
      height_col, width_col, data_im);
}

template <>
void Col2im<float, CUDAContext, StorageOrder::NHWC>(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h,
    const int stride_w, float* data_im, CUDAContext* context) {
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;
  int num_kernels = height * width * channels;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_gpu_kernel_nhwc<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                  CAFFE_CUDA_NUM_THREADS, 0,
                                  context->cuda_stream()>>>(
      num_kernels, data_col, width, channels, kernel_h, kernel_w,
      pad_t, pad_l, stride_h, stride_w, height_col, width_col, data_im);
}

namespace {
__global__ void CopyMatrixKernel(
    const int M, const int N, const char* A, const int lda,
    char* B, const int ldb) {
  CUDA_1D_KERNEL_LOOP(i, M * N) {
    int r = i / N;
    int c = i % N;
    B[r * ldb + c] = A[r * lda + c];
  }
}
}  // namespace

template <>
void CopyMatrix<CUDAContext>(
    const size_t itemsize, const int M, const int N, const void* A,
    const int lda, void* B, const int ldb, CUDAContext* context) {
  CopyMatrixKernel<<<CAFFE_GET_BLOCKS(M * N), CAFFE_CUDA_NUM_THREADS, 0,
                     context->cuda_stream()>>>(
      M, N * itemsize, static_cast<const char*>(A), lda * itemsize,
      static_cast<char*>(B), ldb * itemsize);
}

}  // namespace math
}  // namespace caffe2
