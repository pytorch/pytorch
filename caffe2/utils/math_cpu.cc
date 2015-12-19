// Implementes the math functions for CPU.
#include <random>

#include "caffe2/utils/math.h"
#include "caffe2/utils/mkl_alternate.h"
#include "caffe2/core/context.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

namespace {
// Common Eigen types that we will often use
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> >;
}  // namespace

namespace caffe2 {
namespace math {

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname, OriginalFunc)          \
template <>                                                                    \
void Funcname<T, CPUContext>(                                              \
    const int N, const T* x, T* y,                                     \
    CPUContext* context) {                                                     \
  OriginalFunc(N, x, y);                                                       \
}
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Exp, vsExp)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Exp, vdExp)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log, vsLn)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Log, vdLn)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqr, vsSqr)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sqr, vdSqr)

template <>
void Powx<float, CPUContext>(
    const int N, const float* a, float b, float* y, CPUContext* context) {
  vsPowx(N, a, b, y);
}

template <>
void Powx<double, CPUContext>(
    const int N, const double* a, double b, double* y, CPUContext* context) {
  vdPowx(N, a, b, y);
}


#define DELEGATE_SIMPLE_BINARY_FUNCTION(T, Funcname, OriginalFunc)         \
template <>                                                                    \
void Funcname<T, CPUContext>(                                              \
    const int N, const T* a, const T* b, T* y,                     \
    CPUContext* context) {                                                     \
  OriginalFunc(N, a, b, y);                                                    \
}

DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Add, vsAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Add, vdAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Sub, vsSub)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Sub, vdSub)
DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Mul, vsMul)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Mul, vdMul)
DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Div, vsDiv)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Div, vdDiv)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION

#define CAFFE2_SPECIALIZED_ROWWISEMAX(T)                                       \
template <> void RowwiseMax<T, CPUContext>(                                    \
    const int N, const int D, const T* x, T* y, CPUContext* context) {         \
  EigenVectorMap<T>(y, N) =                                                    \
      ConstEigenMatrixMap<T>(x, D, N).colwise().maxCoeff();                    \
}
CAFFE2_SPECIALIZED_ROWWISEMAX(float)

#define CAFFE2_SPECIALIZED_COLWISEMAX(T)                                       \
template <> void ColwiseMax<T, CPUContext>(                                    \
    const int N, const int D, const T* x, T* y, CPUContext* context) {         \
  EigenVectorMap<T>(y, D) =                                                    \
      ConstEigenMatrixMap<T>(x, D, N).rowwise().maxCoeff();                    \
}
CAFFE2_SPECIALIZED_COLWISEMAX(float)

// AddToRow and AddToCol adds the corresponding row/col vector x to the matrix y
// of shape M x N. The actual implementation uses eigen which is column major,
// so notice the row/column swap in the actual implementation.
template <>
void AddToRow<float, CPUContext>(
    const int M, const int N, const float* x, float* y, CPUContext* context) {
  EigenMatrixMap<float>(y, N, M).colwise() += ConstEigenVectorMap<float>(x, N);
}
template <>
void AddToCol<float, CPUContext>(
    const int M, const int N, const float* x, float* y, CPUContext* context) {
  EigenMatrixMap<float>(y, N, M).rowwise() +=
      ConstEigenVectorMap<float>(x, M).transpose();
}


// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
// A (M*K) * B(K*N) = C(M*N)
template <>
void Gemm<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const float alpha, const float* A,
    const float* B, const float beta, float* C, CPUContext* context) {
  auto C_mat = EigenMatrixMap<float>(C, N, M);
  if (beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }
  switch (TransA) {
  case CblasNoTrans: {
    switch (TransB) {
    case CblasNoTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, N, K) *
          ConstEigenMatrixMap<float>(A, K, M));
      return;
    case CblasTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, K, N).transpose() *
          ConstEigenMatrixMap<float>(A, K, M));
      return;
    default:
      CAFFE_LOG_FATAL << "Unexpected CBLAS_TRANSPOSE for TransB";
    }
  }
  case CblasTrans: {
    switch (TransB) {
    case CblasNoTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, N, K) *
          ConstEigenMatrixMap<float>(A, M, K).transpose());
      return;
    case CblasTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, K, N).transpose() *
          ConstEigenMatrixMap<float>(A, M, K).transpose());
      return;
    default:
      CAFFE_LOG_FATAL << "Unexpected CBLAS_TRANSPOSE for TransB";
    }
  }
  default:
    CAFFE_LOG_FATAL << "Unexpected CBLAS_TRANSPOSE for TransA";
  }
}

template <>
void Gemv<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha,
    const float* A, const float* x, const float beta, float* y,
    CPUContext* context) {
  EigenVectorMap<float> y_vec(y, TransA == CblasNoTrans ? M : N);
  if (beta == 0) {
    // In Caffe2 we often do a lazy initialization, which may contain NaNs in
    // the float values. As a result, if beta is 0, we explicitly do a setzero.
    y_vec.setZero();
  } else {
    y_vec *= beta;
  }
  switch (TransA) {
    case CblasNoTrans: {
      y_vec.noalias() += alpha * (
          ConstEigenMatrixMap<float>(A, N, M).transpose() *
          ConstEigenVectorMap<float>(x, N));
      return;
    }
    case CblasTrans: {
      y_vec.noalias() += alpha * (
          ConstEigenMatrixMap<float>(A, N, M) *
          ConstEigenVectorMap<float>(x, M));
      return;
    }
    default:
      CAFFE_LOG_FATAL << "Gemv float found an unexpected CBLAS_TRANSPOSE input.";
  }
}

#define CAFFE2_SPECIALIZED_SET(T)                                          \
template <>                                                                    \
void Set<T, CPUContext>(const int N, const T alpha, T *Y,          \
                           CPUContext* context) {                              \
  EigenVectorMap<T>(Y, N).setConstant(alpha);                              \
}

CAFFE2_SPECIALIZED_SET(float);
CAFFE2_SPECIALIZED_SET(double);
CAFFE2_SPECIALIZED_SET(int);
#undef CAFFE2_SPECIALIZED_SET

template <>
void RandUniform<float, CPUContext>(
    const int n, const float a, const float b, float* r,
    CPUContext* context) {
  std::uniform_real_distribution<float> distribution(a, b);
  for (int i = 0; i < n; ++i) {
    r[i] = distribution(context->RandGenerator());
  }
}

template <>
void RandUniform<int, CPUContext>(
    const int n, const int a, const int b, int* r,
    CPUContext* context) {
  std::uniform_int_distribution<int> distribution(a, b);
  for (int i = 0; i < n; ++i) {
    r[i] = distribution(context->RandGenerator());
  }
}


template <>
void RandGaussian<float, CPUContext>(
    const int n, const float mean, const float std, float* r,
    CPUContext* context) {
  std::normal_distribution<float> distribution(mean, std);
  for (int i = 0; i < n; ++i) {
    r[i] = distribution(context->RandGenerator());
  }
}

template<>
void Dot<float, CPUContext>(
    const int N, const float* a, const float* b, float* y,
    CPUContext* context) {
  *y = ConstEigenVectorMap<float>(a, N).dot(ConstEigenVectorMap<float>(b, N));
}

template<>
void Dot<double, CPUContext>(
    const int N, const double* a, const double* b, double* y,
    CPUContext* context) {
  *y = ConstEigenVectorMap<double>(a, N).dot(ConstEigenVectorMap<double>(b, N));
}

template<>
void Sum<float, CPUContext>(
    const int N, const float* x, float* y,
    CPUContext* context) {
  *y = ConstEigenVectorMap<float>(x, N).sum();
}

template<>
void Sum<double, CPUContext>(
    const int N, const double* x, double* y,
    CPUContext* context) {
  *y = ConstEigenVectorMap<double>(x, N).sum();
}

template <>
void Select<float, CPUContext>(
      const int N, const int D, const float* x, const int* idx, float* y,
      CPUContext* context) {
  for (int i = 0; i < N; ++i) {
    CAFFE_DCHECK_LT(idx[i], D);
    y[i] = x[i * D + idx[i]];
  }
}

template <>
void Scale<float, CPUContext>(
    const int n, const float alpha, const float *x, float* y,
    CPUContext* context) {
  EigenVectorMap<float>(y, n) = ConstEigenVectorMap<float>(x, n) * alpha;
}

template <>
void Scale<double, CPUContext>(
  const int n, const double alpha, const double *x, double* y,
  CPUContext* context) {
  EigenVectorMap<double>(y, n) = ConstEigenVectorMap<double>(x, n) * alpha;
}

template <>
void Scale<float, CPUContext>(
    const int n, const float* alpha, const float* x, float* y,
    CPUContext* context) {
  EigenVectorMap<float>(y, n) = ConstEigenVectorMap<float>(x, n) * (*alpha);
}

template <>
void Scale<double, CPUContext>(
    const int n, const double* alpha, const double* x, double* y,
    CPUContext* context) {
  EigenVectorMap<double>(y, n) = ConstEigenVectorMap<double>(x, n) * (*alpha);
}


template <>
void Axpy<float, CPUContext>(const int N, const float alpha, const float* x,
                             float* Y, CPUContext* context) {
  EigenVectorMap<float>(Y, N) += ConstEigenVectorMap<float>(x, N) * alpha;
}

template <>
void Axpy<double, CPUContext>(const int N, const double alpha, const double* x,
                             double* Y, CPUContext* context) {
  EigenVectorMap<double>(Y, N) += ConstEigenVectorMap<double>(x, N) * alpha;
}

template <>
void Axpy<float, CPUContext>(const int N, const float* alpha, const float* x,
                             float* Y, CPUContext* context) {
  EigenVectorMap<float>(Y, N) += ConstEigenVectorMap<float>(x, N) * (*alpha);
}

template <>
void Axpy<double, CPUContext>(const int N, const double* alpha, const double* x,
                             double* Y, CPUContext* context) {
  EigenVectorMap<double>(Y, N) += ConstEigenVectorMap<double>(x, N) * (*alpha);
}

template <>
void Axpby<float, CPUContext>(const int N, const float alpha, const float* x,
                              const float beta, float* y,
                              CPUContext* context) {
  EigenVectorMap<float> y_vec(y, N);
  y_vec = y_vec * beta + ConstEigenVectorMap<float>(x, N) * alpha;
}

template <>
void Axpby<double, CPUContext>(const int N, const double alpha,
                               const double* x, const double beta, double* y,
                               CPUContext* context) {
  EigenVectorMap<double> y_vec(y, N);
  y_vec = y_vec * beta + ConstEigenVectorMap<double>(x, N) * alpha;
}

template <>
void Im2col<float, CPUContext, StorageOrder::NCHW>(
    const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h,
    const int stride_w, float* data_col, CPUContext* context) {
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_t + h_offset;
        int w_pad = w * stride_w - pad_l + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

template <>
void Im2col<float, CPUContext, StorageOrder::NHWC>(
    const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h, const int stride_w, float* data_col,
    CPUContext* context) {
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;

  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            memcpy(data_col, data_im + (ih * width + iw) * channels,
                   sizeof(float) * channels);
          } else {
            // This should be simply padded with zero.
            memset(data_col, 0, sizeof(float) * channels);
          }
          data_col += channels;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

template <>
void Col2im<float, CPUContext, StorageOrder::NCHW>(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h,
    const int stride_w, float* data_im, CPUContext* context) {
  Set<float, CPUContext>(height * width * channels, 0, data_im, context);
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_t + h_offset;
        int w_pad = w * stride_w - pad_l + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

template <>
void Col2im<float, CPUContext, StorageOrder::NHWC>(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h,
    const int stride_w, float* data_im, CPUContext* context) {
  Set<float, CPUContext>(height * width * channels, 0, data_im, context);
  int height_col = (height + pad_t + pad_b - kernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - kernel_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      float* data_im_patch = data_im + (h_pad * width + w_pad) * channels;
      for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            Add<float, CPUContext>(
                  channels, data_im_patch, data_col, data_im_patch, context);
          }
          data_im_patch += channels;
          data_col += channels;
        }
        // Jump over remaining number of channels
        data_im_patch += channels * (width - kernel_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

template <>
void CopyMatrix<CPUContext>(
    const size_t itemsize, const int M, const int N, const void* A,
    const int lda, void* B, const int ldb, CPUContext* context) {
  for (int i = 0; i < M; ++i) {
    memcpy(static_cast<char*>(B) + ldb * i * itemsize,
           static_cast<const char*>(A) + lda * i * itemsize,
           itemsize * N);
  }
}

}  // namespace math
}  // namespace caffe2
