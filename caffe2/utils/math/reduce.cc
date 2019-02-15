#include "caffe2/utils/math/reduce.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

#ifdef CAFFE2_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif // CAFFE2_USE_ACCELERATE

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#include "caffe2/core/context.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math/elementwise.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

#define DELEGATE_ROWWISE_REDUCE_FUNCTION(Func, EigenFunc)                    \
  template <typename T>                                                      \
  void Rowwise##Func(                                                        \
      const int rows,                                                        \
      const int cols,                                                        \
      const T alpha,                                                         \
      const T* X,                                                            \
      T* Y,                                                                  \
      CPUContext* /* context */) {                                           \
    EigenVectorMap<T>(Y, rows) =                                             \
        ConstEigenMatrixMap<T>(X, cols, rows).colwise().EigenFunc() * alpha; \
  }
DELEGATE_ROWWISE_REDUCE_FUNCTION(ReduceMin, minCoeff)
DELEGATE_ROWWISE_REDUCE_FUNCTION(ReduceMax, maxCoeff)
DELEGATE_ROWWISE_REDUCE_FUNCTION(ReduceSum, sum)
DELEGATE_ROWWISE_REDUCE_FUNCTION(ReduceMean, mean)
DELEGATE_ROWWISE_REDUCE_FUNCTION(ReduceL1, template lpNorm<1>)
DELEGATE_ROWWISE_REDUCE_FUNCTION(ReduceL2, norm)
#undef DELEGATE_ROWWISE_REDUCE_FUNCTION

#ifndef CAFFE2_USE_EIGEN_FOR_BLAS

#define DELEGATE_ROWWISE_REDUCE_FUNCTION(T, Func, BLASFunc) \
  template <>                                               \
  void Rowwise##Func(                                       \
      const int rows,                                       \
      const int cols,                                       \
      const T alpha,                                        \
      const T* X,                                           \
      T* Y,                                                 \
      CPUContext* /* context */) {                          \
    for (int i = 0; i < rows; ++i) {                        \
      Y[i] = BLASFunc(cols, X + i * cols, 1) * alpha;       \
    }                                                       \
  }
DELEGATE_ROWWISE_REDUCE_FUNCTION(float, ReduceL1, cblas_sasum)
DELEGATE_ROWWISE_REDUCE_FUNCTION(double, ReduceL1, cblas_dasum)
DELEGATE_ROWWISE_REDUCE_FUNCTION(float, ReduceL2, cblas_snrm2)
DELEGATE_ROWWISE_REDUCE_FUNCTION(double, ReduceL2, cblas_dnrm2)
#undef DELEGATE_ROWWISE_REDUCE_FUNCTION

#endif // CAFFE2_USE_EIGEN_FOR_BLAS

#define DELEGATE_COLWISE_REDUCE_FUNCTION(Func, MathFunc)          \
  template <typename T>                                           \
  void Colwise##Func(                                             \
      const int rows,                                             \
      const int cols,                                             \
      const T alpha,                                              \
      const T* X,                                                 \
      T* Y,                                                       \
      CPUContext* context) {                                      \
    std::memcpy(Y, X, sizeof(T) * cols);                          \
    for (int i = 1; i < rows; ++i) {                              \
      MathFunc<T, CPUContext>(cols, Y, X + i * cols, Y, context); \
    }                                                             \
    Scale<T, T, CPUContext>(cols, alpha, Y, Y, context);          \
  }
DELEGATE_COLWISE_REDUCE_FUNCTION(ReduceMin, Min)
DELEGATE_COLWISE_REDUCE_FUNCTION(ReduceMax, Max)
DELEGATE_COLWISE_REDUCE_FUNCTION(ReduceSum, Add)
#undef DELEGATE_COLWISE_REDUCE_FUNCTION

template <typename T>
void ColwiseReduceMean(
    const int rows,
    const int cols,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ColwiseReduceSum<T>(rows, cols, alpha / static_cast<T>(rows), X, Y, context);
}

template <typename T>
void ColwiseReduceL1(
    const int rows,
    const int cols,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ConstEigenArrayMap<T> X_arr(X, cols, rows);
  EigenVectorArrayMap<T> Y_arr(Y, cols);
  Y_arr = X_arr.col(0).abs();
  for (int i = 1; i < rows; ++i) {
    Y_arr += X_arr.col(i).abs();
  }
  Scale<T, T, CPUContext>(cols, alpha, Y, Y, context);
}

template <typename T>
void ColwiseReduceL2(
    const int rows,
    const int cols,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* /* context */) {
  ConstEigenArrayMap<T> X_arr(X, cols, rows);
  EigenVectorArrayMap<T> Y_arr(Y, cols);
  Y_arr = X_arr.col(0).square();
  for (int i = 1; i < rows; ++i) {
    Y_arr += X_arr.col(i).square();
  }
  Y_arr = Y_arr.sqrt() * alpha;
}

template <typename T>
void BothEndsReduceMin(
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  EigenVectorArrayMap<T> Y_arr(Y, N);
  Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().minCoeff();
  for (int i = 1; i < M; ++i) {
    ConstEigenArrayMap<T> X_arr(X + i * N * K, K, N);
    for (int j = 0; j < N; ++j) {
      Y[j] = std::min(Y[j], X_arr.col(j).minCoeff());
    }
  }
  Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
}

template <typename T>
void BothEndsReduceMax(
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  EigenVectorArrayMap<T> Y_arr(Y, N);
  Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().maxCoeff();
  for (int i = 1; i < M; ++i) {
    ConstEigenArrayMap<T> X_arr(X + i * N * K, K, N);
    for (int j = 0; j < N; ++j) {
      Y[j] = std::max(Y[j], X_arr.col(j).maxCoeff());
    }
  }
  Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
}

template <typename T>
void BothEndsReduceSum(
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  EigenVectorArrayMap<T> Y_arr(Y, N);
  Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().sum();
  for (int i = 1; i < M; ++i) {
    Y_arr += ConstEigenArrayMap<T>(X + i * N * K, K, N).colwise().sum();
  }
  Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
}

template <typename T>
void BothEndsReduceMean(
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  EigenVectorArrayMap<T> Y_arr(Y, N);
  Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().mean();
  for (int i = 1; i < M; ++i) {
    Y_arr += ConstEigenArrayMap<T>(X + i * N * K, K, N).colwise().mean();
  }
  Scale<T, T, CPUContext>(N, alpha / static_cast<T>(M), Y, Y, context);
}

template <typename T>
void BothEndsReduceL1(
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  EigenVectorMap<T> Y_vec(Y, N);
  Y_vec = ConstEigenMatrixMap<T>(X, K, N).colwise().template lpNorm<1>();
  for (int i = 1; i < M; ++i) {
    Y_vec += ConstEigenMatrixMap<T>(X + i * N * K, K, N)
                 .colwise()
                 .template lpNorm<1>();
  }
  Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
}

template <typename T>
void BothEndsReduceL2(
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* /* context */) {
  EigenVectorMap<T> Y_vec(Y, N);
  Y_vec = ConstEigenMatrixMap<T>(X, K, N).colwise().squaredNorm();
  for (int i = 1; i < M; ++i) {
    Y_vec +=
        ConstEigenMatrixMap<T>(X + i * N * K, K, N).colwise().squaredNorm();
  }
  Y_vec = Y_vec.cwiseSqrt() * alpha;
}

template <typename T, class Reducer>
void ReduceTensorImpl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const Reducer& reducer,
    const T init,
    const T* X,
    T* Y,
    CPUContext* context) {
  const int X_size =
      std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>());
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  Set<T, CPUContext>(Y_size, init, Y, context);
  std::vector<int> index(ndim, 0);
  for (int X_index = 0; X_index < X_size; ++X_index) {
    const int Y_index = utils::GetIndexFromDims(ndim, Y_dims, index.data());
    Y[Y_index] = reducer(Y[Y_index], X[X_index]);
    utils::IncreaseIndexInDims(ndim, X_dims, index.data());
  }
}

template <typename T>
void ReduceMinImpl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ReduceTensorImpl(
      ndim,
      X_dims,
      Y_dims,
      [](const T a, const T b) { return std::min(a, b); },
      std::numeric_limits<T>::max(),
      X,
      Y,
      context);
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
}

template <typename T>
void ReduceMaxImpl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ReduceTensorImpl(
      ndim,
      X_dims,
      Y_dims,
      [](const T a, const T b) { return std::max(a, b); },
      std::numeric_limits<T>::lowest(),
      X,
      Y,
      context);
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
}

template <typename T>
void ReduceSumImpl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ReduceTensorImpl(ndim, X_dims, Y_dims, std::plus<T>(), T(0), X, Y, context);
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
}

template <typename T>
void ReduceMeanImpl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ReduceTensorImpl(ndim, X_dims, Y_dims, std::plus<T>(), T(0), X, Y, context);
  const int X_size =
      std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>());
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  Scale<T, T, CPUContext>(
      Y_size,
      alpha * static_cast<T>(Y_size) / static_cast<T>(X_size),
      Y,
      Y,
      context);
}

template <typename T>
void ReduceL1Impl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ReduceTensorImpl(
      ndim,
      X_dims,
      Y_dims,
      [](const T a, const T b) { return a + std::abs(b); },
      T(0),
      X,
      Y,
      context);
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
}

template <typename T>
void ReduceL2Impl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  ReduceTensorImpl(
      ndim,
      X_dims,
      Y_dims,
      [](const T a, const T b) { return a + b * b; },
      T(0),
      X,
      Y,
      context);
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  EigenVectorArrayMap<T> Y_arr(Y, Y_size);
  Y_arr = Y_arr.sqrt() * alpha;
}

template <typename T>
void RowwiseMoments(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* var) {
  ConstEigenArrayMap<T> X_arr(X, cols, rows);
  EigenVectorArrayMap<T> mean_arr(mean, rows);
  EigenVectorArrayMap<T> var_arr(var, rows);
  mean_arr = X_arr.colwise().mean();
  var_arr = X_arr.square().colwise().mean() - mean_arr.square().transpose();
}

template <typename T>
void ColwiseMoments(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* var) {
  ConstEigenArrayMap<T> X_arr(X, cols, rows);
  EigenVectorArrayMap<T> mean_arr(mean, cols);
  EigenVectorArrayMap<T> var_arr(var, cols);
  // Eigen rowwise reduction is about 10 times slower than this for-loop.
  mean_arr = X_arr.col(0);
  var_arr = X_arr.col(0).square();
  for (int i = 1; i < rows; ++i) {
    mean_arr += X_arr.col(i);
    var_arr += X_arr.col(i).square();
  }
  const T scale = T(1) / static_cast<T>(rows);
  mean_arr *= scale;
  var_arr = var_arr * scale - mean_arr.square();
}

template <typename T>
void BothEndsMoments(
    const int M,
    const int N,
    const int K,
    const T* X,
    T* mean,
    T* var) {
  EigenVectorArrayMap<T> mean_arr(mean, N);
  EigenVectorArrayMap<T> var_arr(var, N);
  ConstEigenArrayMap<T> X0_arr(X, K, N);
  mean_arr = X0_arr.colwise().sum();
  var_arr = X0_arr.square().colwise().sum();
  for (int i = 1; i < M; ++i) {
    ConstEigenArrayMap<T> X_arr(X + i * N * K, K, N);
    mean_arr += X_arr.colwise().sum();
    var_arr += X_arr.square().colwise().sum();
  }
  const T scale = T(1) / static_cast<T>(M * K);
  mean_arr *= scale;
  var_arr = var_arr * scale - mean_arr.square();
}

template <typename T>
void MomentsImpl(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T* X,
    T* mean,
    T* var,
    CPUContext* /* context */) {
  const int X_size =
      std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>());
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  if (X_size == 0) {
    std::memset(mean, 0, sizeof(T) * Y_size);
    std::memset(var, 0, sizeof(T) * Y_size);
    return;
  }
  if (std::equal(X_dims, X_dims + ndim, Y_dims)) {
    std::memcpy(mean, X, sizeof(T) * Y_size);
    std::memset(var, 0, sizeof(T) * Y_size);
    return;
  }
  int rows;
  int cols;
  if (utils::IsRowwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
    RowwiseMoments<T>(rows, cols, X, mean, var);
    return;
  }
  if (utils::IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
    ColwiseMoments<T>(rows, cols, X, mean, var);
    return;
  }
  int pre;
  int mid;
  int nxt;
  if (utils::IsBothEndsReduce(ndim, X_dims, Y_dims, &pre, &mid, &nxt)) {
    BothEndsMoments<T>(pre, mid, nxt, X, mean, var);
    return;
  }
  std::memset(mean, 0, sizeof(T) * Y_size);
  std::memset(var, 0, sizeof(T) * Y_size);
  std::vector<int> index(ndim, 0);
  for (int X_index = 0; X_index < X_size; ++X_index) {
    const int Y_index = utils::GetIndexFromDims(ndim, Y_dims, index.data());
    mean[Y_index] += X[X_index];
    var[Y_index] += X[X_index] * X[X_index];
    utils::IncreaseIndexInDims(ndim, X_dims, index.data());
  }
  const T scale = static_cast<T>(Y_size) / static_cast<T>(X_size);
  EigenVectorArrayMap<T> mean_arr(mean, Y_size);
  EigenVectorArrayMap<T> var_arr(var, Y_size);
  mean_arr *= scale;
  var_arr = var_arr * scale - mean_arr.square();
}

} // namespace

#define DELEGATE_GLOBAL_REDUCE_FUNCTION(T, Func, EigenFunc) \
  template <>                                               \
  C10_EXPORT void Func<T, CPUContext>(                      \
      const int N,                                          \
      const T* X,                                           \
      T* Y,                                                 \
      Tensor* /* scratch_ptr */,                            \
      CPUContext* /* context */) {                          \
    *Y = ConstEigenVectorArrayMap<T>(X, N).EigenFunc();     \
  }
DELEGATE_GLOBAL_REDUCE_FUNCTION(float, ReduceMin, minCoeff)
DELEGATE_GLOBAL_REDUCE_FUNCTION(std::int32_t, ReduceMin, minCoeff)
DELEGATE_GLOBAL_REDUCE_FUNCTION(std::int64_t, ReduceMin, minCoeff)
DELEGATE_GLOBAL_REDUCE_FUNCTION(float, ReduceMax, maxCoeff)
DELEGATE_GLOBAL_REDUCE_FUNCTION(std::int32_t, ReduceMax, maxCoeff)
DELEGATE_GLOBAL_REDUCE_FUNCTION(std::int64_t, ReduceMax, maxCoeff)
#undef DELEGATE_GLOBAL_REDUCE_FUNCTION

#define DELEGATE_REDUCE_FUNCTION(T, Func, kInit, kIsNorm)                  \
  template <>                                                              \
  C10_EXPORT void Func<T, CPUContext>(                                     \
      const int ndim,                                                      \
      const int* X_dims,                                                   \
      const int* Y_dims,                                                   \
      const T alpha,                                                       \
      const T* X,                                                          \
      T* Y,                                                                \
      CPUContext* context) {                                               \
    const int X_size =                                                     \
        std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>()); \
    const int Y_size =                                                     \
        std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>()); \
    if (X_size == 0) {                                                     \
      Set<T, CPUContext>(Y_size, alpha * kInit, Y, context);               \
      return;                                                              \
    }                                                                      \
    if (alpha == T(0)) {                                                   \
      std::memset(Y, 0, sizeof(T) * Y_size);                               \
      return;                                                              \
    }                                                                      \
    if (std::equal(X_dims, X_dims + ndim, Y_dims)) {                       \
      if (kIsNorm) {                                                       \
        EigenVectorArrayMap<T>(Y, Y_size) =                                \
            ConstEigenVectorArrayMap<T>(X, X_size).abs() * alpha;          \
      } else {                                                             \
        Scale<T, T, CPUContext>(Y_size, alpha, X, Y, context);             \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    int rows;                                                              \
    int cols;                                                              \
    if (utils::IsRowwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {      \
      Rowwise##Func<T>(rows, cols, alpha, X, Y, context);                  \
      return;                                                              \
    }                                                                      \
    if (utils::IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {      \
      Colwise##Func<T>(rows, cols, alpha, X, Y, context);                  \
      return;                                                              \
    }                                                                      \
    int M;                                                                 \
    int N;                                                                 \
    int K;                                                                 \
    if (utils::IsBothEndsReduce(ndim, X_dims, Y_dims, &M, &N, &K)) {       \
      BothEnds##Func<T>(M, N, K, alpha, X, Y, context);                    \
      return;                                                              \
    }                                                                      \
    Func##Impl<T>(ndim, X_dims, Y_dims, alpha, X, Y, context);             \
  }
DELEGATE_REDUCE_FUNCTION(
    float,
    ReduceMin,
    std::numeric_limits<float>::max(),
    false)
DELEGATE_REDUCE_FUNCTION(
    double,
    ReduceMin,
    std::numeric_limits<double>::max(),
    false)
DELEGATE_REDUCE_FUNCTION(
    std::int32_t,
    ReduceMin,
    std::numeric_limits<std::int32_t>::max(),
    false)
DELEGATE_REDUCE_FUNCTION(
    std::int64_t,
    ReduceMin,
    std::numeric_limits<std::int64_t>::max(),
    false)
DELEGATE_REDUCE_FUNCTION(
    float,
    ReduceMax,
    std::numeric_limits<float>::lowest(),
    false)
DELEGATE_REDUCE_FUNCTION(
    double,
    ReduceMax,
    std::numeric_limits<double>::lowest(),
    false)
DELEGATE_REDUCE_FUNCTION(
    std::int32_t,
    ReduceMax,
    std::numeric_limits<std::int32_t>::lowest(),
    false)
DELEGATE_REDUCE_FUNCTION(
    std::int64_t,
    ReduceMax,
    std::numeric_limits<std::int64_t>::lowest(),
    false)
DELEGATE_REDUCE_FUNCTION(float, ReduceSum, 0.0f, false)
DELEGATE_REDUCE_FUNCTION(double, ReduceSum, 0.0, false)
DELEGATE_REDUCE_FUNCTION(std::int32_t, ReduceSum, 0, false)
DELEGATE_REDUCE_FUNCTION(std::int64_t, ReduceSum, 0LL, false)
DELEGATE_REDUCE_FUNCTION(float, ReduceMean, 0.0f, false)
DELEGATE_REDUCE_FUNCTION(double, ReduceMean, 0.0, false)
DELEGATE_REDUCE_FUNCTION(float, ReduceL1, 0.0f, true)
DELEGATE_REDUCE_FUNCTION(double, ReduceL1, 0.0, true)
DELEGATE_REDUCE_FUNCTION(std::int32_t, ReduceL1, 0, true)
DELEGATE_REDUCE_FUNCTION(std::int64_t, ReduceL1, 0LL, true)
DELEGATE_REDUCE_FUNCTION(float, ReduceL2, 0.0f, true)
DELEGATE_REDUCE_FUNCTION(double, ReduceL2, 0.0, true)
#undef DELEGATE_REDUCE_FUNCTION

#define CAFFE2_SPECIALIZED_MOMENTS(T)                            \
  template <>                                                    \
  C10_EXPORT void Moments<T, CPUContext>(                        \
      const int ndim,                                            \
      const int* X_dims,                                         \
      const int* Y_dims,                                         \
      const T* X,                                                \
      T* mean,                                                   \
      T* var,                                                    \
      CPUContext* context) {                                     \
    MomentsImpl<T>(ndim, X_dims, Y_dims, X, mean, var, context); \
  }
CAFFE2_SPECIALIZED_MOMENTS(float)
#undef CAFFE2_SPECIALIZED_MOMENTS

} // namespace math
} // namespace caffe2
