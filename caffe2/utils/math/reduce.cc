#include "caffe2/utils/math/reduce.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

template <typename T>
C10_EXPORT void
RowwiseMoments(const int rows, const int cols, const T* X, T* mean, T* var) {
  ConstEigenArrayMap<T> X_arr(X, cols, rows);
  EigenVectorArrayMap<T> mean_arr(mean, rows);
  EigenVectorArrayMap<T> var_arr(var, rows);
  mean_arr = X_arr.colwise().mean();
  var_arr = X_arr.square().colwise().mean() - mean_arr.square().transpose();
}

template <typename T>
C10_EXPORT void
ColwiseMoments(const int rows, const int cols, const T* X, T* mean, T* var) {
  std::memset(mean, 0, sizeof(T) * cols);
  std::memset(var, 0, sizeof(T) * cols);
  ConstEigenArrayMap<T> X_arr(X, cols, rows);
  EigenVectorArrayMap<T> mean_arr(mean, cols);
  EigenVectorArrayMap<T> var_arr(var, cols);
  // Eigen rowwise reduction is about 10 times slower than this for-loop.
  for (int i = 0; i < rows; ++i) {
    mean_arr += X_arr.col(i);
    var_arr += X_arr.col(i).square();
  }
  const T scale = T(1) / static_cast<T>(rows);
  mean_arr *= scale;
  var_arr = var_arr * scale - mean_arr.square();
}

template <typename T>
C10_EXPORT void BothEndsMoments(
    const int pre,
    const int mid,
    const int nxt,
    const T* X,
    T* mean,
    T* var) {
  std::memset(mean, 0, sizeof(T) * mid);
  std::memset(var, 0, sizeof(T) * mid);
  EigenVectorArrayMap<T> mean_arr(mean, mid);
  EigenVectorArrayMap<T> var_arr(var, mid);
  ConstEigenArrayMap<T> X_arr(X, nxt, pre * mid);
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < mid; ++j) {
      const int c = i * mid + j;
      mean_arr(j) += X_arr.col(c).sum();
      var_arr(j) += X_arr.col(c).square().sum();
    }
  }
  const T scale = T(1) / static_cast<T>(pre * nxt);
  mean_arr *= scale;
  var_arr = var_arr * scale - mean_arr.square();
}

template <typename T>
C10_EXPORT void MomentsImpl(
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
