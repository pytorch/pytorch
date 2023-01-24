#include "caffe2/utils/math/transpose.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#ifdef CAFFE2_USE_HPTT
#include <hptt.h>
#endif // CAFFE2_USE_HPTT

#include "caffe2/core/context.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

template <typename TIndex, typename TData>
void Transpose2D(
    const TIndex rows,
    const TIndex cols,
    const TData* X,
    TData* Y) {
  EigenMatrixMap<TData>(Y, rows, cols) =
      ConstEigenMatrixMap<TData>(X, cols, rows).transpose();
}

#ifdef CAFFE2_USE_MKL

#define DELEGATE_TRANSPOSE_2D(TIndex, TData, MKLFunc)                   \
  template <>                                                           \
  void Transpose2D<TIndex, TData>(                                      \
      const TIndex rows, const TIndex cols, const TData* X, TData* Y) { \
    MKLFunc('R', 'T', rows, cols, TData(1), X, cols, Y, rows);          \
  }
DELEGATE_TRANSPOSE_2D(std::int32_t, float, mkl_somatcopy);
DELEGATE_TRANSPOSE_2D(std::int64_t, float, mkl_somatcopy);
DELEGATE_TRANSPOSE_2D(std::int32_t, double, mkl_domatcopy);
DELEGATE_TRANSPOSE_2D(std::int64_t, double, mkl_domatcopy);
#undef DELEGATE_TRANSPOSE_2D

#endif // CAFFE2_USE_MKL

#ifdef CAFFE2_USE_HPTT

template <typename TIndex, typename TData>
bool TransposeByHPTT(
    const int ndim,
    const TIndex* dims,
    const int* axes,
    const TData* X,
    TData* Y) {
  for (int i = 0; i < ndim; ++i) {
    if (dims[i] <= 0 || dims[i] > std::numeric_limits<int>::max()) {
      return false;
    }
  }

  std::vector<int> axes_cm(ndim);
  std::vector<int> dims_cm(ndim);
  // Convert row-major index to column-major.
  const auto cm_fn = [ndim](const int i) { return ndim - i - 1; };
  for (int i = 0; i < ndim; ++i) {
    axes_cm[i] = cm_fn(axes[cm_fn(i)]);
    dims_cm[i] = dims[cm_fn(i)];
  }
  auto plan = hptt::create_plan(
      axes_cm.data(),
      ndim,
      TData(1),
      X,
      dims_cm.data(),
      nullptr,
      TData(0),
      Y,
      nullptr,
      hptt::ESTIMATE,
      1 /* num_threads */);
  if (plan == nullptr) {
    return false;
  }
  plan->execute();
  return true;
}

#endif // CAFFE2_USE_HPTT

template <typename TIndex, typename TData>
void TransposeND(
    const int ndim,
    const TIndex* dims,
    const int* axes,
    const TData* X,
    TData* Y) {
  std::vector<TIndex> Y_dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    Y_dims[i] = dims[axes[i]];
  }
  // Measure amount of contiguous data we can copy at once
  int pivot = ndim - 1;
  TIndex block_size = 1;
  for (; pivot >= 0 && axes[pivot] == pivot; --pivot) {
    block_size *= Y_dims[pivot];
  }
  ++pivot;
  const TIndex num_blocks = std::accumulate(
      Y_dims.cbegin(),
      Y_dims.cbegin() + pivot,
      TIndex(1),
      std::multiplies<TIndex>());
  std::vector<TIndex> X_strides(pivot);
  utils::ComputeTransposedStrides<TIndex>(pivot, dims, axes, X_strides.data());
  std::vector<TIndex> index(pivot, 0);
  for (TIndex Y_index = 0; Y_index < num_blocks; ++Y_index) {
    const TIndex X_index = std::inner_product(
        X_strides.cbegin(), X_strides.cend(), index.cbegin(), TIndex(0));
    if (block_size == 1) {
      Y[Y_index] = X[X_index];
    } else {
      std::memcpy(
          Y + block_size * Y_index,
          X + block_size * X_index,
          block_size * sizeof(TData));
    }
    utils::IncreaseIndexInDims<TIndex>(pivot, Y_dims.data(), index.data());
  }
}

template <typename TIndex, typename TData>
void TransposeImpl(
    const int ndim,
    const TIndex* dims,
    const int* axes,
    const TData* X,
    TData* Y) {
  const TIndex size =
      std::accumulate(dims, dims + ndim, TIndex(1), std::multiplies<TIndex>());
  if (size == 0) {
    return;
  }
  if (utils::IsIdentityPermutation(ndim, axes)) {
    std::memcpy(Y, X, size * sizeof(TData));
    return;
  }
  if (utils::IsBatchTranspose2D(ndim, axes)) {
    const TIndex H = dims[ndim - 2];
    const TIndex W = dims[ndim - 1];
    const TIndex N = size / (H * W);
    for (TIndex i = 0; i < N; ++i) {
      Transpose2D<TIndex, TData>(H, W, X + i * H * W, Y + i * H * W);
    }
    return;
  }
  TransposeND<TIndex, TData>(ndim, dims, axes, X, Y);
}

#ifdef CAFFE2_USE_HPTT

#define CAFFE2_SPECIALIZED_TRANSPOSE_IMPL(TIndex, TData)                \
  template <>                                                           \
  void TransposeImpl<TIndex, TData>(                                    \
      const int ndim,                                                   \
      const TIndex* dims,                                               \
      const int* axes,                                                  \
      const TData* X,                                                   \
      TData* T) {                                                       \
    const TIndex size = std::accumulate(                                \
        dims, dims + ndim, TIndex(1), std::multiplies<TIndex>());       \
    if (size == 0) {                                                    \
      return;                                                           \
    }                                                                   \
    if (utils::IsIdentityPermutation(ndim, axes)) {                     \
      std::memcpy(Y, X, size * sizeof(TData));                          \
      return;                                                           \
    }                                                                   \
    if (TransposeByHPTT(ndim, dims, axes, X, Y)) {                      \
      return;                                                           \
    }                                                                   \
    if (utils::IsBatchTranspose2D(ndim, axes)) {                        \
      const TIndex H = dims[ndim - 2];                                  \
      const TIndex W = dims[ndim - 1];                                  \
      const TIndex N = size / (H * W);                                  \
      for (TIndex i = 0; i < N; ++i) {                                  \
        Transpose2D<TIndex, TData>(H, W, X + i * H * W, Y + i * H * W); \
      }                                                                 \
      return;                                                           \
    }                                                                   \
    TransposeND<TIndex, TData>(ndim, dims, axes, X, Y);                 \
  }
CAFFE2_SPECIALIZED_TRANSPOSE_IMPL(std::int32_t, float)
CAFFE2_SPECIALIZED_TRANSPOSE_IMPL(std::int64_t, float)
CAFFE2_SPECIALIZED_TRANSPOSE_IMPL(std::int32_t, double)
CAFFE2_SPECIALIZED_TRANSPOSE_IMPL(std::int64_t, double)
#undef CAFFE2_SPECIALIZED_TRANSPOSE_IMPL

#endif // CAFFE2_USE_HPTT

} // namespace

#define CAFFE2_SPECIALIZED_TRANSPOSE(TIndex, TData)       \
  template <>                                             \
  C10_EXPORT void Transpose<TIndex, TData, CPUContext>(   \
      const int ndim,                                     \
      const TIndex* dims,                                 \
      const int* axes,                                    \
      const TData* X,                                     \
      TData* Y,                                           \
      CPUContext* /* context */) {                        \
    TransposeImpl<TIndex, TData>(ndim, dims, axes, X, Y); \
  }
CAFFE2_SPECIALIZED_TRANSPOSE(std::int32_t, float)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int64_t, float)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int32_t, double)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int64_t, double)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int32_t, std::int32_t)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int64_t, std::int32_t)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int32_t, std::int64_t)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int64_t, std::int64_t)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int32_t, std::uint8_t)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int64_t, std::uint8_t)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int32_t, std::uint16_t)
CAFFE2_SPECIALIZED_TRANSPOSE(std::int64_t, std::uint16_t)
#undef CAFFE2_SPECIALIZED_TRANSPOSE

#define CAFFE2_SPECIALIZED_NCHW2NHWC(T)                    \
  template <>                                              \
  C10_EXPORT void NCHW2NHWC<T, CPUContext>(                \
      const int N,                                         \
      const int C,                                         \
      const int HxW,                                       \
      const T* X,                                          \
      T* Y,                                                \
      CPUContext* /* context */) {                         \
    const int stride = C * HxW;                            \
    for (int i = 0; i < N; ++i) {                          \
      Transpose2D(C, HxW, X + i * stride, Y + i * stride); \
    }                                                      \
  }
CAFFE2_SPECIALIZED_NCHW2NHWC(float)
#undef CAFFE2_SPECIALIZED_NCHW2NHWC

#define CAFFE2_SPECIALIZED_NHWC2NCHW(T)                    \
  template <>                                              \
  C10_EXPORT void NHWC2NCHW<T, CPUContext>(                \
      const int N,                                         \
      const int C,                                         \
      const int HxW,                                       \
      const T* X,                                          \
      T* Y,                                                \
      CPUContext* /* context */) {                         \
    const int stride = HxW * C;                            \
    for (int i = 0; i < N; ++i) {                          \
      Transpose2D(HxW, C, X + i * stride, Y + i * stride); \
    }                                                      \
  }
CAFFE2_SPECIALIZED_NHWC2NCHW(float)
#undef CAFFE2_SPECIALIZED_NHWC2NCHW

} // namespace math
} // namespace caffe2
