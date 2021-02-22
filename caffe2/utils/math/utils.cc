#include <c10/util/accumulate.h>
#include "caffe2/core/logging.h"
#include "caffe2/utils/math/utils.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace caffe2 {
namespace math {
namespace utils {

#define CAFFE2_SPECIALIZED_INCREASE_INDEX_IN_DIMS(TIndex)  \
  template <>                                              \
  C10_EXPORT void IncreaseIndexInDims<TIndex>(             \
      const int ndim, const TIndex* dims, TIndex* index) { \
    for (int i = ndim - 1; i >= 0; --i) {                  \
      ++index[i];                                          \
      if (index[i] >= dims[i]) {                           \
        index[i] -= dims[i];                               \
      } else {                                             \
        break;                                             \
      }                                                    \
    }                                                      \
  }
CAFFE2_SPECIALIZED_INCREASE_INDEX_IN_DIMS(std::int32_t)
CAFFE2_SPECIALIZED_INCREASE_INDEX_IN_DIMS(std::int64_t)
#undef CAFFE2_SPECIALIZED_INCREASE_INDEX_IN_DIMS

#define CAFFE2_SPECIALIZED_GET_INDEX_FROM_DIMS(TIndex)        \
  template <>                                                 \
  C10_EXPORT TIndex GetIndexFromDims(                         \
      const int n, const TIndex* dims, const TIndex* index) { \
    TIndex sum = 0;                                           \
    for (int i = 0; i < n; ++i) {                             \
      if (dims[i] > 1) {                                      \
        sum = sum * dims[i] + index[i];                       \
      }                                                       \
    }                                                         \
    return sum;                                               \
  }
CAFFE2_SPECIALIZED_GET_INDEX_FROM_DIMS(std::int32_t)
CAFFE2_SPECIALIZED_GET_INDEX_FROM_DIMS(std::int64_t)
#undef CAFFE2_SPECIALIZED_GET_INDEX_FROM_DIMS

bool IsIdentityPermutation(const int n, const int* perm) {
  for (int i = 0; i < n; ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}

bool CheckReduceDims(const int ndim, const int* X_dims, const int* Y_dims) {
  for (int i = 0; i < ndim; ++i) {
    if (X_dims[i] != Y_dims[i] && Y_dims[i] != 1) {
      return false;
    }
  }
  return true;
}

bool IsRowwiseReduce(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols) {
  *cols = 1;
  int pivot = ndim - 1;
  for (; pivot >= 0 && B_dims[pivot] == 1; --pivot) {
    *cols *= A_dims[pivot];
  }
  *rows = 1;
  for (int i = pivot; i >= 0; --i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *rows *= A_dims[i];
  }
  return true;
}

bool IsColwiseReduce(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols) {
  *rows = 1;
  int pivot = 0;
  for (; pivot < ndim && B_dims[pivot] == 1; ++pivot) {
    *rows *= A_dims[pivot];
  }
  *cols = 1;
  for (int i = pivot; i < ndim; ++i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *cols *= A_dims[i];
  }
  return true;
}

bool IsBothEndsReduce(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* pre,
    int* mid,
    int* nxt) {
  *nxt = 1;
  int r = ndim - 1;
  for (; r >= 0 && B_dims[r] == 1; --r) {
    *nxt *= A_dims[r];
  }
  *pre = 1;
  int l = 0;
  for (; l <= r && B_dims[l] == 1; ++l) {
    *pre *= A_dims[l];
  }
  *mid = 1;
  for (int i = l; i <= r; ++i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *mid *= A_dims[i];
  }
  return true;
}

#define CAFFE2_SPECIALIZED_COMPUTE_BROADCAST_BINARY_OP_DIMS(TIndex)       \
  template <>                                                             \
  C10_EXPORT void ComputeBroadcastBinaryOpDims(                           \
      const int A_ndim,                                                   \
      const TIndex* A_dims,                                               \
      const int B_ndim,                                                   \
      const TIndex* B_dims,                                               \
      TIndex* A_broadcast_dims,                                           \
      TIndex* B_broadcast_dims,                                           \
      TIndex* C_broadcast_dims) {                                         \
    const int ndim = std::max(A_ndim, B_ndim);                            \
    std::fill(A_broadcast_dims, A_broadcast_dims + ndim - A_ndim, 1);     \
    std::fill(B_broadcast_dims, B_broadcast_dims + ndim - B_ndim, 1);     \
    std::copy(A_dims, A_dims + A_ndim, A_broadcast_dims + ndim - A_ndim); \
    std::copy(B_dims, B_dims + B_ndim, B_broadcast_dims + ndim - B_ndim); \
    for (int i = 0; i < ndim; ++i) {                                      \
      CAFFE_ENFORCE(                                                      \
          A_broadcast_dims[i] == B_broadcast_dims[i] ||                   \
          A_broadcast_dims[i] <= 1 || B_broadcast_dims[i] <= 1);          \
      if (A_broadcast_dims[i] == 0 || B_broadcast_dims[i] == 0) {         \
        C_broadcast_dims[i] = 0;                                          \
      } else {                                                            \
        C_broadcast_dims[i] =                                             \
            std::max(A_broadcast_dims[i], B_broadcast_dims[i]);           \
      }                                                                   \
    }                                                                     \
  }
CAFFE2_SPECIALIZED_COMPUTE_BROADCAST_BINARY_OP_DIMS(std::int32_t)
CAFFE2_SPECIALIZED_COMPUTE_BROADCAST_BINARY_OP_DIMS(std::int64_t)
#undef CAFFE2_SPECIALIZED_COMPUTE_BROADCAST_BINARY_OP_DIMS

bool IsRowwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols,
    bool* broadcast_1st) {
  if (ndim == 0) {
    return false;
  }
  int A_pivot = 0;
  for (; A_pivot < ndim && A_dims[A_pivot] == 1; ++A_pivot)
    ;
  int B_pivot = 0;
  for (; B_pivot < ndim && B_dims[B_pivot] == 1; ++B_pivot)
    ;
  if (A_pivot == B_pivot) {
    return false;
  }
  const int pivot = std::max(A_pivot, B_pivot);
  if (A_pivot > B_pivot) {
    *rows = c10::multiply_integers(B_dims + B_pivot, B_dims + pivot);
    *broadcast_1st = true;
  } else {
    *rows = c10::multiply_integers(A_dims + A_pivot, A_dims + pivot);
    *broadcast_1st = false;
  }
  *cols = 1;
  for (int i = pivot; i < ndim; ++i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *cols *= A_dims[i];
  }
  return true;
}

bool IsColwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols,
    bool* broadcast_1st) {
  if (ndim == 0) {
    return false;
  }
  int A_pivot = ndim - 1;
  for (; A_pivot >= 0 && A_dims[A_pivot] == 1; --A_pivot)
    ;
  int B_pivot = ndim - 1;
  for (; B_pivot >= 0 && B_dims[B_pivot] == 1; --B_pivot)
    ;
  if (A_pivot == B_pivot) {
    return false;
  }
  ++A_pivot;
  ++B_pivot;
  const int pivot = std::min(A_pivot, B_pivot);
  if (A_pivot < B_pivot) {
    *cols = c10::multiply_integers(B_dims + pivot, B_dims + B_pivot);
    *broadcast_1st = true;
  } else {
    *cols = c10::multiply_integers(A_dims + pivot, A_dims + A_pivot);
    *broadcast_1st = false;
  }
  *rows = 1;
  for (int i = 0; i < pivot; ++i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *rows *= A_dims[i];
  }
  return true;
}

bool IsBothEndsBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* pre,
    int* mid,
    int* nxt,
    bool* broadcast_1st) {
  if (ndim == 0) {
    return false;
  }
  int A_pre = 0;
  for (; A_pre < ndim && A_dims[A_pre] == 1; ++A_pre)
    ;
  int B_pre = 0;
  for (; B_pre < ndim && B_dims[B_pre] == 1; ++B_pre)
    ;
  int A_nxt = ndim - 1;
  for (; A_nxt >= 0 && A_dims[A_nxt] == 1; --A_nxt)
    ;
  int B_nxt = ndim - 1;
  for (; B_nxt >= 0 && B_dims[B_nxt] == 1; --B_nxt)
    ;
  ++A_nxt;
  ++B_nxt;
  if (A_pre == B_pre || A_nxt == B_nxt) {
    return false;
  }
  if (A_pre > B_pre && A_nxt < B_nxt) {
    *pre = c10::multiply_integers(B_dims + B_pre, B_dims + A_pre);
    *nxt = c10::multiply_integers(B_dims + A_nxt, B_dims + B_nxt);
    *broadcast_1st = true;
  } else if (A_pre < B_pre && A_nxt > B_nxt) {
    *pre = c10::multiply_integers(A_dims + A_pre, A_dims + B_pre);
    *nxt = c10::multiply_integers(A_dims + B_nxt, A_dims + A_nxt);
    *broadcast_1st = false;
  } else {
    return false;
  }
  const int l = std::max(A_pre, B_pre);
  const int r = std::min(A_nxt, B_nxt);
  *mid = 1;
  for (int i = l; i < r; ++i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *mid *= A_dims[i];
  }
  return true;
}

bool IsBatchTranspose2D(const int ndim, const int* axes) {
  if (ndim < 2) {
    return false;
  }
  for (int i = 0; i < ndim - 2; ++i) {
    if (axes[i] != i) {
      return false;
    }
  }
  return axes[ndim - 2] == ndim - 1 && axes[ndim - 1] == ndim - 2;
}

void ComputeTransposeAxesForReduceOp(
    const int num_dims,
    const int num_reduce_axes,
    const int* reduce_axes,
    int* transpose_axes) {
  const int d = num_dims - num_reduce_axes;
  std::copy_n(reduce_axes, num_reduce_axes, transpose_axes + d);
  std::sort(transpose_axes + d, transpose_axes + num_dims);
  int p = 0;
  int q = d;
  for (int i = 0; i < num_dims; ++i) {
    if (q < num_dims && i == transpose_axes[q]) {
      ++q;
    } else {
      transpose_axes[p++] = i;
    }
  }
}

void ComputeTransposeAxesForReduceOp(
    const int ndim,
    const int* dims,
    int* axes) {
  const int d = ndim - std::count(dims, dims + ndim, 1);
  int p = 0;
  int q = d;
  for (int i = 0; i < ndim; ++i) {
    if (dims[i] == 1) {
      axes[q++] = i;
    } else {
      axes[p++] = i;
    }
  }
}

#define CAFFE2_SPECIALIZED_COMPUTE_TRANSPOSED_STRIDES(TIndex)                 \
  template <>                                                                 \
  C10_EXPORT void ComputeTransposedStrides<TIndex>(                           \
      const int ndim, const TIndex* dims, const int* axes, TIndex* strides) { \
    std::vector<TIndex> buff(ndim);                                           \
    TIndex cur_stride = 1;                                                    \
    for (int i = ndim - 1; i >= 0; --i) {                                     \
      buff[i] = cur_stride;                                                   \
      cur_stride *= dims[i];                                                  \
    }                                                                         \
    for (int i = 0; i < ndim; ++i) {                                          \
      strides[i] = buff[axes[i]];                                             \
    }                                                                         \
  }
CAFFE2_SPECIALIZED_COMPUTE_TRANSPOSED_STRIDES(std::int32_t)
CAFFE2_SPECIALIZED_COMPUTE_TRANSPOSED_STRIDES(std::int64_t)
#undef CAFFE2_SPECIALIZED_COMPUTE_TRANSPOSED_STRIDES

} // namespace utils
} // namespace math
} // namespace caffe2
