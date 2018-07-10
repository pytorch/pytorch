#include "caffe2/utils/math_utils.h"

#include <algorithm>
#include <vector>

#include "caffe2/core/logging.h"

namespace caffe2 {
namespace math {
namespace utils {

void IncreaseIndexInDims(const int n, const int* dims, int* index) {
  for (int i = n - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

int GetIndexFromDims(const int n, const int* dims, const int* index) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    if (dims[i] > 1) {
      sum = sum * dims[i] + index[i];
    }
  }
  return sum;
}

bool IsIdentityPermutation(const int n, const int* perm) {
  for (int i = 0; i < n; ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}

void ComputeBroadcastBinaryOpDims(
    const int A_ndim,
    const int* A_dims,
    const int B_ndim,
    const int* B_dims,
    int* A_broadcast_dims,
    int* B_broadcast_dims,
    int* C_broadcast_dims) {
  const int ndim = std::max(A_ndim, B_ndim);
  std::fill(A_broadcast_dims, A_broadcast_dims + ndim - A_ndim, 1);
  std::fill(B_broadcast_dims, B_broadcast_dims + ndim - B_ndim, 1);
  std::copy(A_dims, A_dims + A_ndim, A_broadcast_dims + ndim - A_ndim);
  std::copy(B_dims, B_dims + B_ndim, B_broadcast_dims + ndim - B_ndim);
  for (int i = 0; i < ndim; ++i) {
    CAFFE_ENFORCE(
        A_broadcast_dims[i] == B_broadcast_dims[i] ||
        A_broadcast_dims[i] <= 1 || B_broadcast_dims[i] <= 1);
    if (A_broadcast_dims[i] == 0 || B_broadcast_dims[i] == 0) {
      C_broadcast_dims[i] = 0;
    } else {
      C_broadcast_dims[i] = std::max(A_broadcast_dims[i], B_broadcast_dims[i]);
    }
  }
}

bool IsRowwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* pivot,
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
  *pivot = std::max(A_pivot, B_pivot);
  *broadcast_1st = A_pivot > B_pivot;
  return std::equal(A_dims + *pivot, A_dims + ndim, B_dims + *pivot);
}

bool IsColwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* pivot,
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
  *pivot = std::min(A_pivot, B_pivot) + 1;
  *broadcast_1st = A_pivot < B_pivot;
  return std::equal(A_dims, A_dims + *pivot, B_dims);
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

void ComputeTransposedStrides(
    const int ndim,
    const int* dims,
    const int* axes,
    int* strides) {
  std::vector<int> buff(ndim);
  int cur_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buff[i] = cur_stride;
    cur_stride *= dims[i];
  }
  for (int i = 0; i < ndim; ++i) {
    strides[i] = buff[axes[i]];
  }
}

} // namespace utils
} // namespace math
} // namespace caffe2
