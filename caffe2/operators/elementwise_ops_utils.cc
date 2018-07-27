#include "caffe2/operators/elementwise_ops_utils.h"

namespace caffe2 {
namespace elementwise_ops_utils {

std::vector<int> ComputeBinaryBroadcastForwardDims(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims) {
  const int ndim = std::max(A_dims.size(), B_dims.size());
  std::vector<int> C_dims(ndim);
  int i = A_dims.size() - 1;
  int j = B_dims.size() - 1;
  int k = ndim - 1;
  for (; i >= 0 && j >= 0; --k) {
    const int A_dim = A_dims[i];
    const int B_dim = B_dims[j];
    CAFFE_ENFORCE(A_dim == B_dim || A_dim == 1 || B_dim == 1);
    if (A_dim == 0 || B_dim == 0) {
      C_dims[k] = 0;
    } else {
      C_dims[k] = std::max(A_dims[i], B_dims[j]);
    }
    --i;
    --j;
  }
  for (; i >= 0; --i) {
    C_dims[k--] = A_dims[i];
  }
  for (; j >= 0; --j) {
    C_dims[k--] = B_dims[j];
  }
  return C_dims;
}

void ComputeBinaryBroadcastBackwardAxes(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    std::vector<int>* A_axes,
    std::vector<int>* B_axes) {
  A_axes->clear();
  B_axes->clear();
  const int ndim = std::max(A_dims.size(), B_dims.size());
  int i = A_dims.size() - 1;
  int j = B_dims.size() - 1;
  int k = ndim - 1;
  for (; i >= 0 && j >= 0; --k) {
    CAFFE_ENFORCE(A_dims[i] == B_dims[j] || A_dims[i] == 1 || B_dims[j] == 1);
    if (A_dims[i] != B_dims[j]) {
      if (A_dims[i] == 1) {
        A_axes->push_back(k);
      }
      if (B_dims[j] == 1) {
        B_axes->push_back(k);
      }
    }
    --i;
    --j;
  }
  if (i < 0) {
    for (; k >= 0; --k) {
      A_axes->push_back(k);
    }
  } else {
    for (; k >= 0; --k) {
      B_axes->push_back(k);
    }
  }
  std::reverse(A_axes->begin(), A_axes->end());
  std::reverse(B_axes->begin(), B_axes->end());
}

} // namespace elementwise_ops_utils
} // namespace caffe2
