#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OPS_UTILS_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OPS_UTILS_H_

#include <tuple>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace elementwise_ops_utils {

template <typename Context>
std::tuple<size_t, size_t, size_t> ComputeLegacyBroadcastSizes(
    const Tensor<Context>& A,
    const Tensor<Context>& B,
    int axis) {
  CAFFE_ENFORCE_GE(
      A.ndim(),
      B.ndim(),
      "If you are doing broadcasting, input1 should have "
      "a smaller or equal number of dimensions.");
  if (axis == -1) {
    axis = A.ndim() - B.ndim();
  }
  CAFFE_ENFORCE(
      axis >= 0 && axis <= A.ndim() - B.ndim(),
      "Broadcast axis should be in the range of"
      "[0, A.ndim() - B.ndim()], but axis = ",
      axis);

  int b_dim_start = 0;
  while (b_dim_start < B.ndim() && B.dim(b_dim_start) == 1) {
    ++b_dim_start;
  }
  int b_dim_end = B.ndim() - 1;
  while (b_dim_end >= b_dim_start && B.dim(b_dim_end) == 1) {
    --b_dim_end;
  }
  size_t pre = 1, n = 1, post = 1;
  for (int i = 0; i < axis + b_dim_start; ++i) {
    pre *= A.dim(i);
  }
  for (int i = b_dim_start; i <= b_dim_end; ++i) {
    CAFFE_ENFORCE_EQ(
        A.dim(i + axis), B.dim(i), "Broadcast dimension mismatch.");
    n *= B.dim(i);
  }
  for (int i = axis + b_dim_end + 1; i < A.ndim(); ++i) {
    post *= A.dim(i);
  }
  return std::make_tuple(pre, n, post);
}

std::vector<int> ComputeBinaryBroadcastForwardDims(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims);

void ComputeBinaryBroadcastBackwardAxes(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    std::vector<int>* A_axes,
    std::vector<int>* B_axes);

} // namespace elementwise_ops_utils
} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_OPS_UTILS_H_
