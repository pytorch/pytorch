#include "ATen/native/LinearAlgebraUtils.h"

namespace at { namespace native {

Tensor cloneBatchedColumnMajor(const Tensor& src) {
  // If src is already in batched column major format, then
  // this will be efficient (no reordering of the data will occur)
  // because the first transpose will make the tensor contiguous,
  // and cloning a contiguous tensor is fast.
  auto result = src.transpose(-2, -1).clone();
  result.transpose_(-2, -1);
  return result;
}

}}  // namespace at::native
