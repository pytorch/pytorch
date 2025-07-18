#pragma once

#include <torch/csrc/stable/library.h>
#include <array>
#include <cstdint>

using torch::stable::Tensor;

// We expect this to be the stable version of the transpose op with identical
// semantics to the existing transpose.int op.
inline Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{from(self), from(dim0), from(dim1)};
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::transpose", "int", stack.data()));
  return to<Tensor>(stack[0]);
}
