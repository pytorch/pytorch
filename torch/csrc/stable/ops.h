#pragma once

#include <torch/csrc/stable/library.h>
#include <array>
#include <cstdint>
#include <optional>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>

using torch::stable::Tensor;

// We expect this to be the stable version of the empty_like op that takes in
// no kwargs (device, dtype, layout, memory_format). We will add kwargs
// support in the future.
inline Tensor empty_like(const Tensor& self) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt)};
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::empty_like", "", stack.data()));
  return to<Tensor>(stack[0]);
}

// We expect this to be the stable version of the fill_.Scalar op
// with identical semantics to the existing fill_.Scalar op.
// A subtle nuance is that `value` is typed as a double, but it is
// actually a Scalar. This is because Scalar.h is currently not
// header-only.
inline Tensor fill_(const Tensor& self, double value) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_aten_fill__Scalar(self.get(), value));
  return self;
}

// We expect this to be the stable version of the transpose op with identical
// semantics to the existing transpose.int op.
inline Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{from(self), from(dim0), from(dim1)};
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::transpose", "int", stack.data()));
  return to<Tensor>(stack[0]);
}

// We expect this to be the stable version of the zero_ op with identical
// semantics to the existing zero_ op (except that it will not be called as
// a tensor method but only as a function i.e. zero_(t) not t.zero_()).
inline Tensor zero_(Tensor& self) {
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{from(self)};
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::zero_", "", stack.data()));
  return to<Tensor>(stack[0]);
}
