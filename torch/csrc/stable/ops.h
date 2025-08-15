#pragma once

#include <torch/csrc/stable/library.h>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>

using torch::stable::Tensor;

namespace torch::stable {

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
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::empty_like", "", stack.data()));
  return to<Tensor>(stack[0]);
}

// We expect this to be the stable version of the fill_.Scalar op
// with identical semantics to the existing fill_.Scalar op.
// A subtle nuance is that `value` is typed as a double, but it is
// actually a Scalar. This is because Scalar.h is currently not
// header-only.
inline Tensor fill_(const Tensor& self, double value) {
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_fill__Scalar(self.get(), value));
  return self;
}

// We expect this to be the stable version of the narrow.default op.
// narrow takes in a SymInt for start and length, but these are typed as
// int64_t as SymInt is not yet header-only.
inline Tensor narrow(Tensor& self, int64_t dim, int64_t start, int64_t length) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_narrow(self.get(), dim, start, length, &ret0));
  return Tensor(ret0);
}

// We expect this to be the stable version of the pad.default op.
// pad.default takes in a SymInt[] as the pad argument however pad is typed as
// use std::vector<int64_t> because
// (1) IntArrayRef is not yet header-only
// (2) SymInt is not yet header-only
inline Tensor pad(
    const Tensor& self,
    std::vector<int64_t> pad,
    const std::string& mode = "constant",
    double value = 0.0) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_pad(
      self.get(), pad.data(), pad.size(), mode.c_str(), &value, &ret0));
  return Tensor(ret0);
}

// We expect this to be the stable version of the transpose op with identical
// semantics to the existing transpose.int op.
inline Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{from(self), from(dim0), from(dim1)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::transpose", "int", stack.data()));
  return to<Tensor>(stack[0]);
}

// We expect this to be the stable version of the zero_ op with identical
// semantics to the existing zero_ op (except that it will not be called as
// a tensor method but only as a function i.e. zero_(t) not t.zero_()).
inline Tensor zero_(Tensor& self) {
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{from(self)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::zero_", "", stack.data()));
  return to<Tensor>(stack[0]);
}

} // namespace torch::stable
