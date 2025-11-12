#pragma once

#include <torch/csrc/stable/stableivalue_conversions.h>
#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

// We expect this to be the stable version of the empty_like op that takes in
// no kwargs (device, dtype, layout, memory_format). We will add kwargs
// support in the future.
inline torch::stable::Tensor empty_like(const torch::stable::Tensor& self) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(std::nullopt)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::empty_like", "", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::empty_like", "", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the fill_.Scalar op
// with identical semantics to the existing fill_.Scalar op.
// A subtle nuance is that `value` is typed as a double, but it is
// actually a Scalar. This is because Scalar.h is currently not
// header-only.
inline torch::stable::Tensor fill_(
    const torch::stable::Tensor& self,
    double value) {
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_fill__Scalar(self.get(), value));
  return self;
}

// We expect this to be the stable version of the narrow.default op.
// narrow takes in a SymInt for start and length, but these are typed as
// int64_t as SymInt is not yet header-only.
inline torch::stable::Tensor narrow(
    torch::stable::Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_narrow(self.get(), dim, start, length, &ret0));
  return torch::stable::Tensor(ret0);
}

// We expect this to be a stable version of the new_empty op that takes in
// only dtype information.
inline torch::stable::Tensor new_empty(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<c10::ScalarType> dtype = std::nullopt) {
  int32_t device_type;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(self.get(), &device_type));

  int32_t device_index;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(self.get(), &device_index));

  int32_t target_dtype;
  if (dtype.has_value()) {
    target_dtype = torch::stable::detail::to<int32_t>(
        torch::stable::detail::from(dtype.value()));
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &target_dtype));
  }

  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));

  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_empty(
      self.get(),
      size.data(),
      static_cast<int64_t>(size.size()),
      &target_dtype,
      &layout,
      &device_type,
      device_index,
      nullptr, // pin_memory (nullptr for default)
      &ret0));

  return torch::stable::Tensor(ret0);
}

// We expect this to be a stable version of the new_zeros op that takes in
// only dtype information.
inline torch::stable::Tensor new_zeros(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<c10::ScalarType> dtype = std::nullopt) {
  int32_t device_type;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(self.get(), &device_type));

  int32_t device_index;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(self.get(), &device_index));

  int32_t target_dtype;
  if (dtype.has_value()) {
    target_dtype = torch::stable::detail::to<int32_t>(
        torch::stable::detail::from(dtype.value()));
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &target_dtype));
  }

  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));

  AtenTensorHandle ath;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_zeros(
      self.get(),
      size.data(),
      static_cast<int64_t>(size.size()),
      &target_dtype,
      &layout,
      &device_type,
      device_index,
      nullptr, // pin_memory (nullptr for default)
      &ath));

  return torch::stable::Tensor(ath);
}

// We expect this to be the stable version of the pad.default op.
// pad.default takes in a SymInt[] as the pad argument however pad is typed as
// torch::headeronly::IntHeaderOnlyArrayRef as SymInt is not yet header-only.
inline torch::stable::Tensor pad(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef pad,
    const std::string& mode = "constant",
    double value = 0.0) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_pad(
      self.get(), pad.data(), pad.size(), mode.c_str(), &value, &ret0));
  return torch::stable::Tensor(ret0);
}

// We expect the following two functions to be stable versions of the
// amax.default op with identical semantics to the existing amax.default op. If
// `keepdim` is true, the result will have the same number of dimensions as
// `self`, with the specified dimension having size 1. Otherwise, the result
// will have one fewer dimension than `self`, with the specified dimension
// removed.

// This function is an overload to compute the maximum value along each slice of
// `self` along a single dimension `dim`.
inline torch::stable::Tensor amax(
    const torch::stable::Tensor& self,
    int64_t dim,
    bool keepdim = false) {
  AtenTensorHandle ret = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_amax(self.get(), &dim, 1, keepdim, &ret));
  return torch::stable::Tensor(ret);
}

// This function is an overload to compute the maximum value along each slice of
// `self` reducing over all the dimensions in the vector `dims`. The
// amax.default op takes in a SymInt[] as the dims argument, however dims is
// typed as use IntHeaderOnlyArrayRef here because SymInt is not yet header-only
inline torch::stable::Tensor amax(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef dims,
    bool keepdim = false) {
  AtenTensorHandle ret = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_amax(
      self.get(),
      dims.data(),
      static_cast<int64_t>(dims.size()),
      keepdim,
      &ret));
  return torch::stable::Tensor(ret);
}

// We expect this to be the stable version of the transpose op with identical
// semantics to the existing transpose.int op.
inline torch::stable::Tensor transpose(
    const torch::stable::Tensor& self,
    int64_t dim0,
    int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim0),
      torch::stable::detail::from(dim1)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::transpose", "int", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::transpose", "int", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the zero_ op with identical
// semantics to the existing zero_ op (except that it will not be called as
// a tensor method but only as a function i.e. zero_(t) not t.zero_()).
inline torch::stable::Tensor zero_(torch::stable::Tensor& self) {
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{torch::stable::detail::from(self)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::zero_", "", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::zero_", "", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the copy_ op with
// identical semantics to the existing copy_ op.
inline torch::stable::Tensor copy_(
    torch::stable::Tensor& self,
    const torch::stable::Tensor& src,
    std::optional<bool> non_blocking = std::nullopt) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(src),
      torch::stable::detail::from(non_blocking.value_or(false))};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::copy_", "", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::copy_", "", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the clone op. We will
// add optional memory_format kwarg support in the future.
inline torch::stable::Tensor clone(const torch::stable::Tensor& self) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(std::nullopt)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::clone", "", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::clone", "", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

// New ops should be added here if they use a brand new shim API

// Parallel utility wrapper that provides a stable interface to at::parallel_for
// This function has the same signature as at::parallel_for and allows stable
// ABI code to leverage PyTorch's parallel execution capabilities.
//
// The function f will be called with (begin, end) ranges to process in
// parallel. grain_size controls the minimum work size per thread for efficient
// parallelization.
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  auto callback = [](int64_t cb_begin, int64_t cb_end, void* ctx) {
    const F* func = static_cast<const F*>(ctx);
    (*func)(cb_begin, cb_end);
  };
  TORCH_ERROR_CODE_CHECK(torch_parallel_for(
      begin,
      end,
      grain_size,
      callback,
      const_cast<void*>(static_cast<const void*>(&f))));
}

// Get the number of threads for the parallel backend
// This provides a stable interface to at::get_num_threads
inline uint32_t get_num_threads() {
  uint32_t num_threads;
  TORCH_ERROR_CODE_CHECK(torch_get_num_threads(&num_threads));
  return num_threads;
}

// We expect this to be the stable version of the empty op that takes in
// device and dtype parameters. The empty op creates a tensor with uninitialized
// values of the specified size, dtype, and device.
inline torch::stable::Tensor empty(
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<c10::ScalarType> dtype = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(size),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(device),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::empty", "memory_format", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

#endif

HIDDEN_NAMESPACE_END(torch, stable)
