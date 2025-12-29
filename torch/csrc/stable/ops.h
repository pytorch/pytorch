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

#if TORCH_FEATURE_VERSION < TORCH_VERSION_2_10_0
// We expect this to be a stable version of the new_empty op that takes in
// only dtype information.
// This is gated to < 2.10 to avoid ambiguity with the full new_empty overload
// in the 2.10+ block, which has the same first three parameters with defaults.
inline torch::stable::Tensor new_empty(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
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
// This is gated to < 2.10 to avoid ambiguity with the full new_zeros overload
// in the 2.10+ block, which has the same first three parameters with defaults.
inline torch::stable::Tensor new_zeros(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
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
#endif // TORCH_FEATURE_VERSION < TORCH_VERSION_2_10_0

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

// We expect this to be the stable version of the flatten.using_ints op.
inline torch::stable::Tensor flatten(
    const torch::stable::Tensor& self,
    int64_t start_dim = 0,
    int64_t end_dim = -1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(start_dim),
      torch::stable::detail::from(end_dim)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::flatten", "using_ints", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::flatten", "using_ints", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the unsqueeze op with identical
// semantics to the existing unsqueeze op.
inline torch::stable::Tensor unsqueeze(
    const torch::stable::Tensor& self,
    int64_t dim) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dim)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::unsqueeze", "", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::unsqueeze", "", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the squeeze.dim op with identical
// semantics to the existing squeeze.dim op.
inline torch::stable::Tensor squeeze(
    const torch::stable::Tensor& self,
    int64_t dim) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dim)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::squeeze", "dim", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::squeeze", "dim", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the select.int op with identical
// semantics to the existing select.int op.
// Note: index is typed as int64_t because SymInt is not yet header-only.
inline torch::stable::Tensor select(
    const torch::stable::Tensor& self,
    int64_t dim,
    int64_t index) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(index)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::select", "int", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::select", "int", stack.data()));
#endif
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the matmul op with identical
// semantics to the existing matmul op.
inline torch::stable::Tensor matmul(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(other)};
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::matmul", "", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::matmul", "", stack.data()));
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

// We expect this to be the stable version of the empty.memory_format op that
// takes in device and dtype parameters. This function is only available in 2.10
// because it uses the stableivalue conversion for HeaderOnlyArrayRef<T>, which
// is only available in 2.10.
inline torch::stable::Tensor empty(
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::headeronly::Layout> layout = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt,
    std::optional<torch::headeronly::MemoryFormat> memory_format =
        std::nullopt) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(size),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(layout),
      torch::stable::detail::from(device),
      torch::stable::detail::from(pin_memory),
      torch::stable::detail::from(memory_format)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::empty", "memory_format", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the reshape op.
// This function is only available in 2.10 because it uses the stableivalue
// conversion for HeaderOnlyArrayRef<T>, which is only available in 2.10.
inline torch::stable::Tensor reshape(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef shape) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(shape)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::reshape", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the view op.
// This function is only available in 2.10 because it uses the stableivalue
// conversion for HeaderOnlyArrayRef<T>, which is only available in 2.10.
inline torch::stable::Tensor view(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef size) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(size)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::view", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor from_blob(
    void* data,
    torch::headeronly::IntHeaderOnlyArrayRef sizes,
    torch::headeronly::IntHeaderOnlyArrayRef strides,
    torch::stable::Device device,
    torch::headeronly::ScalarType dtype,
    int64_t storage_offset = 0,
    torch::headeronly::Layout layout = torch::headeronly::Layout::Strided) {
  auto shim_dtype =
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
  auto shim_device_type = torch::stable::detail::to<int32_t>(
      torch::stable::detail::from(device.type()));
  auto shim_layout =
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(layout));
  AtenTensorHandle ath;
  TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      storage_offset,
      shim_dtype,
      shim_device_type,
      device.index(),
      &ath,
      shim_layout,
      nullptr,
      0));
  return torch::stable::Tensor(ath);
}

// We expect this to be the stable version of the to.dtype_layout op.
// This function is only available in 2.10 because it uses the stableivalue
// conversion for torch::stable::Device, which is only available in 2.10.
inline torch::stable::Tensor to(
    const torch::stable::Tensor& self,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::headeronly::Layout> layout = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt,
    bool non_blocking = false,
    bool copy = false,
    std::optional<torch::headeronly::MemoryFormat> memory_format =
        std::nullopt) {
  const auto num_args = 8;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(layout),
      torch::stable::detail::from(device),
      torch::stable::detail::from(pin_memory),
      torch::stable::detail::from(non_blocking),
      torch::stable::detail::from(copy),
      torch::stable::detail::from(memory_format)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::to", "dtype_layout", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// Convenience overload for to(device)
// We add this for convenience since stable does not support .to(TensorOptions)
inline torch::stable::Tensor to(
    const torch::stable::Tensor& self,
    torch::stable::Device device,
    bool non_blocking = false,
    bool copy = false) {
  return to(
      self,
      std::nullopt,
      std::nullopt,
      device,
      std::nullopt,
      non_blocking,
      copy,
      std::nullopt);
}

// We expect this to be the stable version of the contiguous op.
// This function is only available in 2.10 because it uses the stableivalue
// conversion for MemoryFormat, which is only available in 2.10.
// Contiguous is also a method on (non-stable Tensor), for now we only
// support the function version.
inline torch::stable::Tensor contiguous(
    const torch::stable::Tensor& self,
    torch::headeronly::MemoryFormat memory_format =
        torch::headeronly::MemoryFormat::Contiguous) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(memory_format)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::contiguous", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the new_empty op with all kwargs.
// This function is only available in 2.10 because it uses the stableivalue
// conversion for Device, HeaderOnlyArrayRef, Layout which are only available
// in 2.10. In versions < 2.10, a simpler new_empty overload that only takes
// dtype is available instead.
inline torch::stable::Tensor new_empty(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::headeronly::Layout> layout = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(size),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(layout),
      torch::stable::detail::from(device),
      torch::stable::detail::from(pin_memory)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::new_empty", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the new_zeros op with all kwargs.
// This function is only available in 2.10 because it uses the stableivalue
// conversion for Device, HeaderOnlyArrayRef, Layout which are only available
// in 2.10. In versions < 2.10, a simpler new_zeros overload that only takes
// dtype is available instead.
inline torch::stable::Tensor new_zeros(
    const torch::stable::Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::headeronly::Layout> layout = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(size),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(layout),
      torch::stable::detail::from(device),
      torch::stable::detail::from(pin_memory)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::new_zeros", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the sum.dim_IntList op.
// This function computes the sum of the input tensor along the specified
// dimensions and returns a new tensor containing the result. This function is
// only available in 2.10 because it uses the stableivalue conversion for
// HeaderOnlyArrayRef<T>, which is only available in 2.10. The dim parameter is
// optional - if not provided, sums over all dimensions.
inline torch::stable::Tensor sum(
    const torch::stable::Tensor& self,
    std::optional<torch::headeronly::IntHeaderOnlyArrayRef> dim = std::nullopt,
    bool keepdim = false,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
  const auto num_args = 4;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(keepdim),
      torch::stable::detail::from(dtype)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sum", "dim_IntList", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the sum.IntList_out op.
// This function takes an output tensor and computes the sum of the input tensor
// along the specified dimensions. The output tensor is modified in-place and
// returned. Following C++ convention, the out parameter comes first. This
// function is only available in 2.10 because it uses the stableivalue
// conversion for HeaderOnlyArrayRef<T>, which is only available in 2.10.
inline torch::stable::Tensor& sum_out(
    torch::stable::Tensor& out,
    const torch::stable::Tensor& self,
    std::optional<torch::headeronly::IntHeaderOnlyArrayRef> dim = std::nullopt,
    bool keepdim = false,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
  const auto num_args = 5;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(keepdim),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(out)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sum", "IntList_out", stack.data(), TORCH_ABI_VERSION));
  // Clean up the handle in stack[0], discard the temporary
  (void)torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
  return out;
}

// We expect this to be the stable version of the subtract.Tensor op.
// Note: alpha is typed as double because the underlying C shim API
// uses double for the Scalar parameter. We don't use torch_call_dispatcher
// as the stableivalue conversion for Scalar is not yet available as of
// 2.10
inline torch::stable::Tensor subtract(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other,
    double alpha = 1.0) {
  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_subtract_Tensor(self.get(), other.get(), alpha, &ret0));
  return torch::stable::Tensor(ret0);
}

// We expect this to be the stable version of the full.default op.
// Note: fill_value is typed as double because the underlying C shim API
// uses double for the Scalar parameter. We don't use torch_call_dispatcher
// as the stableivalue conversion for Scalar is not yet available as of
// 2.10
inline torch::stable::Tensor full(
    torch::headeronly::IntHeaderOnlyArrayRef size,
    double fill_value,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::headeronly::Layout> layout = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  int32_t* dtype_ptr = nullptr;
  int32_t dtype_val;
  if (dtype.has_value()) {
    dtype_val = torch::stable::detail::to<int32_t>(
        torch::stable::detail::from(dtype.value()));
    dtype_ptr = &dtype_val;
  }

  int32_t* layout_ptr = nullptr;
  int32_t layout_val;
  if (layout.has_value()) {
    layout_val = torch::stable::detail::to<int32_t>(
        torch::stable::detail::from(layout.value()));
    layout_ptr = &layout_val;
  }

  int32_t* device_type_ptr = nullptr;
  int32_t device_type_val;
  int32_t device_index = 0;
  if (device.has_value()) {
    device_type_val = torch::stable::detail::to<int32_t>(
        torch::stable::detail::from(device.value().type()));
    device_type_ptr = &device_type_val;
    device_index = device.value().index();
  }

  int32_t* pin_memory_ptr = nullptr;
  int32_t pin_memory_val;
  if (pin_memory.has_value()) {
    pin_memory_val = pin_memory.value() ? 1 : 0;
    pin_memory_ptr = &pin_memory_val;
  }

  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_full(
      size.data(),
      static_cast<int64_t>(size.size()),
      fill_value,
      dtype_ptr,
      layout_ptr,
      device_type_ptr,
      device_index,
      pin_memory_ptr,
      &ret0));

  return torch::stable::Tensor(ret0);
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

HIDDEN_NAMESPACE_END(torch, stable)
