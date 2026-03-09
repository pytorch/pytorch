#pragma once

#include <torch/csrc/stable/stableivalue_conversions.h>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

/// A function pointer type for data deleters used with from_blob.
/// The deleter is called with the data pointer when the tensor's storage
/// is deallocated.
using DeleterFnPtr = void (*)(void*);

/// Stable version of the empty_like op.
///
/// Creates a new uninitialized tensor with the same size, dtype, layout, and
/// device as the input tensor. This version does not support kwargs (device,
/// dtype, layout, memory_format) - kwargs support may be added in the future.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The input tensor whose properties will be used for the new
/// tensor.
/// @return A new uninitialized tensor with the same properties as self.
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

/// Stable version of the fill_.Scalar op.
///
/// Fills the input tensor with the specified scalar value in-place and returns
/// it. This has identical semantics to the existing fill_.Scalar op.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @note The value parameter is typed as double
///       This is because Scalar.h is currently not header-only.
///
/// @param self The tensor to fill.
/// @param value The scalar value to fill the tensor with.
/// @return The input tensor, now filled with the specified value.
inline torch::stable::Tensor fill_(
    const torch::stable::Tensor& self,
    double value) {
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_fill__Scalar(self.get(), value));
  return self;
}

/// Stable version of the narrow.default op.
///
/// Returns a new tensor that is a narrowed version of the input tensor. The
/// dimension dim is narrowed from start to start + length.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @note The start and length parameters
///       is not yet header-only.
///
/// @param self The input tensor to narrow.
/// @param dim The dimension along which to narrow.
/// @param start The starting index for the narrowed dimension.
/// @param length The length of the narrowed dimension.
/// @return A new tensor that is a narrowed view of the input.
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
/// Stable version of the new_empty op (2.9 version).
///
/// Creates a new uninitialized tensor with the specified size, inheriting
/// device and layout from the input tensor. This version only supports the
/// dtype kwarg. For the full kwargs version, use PyTorch 2.10+.
///
/// Minimum compatible version: PyTorch 2.9. For full kwargs support, use
/// PyTorch 2.10+.
///
/// @param self The input tensor whose device
/// @param size The desired size of the output tensor.
/// @param dtype Optional scalar type for the tensor elements. If not provided,
///              inherits from self.
/// @return A new uninitialized tensor with the specified properties.
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

/// Stable version of the new_zeros op (2.9 version).
///
/// Creates a new tensor filled with zeros with the specified size, inheriting
/// device and layout from the input tensor. This version only supports the
/// dtype kwarg. For the full kwargs version, use PyTorch 2.10+.
///
/// Minimum compatible version: PyTorch 2.9. For full kwargs support, use
/// PyTorch 2.10+.
///
/// @param self The input tensor whose device and layout will be inherited.
/// @param size The desired size of the output tensor.
/// @param dtype Optional scalar type for the tensor elements. If not provided,
///              inherits from self.
/// @return A new zero-filled tensor with the specified properties.
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

/// Stable version of the pad.default op.
///
/// Pads the input tensor according to the specified padding sizes. The padding
/// is applied symmetrically to each dimension, with the padding sizes specified
/// in reverse order (last dimension first).
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @note The pad parameter is typed
///       not yet header-only.
///
/// @param self The input tensor to pad.
/// @param pad The padding sizes for each dimension (in pairs, starting from
///            the last dimension).
/// @param mode The padding mode: "constant", "reflect", "replicate", or
///             "circular". Defaults to "constant".
/// @param value The fill value for constant padding. Defaults to 0.0.
/// @return A new padded tensor.
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

/// Stable version of the amax.default op (single dimension).
///
/// Computes the maximum value along the specified dimension. If keepdim is
/// true, the output tensor has the same number of dimensions as the input,
/// with the reduced dimension having size 1. Otherwise, the reduced dimension
/// is removed.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The input tensor.
/// @param dim The dimension along which to compute the maximum.
/// @param keepdim Whether to retain
/// @return A tensor containing the maximum values along the specified
/// dimension.
inline torch::stable::Tensor amax(
    const torch::stable::Tensor& self,
    int64_t dim,
    bool keepdim = false) {
  AtenTensorHandle ret = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_amax(self.get(), &dim, 1, keepdim, &ret));
  return torch::stable::Tensor(ret);
}

/// Stable version of the amax.default op (multiple dimensions).
///
/// Computes the maximum value reducing over all the specified dimensions. If
/// keepdim is true, the output tensor has the same number of dimensions as the
/// input, with the reduced dimensions having size 1. Otherwise, the reduced
/// dimensions are removed.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @note The dims parameter is typed
///       is not yet header-only.
///
/// @param self The input tensor.
/// @param dims The dimensions along which to compute the maximum.
/// @param keepdim Whether to retain the reduced dimensions. Defaults to false.
/// @return A tensor containing the maximum values.
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

/// Stable version of the transpose.int op.
///
/// Returns a tensor that is a transposed version of the input, with dimensions
/// dim0 and dim1 swapped. The returned tensor shares storage with the input.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The input tensor.
/// @param dim0 The first dimension to transpose.
/// @param dim1 The second dimension to transpose.
/// @return A transposed view of the input tensor.
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

/// Stable version of the zero_ op.
///
/// Fills the input tensor with zeros in-place and returns it. Unlike the
/// tensor method version (t.zero_()), this is called as a function: zero_(t).
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The tensor to fill with zeros.
/// @return The input tensor, now filled with zeros.
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

/// Stable version of the copy_ op.
///
/// Copies the elements from the source tensor into the destination tensor
/// in-place and returns the destination tensor. The tensors must be
/// broadcastable.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The destination tensor (modified in-place).
/// @param src The source tensor to copy from.
/// @param non_blocking If true, the copy may occur asynchronously with respect
///                     to the host. Defaults to false.
/// @return The destination tensor with copied values.
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

/// Stable version of the clone op.
///
/// Returns a copy of the input tensor. The returned tensor has the same data
/// and type as the input, but is stored in a new memory location.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @note Optional memory_format kwarg support
///
/// @param self The input tensor to clone.
/// @return A new tensor with copied data.
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

/// Stable version of the flatten.using_ints op.
///
/// Flattens the input tensor by reshaping it into a one-dimensional tensor.
/// If start_dim or end_dim are specified, only dimensions starting from
/// start_dim to end_dim are flattened.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The input tensor to flatten.
/// @param start_dim The first dimension to flatten. Defaults to 0.
/// @param end_dim The last dimension to flatten. Defaults to -1 (last dim).
/// @return A flattened tensor.
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

/// Stable version of the unsqueeze op.
///
/// Returns a new tensor with a dimension of size one inserted at the specified
/// position. The returned tensor shares the same underlying data with the input
/// tensor.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The input tensor.
/// @param dim The index at which to insert
///            values are supported.
/// @return A tensor with an additional dimension.
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

/// Stable version of the squeeze.dim op.
///
/// Returns a tensor with the dimension of size one at the specified position
/// removed. The returned tensor shares the same underlying data with the input
/// tensor.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The input tensor.
/// @param dim The dimension to squeeze.
///            the tensor is returned unchanged.
/// @return A tensor with the specified dimension removed (if size was 1).
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

/// Stable version of the select.int op.
///
/// Slices the input tensor along the specified dimension at the given index.
/// This function returns a view of the original tensor with the given dimension
/// removed.
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @note The index parameter is typed
///       header-only.
///
/// @param self The input tensor.
/// @param dim The dimension to slice.
/// @param index The index to select along the dimension.
/// @return A tensor with one fewer dimension.
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

/// Stable version of the matmul op.
///
/// Performs matrix multiplication between two tensors. The behavior depends on
/// the dimensionality of the tensors (see PyTorch documentation for details on
/// broadcasting rules for matmul).
///
/// Minimum compatible version: PyTorch 2.9.
///
/// @param self The first input tensor.
/// @param other The second input tensor.
/// @return The result of matrix multiplication.
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

/// Stable parallel_for utility.
///
/// Provides a stable interface to at::parallel_for for parallel execution.
/// The function f will be called with (begin, end) ranges to process in
/// parallel. grain_size controls the minimum work size per thread for efficient
/// parallelization.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @tparam F The callable type
/// @param begin The start of the iteration range.
/// @param end The end of the iteration range (exclusive).
/// @param grain_size The minimum number of iterations per thread.
/// @param f The function to execute in parallel.
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

/// Gets the number of threads for the parallel backend.
///
/// Provides a stable interface to at::get_num_threads.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @return The number of threads
inline uint32_t get_num_threads() {
  uint32_t num_threads;
  TORCH_ERROR_CODE_CHECK(torch_get_num_threads(&num_threads));
  return num_threads;
}

/// Stable version of the empty.memory_format op.
///
/// Creates a new uninitialized tensor with the specified size and options.
/// This function supports full tensor creation options including device,
/// dtype, layout, and memory format.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param size The desired size of the output tensor.
/// @param dtype Optional scalar type for the tensor elements.
/// @param layout Optional memory layout (e.g., strided, sparse).
/// @param device Optional device to place the tensor on.
/// @param pin_memory Optional flag to use pinned memory (for CUDA tensors).
/// @param memory_format Optional memory format for the tensor.
/// @return A new uninitialized tensor with the specified properties.
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

/// Stable version of the reshape op.
///
/// Returns a tensor with the same data and number of elements as the input,
/// but with the specified shape. When possible, the returned tensor will be
/// a view of the input.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor.
/// @param shape The desired output shape.
/// @return A tensor with the specified shape.
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

/// Stable version of the view op.
///
/// Returns a new tensor with the same data as the input tensor but with a
/// different shape. The returned tensor shares the same data and must have
/// the same number of elements.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor.
/// @param size The desired output shape.
/// @return A view tensor with the specified shape.
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

/// Creates a tensor from an existing data blob.
///
/// Creates a tensor that uses the provided data pointer as its storage.
/// The tensor does not own the data, so the caller must ensure the data
/// remains valid for the lifetime of the tensor.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param data Pointer to the data buffer.
/// @param sizes The size of each dimension of the tensor.
/// @param strides The stride for each dimension.
/// @param device The device where the data resides.
/// @param dtype The scalar type of the data.
/// @param storage_offset The offset into the data buffer. Defaults to 0.
/// @param layout The memory layout. Defaults to Strided.
/// @return A tensor backed by the provided data.
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

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
/// Creates a tensor from an existing data blob with a custom deleter.
///
/// This is the same as the from_blob function above, but allows specifying a
/// custom deleter function that will be called when the tensor's storage is
/// deallocated.
/// Minimum compatible version: PyTorch 2.11.
///
/// @param data Pointer to the data buffer.
/// @param sizes The size of each dimension of the tensor.
/// @param strides The stride for each dimension.
/// @param device The device where the data resides.
/// @param dtype The scalar type of the data.
/// @param deleter Function to call when the tensor is deallocated. May be
///                nullptr if no cleanup is needed.
/// @param storage_offset The offset into the data buffer. Defaults to 0.
/// @param layout The memory layout. Defaults to Strided.
/// @return A tensor backed by the provided data.
inline torch::stable::Tensor from_blob(
    void* data,
    torch::headeronly::IntHeaderOnlyArrayRef sizes,
    torch::headeronly::IntHeaderOnlyArrayRef strides,
    torch::stable::Device device,
    torch::headeronly::ScalarType dtype,
    DeleterFnPtr deleter,
    int64_t storage_offset = 0,
    torch::headeronly::Layout layout = torch::headeronly::Layout::Strided) {
  auto shim_dtype =
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
  auto shim_device_type = torch::stable::detail::to<int32_t>(
      torch::stable::detail::from(device.type()));
  auto shim_layout =
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(layout));
  AtenTensorHandle ath;
  TORCH_ERROR_CODE_CHECK(torch_from_blob(
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
      0,
      deleter));
  return torch::stable::Tensor(ath);
}
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0
/// Creates a tensor from an existing data blob with a generic callable deleter.
///
/// This overload accepts any callable as the deleter, including capturing
/// lambdas, which the from_blob above doesn't.
///
/// Minimum compatible version: PyTorch 2.12.
///
/// @tparam F The callable type. Must be invocable with (void*).
/// @param data Pointer to the data buffer.
/// @param sizes The size of each dimension of the tensor.
/// @param strides The stride for each dimension.
/// @param device The device where the data resides.
/// @param dtype The scalar type of the data.
/// @param deleter Callable to invoke when the tensor is deallocated.
/// @param storage_offset The offset into the data buffer. Defaults to 0.
/// @param layout The memory layout. Defaults to Strided.
/// @return A tensor backed by the provided data.

// The enable_if_t part below ensures that:
// 1. The other, simpler from_blob is called if the deleter is compatible
//    with it (i.e. if it can be converted to DeleterFnPtr).
// 2. Non-callable types (like int) don't accidentally match this template
//    when passed as storage_offset.
template <
    class F,
    std::enable_if_t<
        !std::is_convertible_v<F, DeleterFnPtr> &&
            std::is_invocable_v<F, void*>,
        int> = 0>
inline torch::stable::Tensor from_blob(
    void* data,
    torch::headeronly::IntHeaderOnlyArrayRef sizes,
    torch::headeronly::IntHeaderOnlyArrayRef strides,
    torch::stable::Device device,
    torch::headeronly::ScalarType dtype,
    F deleter,
    int64_t storage_offset = 0,
    torch::headeronly::Layout layout = torch::headeronly::Layout::Strided) {
  auto shim_dtype =
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
  auto shim_device_type = torch::stable::detail::to<int32_t>(
      torch::stable::detail::from(device.type()));
  auto shim_layout =
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(layout));

  // The deleter we receive may be a capturing lambda, which is typically
  // allocated on the stack. It needs to outlive the from_blob call and other
  // stack frames, so we have to copy it into a heap object.
  // This is a similar pattern that is used in InefficientStdFunctionContext.
  F* heap_allocated_deleter = new F(std::move(deleter));

  // Also, since F may be a capturing lambda, it cannot be passed to the C layer
  // directly. We have to do type erasure and hide it behind two separate things
  // that C understands:
  // - a C-like callback (deleter_callback), which will eventually be called
  //   within at::from_blob. Note that deleter_callback is also a lambda, but C
  //   understands it because it can be implicitly cast to
  //   void (*)(void*, void*).
  // - a context associated to the callback. Here, the context is literally the
  //   heap_allocated_deleter, which is called within the callback.
  //
  // Note that all of heap_allocated_deleter, deleter_ctx and func are all the
  // exact same thing: the heap-allocated deleter.
  auto deleter_callback = [](void* data, void* deleter_ctx) {
    F* func = static_cast<F*>(deleter_ctx);
    (*func)(data);
    delete func;
  };

  AtenTensorHandle ath;
  TORCH_ERROR_CODE_CHECK(torch_from_blob_v2(
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
      0,
      deleter_callback,
      static_cast<void*>(heap_allocated_deleter)));
  return torch::stable::Tensor(ath);
}
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0

/// Stable version of the to.dtype_layout op.
///
/// Converts a tensor to the specified dtype, layout, device, and/or memory
/// format. Returns a new tensor with the specified properties.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor.
/// @param dtype Optional target scalar type.
/// @param layout Optional target memory layout.
/// @param device Optional target device.
/// @param pin_memory Optional flag to use pinned memory.
/// @param non_blocking If true, the operation may be asynchronous. Defaults to
///                     false.
/// @param copy If true, always create a copy. Defaults to false.
/// @param memory_format Optional target memory format.
/// @return A tensor with the specified properties.
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

/// Convenience overload for moving a tensor to a device.
///
/// Moves the tensor to the specified device. This is a convenience wrapper
/// around the full to() function.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor.
/// @param device The target device.
/// @param non_blocking If true, the operation may be asynchronous. Defaults to
///                     false.
/// @param copy If true, always create a copy. Defaults to false.
/// @return A tensor on the specified device.
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

/// Stable version of the contiguous op.
///
/// Returns a contiguous in memory tensor containing the same data as the input
/// tensor. If the input tensor is already contiguous in the specified memory
/// format, the input tensor is returned.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor.
/// @param memory_format The desired memory format.
/// @return A contiguous tensor.
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

/// Stable version of the new_empty op (2.10 version with full kwargs).
///
/// Creates a new uninitialized tensor with the specified size and options.
/// This version supports all tensor creation kwargs. For versions < 2.10,
/// a simpler overload that only takes dtype is available.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor whose properties may be inherited if kwargs are
///             not provided.
/// @param size The desired size of the output tensor.
/// @param dtype Optional scalar type for the tensor elements.
/// @param layout Optional memory layout (e.g., strided, sparse).
/// @param device Optional device to place the tensor on.
/// @param pin_memory Optional flag to use pinned memory (for CUDA tensors).
/// @return A new uninitialized tensor with the specified properties.
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

/// Stable version of the new_zeros op (2.10 version with full kwargs).
///
/// Creates a new zero-filled tensor with the specified size and options.
/// This version supports all tensor creation kwargs. For versions < 2.10,
/// a simpler overload that only takes dtype is available.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor whose properties may be inherited if kwargs are
///             not provided.
/// @param size The desired size of the output tensor.
/// @param dtype Optional scalar type for the tensor elements.
/// @param layout Optional memory layout (e.g., strided, sparse).
/// @param device Optional device to place the tensor on.
/// @param pin_memory Optional flag to use pinned memory (for CUDA tensors).
/// @return A new zero-filled tensor with the specified properties.
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

/// Stable version of the sum.dim_IntList op.
///
/// Computes the sum of the input tensor along the specified dimensions.
/// If dim is not provided, sums over all dimensions.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param self The input tensor.
/// @param dim Optional dimensions to reduce. If not provided, reduces all
///            dimensions.
/// @param keepdim Whether to retain the reduced dimensions. Defaults to false.
/// @param dtype Optional output dtype. If not provided, uses the input dtype.
/// @return A tensor containing the sum.
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

/// Stable version of the sum.IntList_out op.
///
/// Computes the sum of the input tensor along the specified dimensions,
/// storing the result in the provided output tensor. Following C++ convention,
/// the out parameter comes first.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @param out The output tensor (modified in-place).
/// @param self The input tensor.
/// @param dim Optional dimensions to reduce.
/// @param keepdim Whether to retain the reduced dimensions. Defaults to false.
/// @param dtype Optional output dtype.
/// @return Reference to the output tensor.
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

/// Stable version of the subtract.Tensor op.
///
/// Subtracts the other tensor from self, with an optional scaling factor alpha.
/// Computes: self - alpha * other.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @note The alpha parameter is typed as double
///       API uses double for the Scalar parameter.
///
/// @param self The input tensor.
/// @param other The tensor to subtract.
/// @param alpha The scaling factor for other. Defaults to 1.0.
/// @return The result of self - alpha * other.
inline torch::stable::Tensor subtract(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other,
    double alpha = 1.0) {
  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_subtract_Tensor(self.get(), other.get(), alpha, &ret0));
  return torch::stable::Tensor(ret0);
}

/// Stable version of the full.default op.
///
/// Creates a tensor of the specified size filled with the given value.
///
/// Minimum compatible version: PyTorch 2.10.
///
/// @note The fill_value parameter is typed
///       C shim API uses double for the Scalar parameter.
///
/// @param size The desired size of the output tensor.
/// @param fill_value The value to fill the tensor with.
/// @param dtype Optional scalar type for the tensor elements.
/// @param layout Optional memory layout.
/// @param device Optional device to place the tensor on.
/// @param pin_memory Optional flag to use pinned memory.
/// @return A new tensor filled with the specified value.
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
