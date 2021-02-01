#pragma once

// ${generated_comment}

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <ATen/core/Generator.h>
#include <c10/util/Deprecated.h>
#include <ATen/DeviceGuard.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Reduction.h>
#include <c10/util/Optional.h>
#include <ATen/TensorUtils.h>
#include <ATen/Context.h>
#include <ATen/TracerMode.h>
#include <ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h>

namespace at {

// These functions are defined in ATen/Utils.cpp.
#define TENSOR(T, S)                                                          \
  TORCH_API Tensor tensor(ArrayRef<T> values, const TensorOptions& options); \
  inline Tensor tensor(                                                       \
      std::initializer_list<T> values, const TensorOptions& options) {        \
    return at::tensor(ArrayRef<T>(values), options);                          \
  }                                                                           \
  inline Tensor tensor(T value, const TensorOptions& options) {               \
    return at::tensor(ArrayRef<T>(value), options);                           \
  }                                                                           \
  inline Tensor tensor(ArrayRef<T> values) {                                  \
    return at::tensor(std::move(values), at::dtype(k##S));                    \
  }                                                                           \
  inline Tensor tensor(std::initializer_list<T> values) {                     \
    return at::tensor(ArrayRef<T>(values));                                   \
  }                                                                           \
  inline Tensor tensor(T value) {                                             \
    return at::tensor(ArrayRef<T>(value));                                    \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

${function_declarations}

namespace redispatch {
    ${function_redispatch_declarations}
} // namespace redispatch

// Special C++ only overloads for std()-like functions (See gh-40287)
// These are needed because int -> bool conversion takes precedence over int -> IntArrayRef
// So, for example std(0) would select the std(unbiased=False) overload
TORCH_API Tensor var(const Tensor& self, int dim);
TORCH_API std::tuple<Tensor,Tensor> var_mean(const Tensor& self, int dim);
TORCH_API Tensor std(const Tensor& self, int dim);
TORCH_API std::tuple<Tensor,Tensor> std_mean(const Tensor& self, int dim);

namespace {
  inline std::vector<int64_t> zero_sizes(const TensorOptions& options) {
    if (options.has_memory_format()) {
      auto memory_format = *options.memory_format_opt();
      if (at::MemoryFormat::ChannelsLast == memory_format) {
        return {0, 0, 0, 0};
      }
      if (at::MemoryFormat::ChannelsLast3d == memory_format) {
        return {0, 0, 0, 0, 0};
      }
    }
    return {0};
  }
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const c10::optional<Device> target_device = c10::nullopt) {
  AutoNonVariableTypeMode guard;  // TODO: remove
  tracer::impl::NoTracerDispatchMode tracer_guard;
  auto device = (target_device.has_value()?
    target_device.value() : globalContext().getDeviceFromPtr(data, options.device().type()));
  if (options.device().has_index()) {
    TORCH_CHECK(
        options.device() == device,
        "Specified device ", options.device(),
        " does not match device of data ", device);
  }
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
      InefficientStdFunctionContext::makeDataPtr(data, deleter, device),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  return empty(IntArrayRef(zero_sizes(options)), options).set_(storage, 0, sizes, strides);
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {}) {
  return from_blob(data, sizes, detail::defaultStrides(sizes), deleter, options);
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options = {}) {
  AutoNonVariableTypeMode guard;  // TODO: remove
  tracer::impl::NoTracerDispatchMode tracer_guard;
  auto device = globalContext().getDeviceFromPtr(data, options.device().type());
  if (options.device().has_index()) {
    TORCH_CHECK(
        options.device() == device,
        "Specified device ", options.device(),
        " does not match device of data ", device);
  }
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
      DataPtr(data, nullptr, [](void*) {}, device),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  return empty(IntArrayRef(zero_sizes(options)), options).set_(storage, 0, sizes, strides);
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const TensorOptions& options = {}) {
  return from_blob(data, sizes, detail::defaultStrides(sizes), options);
}

inline int64_t numel(const Tensor& tensor) {
  return tensor.numel();
}

inline int64_t size(const Tensor& tensor, int64_t dim) {
  return tensor.size(dim);
}

inline int64_t stride(const Tensor& tensor, int64_t dim) {
  return tensor.stride(dim);
}

} // namespace at
