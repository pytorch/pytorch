#pragma once

// ${generated_comment}

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <ATen/core/Generator.h>
#include <c10/util/Deprecated.h>
#include <ATen/NativeFunctions.h>
#include <ATen/DeviceGuard.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Reduction.h>
#include <c10/util/Optional.h>
#include <ATen/TensorUtils.h>
#include <ATen/Context.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/TypeDefault.h>
#include <ATen/CPUType.h>
#include <ATen/QuantizedCPUType.h>

namespace at {

using native::tensor;

${function_declarations}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {}) {
  AutoNonVariableTypeMode guard;
  auto device = globalContext().getDeviceFromPtr(data, options.device().type());
  if (options.device().has_index()) {
    TORCH_CHECK(
        options.device() == device,
        "Specified device ", options.device(),
        " does not match device of data ", device);
  }
  auto storage = Storage(
      Storage::use_byte_size_t(),
      options.dtype(),
      detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
      InefficientStdFunctionContext::makeDataPtr(data, deleter, device),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  return empty({0}, options).set_(storage, 0, sizes, strides);
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
  AutoNonVariableTypeMode guard;
  auto device = globalContext().getDeviceFromPtr(data, options.device().type());
  if (options.device().has_index()) {
    TORCH_CHECK(
        options.device() == device,
        "Specified device ", options.device(),
        " does not match device of data ", device);
  }
  auto storage = Storage(
      Storage::use_byte_size_t(),
      options.dtype(),
      detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
      DataPtr(data, nullptr, [](void*) {}, device),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  return empty({0}, options).set_(storage, 0, sizes, strides);
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

// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
${function_definitions}

}
