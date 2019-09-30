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
#include <ATen/core/ATenDispatch.h>
#include <ATen/Context.h>
#include <ATen/core/EnableNamedTensor.h>

namespace at {

using native::tensor;

${function_declarations}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {}) {
  auto device = globalContext().getDeviceFromPtr(data, options.device().type());
  if (options.device().has_index()) {
    TORCH_CHECK(
        options.device() == device,
        "Specified device ", options.device(),
        " does not match device of data ", device);
  }
  auto storage = Storage(
      options.dtype(),
      detail::computeStorageSize(sizes, strides),
      InefficientStdFunctionContext::makeDataPtr(
          data, deleter, device),
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
  return from_blob(data, sizes, strides, [](void*) {}, options);
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const TensorOptions& options = {}) {
  return from_blob(data, sizes, detail::defaultStrides(sizes), [](void*) {}, options);
}

// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
${function_definitions}

}
