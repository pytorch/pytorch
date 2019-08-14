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

namespace detail {

static inline Backend infer_backend(const Tensor & t) {
  TORCH_CHECK(t.defined(), "undefined Tensor");
  return tensorTypeIdToBackend(t.type_id());
}
static inline Backend infer_backend(const TensorList & tl) {
  TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
  return tensorTypeIdToBackend(tl[0].type_id());
}

static inline bool infer_is_variable(const Tensor & t) {
  TORCH_CHECK(t.defined(), "undefined Tensor");
  return t.is_variable();
}
static inline bool infer_is_variable(const TensorList & tl) {
  TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
  return tl[0].is_variable();
}


} // namespace detail

// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
${function_definitions}

}
