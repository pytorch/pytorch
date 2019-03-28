#pragma once

// ${generated_comment}

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/TensorMethods.h>
#include <c10/core/TensorOptions.h>

#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace c10 {
class Scalar;
}
namespace at {
struct Generator;
class Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {}) {
  return at::getType(options).tensorFromBlob(data, sizes, deleter);
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {}) {
  return at::getType(options).tensorFromBlob(data, sizes, strides, deleter);
}

// These functions are defined in native/TensorFactories.cpp.
#define TENSOR(T, S, _1)                                                      \
  CAFFE2_API Tensor tensor(ArrayRef<T> values, const TensorOptions& options); \
  inline Tensor tensor(                                                       \
      std::initializer_list<T> values, const TensorOptions& options) {        \
    return native::tensor(ArrayRef<T>(values), options);                      \
  }                                                                           \
  inline Tensor tensor(T value, const TensorOptions& options) {               \
    return native::tensor(ArrayRef<T>(value), options);                       \
  }                                                                           \
  inline Tensor tensor(ArrayRef<T> values) {                                  \
    return native::tensor(std::move(values), at::dtype(k##S));                \
  }                                                                           \
  inline Tensor tensor(std::initializer_list<T> values) {                     \
    return native::tensor(ArrayRef<T>(values));                               \
  }                                                                           \
  inline Tensor tensor(T value) {                                             \
    return native::tensor(ArrayRef<T>(value));                                \
  }
AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(TENSOR)
#undef TENSOR

${native_function_declarations}

} // namespace native
} // namespace at
