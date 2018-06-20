#pragma once

// ${generated_comment}

#include <ATen/ScalarType.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorMethods.h>
#include <ATen/TensorOptions.h>

#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace at {
struct Generator;
class Scalar;
struct Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {

inline Tensor from_blob(
    void* data,
    IntList sizes,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {}) {
  return options.type().tensorFromBlob(data, sizes, deleter);
}

inline Tensor from_blob(
    void* data,
    IntList sizes,
    const TensorOptions& options = {}) {
  return native::from_blob(data, sizes, [](void*) {}, options);
}

${native_function_declarations}

} // namespace native
} // namespace at
