#pragma once

// ${generated_comment}

#include <ATen/Context.h>
#include <ATen/NativeMetaFunctions.h>
#include <ATen/core/Reduction.h>
#include <c10/core/ScalarType.h>
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

${native_function_declarations}

} // namespace native

// From build/aten/src/ATen/NativeFunctions.h
namespace native {
  struct TORCH_API structured_add_out : public at::meta::structured_add_Tensor {
    void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
  };
} //namespace native

} // namespace at
