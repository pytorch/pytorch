#pragma once

#include <ATen/core/Tensor.h>

namespace at {
namespace native {

Tensor& quantized_copy_from_float_(Tensor& self, const Tensor& src);
}
} // namespace at
