#pragma once

#include <ATen/core/Tensor.h>

namespace at {
namespace native {

const Tensor& quantized_copy_from_float_cpu_(const Tensor& self, const Tensor& src);
}
} // namespace at
