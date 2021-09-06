#pragma once

#include <ATen/core/Tensor.h>

namespace at {
namespace native {

Tensor& quantized_copy_from_float_cpu_(Tensor& self, const Tensor& src);
Tensor& copy_quantized_cpu_(Tensor& self, const Tensor& src);
Tensor& copy_quantized_gpu_(Tensor& self, const Tensor& src);
Tensor& copy_quantized_xpu_(Tensor& self, const Tensor& src);
}
} // namespace at
