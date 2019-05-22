#pragma once

#include <ATen/core/Tensor.h>

namespace at {
namespace native {

Tensor& quantized_copy_(Tensor& self, const Tensor& src);

}
}
