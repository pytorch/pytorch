#pragma once

#include <ATen/Tensor.h>

namespace at {
namespace native {

Tensor& quantized_copy_from_float_(Tensor& self, const Tensor& src);

}
}
