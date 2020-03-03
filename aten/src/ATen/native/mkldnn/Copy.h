#pragma once

#include <ATen/core/Tensor.h>

namespace at {
namespace native {

Tensor& mkldnn_copy_from(Tensor& self, const Tensor& src);

}
} // namespace at
