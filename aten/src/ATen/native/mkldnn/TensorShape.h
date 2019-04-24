#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size);

} // namespace native
} // namespace at
