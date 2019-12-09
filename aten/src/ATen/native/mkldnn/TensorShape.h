#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size);

Tensor mkldnn_clone(const Tensor& self);

} // namespace native
} // namespace at
