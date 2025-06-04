#pragma once

#include <ATen/ATen.h>
#include <c10/core/SymIntArrayRef.h>

namespace at::native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size);

Tensor mkldnn_view_symint(const Tensor& self, c10::SymIntArrayRef size);

Tensor mkldnn_clone(const Tensor& self);

} // namespace at
