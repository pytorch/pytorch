#pragma once

#include <ATen/ATen.h>
#include <c10/core/SymIntArrayRef.h>

namespace at::native {

Tensor onednn_view(const Tensor& self, IntArrayRef size);

Tensor onednn_view_symint(const Tensor& self, c10::SymIntArrayRef size);

Tensor onednn_clone(const Tensor& self);

} // namespace at
