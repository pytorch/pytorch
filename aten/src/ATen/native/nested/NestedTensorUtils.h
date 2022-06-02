#pragma once

#include <ATen/NestedTensorImpl.h>
#include <c10/macros/Macros.h>

#include <vector>

namespace at {
namespace native {

struct NestedTensorImpl;
Tensor NestedTensor_to_buffer(const Tensor& self);
Tensor NestedTensor_from_buffer(const Tensor& buffer, const Tensor& shape);

} // namespace native
} // namespace at
