#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

// expects as input a complex tensor and returns back a float tensor
// containing the complex values in the last two dimensions
Tensor view_complex_as_float(const Tensor& self);

}} // namespace at::native
