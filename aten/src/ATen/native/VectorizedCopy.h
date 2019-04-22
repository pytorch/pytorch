#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor vectorized_contig_tensor_repeat(const Tensor& self, IntArrayRef repeats);

} // native
} // at
