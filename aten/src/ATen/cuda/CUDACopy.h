#pragma once

#include "ATen/ATen.h"

namespace at {
namespace cuda {

Tensor copy_cuda(Tensor& dst, const Tensor& src);

} // namespace cuda
} // namespace at
