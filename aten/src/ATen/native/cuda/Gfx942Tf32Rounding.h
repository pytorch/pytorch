#pragma once

#include <ATen/core/Tensor.h>

#ifdef USE_ROCM
namespace at::native {

Tensor round_gfx942_tf32_rne_cuda(const Tensor& input);

} // namespace at::native
#endif // USE_ROCM
