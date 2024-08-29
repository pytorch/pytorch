#pragma once

#ifdef USE_C10D_NCCL

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

namespace c10d {

// Check for NaNs in a tensor on a given stream. If any are found, throw a
// device-side error.
void checkForNan(const at::Tensor& tensor, at::cuda::CUDAStream& stream);

} // namespace c10d

#endif // USE_C10D_NCCL
