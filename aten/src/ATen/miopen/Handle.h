#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

// Note [MIOpen calls during HIP graph capture]
// MIOpen 3.6 (ROCm 10.2) convolution find evaluates naive solvers on its own
// streams and abandons them once they exceed MIOPEN_NAIVE_TIMEOUT_FACTOR
// percent of the best non-naive time. Every later MIOpen kernel launch polls
// those streams with hipStreamQuery and frees their scratch with hipFree; HIP
// rejects both during a global or thread-local stream capture, invalidating
// the capture. Callers hold a relaxed capture-mode guard around MIOpen calls;
// the polled streams and scratch are MIOpen's own and never part of the
// caller's graph. Remove once the MIOpen fix (ROCm/rocm-libraries#12265)
// ships.
TORCH_CUDA_CPP_API miopenHandle_t getMiopenHandle();
} // namespace at::native
