#pragma once

#include "ATen/cuda/ATenCUDAGeneral.h"
#include "ATen/Half.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace at {
template <> AT_CUDA_API half convert(Half aten_half);
template <> AT_CUDA_API Half convert(half cuda_half);
template <> AT_CUDA_API half convert(double value);
#if CUDA_VERSION >= 9000 || defined(__HIP_PLATFORM_HCC__)
template <> __half HalfFix(Half h);
template <> Half HalfFix(__half h);
#endif
} // namespace at
