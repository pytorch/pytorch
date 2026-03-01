#pragma once

#if !defined(USE_ROCM)
#include <cuda.h>  // for CUDA_VERSION
#endif

#if !defined(USE_ROCM)
#include <cub/version.cuh>
#else
#define CUB_VERSION 200001
#endif

#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() true

// There were many bc-breaking changes in major version release of CCCL v3.0.0
// Please see https://nvidia.github.io/cccl/cccl/3.0_migration_guide.html
#if CUB_VERSION >= 200800
#define CUB_V3_PLUS() true
#else
#define CUB_V3_PLUS() false
#endif
