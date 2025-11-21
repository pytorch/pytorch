#pragma once

#if !defined(USE_ROCM)
#include <cuda.h>  // for CUDA_VERSION
#endif

#if !defined(USE_ROCM)
#include <cub/version.cuh>
#else
#define CUB_VERSION 200001
#endif

// cub support for CUB_WRAPPED_NAMESPACE is added to cub 1.13.1 in:
// https://github.com/NVIDIA/cub/pull/326
// CUB_WRAPPED_NAMESPACE is defined globally in cmake/Dependencies.cmake
// starting from CUDA 11.5
#if defined(CUB_WRAPPED_NAMESPACE) || defined(THRUST_CUB_WRAPPED_NAMESPACE)
#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() true
#else
#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() false
#endif

// There were many bc-breaking changes in major version release of CCCL v3.0.0
// Please see https://nvidia.github.io/cccl/cccl/3.0_migration_guide.html
#if CUB_VERSION >= 200800
#define CUB_V3_PLUS() true
#else
#define CUB_V3_PLUS() false
#endif
