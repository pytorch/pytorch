#pragma once

#if !defined(USE_ROCM)
#include <cuda.h> // for CUDA_VERSION
#include <cub/version.cuh>
#else
// Check if we can find HIPCUB_CCCL_VERSION. It is exposed
// transitively by hipcub/config.hpp.
#include <hipcub/config.hpp>
// Older versions of hipCUB do not support the CUB V3 API.
#if defined(HIPCUB_CCCL_VERSION) && HIPCUB_CCCL_VERSION >= 300000
#define CUB_VERSION HIPCUB_CCCL_VERSION
#else
// If we cannot find a compatible CCCL version, fallback to
// a very old version.
#define CUB_VERSION 200001
#endif
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
#if CUB_VERSION >= 300400
#define CUB_V3_4_PLUS() true
#define CUB_V3_PLUS() false
#elif CUB_VERSION >= 200800
// CCCL 2.8 introduced new CUB v3 API.
#define CUB_V3_4_PLUS() false
#define CUB_V3_PLUS() true
#else
// Pre CCCL 2.8.
#define CUB_V3_4_PLUS() false
#define CUB_V3_PLUS() false
#endif
