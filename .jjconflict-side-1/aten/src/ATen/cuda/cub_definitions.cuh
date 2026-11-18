#pragma once

#if !defined(USE_ROCM)
#include <cuda.h> // for CUDA_VERSION
#include <cub/version.cuh>
#else
// Check if we can find HIPCUB_CCCL_VERSION.
#include <hipcub/hipcub_version.hpp>
// Older versions of hipCUB do not support the CUB V3 API.
#if defined(HIPCUB_CCCL_VERSION) && HIPCUB_CCCL_VERSION >= 300000
#define CUB_VERSION HIPCUB_CCCL_VERSION
#else
// If we cannot find a compatible CCCL version, fallback to
// a very old version.
#define CUB_VERSION 200001
#endif
#endif

#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() true

// There were many bc-breaking changes in major version release of CCCL v3.0.0
// Please see https://github.com/NVIDIA/cccl/blob/main/docs/cccl/3.0_migration_guide.rst
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
