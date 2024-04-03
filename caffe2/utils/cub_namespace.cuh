#pragma once

// cub sort support for CUB_WRAPPED_NAMESPACE is added to cub 1.13.1 in:
// https://github.com/NVIDIA/cub/pull/326
// CUB_WRAPPED_NAMESPACE is defined globally in cmake/Dependencies.cmake
// starting from CUDA 11.5
#if defined(CUB_WRAPPED_NAMESPACE) || defined(THRUST_CUB_WRAPPED_NAMESPACE)
#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() true
#else
#define USE_GLOBAL_CUB_WRAPPED_NAMESPACE() false
#endif

#if USE_GLOBAL_CUB_WRAPPED_NAMESPACE()
namespace caffe2 {
namespace cub = ::CUB_WRAPPED_NAMESPACE::cub;
}
#endif
