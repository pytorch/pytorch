#pragma once
#include <ATen/detail/CUDAHooksInterface.h>
namespace at { namespace cuda {
// Forward-declares at::cuda::NVRTC
struct NVRTC;

namespace detail {
extern NVRTC lazyNVRTC;
}

}}  // at::cuda::detail
