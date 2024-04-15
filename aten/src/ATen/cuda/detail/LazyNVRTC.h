#pragma once
#include <ATen/detail/CUDAHooksInterface.h>
namespace at::cuda {
// Forward-declares at::cuda::NVRTC
struct NVRTC;

namespace detail {
extern NVRTC lazyNVRTC;
} // namespace detail

}  // namespace at::cuda
