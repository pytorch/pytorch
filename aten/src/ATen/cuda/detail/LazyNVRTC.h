#pragma once
#include <ATen/detail/CUDAHooksInterface.h>

// jiterator headers (temporarily parked here)
#include <torch/csrc/jit/frontend/code_template.h>

namespace at { namespace cuda {
// Forward-declares at::cuda::NVRTC
struct NVRTC;

namespace detail {
extern NVRTC lazyNVRTC;

// jiterator functions (temporarily parked here)

// Loads the file at the specified path and returns it as a CodeTemplate
torch::jit::CodeTemplate load_code_template(const std::string& path);
}

}}  // at::cuda::detail
