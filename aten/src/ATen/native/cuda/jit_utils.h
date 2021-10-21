#pragma once
#include <string>
#include <ATen/cuda/detail/LazyNVRTC.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>


namespace at{

namespace cuda{

namespace jit {

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = nullptr;
};
std::string generate_code(int nTensors, bool contiguous, bool dynamic_casting);
NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name);
void launch_jitted_pwise_function(
    NvrtcFunction function,
    std::array<void*, 7>& args,
    const int nBlocks,
    const int kBlockSize);
} // namespace jit
}
}
