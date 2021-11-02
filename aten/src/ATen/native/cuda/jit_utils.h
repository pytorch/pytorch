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

// Silly functions
//std::string silly_generate_code();

// void launch_silly_jitted_pwise_function(
//     NvrtcFunction function,
//     std::array<void*, 4>& args,
//     const int nBlocks,
//     const int kBlockSize);

// Serious functions
std::string generate_code(
    int nTensors,
    const std::string& func,
    const std::string& name,
    const std::string& common_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    bool vectorized=false,
    int vec_size=0);
NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name);
void launch_jitted_pwise_function(
    NvrtcFunction function,
    std::array<void*, 6>& args,
    const int nBlocks,
    const int kBlockSize);

template <typename T> inline std::string typeName(){
    TORCH_INTERNAL_ASSERT(false, "invalid type");
    return "void";
}

#define TYPE_NAME_FN(ctype, name) \
template <> inline std::string typeName<ctype>(){ \
    return std::string(#ctype);    \
}

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(TYPE_NAME_FN)


} // namespace jit
}
}
