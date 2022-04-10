#pragma once
#include <ATen/cuda/CUDAConfig.h>
#include <string>

// AT_USE_JITERATOR(), controls whether we jit some elementwise kernels
// Currently unsupported on ROCm GPUs
#if !AT_ROCM_ENABLED()
    #define AT_USE_JITERATOR() true
    #define jiterator_stringify(...) std::string(#__VA_ARGS__);
#else
    #define AT_USE_JITERATOR() false
    #define jiterator_stringify(...) static_assert(false, "Jiterator is not supported on ROCm");
#endif // USE_ROCM
