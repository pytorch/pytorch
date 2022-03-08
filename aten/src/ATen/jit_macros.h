#pragma once
#include <ATen/cuda/CUDAConfig.h>
#include <c10/macros/Macros.h>
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

#if !AT_ROCM_ENABLED() && !defined(__CUDACC__)
    // CPU only case
    // Only needs the function
    #define CODE(...) __VA_ARGS__
    #define jiterator_code_stringify(code, str_name) code
#elif AT_ROCM_ENABLED()
    // ROCm case
    // Only needs the function with __host__ __device__ attribute
    #define CODE(...) __VA_ARGS__
    #define jiterator_code_stringify(code, str_name) C10_HOST_DEVICE code
#else
    // CUDA case
    // CODE and stringify are helper macro to deal with `,` in code which preprocessor
    // splits into multiple arguments.
    #define CODE(...) __VA_ARGS__
    #define stringify(...) #__VA_ARGS__
    #define jiterator_code_stringify(code, str_name) \
        code \
        const std::string str_name = std::string(stringify(code));
#endif
