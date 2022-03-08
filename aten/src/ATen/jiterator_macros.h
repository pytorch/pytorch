#pragma once
#include <c10/macros/Macros.h>
#include <string>

// jiterator_code_stringify macro is used to define code (for CPU/ROCm)
// and generate code string for `jiterator` (only when compiling for CUDA).
// Usage :
//      jiterator_code_stringify(
//          code(template <typename T> T identity(T x) { return x; }),
//          identity_string);
// This will define the function `identity` as present in code and
// also define `std::string identity_string` if this is being compiled
// for CUDA.

// `jiterator_code` macro is to deal with `,` in the kernel code.
// These `,`s confuse the preprocessor into thinking we are passing
// multiple arguments to the macro.
#define jiterator_code(...) __VA_ARGS__
#if defined(__HIPCC__)
    // CPU and ROCm case
    // Needs the function with __host__ __device__ attribute
    #define jiterator_code_stringify(code, str_name) C10_HOST_DEVICE code
#elif defined(__CUDACC__)
    // CPU and CUDA case
    #define stringify_code(...) #__VA_ARGS__
    #define jiterator_code_stringify(code, str_name)                    \
        code /* define the function */                                  \
        const std::string str_name = std::string(stringify_code(code));
#else
    // CPU only case
    // Only needs the function
    #define jiterator_code_stringify(code, str_name) code
#endif
