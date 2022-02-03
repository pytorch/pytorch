#pragma once

// USE_JITERATOR, controls whether we jit some elementwise kernels
// Currently unsupported on ROCm GPUs
#ifndef USE_ROCM
    #define USE_JITERATOR true
    #define jiterator_stringify(...) std::string(#__VA_ARGS__);
#else
    // TODO: update this to become a static assertion
    #define jiterator_stringify(...) std::string("USE_JITERATOR is undefined");
#endif // USE_ROCM
