#pragma once

// USE_JITERATOR, controls whether we jit some elementwise kernels
// Currently unsupported on ROCm GPUs
#ifndef USE_ROCM
    #define USE_JITERATOR true
    #define jiterator_stringify(...) std::string(#__VA_ARGS__);
#else
    #define jiterator_stringify(...) std::string("USE_JITERATOR is undefined");
#endif // USE_ROCM

// USE_JITERATOR_WITH_CACHE, controls whether jitted kernels are cached
// Currently unsupported on Windows
#ifndef _WIN32
    #define USE_JITERATOR_WITH_CACHE true
#endif // _WIN32

// #define JITERATOR_CACHE_PATH "~/.cache/torch/kernels/"

#define JITERATOR_CACHE_PATH "/private/home/mruberry/"
