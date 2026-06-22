#pragma once
#include <ATen/cuda/CUDAConfig.h>
#include <string>

// AT_USE_JITERATOR(), controls whether we jit some elementwise kernels.
// AT_DISABLE_JITERATOR is set by CMake when jiterator should be off.
// Currently set under USE_ROCM + USE_ASAN, because jiterator JITs kernels
// through hiprtc and we haven't set up an ASAN-aware hiprtc runtime.
#ifdef AT_DISABLE_JITERATOR
#define AT_USE_JITERATOR() false
#else
#define AT_USE_JITERATOR() true
#endif
#define jiterator_stringify(...) std::string(#__VA_ARGS__);
