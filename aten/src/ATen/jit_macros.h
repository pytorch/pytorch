#pragma once
#include <ATen/cuda/CUDAConfig.h>
#include <string>

// AT_USE_JITERATOR(), controls whether we jit some elementwise kernels
#define AT_USE_JITERATOR() true
#define jiterator_stringify(...) std::string(#__VA_ARGS__);
