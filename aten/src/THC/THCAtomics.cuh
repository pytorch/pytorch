#pragma once
// TODO: Remove once torchvision has been updated to use the ATen header
#ifdef __CUDACC__
#include <ATen/cuda/Atomic.cuh>
#else
#include <ATen/hip/Atomic.cuh>
#endif
