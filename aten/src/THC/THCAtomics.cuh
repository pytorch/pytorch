#pragma once
// TODO: Remove once torchvision has been updated to use the ATen header
#ifdef __CUDACC__
#include <ATen/cuda/Atomics.cuh>
#else
#include <ATen/hip/Atomics.cuh>
#endif
