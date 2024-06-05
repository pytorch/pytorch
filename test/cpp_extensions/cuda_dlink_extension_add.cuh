#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

__device__ void add(const float* a, const float* b, float* output);
