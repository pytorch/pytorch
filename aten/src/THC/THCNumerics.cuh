#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#include "ATen/cuda/CUDANumerics.cuh"

template <typename T>
using THCNumerics = at::cuda::CUDANumerics<T>;

using at::cuda::ScalarConvert;
using at::cuda::scalar_cast;

#endif // THC_NUMERICS_INC
