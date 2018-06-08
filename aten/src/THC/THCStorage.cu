#include "THCStorage.hpp"

#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#if CUDA_VERSION >= 7000 || defined(__HIP_PLATFORM_HCC__)
#include <thrust/system/cuda/execution_policy.h>
#endif

#include "THCHalf.h"

#include "generic/THCStorage.cu"
#include "THCGenerateAllTypes.h"
