#include <THC/THCStorage.hpp>

#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 7000) || defined(USE_ROCM)
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateByteType.h>
