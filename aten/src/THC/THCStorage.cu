#include <THC/THCStorage.hpp>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <THC/THCThrustAllocator.cuh>
#if CUDA_VERSION >= 7000 || defined(__HIP_PLATFORM_HCC__)
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <TH/THHalf.h>

// clang-format off
#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateBFloat16Type.h>
// clang-format on
