#include <THC/THCStorage.hpp>

#include <THC/THCThrustAllocator.cuh>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/system/cuda/execution_policy.h>

#include <TH/THHalf.h>

#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorage.cu>
#include <THC/THCGenerateBFloat16Type.h>
