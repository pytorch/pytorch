#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THCStorage.h"
// Should work with THStorageClass
#include <TH/THStorageFunctions.hpp>

#include "ATen/ScalarType.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace at {

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_HCC__)
template <>
struct CTypeToScalarType<__half> : public CTypeToScalarType<Half> {};
#endif

}

THC_API THCStorage* THCStorage_new(THCState* state, caffe2::TypeMeta);

THC_API void THCStorage_retain(THCState *state, THCStorage *storage);

THC_API void THCStorage_resize(THCState *state, THCStorage *storage, ptrdiff_t size);
THC_API int THCStorage_getDevice(THCState* state, const THCStorage* storage);

THC_API THCStorage* THCStorage_newWithDataAndAllocator(
  THCState *state, at::ScalarType scalar_type,
  at::DataPtr&& data, ptrdiff_t size,
  at::Allocator* allocator);
