#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <THC/THCStorage.h>
// Should work with THStorageClass
#include <TH/THStorageFunctions.hpp>

#include <c10/core/ScalarType.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

THC_API THCStorage* THCStorage_new(THCState* state);

THC_API void THCStorage_retain(THCState *state, THCStorage *storage);

THC_API void THCStorage_resizeBytes(
    THCState* state,
    THCStorage* storage,
    ptrdiff_t size_bytes);
THC_API int THCStorage_getDevice(THCState* state, const THCStorage* storage);

THC_API THCStorage* THCStorage_newWithDataAndAllocator(
    THCState* state,
    at::DataPtr&& data,
    ptrdiff_t size,
    at::Allocator* allocator);
