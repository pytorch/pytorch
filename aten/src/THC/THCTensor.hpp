#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <THC/THCTensor.h>
#include <TH/THTensor.hpp>
#include <THC/THCStorage.hpp>
#include <THC/THCGeneral.hpp>

#include <ATen/ATen.h>

TORCH_CUDA_CU_API void THCTensor_setStorage(
    THCState* state,
    THCTensor* self,
    THCStorage* storage_,
    ptrdiff_t storageOffset_,
    at::IntArrayRef size_,
    at::IntArrayRef stride_);
