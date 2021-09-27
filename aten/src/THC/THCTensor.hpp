#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <THC/THCTensor.h>
#include <TH/THTensor.hpp>
#include <THC/THCStorage.hpp>
#include <THC/THCGeneral.hpp>

#include <ATen/ATen.h>


TORCH_CUDA_CU_API THCTensor* THCTensor_new(
    THCState* state,
    caffe2::TypeMeta type_meta);

TORCH_CUDA_CU_API void THCTensor_resizeNd(
    THCState* state,
    THCTensor* tensor,
    int nDimension,
    const int64_t* size,
    const int64_t* stride);
TORCH_CUDA_CU_API void THCTensor_resizeAs(
    THCState* state,
    THCTensor* tensor,
    THCTensor* src);

TORCH_CUDA_CU_API void THCTensor_setStorage(
    THCState* state,
    THCTensor* self,
    THCStorage* storage_,
    ptrdiff_t storageOffset_,
    at::IntArrayRef size_,
    at::IntArrayRef stride_);

TORCH_CUDA_CU_API void THCTensor_retain(THCState* state, THCTensor* self);
TORCH_CUDA_CU_API void THCTensor_free(THCState* state, THCTensor* self);

TORCH_CUDA_CU_API int THCTensor_getDevice(
    THCState* state,
    const THCTensor* tensor);
