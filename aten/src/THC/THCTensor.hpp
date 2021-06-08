#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <THC/THCTensor.h>
#include <TH/THTensor.hpp>
#include <THC/THCStorage.hpp>
#include <THC/THCGeneral.hpp>

#include <atomic>
#include <ATen/ATen.h>

// See [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
TORCH_CUDA_CU_API int THCTensor_nDimension(
    THCState* state,
    const THCTensor* self);
TORCH_CUDA_CU_API int THCTensor_nDimensionLegacyNoScalars(
    THCState* state,
    const THCTensor* self);
TORCH_CUDA_CU_API int THCTensor_nDimensionLegacyAll(
    THCState* state,
    const THCTensor* self);

TORCH_CUDA_CU_API int64_t
THCTensor_size(THCState* state, const THCTensor* self, int dim);
TORCH_CUDA_CU_API int64_t
THCTensor_sizeLegacyNoScalars(THCState* state, const THCTensor* self, int dim);
TORCH_CUDA_CU_API int64_t
THCTensor_stride(THCState* state, const THCTensor* self, int dim);
TORCH_CUDA_CU_API int64_t THCTensor_strideLegacyNoScalars(
    THCState* state,
    const THCTensor* self,
    int dim);

TORCH_CUDA_CU_API THCTensor* THCTensor_new(
    THCState* state,
    caffe2::TypeMeta type_meta);

TORCH_CUDA_CU_API void THCTensor_resize(
    THCState* state,
    THCTensor* tensor,
    at::IntArrayRef size,
    at::IntArrayRef stride);
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

TORCH_CUDA_CU_API void THCTensor_set(
    THCState* state,
    THCTensor* self,
    THCTensor* src);
TORCH_CUDA_CU_API void THCTensor_setStorage(
    THCState* state,
    THCTensor* self,
    THCStorage* storage_,
    ptrdiff_t storageOffset_,
    at::IntArrayRef size_,
    at::IntArrayRef stride_);

TORCH_CUDA_CU_API void THCTensor_squeeze1d(
    THCState* state,
    THCTensor* self,
    THCTensor* src,
    int dimension_);
TORCH_CUDA_CU_API void THCTensor_unsqueeze1d(
    THCState* state,
    THCTensor* self,
    THCTensor* src,
    int dimension_);

TORCH_CUDA_CU_API bool THCTensor_allContiguous(
    THCState* state,
    THCTensor** inputs,
    int numInputs);
TORCH_CUDA_CU_API ptrdiff_t
THCTensor_nElement(THCState* state, const THCTensor* self);

TORCH_CUDA_CU_API void THCTensor_retain(THCState* state, THCTensor* self);
TORCH_CUDA_CU_API void THCTensor_free(THCState* state, THCTensor* self);

TORCH_CUDA_CU_API int THCTensor_getDevice(
    THCState* state,
    const THCTensor* tensor);
TORCH_CUDA_CU_API bool THCTensor_allSameDevice(
    THCState* state,
    THCTensor** inputs,
    int numInputs);

/* Can we use 32 bit math for indexing? */
TORCH_CUDA_CU_API bool THCTensor_canUse32BitIndexMath(
    THCState* state,
    const THCTensor* t,
    ptrdiff_t max_elem = INT32_MAX);
/* Are all tensors 32-bit indexable? */
TORCH_CUDA_CU_API bool THCTensor_all32BitIndexable(
    THCState* state,
    THCTensor** inputs,
    int numInputs);
TORCH_CUDA_CU_API void THCTensor_preserveReduceDimSemantics(
    THCState* state,
    THCTensor* tensor,
    int in_dims,
    int64_t dimension,
    int keepdim);
/* Returns false if there is no possibility that the tensor    */
/* has more than one index that references the same datapoint, */
/* true otherwise.                                             */
TORCH_CUDA_CU_API bool THCTensor_maybeOverlappingIndices(
    THCState* state,
    const THCTensor* t);

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateBFloat16Type.h>
