#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorage.h"
#else

#define THCStorage THStorage

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THCStorage type.
#define THCudaStorage                       THCStorage
#define THCudaDoubleStorage                 THCStorage
#define THCudaHalfStorage                   THCStorage
#define THCudaByteStorage                   THCStorage
#define THCudaCharStorage                   THCStorage
#define THCudaShortStorage                  THCStorage
#define THCudaIntStorage                    THCStorage
#define THCudaLongStorage                   THCStorage
#define THCudaBoolStorage                   THCStorage
#define THCudaBFloat16Storage               THCStorage
#define THCudaComplexFloatStorage           THCStorage
#define THCudaComplexDoubleStorage          THCStorage

TORCH_CUDA_CU_API scalar_t* THCStorage_(
    data)(THCState* state, const THCStorage*);
TORCH_CUDA_CU_API int THCStorage_(elementSize)(THCState* state);

/* slow access -- checks everything */
TORCH_CUDA_CU_API void THCStorage_(
    set)(THCState* state, THCStorage*, ptrdiff_t, scalar_t);
TORCH_CUDA_CU_API scalar_t
    THCStorage_(get)(THCState* state, const THCStorage*, ptrdiff_t);

TORCH_CUDA_CU_API THCStorage* THCStorage_(new)(THCState* state);
TORCH_CUDA_CU_API THCStorage* THCStorage_(
    newWithSize)(THCState* state, ptrdiff_t size);
TORCH_CUDA_CU_API THCStorage* THCStorage_(
    newWithSize1)(THCState* state, scalar_t);
TORCH_CUDA_CU_API THCStorage* THCStorage_(newWithMapping)(
    THCState* state,
    const char* filename,
    ptrdiff_t size,
    int shared);

TORCH_CUDA_CU_API THCStorage* THCStorage_(newWithAllocator)(
    THCState* state,
    ptrdiff_t size,
    at::Allocator* allocator);
TORCH_CUDA_CU_API THCStorage* THCStorage_(newWithDataAndAllocator)(
    THCState* state,
    at::DataPtr&& data,
    ptrdiff_t size,
    at::Allocator* allocator);

TORCH_CUDA_CU_API void THCStorage_(
    setFlag)(THCState* state, THCStorage* storage, const char flag);
TORCH_CUDA_CU_API void THCStorage_(
    clearFlag)(THCState* state, THCStorage* storage, const char flag);
TORCH_CUDA_CU_API void THCStorage_(
    retain)(THCState* state, THCStorage* storage);

TORCH_CUDA_CU_API void THCStorage_(free)(THCState* state, THCStorage* storage);
TORCH_CUDA_CU_API void THCStorage_(
    resizeBytes)(THCState* state, THCStorage* storage, ptrdiff_t size_bytes);
TORCH_CUDA_CU_API void THCStorage_(
    fill)(THCState* state, THCStorage* storage, scalar_t value);

TORCH_CUDA_CU_API int THCStorage_(
    getDevice)(THCState* state, const THCStorage* storage);

#endif
