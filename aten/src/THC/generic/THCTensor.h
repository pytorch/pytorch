#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensor.h"
#else

#define THCTensor THTensor

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THCTensor type.
#define THCudaTensor                THCTensor
#define THCudaDoubleTensor          THCTensor
#define THCudaHalfTensor            THCTensor
#define THCudaByteTensor            THCTensor
#define THCudaCharTensor            THCTensor
#define THCudaShortTensor           THCTensor
#define THCudaIntTensor             THCTensor
#define THCudaLongTensor            THCTensor
#define THCudaBoolTensor            THCTensor
#define THCudaBFloat16Tensor        THCTensor
#define THCudaComplexFloatTensor    THCTensor
#define THCudaComplexDoubleTensor   THCTensor

/**** access methods ****/
TORCH_CUDA_CU_API THCStorage* THCTensor_(
    storage)(THCState* state, const THCTensor* self);
/**** creation methods ****/
TORCH_CUDA_CU_API THCTensor* THCTensor_(new)(THCState* state);
TORCH_CUDA_CU_API THCTensor* THCTensor_(newWithStorage1d)(
    THCState* state,
    THCStorage* storage_,
    ptrdiff_t storageOffset_,
    int64_t size0_,
    int64_t stride0_);

TORCH_CUDA_CU_API void THCTensor_(free)(THCState* state, THCTensor* self);

#endif
