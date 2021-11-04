#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorageCopy.h"
#else

/* Support for copy between different Storage types */

TORCH_CUDA_CU_API void THCStorage_(
    copy)(THCState* state, THCStorage* storage, THCStorage* src);
TORCH_CUDA_CU_API void THCStorage_(
    copyByte)(THCState* state, THCStorage* storage, struct THByteStorage* src);

TORCH_CUDA_CU_API void THCStorage_(copyCudaByte)(
    THCState* state,
    THCStorage* storage,
    struct THCudaByteStorage* src);

TORCH_CUDA_CU_API void TH_CONCAT_2(
    THByteStorage_copyCuda,
    Real)(THCState* state, THByteStorage* self, struct THCStorage* src);

TORCH_CUDA_CU_API void THStorage_(
    copyCuda)(THCState* state, THStorage* self, THCStorage* src);
TORCH_CUDA_CU_API void THCStorage_(
    copyCuda)(THCState* state, THCStorage* self, THCStorage* src);
TORCH_CUDA_CU_API void THCStorage_(
    copyCPU)(THCState* state, THCStorage* self, THStorage* src);

#endif
