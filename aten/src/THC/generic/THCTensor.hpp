#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensor.hpp"
#else

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

// NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
// new functions in here, they should probably be un-genericized.

TORCH_CUDA_CU_API void THCTensor_(setStorage)(
    THCState* state,
    THCTensor* self,
    THCStorage* storage_,
    ptrdiff_t storageOffset_,
    at::IntArrayRef size_,
    at::IntArrayRef stride_);

TORCH_CUDA_CU_API void THCTensor_(resize)(
    THCState* state,
    THCTensor* self,
    at::IntArrayRef size,
    at::IntArrayRef stride);

#endif
