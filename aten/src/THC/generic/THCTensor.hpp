#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensor.hpp"
#else

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

// NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
// new functions in here, they should probably be un-genericized.

THC_API void THCTensor_(setStorage)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                                    at::IntArrayRef size_, at::IntArrayRef stride_);
/* strides.data() might be nullptr */
THC_API THCTensor *THCTensor_(newWithStorage)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                                              at::IntArrayRef sizes, at::IntArrayRef strides);

THC_API void THCTensor_(resize)(THCState *state, THCTensor *self, at::IntArrayRef size, at::IntArrayRef stride);
// this is not an actual function, but removing it seems to cause compilationproblems on Windows + CUDA.
THC_API THCTensor *THCTensor_(newWithSize)(THCState *state);

#endif
