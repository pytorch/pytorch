#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.hpp"
#else

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

// NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
// new functions in here, they should probably be un-genericized.

THC_API void THCTensor_(setStorage)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                                    at::IntList size_, at::IntList stride_);
THC_API THCTensor *THCTensor_(newView)(THCState *state, THCTensor *tensor, at::IntList size);
/* strides.data() might be nullptr */
THC_API THCTensor *THCTensor_(newWithStorage)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                                              at::IntList sizes, at::IntList strides);


#endif
