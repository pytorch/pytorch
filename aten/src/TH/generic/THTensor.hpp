#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensor.hpp"
#else

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

// NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
// new functions in here, they should probably be un-genericized.

TH_CPP_API void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                                      at::IntArrayRef size_, at::IntArrayRef stride_);
/* strides.data() might be NULL */
TH_CPP_API THTensor *THTensor_(newWithStorage)(THStorage *storage, ptrdiff_t storageOffset,
                                               at::IntArrayRef sizes, at::IntArrayRef strides);

TH_CPP_API void THTensor_(resize)(THTensor *self, at::IntArrayRef size, at::IntArrayRef stride);
TH_CPP_API THTensor *THTensor_(newWithSize)(at::IntArrayRef size, at::IntArrayRef stride);

#endif
