#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensor.hpp"
#else

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

// NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
// new functions in here, they should probably be un-genericized.

TH_CPP_API void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                                      at::IntListRef size_, at::IntListRef stride_);
TH_CPP_API THTensor *THTensor_(newView)(THTensor *tensor, at::IntListRef size);
/* strides.data() might be NULL */
TH_CPP_API THTensor *THTensor_(newWithStorage)(THStorage *storage, ptrdiff_t storageOffset,
                                               at::IntListRef sizes, at::IntListRef strides);

TH_CPP_API void THTensor_(resize)(THTensor *self, at::IntListRef size, at::IntListRef stride);
TH_CPP_API THTensor *THTensor_(newWithSize)(at::IntListRef size, at::IntListRef stride);

#endif
