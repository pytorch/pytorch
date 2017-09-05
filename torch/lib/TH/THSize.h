#ifndef TH_SIZE_INC
#define TH_SIZE_INC

#include "THGeneral.h"
#include <stddef.h>

// THTensor functions that would work on a THSize if we had such a class in C++,
// i.e. THTensor functions that depend only on the shape of the tensor, not the type.

TH_API int THSize_isSameSizeAs(const long *sizeA, long dimsA, const long *sizeB, long dimsB);
TH_API ptrdiff_t THSize_nElement(long dims, long *size);

#endif
