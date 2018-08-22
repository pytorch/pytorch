#ifndef TH_SIZE_INC
#define TH_SIZE_INC

#include <TH/THGeneral.h>
#include <stddef.h>

// THTensor functions that would work on a THSize if we had such a class in C++,
// i.e. THTensor functions that depend only on the shape of the tensor, not the type.

TH_API int THSize_isSameSizeAs(const int64_t *sizeA, int64_t dimsA, const int64_t *sizeB, int64_t dimsB);
TH_API ptrdiff_t THSize_nElement(int64_t dims, int64_t *size);

#endif
