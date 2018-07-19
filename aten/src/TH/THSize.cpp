#include "THSize.h"

int THSize_isSameSizeAs(const int64_t *sizeA, int64_t dimsA, const int64_t *sizeB, int64_t dimsB) {
  int d;
  if (dimsA != dimsB)
    return 0;
  for(d = 0; d < dimsA; ++d)
  {
    if(sizeA[d] != sizeB[d])
      return 0;
  }
  return 1;
}

ptrdiff_t THSize_nElement(int64_t dims, int64_t *size) {
  if(dims == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < dims; d++)
      nElement *= size[d];
    return nElement;
  }
}
