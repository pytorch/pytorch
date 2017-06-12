#include "THSize.h"

int THSize_isSameSizeAs(const long *sizeA, long dimsA, const long *sizeB, long dimsB) {
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

ptrdiff_t THSize_nElement(long dims, long *size) {
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
