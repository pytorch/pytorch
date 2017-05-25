int THLongStorage_isSameSizeAs(const long *sizeA, long dimsA, const long *sizeB, long dimsB) {
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
