#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.hpp"
#else

#include <atomic>

typedef struct THCSTensor
{  // Stored in COO format, indices + values
    int64_t *size;
    ptrdiff_t nnz;
    int nDimensionI; // dimension of indices
    int nDimensionV; // dimension of values

    // 2-D tensor of nDim x nnz of indices. May have nnz dim bigger than nnz
    // as buffer, so we keep track of both
    THCIndexTensor *indices;
    THCTensor *values;
    // Some math operations can only be performed on ordered sparse tensors
    int coalesced;
    std::atomic<int> refcount;

    // NOTE: this returns the "old" TH dimension view where no dimensions represents an empty tensor.
    // There will be a dim() function that gives the new view that supports 0-sized dimensions.
    inline int64_t _dim() const {
      return nDimensionI + nDimensionV;
    }

    inline int64_t dim() const {
      // FixME: nDimensionI and nDimensionV should be set correctly by THCS
      return (nDimensionI + nDimensionV) == 0 ? 1 : (nDimensionI + nDimensionV);
    }
} THCSTensor;

#endif
