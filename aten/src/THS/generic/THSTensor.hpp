#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensor.hpp"
#else

typedef struct THSTensor
{  // Stored in COO format, indices + values
    int64_t *size;
    ptrdiff_t nnz;
    int nDimensionI; // dimension of indices
    int nDimensionV; // dimension of values

    // 2-D tensor of nDim x nnz of indices. May have nnz dim bigger than nnz
    // as buffer, so we keep track of both
    THLongTensor *indices;
    THTensor *values;
    // A sparse tensor is 'coalesced' if every index occurs at most once in
    // the indices tensor, and the indices are in sorted order.
    // Most math operations can only be performed on ordered sparse tensors
    int coalesced;
    std::atomic<int> refcount;

} THSTensor;

#endif
