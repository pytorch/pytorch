#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensor.h"
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
    int refcount;

} THSTensor;

/**** access methods ****/
TH_API int THSTensor_(nDimension)(const THSTensor *self);
TH_API int THSTensor_(nDimensionI)(const THSTensor *self);
TH_API int THSTensor_(nDimensionV)(const THSTensor *self);
TH_API int64_t THSTensor_(size)(const THSTensor *self, int dim);
TH_API ptrdiff_t THSTensor_(nnz)(const THSTensor *self);
TH_API THLongStorage *THSTensor_(newSizeOf)(THSTensor *self);
TH_API THLongTensor *THSTensor_(newIndices)(const THSTensor *self);
TH_API THTensor *THSTensor_(newValues)(const THSTensor *self);

/**** creation methods ****/
TH_API THSTensor *THSTensor_(new)(void);
TH_API THSTensor *THSTensor_(newWithTensor)(THLongTensor *indices, THTensor *values);
TH_API THSTensor *THSTensor_(newWithTensorAndSizeUnsafe)(THLongTensor *indices, THTensor *values, THLongStorage *sizes);
TH_API THSTensor *THSTensor_(newWithTensorAndSize)(THLongTensor *indices, THTensor *values, THLongStorage *sizes);

// Note the second argument is ignored. It exists only to match the signature of THTensor_(new).
TH_API THSTensor *THSTensor_(newWithSize)(THLongStorage *size_, THLongStorage *_ignored);
TH_API THSTensor *THSTensor_(newWithSize1d)(int64_t size0_);
TH_API THSTensor *THSTensor_(newWithSize2d)(int64_t size0_, int64_t size1_);
TH_API THSTensor *THSTensor_(newWithSize3d)(int64_t size0_, int64_t size1_, int64_t size2_);
TH_API THSTensor *THSTensor_(newWithSize4d)(int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

TH_API THSTensor *THSTensor_(newClone)(THSTensor *self);
TH_API THSTensor *THSTensor_(newTranspose)(THSTensor *self, int dimension1_, int dimension2_);

/**** reshaping methods ***/
TH_API THSTensor *THSTensor_(resize)(THSTensor *self, THLongStorage *size);
TH_API THSTensor *THSTensor_(resizeAs)(THSTensor *self, THSTensor *src);
TH_API THSTensor *THSTensor_(resize1d)(THSTensor *self, int64_t size0);
TH_API THSTensor *THSTensor_(resize2d)(THSTensor *self, int64_t size0, int64_t size1);
TH_API THSTensor *THSTensor_(resize3d)(THSTensor *self, int64_t size0, int64_t size1, int64_t size2);
TH_API THSTensor *THSTensor_(resize4d)(THSTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3);

TH_API THTensor *THSTensor_(toDense)(THSTensor *self);
TH_API void THSTensor_(copy)(THSTensor *self, THSTensor *src);

TH_API void THSTensor_(transpose)(THSTensor *self, int dimension1_, int dimension2_);
TH_API int THSTensor_(isCoalesced)(const THSTensor *self);
TH_API int THSTensor_(isSameSizeAs)(const THSTensor *self, const THSTensor *src);
TH_API THSTensor *THSTensor_(newCoalesce)(THSTensor *self);

TH_API void THTensor_(sparseMask)(THSTensor *r_, THTensor *t, THSTensor *mask);

TH_API void THSTensor_(free)(THSTensor *self);
TH_API void THSTensor_(retain)(THSTensor *self);


/* TODO (check function signatures too, might be wrong)
TH_API void THSTensor_(freeCopyTo)(THSTensor *self, THSTensor *dst);

TH_API void THSTensor_(narrow)(THSTensor *self, THSTensor *src, int dimension_, int64_t firstIndex_, int64_t size_);
TH_API void THSTensor_(select)(THSTensor *self, THSTensor *src, int dimension_, int64_t sliceIndex_);
*/

// internal methods
TH_API THSTensor* THSTensor_(rawResize)(THSTensor *self, int nDimI, int nDimV, int64_t *size);
THSTensor* THSTensor_(_move)(THSTensor *self, THLongTensor *indices, THTensor *values);
THSTensor* THSTensor_(_set)(THSTensor *self, THLongTensor *indices, THTensor *values);

#endif
