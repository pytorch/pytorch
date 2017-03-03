#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.h"
#else

typedef struct THCSTensor
{  // Stored in COO format, indices + values
    long *size;
    ptrdiff_t nnz;
    int nDimensionI; // dimension of indices
    int nDimensionV; // dimension of values

    // 2-D tensor of nDim x nnz of indices. May have nnz dim bigger than nnz
    // as buffer, so we keep track of both
    THCIndexTensor *indices;
    THCTensor *values;
    // Math operations can only be performed on ordered sparse tensors
    int contiguous;
    int refcount;

} THCSTensor;

/**** access methods ****/
TH_API int THCSTensor_(nDimension)(THCState *state, const THCSTensor *self);
TH_API int THCSTensor_(nDimensionI)(THCState *state, const THCSTensor *self);
TH_API int THCSTensor_(nDimensionV)(THCState *state, const THCSTensor *self);
TH_API long THCSTensor_(size)(THCState *state, const THCSTensor *self, int dim);
TH_API ptrdiff_t THCSTensor_(nnz)(THCState *state, const THCSTensor *self);
TH_API THLongStorage *THCSTensor_(newSizeOf)(THCState *state, THCSTensor *self);
TH_API THCIndexTensor *THCSTensor_(indices)(THCState *state, const THCSTensor *self);
TH_API THCTensor *THCSTensor_(values)(THCState *state, const THCSTensor *self);

/**** creation methods ****/
TH_API THCSTensor *THCSTensor_(new)(THCState *state);
TH_API THCSTensor *THCSTensor_(newWithTensor)(THCState *state, THCIndexTensor *indices, THCTensor *values);
TH_API THCSTensor *THCSTensor_(newWithTensorAndSize)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes);

TH_API THCSTensor *THCSTensor_(newWithSize)(THCState *state, THLongStorage *size_);
TH_API THCSTensor *THCSTensor_(newWithSize1d)(THCState *state, long size0_);
TH_API THCSTensor *THCSTensor_(newWithSize2d)(THCState *state, long size0_, long size1_);
TH_API THCSTensor *THCSTensor_(newWithSize3d)(THCState *state, long size0_, long size1_, long size2_);
TH_API THCSTensor *THCSTensor_(newWithSize4d)(THCState *state, long size0_, long size1_, long size2_, long size3_);

TH_API THCSTensor *THCSTensor_(newClone)(THCState *state, THCSTensor *self);
TH_API THCSTensor *THCSTensor_(newContiguous)(THCState *state, THCSTensor *self);
TH_API THCSTensor *THCSTensor_(newTranspose)(THCState *state, THCSTensor *self, int dimension1_, int dimension2_);

/**** reshaping methods ***/
TH_API THCSTensor *THCSTensor_(resize)(THCState *state, THCSTensor *self, THLongStorage *size);
TH_API THCSTensor *THCSTensor_(resizeAs)(THCState *state, THCSTensor *self, THCSTensor *src);
TH_API THCSTensor *THCSTensor_(resize1d)(THCState *state, THCSTensor *self, long size0);
TH_API THCSTensor *THCSTensor_(resize2d)(THCState *state, THCSTensor *self, long size0, long size1);
TH_API THCSTensor *THCSTensor_(resize3d)(THCState *state, THCSTensor *self, long size0, long size1, long size2);
TH_API THCSTensor *THCSTensor_(resize4d)(THCState *state, THCSTensor *self, long size0, long size1, long size2, long size3);

TH_API THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self);

TH_API void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int dimension1_, int dimension2_);
TH_API int THCSTensor_(isContiguous)(THCState *state, const THCSTensor *self);
TH_API void THCSTensor_(contiguous)(THCState *state, THCSTensor *self);

TH_API void THCTensor_(sparseMask)(THCState *state, THCSTensor *r_, THCTensor *t, THCSTensor *mask);

TH_API void THCSTensor_(free)(THCState *state, THCSTensor *self);
TH_API void THCSTensor_(retain)(THCState *state, THCSTensor *self);

/* CUDA-specific functions */
TH_API int THCSTensor_(getDevice)(THCState *state, const THCSTensor *self);

#endif
