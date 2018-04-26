#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.h"
#else

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
    int refcount;

} THCSTensor;

/**** access methods ****/
TH_API int THCSTensor_(nDimension)(THCState *state, const THCSTensor *self);
TH_API int THCSTensor_(nDimensionI)(THCState *state, const THCSTensor *self);
TH_API int THCSTensor_(nDimensionV)(THCState *state, const THCSTensor *self);
TH_API int64_t THCSTensor_(size)(THCState *state, const THCSTensor *self, int dim);
TH_API ptrdiff_t THCSTensor_(nnz)(THCState *state, const THCSTensor *self);
TH_API THLongStorage *THCSTensor_(newSizeOf)(THCState *state, THCSTensor *self);
TH_API THCIndexTensor *THCSTensor_(newIndices)(THCState *state, const THCSTensor *self);
TH_API THCTensor *THCSTensor_(newValues)(THCState *state, const THCSTensor *self);

/**** creation methods ****/
TH_API THCSTensor *THCSTensor_(new)(THCState *state);
TH_API THCSTensor *THCSTensor_(newWithTensor)(THCState *state, THCIndexTensor *indices, THCTensor *values);
TH_API THCSTensor *THCSTensor_(newWithTensorAndSizeUnsafe)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes);
TH_API THCSTensor *THCSTensor_(newWithTensorAndSize)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes);

TH_API THCSTensor *THCSTensor_(newWithSize)(THCState *state, THLongStorage *size_, THLongStorage *_ignored);
TH_API THCSTensor *THCSTensor_(newWithSize1d)(THCState *state, int64_t size0_);
TH_API THCSTensor *THCSTensor_(newWithSize2d)(THCState *state, int64_t size0_, int64_t size1_);
TH_API THCSTensor *THCSTensor_(newWithSize3d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_);
TH_API THCSTensor *THCSTensor_(newWithSize4d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

TH_API THCSTensor *THCSTensor_(newClone)(THCState *state, THCSTensor *self);
TH_API THCSTensor *THCSTensor_(newTranspose)(THCState *state, THCSTensor *self, int dimension1_, int dimension2_);

/**** reshaping methods ***/
TH_API int THCSTensor_(isSameSizeAs)(THCState *state, const THCSTensor *self, const THCSTensor* src);
TH_API int THCSTensor_(isSameSizeAsDense)(THCState *state, const THCSTensor *self, const THCTensor* src);
TH_API THCSTensor *THCSTensor_(resize)(THCState *state, THCSTensor *self, THLongStorage *size);
TH_API THCSTensor *THCSTensor_(resizeAs)(THCState *state, THCSTensor *self, THCSTensor *src);
TH_API THCSTensor *THCSTensor_(resize1d)(THCState *state, THCSTensor *self, int64_t size0);
TH_API THCSTensor *THCSTensor_(resize2d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1);
TH_API THCSTensor *THCSTensor_(resize3d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2);
TH_API THCSTensor *THCSTensor_(resize4d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3);

TH_API THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self);
TH_API void THCSTensor_(copy)(THCState *state, THCSTensor *self, THCSTensor *src);

TH_API void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int dimension1_, int dimension2_);
TH_API int THCSTensor_(isCoalesced)(THCState *state, const THCSTensor *self);
TH_API THCSTensor *THCSTensor_(newCoalesce)(THCState *state, THCSTensor *self);

TH_API void THCTensor_(sparseMask)(THCState *state, THCSTensor *r_, THCTensor *t, THCSTensor *mask);

TH_API void THCSTensor_(free)(THCState *state, THCSTensor *self);
TH_API void THCSTensor_(retain)(THCState *state, THCSTensor *self);

/* CUDA-specific functions */
TH_API int THCSTensor_(getDevice)(THCState *state, const THCSTensor *self);
// NB: nTensors is the number of TOTAL tensors, not the number of dense tensors.
// That is to say, nSparseTensors + nDenseTensors == nTensors
TH_API int THCSTensor_(checkGPU)(THCState *state, unsigned int nSparseTensors, unsigned int nTensors, ...);

/* internal methods */
TH_API THCSTensor* THCSTensor_(rawResize)(THCState *state, THCSTensor *self, int nDimI, int nDimV, int64_t *size);
TH_API THCTensor *THCSTensor_(newValuesWithSizeOf)(THCState *state, THCTensor *values, int64_t nnz);
TH_API THCSTensor* THCSTensor_(_move)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values);
TH_API THCSTensor* THCSTensor_(_set)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values);
// forceClone is intended to use as a boolean
TH_API THCIndexTensor* THCSTensor_(newFlattenedIndices)(THCState *state, THCSTensor *self, int forceClone);

#endif
