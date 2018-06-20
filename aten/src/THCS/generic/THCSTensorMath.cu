#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensorMath.cu"
#else

#include "THCThrustAllocator.cuh"
#include "THCNumerics.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define I_INFO(tensor) getTensorInfo<int64_t, THCIndexTensor, uint64_t>(state, tensor)
#define V_INFO(tensor) getTensorInfo<real, THCTensor, uint64_t>(state, tensor)

THCudaIntTensor *THCSTensor_(toCSR)(THCState *state, THCIndexTensor *rowIndices, int64_t dim, int64_t nnz) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(zero)(THCState *state, THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(zerosLike)(THCState *state, THCSTensor *r_, THCSTensor *input)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCTensor_(spaddcmul)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCTensor_(spaddcdiv)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(spaddmm)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCSTensor *sparse_, THCTensor *dense) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(sspaddmm)(THCState *state, THCSTensor *r_, real beta, THCSTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(hspmm)(THCState *state, THCSTensor *r_, real alpha, THCSTensor *sparse_, THCTensor *dense) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(spcadd)(THCState *state, THCTensor *r_, THCTensor *dense, real value, THCSTensor *sparse) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(mul)(THCState *state, THCSTensor *r_, THCSTensor *t, real value) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(div)(THCState *state, THCSTensor *r_, THCSTensor *t, real value) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(isSameSizeIgnoringDensity)(THCState *state, const THCSTensor *self, const THCSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(isSameDensity)(THCState *state, const THCSTensor *self, const THCSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}


void THCSTensor_(cadd)(THCState *state, THCSTensor *r_, THCSTensor *t, real value, THCSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(csub)(THCState *state, THCSTensor *r_, THCSTensor *t, real value, THCSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(cmul)(THCState *state, THCSTensor *r_, THCSTensor *t_, THCSTensor *src_) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(pow)(THCState *state, THCSTensor *r_, THCSTensor *t_, real value) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

#if defined(THCS_REAL_IS_FLOAT) || defined(THCS_REAL_IS_DOUBLE) || defined(THCS_REAL_IS_HALF)
accreal THCSTensor_(normall)(THCState *state, THCSTensor *self, real value) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}
#endif

#endif
