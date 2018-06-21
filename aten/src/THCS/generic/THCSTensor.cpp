#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.cpp"
#else

/******************************************************************************
 * access methods
 ******************************************************************************/

int THCSTensor_(_nDimension)(THCState *state, const THCSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(nDimensionI)(THCState *state, const THCSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(nDimensionV)(THCState *state, const THCSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int64_t THCSTensor_(size)(THCState *state, const THCSTensor *self, int dim)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

ptrdiff_t THCSTensor_(nnz)(THCState *state, const THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THLongStorage *THCSTensor_(newSizeOf)(THCState *state, THCSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCIndexTensor *THCSTensor_(newIndices)(THCState *state, const THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCTensor *THCSTensor_(newValues)(THCState *state, const THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}


/******************************************************************************
 * creation methods
 ******************************************************************************/

/*** Helper methods ***/
static void THCSTensor_(rawInit)(THCState *state, THCSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor* THCSTensor_(rawResize)(THCState *state, THCSTensor *self, int nDimI, int nDimV, int64_t *size) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

// directly assign without cloning or retaining (internal method)
THCSTensor* THCSTensor_(_move)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor* THCSTensor_(_set)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

static inline THCSTensor* THCSTensor_(_newWithDimsAndTensor)(THCState *state, int64_t nDimI, int64_t nDimV, int64_t *sizes, THCIndexTensor *indices, THCTensor *values) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/*** end helper methods ***/

/* Empty init */
THCSTensor *THCSTensor_(new)(THCState *state)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/* Pointer-copy init */
THCSTensor *THCSTensor_(newWithTensor)(THCState *state, THCIndexTensor *indices, THCTensor *values)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newWithTensorAndSizeUnsafe)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newWithTensorAndSize)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newWithSize)(THCState *state, THLongStorage *size, THLongStorage *_ignored)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newWithSize1d)(THCState *state, int64_t size0)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newWithSize2d)(THCState *state, int64_t size0, int64_t size1)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newWithSize3d)(THCState *state, int64_t size0, int64_t size1, int64_t size2)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newWithSize4d)(THCState *state, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newClone)(THCState *state, THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newTranspose)(THCState *state, THCSTensor *self, int d1, int d2) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCTensor *THCSTensor_(newValuesWithSizeOf)(THCState *state, THCTensor *values, int64_t nnz) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

int THCSTensor_(isSameSizeAs)(THCState *state, const THCSTensor *self, const THCSTensor* src)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(isSameSizeAsDense)(THCState *state, const THCSTensor *self, const THCTensor* src)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(resizeLegacy)(THCState *state, THCSTensor *self, THLongStorage *size)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(resizeAs)(THCState *state, THCSTensor *self, THCSTensor *src)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(resize1d)(THCState *state, THCSTensor *self, int64_t size0)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(resize2d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(resize3d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(resize4d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(copy)(THCState *state, THCSTensor *self, THCSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(isCoalesced)(THCState *state, const THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(free)(THCState *state, THCSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCSTensor_(retain)(THCState *state, THCSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(checkGPU)(THCState *state, unsigned int nSparseTensors, unsigned int nTensors, ...)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCTensor_(sparseMask)(THCState *state, THCSTensor *r_, THCTensor *t, THCSTensor *mask) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

#endif
