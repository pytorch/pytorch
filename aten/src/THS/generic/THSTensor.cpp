#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensor.cpp"
#else

/******************************************************************************
 * access methods
 ******************************************************************************/

int THSTensor_(_nDimension)(const THSTensor *self)
{
  THError("Internal error! THSTensor_(_nDimension)(self) shouldn't be called; use self.dim() instead");
}

int THSTensor_(nDimensionI)(const THSTensor *self)
{
  THError("Internal error! THSTensor_(nDimensionI)(self) shouldn't be called; use self._sparseDims() instead");
}

int THSTensor_(nDimensionV)(const THSTensor *self)
{
  THError("Internal error! THSTensor_(nDimensionV)(self) shouldn't be called; use self._denseDims() instead");
}

int64_t THSTensor_(size)(const THSTensor *self, int dim)
{
  THError("Internal error! THSTensor_(size)(self, dim) shouldn't be called; use self.size(dim) instead");
}

ptrdiff_t THSTensor_(nnz)(const THSTensor *self) {
  THError("Internal error! THSTensor_(nnz)(self) shouldn't be called; use self._nnz() instead");
}

THLongStorage *THSTensor_(newSizeOf)(THSTensor *self)
{
  THError("Internal error! THSTensor_(newSizeOf)(self) shouldn't be called; use dtype.tensor(self.size()) instead");
}

THLongTensor *THSTensor_(newIndices)(const THSTensor *self) {
  THError("Internal error! THSTensor_(newIndices)(self) shouldn't be called; use self._indices() instead");
}

THTensor *THSTensor_(newValues)(const THSTensor *self) {
  THError("Internal error! THSTensor_(newValues)(self) shouldn't be called; use self._values() instead");
}


/******************************************************************************
 * creation methods
 ******************************************************************************/

/*** Helper methods ***/
static void THSTensor_(rawInit)(THSTensor *self)
{
  THError("Internal error! THSTensor_(rawInit)(self) shouldn't be called; dtype.tensor() allocated sparse tensors should already be initialized");
}

THSTensor* THSTensor_(rawResize)(THSTensor *self, int nDimI, int nDimV, int64_t *size) {
  THError("Internal error! THSTensor_(rawResize)(self, nDimI, nDimV, size) shouldn't be called; use _get_sparse_impl(self)->raw_resize_(sparseDims, denseDims, size) instead");
}

// directly assign without cloning or retaining (internal method)
THSTensor* THSTensor_(_move)(THSTensor *self, THLongTensor *indices, THTensor *values) {
  THError("Internal error! THSTensor_(_move)(self, indices, values) shouldn't be called; use _alias_into_sparse(self, indices, values) instead");
}

THSTensor* THSTensor_(_set)(THSTensor *self, THLongTensor *indices, THTensor *values) {
  THError("Internal error! THSTensor_(_set)(self, indices, values) shouldn't be called; use _copy_into_sparse(self, indices, values) instead");
}

static inline THSTensor* THSTensor_(_newWithDimsAndTensor)(int64_t nDimI, int64_t nDimV, int64_t *sizes, THLongTensor *indices, THTensor *values) {
  THError("Internal error! THSTensor_(_newWithDimsAndTensor)(nDimI, nDimV, sizes, indices, values) shouldn't be called; use _new_with_dims_and_tensor_sparse(dtype, nDimI, nDimV, sizes, indices, values) instead");
}

/*** end helper methods ***/

/* Empty init */
THSTensor *THSTensor_(new)(void)
{
  THError("Internal error! THSTensor_(new)() shouldn't be called; use dtype.tensor() instead");
}

/* Pointer-copy init */
THSTensor *THSTensor_(newWithTensor)(THLongTensor *indices, THTensor *values)
{
  THError("Internal error! THSTensor_(newWithTensor)(indices, values) shouldn't be called; use dtype.sparse_coo_tensor(indices, values) instead");
}

THSTensor *THSTensor_(newWithTensorAndSizeUnsafe)(THLongTensor *indices, THTensor *values, THLongStorage *sizes)
{
  THError("Internal error! THSTensor_(newWithTensorAndSizeUnsafe)(indices, values, sizes) shouldn't be called; use dtype._sparse_coo_tensor_unsafe(indices, values, unsafe) instead");
}

THSTensor *THSTensor_(newWithTensorAndSize)(THLongTensor *indices, THTensor *values, THLongStorage *sizes)
{
  THError("Internal error! THSTensor_(newWithTensorAndSize)(indices, values, sizes) shouldn't be called; use dtype.sparse_coo_tensor(indices, values, sizes) instead");
}

THSTensor *THSTensor_(newWithSize)(THLongStorage *size, THLongStorage *_ignored)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(newWithSize1d)(int64_t size0)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(newWithSize2d)(int64_t size0, int64_t size1)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(newWithSize3d)(int64_t size0, int64_t size1, int64_t size2)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(newWithSize4d)(int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(newClone)(THSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(newTranspose)(THSTensor *self, int d1, int d2) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

int THSTensor_(isSameSizeAs)(const THSTensor *self, const THSTensor* src)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(resizeAs)(THSTensor *self, THSTensor *src)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(resize1d)(THSTensor *self, int64_t size0)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(resize2d)(THSTensor *self, int64_t size0, int64_t size1)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(resize3d)(THSTensor *self, int64_t size0, int64_t size1, int64_t size2)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(resize4d)(THSTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THTensor *THSTensor_(toDense)(THSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(copy)(THSTensor *self, THSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

// In place transpose
void THSTensor_(transpose)(THSTensor *self, int d1, int d2) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THSTensor_(isCoalesced)(const THSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/* Internal slice operations. Buffers can be reused across calls to avoid
allocating tensors every time */

void THSTensor_(mulSlice)(
  THTensor *dstBuffer, THTensor *src1Buffer, THTensor *src2Buffer,
  THTensor *dst, THTensor *src1, THTensor *src2,
  int64_t dim, int64_t dstIdx, int64_t src1Idx, int64_t src2Idx) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(divSlice)(
  THTensor *dstBuffer, THTensor *src1Buffer, THTensor *src2Buffer,
  THTensor *dst, THTensor *src1, THTensor *src2,
  int64_t dim, int64_t dstIdx, int64_t src1Idx, int64_t src2Idx) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THTensor *THSTensor_(newValuesWithSizeOf)(THTensor *values, int64_t nnz) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THSTensor *THSTensor_(newCoalesce)(THSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THTensor_(sparseMask)(THSTensor *r_, THTensor *t, THSTensor *mask) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(free)(THSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(retain)(THSTensor *self)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

#endif
