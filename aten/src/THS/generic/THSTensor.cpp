#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensor.cpp"
#else

/******************************************************************************
 * access methods
 ******************************************************************************/

int THSTensor_(nDimension)(const THSTensor *self)
{
  THError("Internal error! THSTensor_(nDimension)(self) shouldn't be called; use self.dim() instead");
}

int THSTensor_(nDimensionI)(const THSTensor *self)
{
  THError("Internal error! THSTensor_(nDimensionI)(self) shouldn't be called; use self._dimI() instead");
}

int THSTensor_(nDimensionV)(const THSTensor *self)
{
  THError("Internal error! THSTensor_(nDimensionV)(self) shouldn't be called; use self._dimV() instead");
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
  THError("Internal error! THSTensor_(rawResize)(self, nDimI, nDimV, size) shouldn't be called; use _get_sparse_impl(self)->raw_resize_(dimI, dimV, size) instead");
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
  THSTensor *self = THSTensor_(new)();
  THSTensor_(rawResize)(self, size->size, 0, THLongStorage_data(size));
  return self;
}

THSTensor *THSTensor_(newWithSize1d)(int64_t size0)
{
  int64_t size[1] = {size0};
  THSTensor *self = THSTensor_(new)();
  THSTensor_(rawResize)(self, 1, 0, size);
  return self;
}

THSTensor *THSTensor_(newWithSize2d)(int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THSTensor *self = THSTensor_(new)();
  THSTensor_(rawResize)(self, 2, 0, size);
  return self;
}

THSTensor *THSTensor_(newWithSize3d)(int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THSTensor *self = THSTensor_(new)();
  THSTensor_(rawResize)(self, 3, 0, size);
  return self;
}

THSTensor *THSTensor_(newWithSize4d)(int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THSTensor *self = THSTensor_(new)();
  THSTensor_(rawResize)(self, 4, 0, size);
  return self;
}

THSTensor *THSTensor_(newClone)(THSTensor *self) {
  THSTensor *other = THSTensor_(new)();
  THSTensor_(rawResize)(other, self->nDimensionI, self->nDimensionV, self->size);

  THSTensor_(_set)(other, self->indices, self->values);

  other->coalesced = self->coalesced;
  other->nnz = self->nnz;
  return other;
}

THSTensor *THSTensor_(newTranspose)(THSTensor *self, int d1, int d2) {
  THSTensor *other = THSTensor_(newClone)(self);
  THSTensor_(transpose)(other, d1, d2);
  return other;
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

int THSTensor_(isSameSizeAs)(const THSTensor *self, const THSTensor* src)
{
  int d;
  if (self->nDimensionI != src->nDimensionI || self->nDimensionV != src->nDimensionV)
    return 0;
  for(d = 0; d < self->nDimensionI + self->nDimensionV; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

THSTensor *THSTensor_(resizeLegacy)(THSTensor *self, THLongStorage *size)
{
  THSTensor_(rawResize)(self, size->size, 0, THLongStorage_data(size));
  return self;
}

THSTensor *THSTensor_(resizeAs)(THSTensor *self, THSTensor *src)
{
  if(!THSTensor_(isSameSizeAs)(self, src)) {
    THSTensor_(rawResize)(self, src->nDimensionI, src->nDimensionV, src->size);
  }
  return self;
}

THSTensor *THSTensor_(resize1d)(THSTensor *self, int64_t size0)
{
  int64_t size[1] = {size0};
  THSTensor_(rawResize)(self, 1, 0, size);
  return self;
}

THSTensor *THSTensor_(resize2d)(THSTensor *self, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THSTensor_(rawResize)(self, 2, 0, size);
  return self;
}

THSTensor *THSTensor_(resize3d)(THSTensor *self, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THSTensor_(rawResize)(self, 3, 0, size);
  return self;
}

THSTensor *THSTensor_(resize4d)(THSTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THSTensor_(rawResize)(self, 4, 0, size);
  return self;
}

THTensor *THSTensor_(toDense)(THSTensor *self) {
  THLongStorage *size;
  THTensor *dst;

  // set up the new tensor
  size = THSTensor_(newSizeOf)(self);
  dst = THTensor_(newWithSize)(size, NULL);
  THLongStorage_free(size);
  THTensor_(zero)(dst);

  // real one = ScalarConvert<int, real>::to(1);
  THSTensor_(spcadd)(dst, dst, 1, self);
  return dst;
}

void THSTensor_(copy)(THSTensor *self, THSTensor *src) {
  if (self == src) return;
  THSTensor_(rawResize)(self, src->nDimensionI, src->nDimensionV, src->size);
  THSTensor_(_set)(self, src->indices, src->values);
  self->nnz = src->nnz;
  self->coalesced = src->coalesced;
}

// In place transpose
void THSTensor_(transpose)(THSTensor *self, int d1, int d2) {
  int64_t nDimI = THSTensor_(nDimensionI)(self);
  THArgCheck(d1 < nDimI && d2 < nDimI, 0, "Transposed dimensions should be sparse. Got nDimI: %" PRId64 ", d1: %" PRId64 ", d2: %" PRId64, nDimI, d1, d2);
  THLongTensor *indices = THSTensor_(newIndices)(self);
  ptrdiff_t i;
  for (i = 0; i < THSTensor_(nnz)(self); i++) {
    int64_t tmp = THLongTensor_fastGet2d(indices, d1, i);
    THLongTensor_fastSet2d(indices, d1, i,
        THLongTensor_fastGet2d(indices, d2, i));
    THLongTensor_fastSet2d(indices, d2, i, tmp);
  }
  i = self->size[d1];
  self->size[d1] = self->size[d2];
  self->size[d2] = i;
  THLongTensor_free(indices);
}

int THSTensor_(isCoalesced)(const THSTensor *self) {
  return self->coalesced;
}

/* Internal slice operations. Buffers can be reused across calls to avoid
allocating tensors every time */

void THSTensor_(mulSlice)(
  THTensor *dstBuffer, THTensor *src1Buffer, THTensor *src2Buffer,
  THTensor *dst, THTensor *src1, THTensor *src2,
  int64_t dim, int64_t dstIdx, int64_t src1Idx, int64_t src2Idx) {
  if (src1->_dim() > 1) {
    THTensor_(select)(src1Buffer, src1, dim, src1Idx);
    THTensor_(select)(src2Buffer, src2, dim, src2Idx);
    THTensor_(select)(dstBuffer, dst, dim, dstIdx);
    THTensor_(cmul)(dstBuffer, src1Buffer, src2Buffer);
  } else {
    THTensor_(fastSet1d)(dst, dstIdx, THTensor_(fastGet1d)(src1, src1Idx) * THTensor_(fastGet1d)(src2, src2Idx));
  }
}

void THSTensor_(divSlice)(
  THTensor *dstBuffer, THTensor *src1Buffer, THTensor *src2Buffer,
  THTensor *dst, THTensor *src1, THTensor *src2,
  int64_t dim, int64_t dstIdx, int64_t src1Idx, int64_t src2Idx) {
  if (src1->_dim() > 1) {
    THTensor_(select)(src1Buffer, src1, dim, src1Idx);
    THTensor_(select)(src2Buffer, src2, dim, src2Idx);
    THTensor_(select)(dstBuffer, dst, dim, dstIdx);
    THTensor_(cdiv)(dstBuffer, src1Buffer, src2Buffer);
  } else {
    THTensor_(fastSet1d)(dst, dstIdx, THTensor_(fastGet1d)(src1, src1Idx) / THTensor_(fastGet1d)(src2, src2Idx));
  }
}

THTensor *THSTensor_(newValuesWithSizeOf)(THTensor *values, int64_t nnz) {
  THTensor *new_values;
  if (THTensor_(nDimension)(values) == 0) { // values tensor uninitialized
    new_values = THTensor_(newWithSize1d)(nnz);
  } else {
    THLongStorage *size = THTensor_(newSizeOf)(values);
    THLongStorage_data(size)[0] = nnz;
    new_values = THTensor_(newWithSize)(size, NULL);
    THLongStorage_free(size);
  }
  return new_values;
}

THSTensor *THSTensor_(newCoalesce)(THSTensor *self) {
  if (self->nnz < 2) {
    self->coalesced = 1;
  }
  if (self->coalesced) {
    THSTensor_(retain)(self);
    return self;
  }
  THLongTensor *indices = THSTensor_(newIndices)(self);
  THTensor *values_ = THSTensor_(newValues)(self);
  THTensor *values = THTensor_(newContiguous)(values_);
  int64_t nDimI = THSTensor_(nDimensionI)(self);
  int64_t nDimV = THSTensor_(nDimensionV)(self);

  THLongTensor *indicesScalar = THLongTensor_newWithSize1d(self->nnz);
  THLongTensor *indicesSlice = THLongTensor_new();
  THLongTensor *indicesBuffer = THLongTensor_newWithSize1d(self->nnz);
  THLongTensor *indicesPermutation = THLongTensor_newWithSize1d(self->nnz);
  THLongTensor_zero(indicesScalar);
  int64_t factor = 1;
  for (int64_t d = nDimI - 1; d >= 0; d--) {
    THLongTensor_select(indicesSlice, indices, 0, d);
    THLongTensor_cadd(indicesScalar, indicesScalar, factor, indicesSlice);
    factor *= self->size[d];
  }

  THLongTensor *newIndices = THLongTensor_new();
  THTensor *newValues = THTensor_(new)();
  THLongTensor_resizeAs(newIndices, indices);
  THTensor_(resizeAs)(newValues, values_);
  // THSTensor_(_move)(self, newIndices, newValues);
  THSTensor *dst = THSTensor_(new)();
  THSTensor_(rawResize)(dst, nDimI, nDimV, self->size);
  THSTensor_(_move)(dst, newIndices, newValues);

  THLongTensor_sort(indicesBuffer, indicesPermutation, indicesScalar, 0, 0);

  int64_t i = -1;
  int64_t prev = -1;
  int64_t blockSize = values->stride[0];
  for (int64_t j = 0; j < self->nnz; j++) {
    int64_t pos = THLongTensor_fastGet1d(indicesPermutation, j);
    int64_t curr = THLongTensor_fastGet1d(indicesBuffer, j);
    if (curr == prev) {
      THBlas_(axpy)(blockSize, 1,
        THTensor_(data)(values) + pos * blockSize, 1,
        THTensor_(data)(newValues) + i * blockSize, 1);
    } else {
      ++i;
      for (int64_t d = 0; d < nDimI; d++) {
        THLongTensor_fastSet2d(newIndices, d, i, THLongTensor_fastGet2d(indices, d, pos));
      }
      THBlas_(copy)(blockSize,
        THTensor_(data)(values) + pos * blockSize, 1,
        THTensor_(data)(newValues) + i * blockSize, 1);
    }
    prev = curr;
  }
  dst->nnz = i + 1;
  dst->coalesced = 1;
  THLongTensor_free(indicesScalar);
  THLongTensor_free(indicesBuffer);
  THLongTensor_free(indicesPermutation);
  THLongTensor_free(indicesSlice);
  THLongTensor_free(indices);
  THTensor_(free)(values_);
  THTensor_(free)(values);

  return dst;
}

void THTensor_(sparseMask)(THSTensor *r_, THTensor *t, THSTensor *mask) {
  THArgCheck(mask->coalesced, 2, "mask is uncoalesced");
  THSTensor_(resizeAs)(r_, mask);
  if (mask->nnz == 0) {
    THSTensor_(zero)(r_);
    return;
  }
  int64_t nDim = THTensor_(nDimension)(t);
  int64_t nDimI = THSTensor_(nDimensionI)(mask);
  THLongTensor *mask_indices_ = THSTensor_(newIndices)(mask);
  THTensor *mask_values_ = THSTensor_(newValues)(mask);
  THTensor *r_values_ = THTensor_(new)();
  THTensor_(resizeAs)(r_values_, mask_values_);
  THSTensor_(_move)(r_, THLongTensor_newClone(mask_indices_), r_values_);
  r_->coalesced = mask->coalesced;
  r_->nnz = mask->nnz;

  if (nDim > nDimI) {
    THTensor *srcBuffer = THTensor_(new)();
    THTensor *dstBuffer = THTensor_(new)();
    for (int64_t i = 0; i < r_->nnz; i++) {
      THTensor_(set)(srcBuffer, t);
      for (int64_t d = 0; d < nDimI; d++) {
        THTensor_(select)(srcBuffer, srcBuffer, 0, THLongTensor_fastGet2d(mask_indices_, d, i));
      }
      THTensor_(select)(dstBuffer, r_values_, 0, i);
      THTensor_(copy)(dstBuffer, srcBuffer);
    }
    THTensor_(free)(srcBuffer);
    THTensor_(free)(dstBuffer);
  } else {
    for (int64_t i = 0; i < r_->nnz; i++) {
      int64_t idx = 0;
      for (int64_t d = 0; d < nDimI; d++) {
        idx += THLongTensor_fastGet2d(mask_indices_, d, i) * t->stride[d];
      }
      real val = (THStorage_(data)(t->storage) + t->storageOffset)[idx];
      THTensor_(fastSet1d)(r_values_, i, val);
    }
  }

  THLongTensor_free(mask_indices_);
  THTensor_(free)(mask_values_);
}

void THSTensor_(free)(THSTensor *self)
{
  if(!self)
    return;
  if(--self->refcount == 0)
  {
    THFree(self->size);
    THLongTensor_free(self->indices);
    THTensor_(free)(self->values);
    self->refcount.~atomic<int>();
    THFree(self);
  }
}

void THSTensor_(retain)(THSTensor *self)
{
  self->refcount++;
}

#endif
