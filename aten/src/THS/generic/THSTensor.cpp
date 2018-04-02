#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensor.cpp"
#else

/******************************************************************************
 * access methods
 ******************************************************************************/

int THSTensor_(nDimension)(const THSTensor *self)
{
  return self->nDimensionI + self->nDimensionV;
}

int THSTensor_(nDimensionI)(const THSTensor *self)
{
  return self->nDimensionI;
}

int THSTensor_(nDimensionV)(const THSTensor *self)
{
  return self->nDimensionV;
}

int64_t THSTensor_(size)(const THSTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimensionI + self->nDimensionV),
      1, "dimension %d out of range of %dD tensor",
      dim+1, THSTensor_(nDimension)(self));
  return self->size[dim];
}

ptrdiff_t THSTensor_(nnz)(const THSTensor *self) {
  return self->nnz;
}

THLongStorage *THSTensor_(newSizeOf)(THSTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimensionI + self->nDimensionV);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongTensor *THSTensor_(newIndices)(const THSTensor *self) {
  if (self->nnz == 0) {
    // Narrows don't work on 0-length tensors
    THLongTensor_retain(self->indices);
    return self->indices;
  }
  return THLongTensor_newNarrow(self->indices, 1, 0, self->nnz);
}

THTensor *THSTensor_(newValues)(const THSTensor *self) {
  if (self->nnz == 0) {
    THTensor_(retain)(self->values);
    return self->values;
  }
  return THTensor_(newNarrow)(self->values, 0, 0, self->nnz);
}


/******************************************************************************
 * creation methods
 ******************************************************************************/

/*** Helper methods ***/
static void THSTensor_(rawInit)(THSTensor *self)
{
  self->refcount = 1;
  self->size = NULL;
  self->indices = THLongTensor_new();
  self->values = THTensor_(new)();
  self->nDimensionI = 0;
  self->nDimensionV = 0;
  self->coalesced = 0;
  self->nnz = 0;
  // self->flag = TH_TENSOR_REFCOUNTED;
}

THSTensor* THSTensor_(rawResize)(THSTensor *self, int nDimI, int nDimV, int64_t *size) {
  // Only resize valid sizes into tensor.
  self->size = (int64_t *)THRealloc(self->size, sizeof(int64_t)*(nDimI + nDimV));

  for (int64_t d = 0; d < nDimI + nDimV; d++) {
    self->size[d] = size[d];
  }
  self->nDimensionI = nDimI;
  self->nDimensionV = nDimV;

  return self;
}

// directly assign without cloning or retaining (internal method)
THSTensor* THSTensor_(_move)(THSTensor *self, THLongTensor *indices, THTensor *values) {
  int empty = THTensor_(nDimension)(values) == 0;
  if (!empty) {
    THArgCheck(THLongTensor_nDimension(indices) == 2, 1,
        "indices must be nDim x nnz");
    THArgCheck(THLongTensor_size(indices, 1) == THTensor_(size)(values, 0), 1,
        "indices and values must have same nnz");
    THArgCheck(THLongTensor_size(indices, 0) == self->nDimensionI, 2,
        "indices has incorrect first dimension, expected %d, got %d", self->nDimensionI, THLongTensor_size(indices, 0));
    THArgCheck(THTensor_(nDimension)(values) == self->nDimensionV + 1, 3,
        "values has incorrect number of dimensions, expected %d, got %d", self->nDimensionV + 1, THTensor_(nDimension)(values));
  } else {
    THArgCheck(THLongTensor_nDimension(indices) == 0, 2,
        "if values is empty, indices must be empty too");
  }
  THLongTensor_free(self->indices);
  THTensor_(free)(self->values);
  self->indices = indices;
  self->values = values;
  self->nnz = empty ? 0 : THTensor_(size)(values, 0);
  self->coalesced = 0;

  return self;
}

THSTensor* THSTensor_(_set)(THSTensor *self, THLongTensor *indices, THTensor *values) {
  // Note: Not like torch.set, this is an internal method
  return THSTensor_(_move)(
    self, THLongTensor_newClone(indices), THTensor_(newClone)(values));
}

static inline THSTensor* THSTensor_(_newWithDimsAndTensor)(int64_t nDimI, int64_t nDimV, int64_t *sizes, THLongTensor *indices, THTensor *values) {
  THSTensor *self = THSTensor_(new)();
  THSTensor_(rawResize)(self, nDimI, nDimV, sizes);

  // NB: by default, we do NOT clone indices/values into the sparse tensor.
  // Efficient API by default!
  THSTensor_(_move)(self, THLongTensor_newWithTensor(indices), THTensor_(newWithTensor)(values));
  return self;
}

/*** end helper methods ***/

/* Empty init */
THSTensor *THSTensor_(new)(void)
{
  THSTensor *self = (THSTensor *)THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THSTensor *THSTensor_(newWithTensor)(THLongTensor *indices, THTensor *values)
{
  // If sizes are not given, it is inferred as max index of each dim.
  int64_t nDimI = THLongTensor_size(indices, 0);
  int64_t nDimV = THTensor_(nDimension)(values) - 1;

  THLongTensor *ignore = THLongTensor_new();
  THLongTensor *computed_indices_sizes = THLongTensor_new();
  THLongTensor *computed_sizes = THLongTensor_newWithSize1d(nDimI + nDimV);
  THLongTensor_max(computed_indices_sizes, ignore, indices, 1, 1);
  THLongTensor_add(computed_indices_sizes, computed_indices_sizes, 1);
  for (int d = 0; d < nDimI; d++) {
      THTensor_fastSet1d(computed_sizes, d, THTensor_fastGet1d(computed_indices_sizes, d));
  }
  for (int d = 0; d < nDimV; d++) {
      THTensor_fastSet1d(computed_sizes, nDimI + d, THTensor_(size)(values, d + 1));
  }

  THSTensor *self = THSTensor_(_newWithDimsAndTensor)(nDimI, nDimV, THLongTensor_data(computed_sizes), indices, values);
  THLongTensor_free(computed_indices_sizes);
  THLongTensor_free(computed_sizes);
  THLongTensor_free(ignore);
  return self;
}

THSTensor *THSTensor_(newWithTensorAndSizeUnsafe)(THLongTensor *indices, THTensor *values, THLongStorage *sizes)
{
  if (sizes == NULL)
  {
    return THSTensor_(newWithTensor)(indices, values);
  }
  if (THLongTensor_nDimension(indices) == 0 && THTensor_(nDimension)(values) == 0)
  {
    return THSTensor_(newWithSize)(sizes, NULL);
  }

  int64_t nDimI = THLongTensor_size(indices, 0);
  int64_t nDimV = THTensor_(nDimension)(values) - 1;

  return THSTensor_(_newWithDimsAndTensor)(nDimI, nDimV, THLongStorage_data(sizes), indices, values);
}

THSTensor *THSTensor_(newWithTensorAndSize)(THLongTensor *indices, THTensor *values, THLongStorage *sizes)
{
  if (sizes == NULL)
  {
    return THSTensor_(newWithTensor)(indices, values);
  }
  if (THLongTensor_nDimension(indices) == 0 && THTensor_(nDimension)(values) == 0)
  {
    return THSTensor_(newWithSize)(sizes, NULL);
  }

  int64_t nDimI = THLongTensor_size(indices, 0);
  int64_t nDimV = THTensor_(nDimension)(values) - 1;
  THArgCheck(THLongStorage_size(sizes) == nDimI + nDimV, 2,
      "number of dimensions must be nDimI + nDimV");

  THLongTensor *max_indices = THLongTensor_new();
  THLongTensor *ignore = THLongTensor_new();
  THLongTensor_max(max_indices, ignore, indices, 1, 0);
  THLongTensor_free(ignore);
  for (int d = 0; d < nDimI; d++) {
    int64_t max_index_in_dim = THTensor_fastGet1d(max_indices, d);
    int64_t dim_size = sizes->data[d];
    THArgCheck(max_index_in_dim < dim_size, 2,
        "sizes is inconsistent with indices: for dim %d, size is %lld but found index %lld",
        d, (long long)dim_size, (long long)max_index_in_dim);
  }
  for (int d = 0; d < nDimV; d++) {
    int64_t values_size = THTensor_(size)(values, d + 1);
    int64_t specified_size = sizes->data[nDimI + d];
    THArgCheck(values_size <= specified_size, 2,
        "values and sizes are inconsistent: sizes[%d] is %lld but values.size(%d) is %lld",
        d + nDimI, (long long)specified_size, d + 1, (long long)values_size);
  }
  THLongTensor_free(max_indices);

  return THSTensor_(_newWithDimsAndTensor)(nDimI, nDimV, THLongStorage_data(sizes), indices, values);
}

THSTensor *THSTensor_(newWithSize)(THLongStorage *size, THLongStorage *_ignored)
{
  THSTensor *self = THSTensor_(new)();
  THSTensor_(rawResize)(self, size->size, 0, size->data);
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

THSTensor *THSTensor_(resize)(THSTensor *self, THLongStorage *size)
{
  THSTensor_(rawResize)(self, size->size, 0, size->data);
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
    int64_t tmp = THTensor_fastGet2d(indices, d1, i);
    THTensor_fastSet2d(indices, d1, i,
        THTensor_fastGet2d(indices, d2, i));
    THTensor_fastSet2d(indices, d2, i, tmp);
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
  if (src1->nDimension > 1) {
    THTensor_(select)(src1Buffer, src1, dim, src1Idx);
    THTensor_(select)(src2Buffer, src2, dim, src2Idx);
    THTensor_(select)(dstBuffer, dst, dim, dstIdx);
    THTensor_(cmul)(dstBuffer, src1Buffer, src2Buffer);
  } else {
    THTensor_fastSet1d(dst, dstIdx, THTensor_fastGet1d(src1, src1Idx) * THTensor_fastGet1d(src2, src2Idx));
  }
}

void THSTensor_(divSlice)(
  THTensor *dstBuffer, THTensor *src1Buffer, THTensor *src2Buffer,
  THTensor *dst, THTensor *src1, THTensor *src2,
  int64_t dim, int64_t dstIdx, int64_t src1Idx, int64_t src2Idx) {
  if (src1->nDimension > 1) {
    THTensor_(select)(src1Buffer, src1, dim, src1Idx);
    THTensor_(select)(src2Buffer, src2, dim, src2Idx);
    THTensor_(select)(dstBuffer, dst, dim, dstIdx);
    THTensor_(cdiv)(dstBuffer, src1Buffer, src2Buffer);
  } else {
    THTensor_fastSet1d(dst, dstIdx, THTensor_fastGet1d(src1, src1Idx) / THTensor_fastGet1d(src2, src2Idx));
  }
}

THTensor *THSTensor_(newValuesWithSizeOf)(THTensor *values, int64_t nnz) {
  THTensor *new_values;
  if (THTensor_(nDimension)(values) == 0) { // values tensor uninitialized
    new_values = THTensor_(newWithSize1d)(nnz);
  } else {
    THLongStorage *size = THTensor_(newSizeOf)(values);
    size->data[0] = nnz;
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
    int64_t pos = THTensor_fastGet1d(indicesPermutation, j);
    int64_t curr = THTensor_fastGet1d(indicesBuffer, j);
    if (curr == prev) {
      THBlas_(axpy)(blockSize, 1,
        THTensor_(data)(values) + pos * blockSize, 1,
        THTensor_(data)(newValues) + i * blockSize, 1);
    } else {
      ++i;
      for (int64_t d = 0; d < nDimI; d++) {
        THTensor_fastSet2d(newIndices, d, i, THTensor_fastGet2d(indices, d, pos));
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
        THTensor_(select)(srcBuffer, srcBuffer, 0, THTensor_fastGet2d(mask_indices_, d, i));
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
        idx += THTensor_fastGet2d(mask_indices_, d, i) * t->stride[d];
      }
      real val = (t->storage->data + t->storageOffset)[idx];
      THTensor_fastSet1d(r_values_, i, val);
    }
  }

  THLongTensor_free(mask_indices_);
  THTensor_(free)(mask_values_);
}

void THSTensor_(free)(THSTensor *self)
{
  if(!self)
    return;
  if(THAtomicDecrementRef(&self->refcount))
  {
    THFree(self->size);
    THLongTensor_free(self->indices);
    THTensor_(free)(self->values);
    THFree(self);
  }
}

void THSTensor_(retain)(THSTensor *self)
{
  THAtomicIncrementRef(&self->refcount);
}

#endif
