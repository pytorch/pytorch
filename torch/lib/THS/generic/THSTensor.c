#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensor.c"
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

long THSTensor_(size)(const THSTensor *self, int dim)
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

static void THSTensor_(rawResize)(THSTensor *self, int nDimI, int nDimV, long *size) {
  // Only resize valid sizes into tensor.
  self->size = THRealloc(self->size, sizeof(long)*(nDimI + nDimV));

  for (long d = 0; d < nDimI + nDimV; d++) {
    self->size[d] = size[d];
  }
  self->nDimensionI = nDimI;
  self->nDimensionV = nDimV;
  self->coalesced = 0;
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


/*** end helper methods ***/

/* Empty init */
THSTensor *THSTensor_(new)(void)
{
  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THSTensor *THSTensor_(newWithTensor)(THLongTensor *indices, THTensor *values)
{
  return THSTensor_(newWithTensorAndSize)(indices, values, NULL);
}

THSTensor *THSTensor_(newWithTensorAndSize)(THLongTensor *indices, THTensor *values, THLongStorage *sizes)
{  // If sizes are not given, it is inferred as max index of each dim.
  long nDimI, nDimV;
  THLongTensor *ignore;

  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);

  nDimI = THLongTensor_size(indices, 0);
  nDimV = THTensor_(nDimension)(values) - 1;
  if (!sizes) {
    ignore = THLongTensor_new();
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
    THSTensor_(rawResize)(self, nDimI, nDimV, THLongTensor_data(computed_sizes));
    THLongTensor_free(computed_indices_sizes);
    THLongTensor_free(computed_sizes);
    THLongTensor_free(ignore);
  }
  else {
    THArgCheck(THLongStorage_size(sizes) == nDimI + nDimV, 2,
        "number of dimensions must be nDimI + nDimV");
    THSTensor_(rawResize)(self, nDimI, nDimV, THLongStorage_data(sizes));
  }
  THSTensor_(_set)(self, indices, values);

  return self;
}

THSTensor *THSTensor_(newWithSize)(THLongStorage *size)
{
  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  THSTensor_(rawResize)(self, size->size, 0, size->data);

  return self;
}

THSTensor *THSTensor_(newWithSize1d)(long size0)
{
  long size[1] = {size0};

  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  THSTensor_(rawResize)(self, 1, 0, size);

  return self;
}

THSTensor *THSTensor_(newWithSize2d)(long size0, long size1)
{
  long size[2] = {size0, size1};

  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  THSTensor_(rawResize)(self, 2, 0, size);

  return self;
}

THSTensor *THSTensor_(newWithSize3d)(long size0, long size1, long size2)
{
  long size[3] = {size0, size1, size2};

  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  THSTensor_(rawResize)(self, 3, 0, size);

  return self;
}

THSTensor *THSTensor_(newWithSize4d)(long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
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

THSTensor *THSTensor_(resize1d)(THSTensor *self, long size0)
{
  long size[1] = {size0};
  THSTensor_(rawResize)(self, 1, 0, size);
  return self;
}

THSTensor *THSTensor_(resize2d)(THSTensor *self, long size0, long size1)
{
  long size[2] = {size0, size1};
  THSTensor_(rawResize)(self, 2, 0, size);
  return self;
}

THSTensor *THSTensor_(resize3d)(THSTensor *self, long size0, long size1, long size2)
{
  long size[3] = {size0, size1, size2};
  THSTensor_(rawResize)(self, 3, 0, size);
  return self;
}

THSTensor *THSTensor_(resize4d)(THSTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};
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
  long nDimI = THSTensor_(nDimensionI)(self);
  long nDimV = THSTensor_(nDimensionV)(self);
  THArgCheck(d1 < nDimI && d2 < nDimI, 0, "Transposed dimensions should be sparse. Got nDimI: %ld, d1: %ld, d2: %ld", nDimI, d1, d2);
  THLongTensor *indices = THSTensor_(newIndices)(self);
  ptrdiff_t i;
  for (i = 0; i < THSTensor_(nnz)(self); i++) {
    long tmp = THTensor_fastGet2d(indices, d1, i);
    THTensor_fastSet2d(indices, d1, i,
        THTensor_fastGet2d(indices, d2, i));
    THTensor_fastSet2d(indices, d2, i, tmp);
  }
  i = self->size[d1];
  self->size[d1] = self->size[d2];
  self->size[d2] = i;
  self->coalesced = 0;
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
  long dim, long dstIdx, long src1Idx, long src2Idx) {
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
  long dim, long dstIdx, long src1Idx, long src2Idx) {
  if (src1->nDimension > 1) {
    THTensor_(select)(src1Buffer, src1, dim, src1Idx);
    THTensor_(select)(src2Buffer, src2, dim, src2Idx);
    THTensor_(select)(dstBuffer, dst, dim, dstIdx);
    THTensor_(cdiv)(dstBuffer, src1Buffer, src2Buffer);
  } else {
    THTensor_fastSet1d(dst, dstIdx, THTensor_fastGet1d(src1, src1Idx) / THTensor_fastGet1d(src2, src2Idx));
  }
}

THTensor *THSTensor_(newValuesWithSizeOf)(THTensor *values, long nnz) {
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
  long nDimI = THSTensor_(nDimensionI)(self);
  long nDimV = THSTensor_(nDimensionV)(self);

  THLongTensor *indicesScalar = THLongTensor_newWithSize1d(self->nnz);
  THLongTensor *indicesSlice = THLongTensor_new();
  THLongTensor *indicesBuffer = THLongTensor_newWithSize1d(self->nnz);
  THLongTensor *indicesPermutation = THLongTensor_newWithSize1d(self->nnz);
  THLongTensor_zero(indicesScalar);
  long factor = 1;
  for (long d = nDimI - 1; d >= 0; d--) {
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

  long i = -1;
  long prev = -1;
  long blockSize = values->stride[0];
  for (long j = 0; j < self->nnz; j++) {
    long pos = THTensor_fastGet1d(indicesPermutation, j);
    long curr = THTensor_fastGet1d(indicesBuffer, j);
    if (curr == prev) {
      THBlas_(axpy)(blockSize, 1,
        THTensor_(data)(values) + pos * blockSize, 1,
        THTensor_(data)(newValues) + i * blockSize, 1);
    } else {
      ++i;
      for (long d = 0; d < nDimI; d++) {
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

// Given an index tensor indices into a dense tensor t, select just the
// values index and create a corresponding sparse tensor.
// Precondition: indices is coalesced
void THTensor_(sparseSelect)(THSTensor *r_, THTensor *t, THLongTensor *indices)
{
  THArgCheck(THLongTensor_nDimension(indices) == 2, 2, "indices must be nDim x nnz");
  long nDim = THTensor_(nDimension)(t);
  long nDimI = THLongTensor_size(indices, 0);
  long nDimV = nDim - nDimI;
  long nnz = THLongTensor_size(indices, 1);
  THArgCheck(nDimI <= nDim, 2, "nDim of indices must be <= nDim of tensor");
  // NB: In principle, we should check if resizing is necessary, but
  // since this operator is not currently used inplace, that test will
  // always return false.  So don't bother.
  THSTensor_(rawResize)(r_, nDimI, nDimV, t->size);
  if (nnz == 0) {
    THSTensor_(zero)(r_);
    return;
  }
  THTensor *r_values_ = THTensor_(new)();
  long *size = THAlloc(sizeof(long) * (nDimV + 1));
  size[0] = nnz;
  for (int i = 0; i < nDimV; i++) {
    size[i+1] = t->size[nDimI + i];
  }
  THTensor_(resizeNd)(r_values_, nDimV + 1, size, NULL);
  THFree(size);
  THLongTensor *r_indices_ = THLongTensor_newClone(indices);
  THSTensor_(_move)(r_, r_indices_, r_values_);
  r_->coalesced = 1; // OK due to pre-condition!

  if (nDim > nDimI) {
    THTensor *srcBuffer = THTensor_(new)();
    THTensor *dstBuffer = THTensor_(new)();
    for (long i = 0; i < r_->nnz; i++) {
      THTensor_(set)(srcBuffer, t);
      for (long d = 0; d < nDimI; d++) {
        THTensor_(select)(srcBuffer, srcBuffer, 0, THTensor_fastGet2d(indices, d, i));
      }
      THTensor_(select)(dstBuffer, r_values_, 0, i);
      THTensor_(copy)(dstBuffer, srcBuffer);
    }
    THTensor_(free)(srcBuffer);
    THTensor_(free)(dstBuffer);
  } else {
    for (long i = 0; i < r_->nnz; i++) {
      long idx = 0;
      for (long d = 0; d < nDimI; d++) {
        long j = THTensor_fastGet2d(indices, d, i);
        if (j < TH_INDEX_BASE || j >= TH_INDEX_BASE + t->size[d]) {
          THError("index out of range");
        }
        idx += j * t->stride[d];
      }
      real val = (t->storage->data + t->storageOffset)[idx];
      THTensor_fastSet1d(r_values_, i, val);
    }
  }
}

void THTensor_(sparseCopy)(THTensor *r_, THTensor *dense, THSTensor *sparse_)
{
  THTensor_(resizeAs)(r_, dense);
  THSTensor *sparse = THSTensor_(newCoalesce)(sparse_);

  long k;
  THLongTensor  *indices = THSTensor_(newIndices)(sparse);
  THTensor      *values = THSTensor_(newValues)(sparse);
  THLongStorage *storage = THSTensor_(newSizeOf)(sparse);
  long          *sizes = storage->data;
  long          nDim = THTensor_(nDimension)(dense);
  long          nDimI = THSTensor_(nDimensionI)(sparse);

  if (r_ != dense) THTensor_(copy)(r_, dense);

  if (nDim > nDimI) {
    THTensor *srcBuffer = THTensor_(new)();
    THTensor *dstBuffer = THTensor_(new)();
    for (k = 0; k < sparse->nnz; k++) {
      THTensor_(set)(dstBuffer, r_);
      for (long d = 0; d < sparse->nDimensionI; d++) {
        THTensor_(select)(dstBuffer, dstBuffer, 0, THTensor_fastGet2d(indices, d, k));
      }
      THTensor_(select)(srcBuffer, values, 0, k);
      THTensor_(copy)(dstBuffer, srcBuffer);
    }
    THTensor_(free)(srcBuffer);
    THTensor_(free)(dstBuffer);
  } else {
    #pragma omp parallel for private(k)
    for (k = 0; k < sparse->nnz; k++) {
      long index = r_->storageOffset;
      for (long d = 0; d < sparse->nDimensionI; d++) {
        index += r_->stride[d] * THTensor_fastGet2d(indices, d, k);
      }
      r_->storage->data[index] = THTensor_fastGet1d(values, k);
    }
  }

  THLongTensor_free(indices);
  THTensor_(free)(values);
  THLongStorage_free(storage);
  THSTensor_(free)(sparse);
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
