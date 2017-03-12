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

THLongTensor *THSTensor_(indices)(const THSTensor *self) {
  if (self->nnz == 0) {
    // Narrows don't work on 0-length tensors
    THLongTensor_retain(self->indices);
    return self->indices;
  }
  return THLongTensor_newNarrow(self->indices, 1, 0, self->nnz);
}

THTensor *THSTensor_(values)(const THSTensor *self) {
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
  self->contiguous = 0;
  self->nnz = 0;
  // self->flag = TH_TENSOR_REFCOUNTED;
}

static void THSTensor_(rawResize)(THSTensor *self, int nDimI, int nDimV, long *size) {
  // Only resize valid sizes into tensor.
  self->size = THRealloc(self->size, sizeof(long)*(nDimI + nDimV));

  long d, nDimI_ = 0, nDimV_ = 0;
  for (d = 0; d < nDimI; d++) {
    if (size[d] > 0) {
      self->size[nDimI_++] = size[d];
    }
  }
  for (d = nDimI; d < nDimI + nDimV; d++) {
    if (size[d] > 0) {
      self->size[nDimI_ + nDimV_++] = size[d];
    }
  }
  self->nDimensionI = nDimI_;
  self->nDimensionV = nDimV_;
  self->contiguous = 0;
}

// directly assign without cloning or retaining (internal method)
THSTensor* THSTensor_(move)(THSTensor *self, THLongTensor *indices, THTensor *values) {
  int empty = THTensor_(nDimension)(values) == 0;
  if (!empty) {
    THArgCheck(THLongTensor_nDimension(indices) == 2, 1,
        "indices must be nDim x nnz");
    THArgCheck(THLongTensor_size(indices, 1) == THTensor_(size)(values, 0), 1,
        "indices and values must have same nnz");
  }
  THLongTensor_free(self->indices);
  THTensor_(free)(self->values);
  self->indices = indices;
  self->values = values;
  self->nnz = empty ? 0 : THTensor_(size)(values, 0);

  return self;
}

THSTensor* THSTensor_(_set)(THSTensor *self, THLongTensor *indices, THTensor *values) {
  // Note: Not like torch.set, this is an internal method
  return THSTensor_(move)(
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
  THSTensor_(_set)(self, indices, values);

  nDimI = THLongTensor_size(indices, 0);
  nDimV = THTensor_(nDimension)(values) - 1;
  if (!sizes) {
    ignore = THLongTensor_new();
    THLongTensor *computed_indices_sizes = THLongTensor_new();
    THLongTensor *computed_sizes = THLongTensor_newWithSize1d(nDimI + nDimV);
    THLongTensor_max(computed_indices_sizes, ignore, indices, 1);
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
  return THSTensor_(newWithSize4d)(size0, -1, -1, -1);
}

THSTensor *THSTensor_(newWithSize2d)(long size0, long size1)
{
  return THSTensor_(newWithSize4d)(size0, size1, -1, -1);
}

THSTensor *THSTensor_(newWithSize3d)(long size0, long size1, long size2)
{
  return THSTensor_(newWithSize4d)(size0, size1, size2, -1);
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

  other->nnz = self->nnz;
  return other;
}

THSTensor *THSTensor_(newContiguous)(THSTensor *self) {
  THSTensor *other = THSTensor_(newClone)(self);
  THSTensor_(contiguous)(other);
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
  return THSTensor_(resize4d)(self, size0, -1, -1, -1);
}

THSTensor *THSTensor_(resize2d)(THSTensor *self, long size0, long size1)
{
  return THSTensor_(resize4d)(self, size0, size1, -1, -1);
}

THSTensor *THSTensor_(resize3d)(THSTensor *self, long size0, long size1, long size2)
{
  return THSTensor_(resize4d)(self, size0, size1, size2, -1);
}

THSTensor *THSTensor_(resize4d)(THSTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};
  THSTensor_(rawResize)(self, 4, 0, size);
  return self;
}

THTensor *THSTensor_(toDense)(THSTensor *self) {
  int d, k, l, index;
  ptrdiff_t nnz;
  long nDimI, nDimV, indskip, blocksize = 1;
  long *sizes;
  THLongStorage *storage;

  THTensor *other_, *values_;
  real *other, *values;
  THLongTensor *indices_;
  long *indices;

  THSTensor_(contiguous)(self);

  // set up the new tensor
  storage = THSTensor_(newSizeOf)(self);
  other_ = THTensor_(newWithSize)(storage, NULL);
  THTensor_(zero)(other_);
  other = THTensor_(data)(other_);

  nnz = THSTensor_(nnz)(self);
  if (nnz == 0) {
    THLongStorage_free(storage);
    return other_;
  }

  // Some necessary dimensions and sizes
  nDimI = THSTensor_(nDimensionI)(self);
  nDimV = THSTensor_(nDimensionV)(self);
  sizes = storage->data;
  for (int i = 0; i < nDimV; i++) {
    blocksize *= sizes[nDimI + i];
  }

  // These should be contiguous...
  values_ = THSTensor_(values)(self);
  indices_ = self->indices;
  values = THTensor_(data)(values_);
  indices = THLongTensor_data(indices_);
  indskip = THLongTensor_size(indices_, 1); // To index indices

  #pragma omp parallel for private(k, d, l, index)
  for (k = 0; k < nnz; k++) {
    for (d = 0, index = 0; d < nDimI; d++)
      index = sizes[d] * index + indices[d * indskip + k];
    for (l = 0; l < blocksize; l++) {
      other[index * blocksize + l] = values[k * blocksize + l];
    }
  }

  THTensor_(free)(values_);
  THLongStorage_free(storage);
  return other_;
}

void THSTensor_(copy)(THSTensor *self, THSTensor *src) {
  if (self == src) return;
  THSTensor_(rawResize)(self, src->nDimensionI, src->nDimensionV, src->size);
  THSTensor_(_set)(self, src->indices, src->values);
  self->nnz = src->nnz;
}

// In place transpose
void THSTensor_(transpose)(THSTensor *self, int d1, int d2) {
  THLongTensor *indices = THSTensor_(indices)(self);
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
  self->contiguous = 0;
  THLongTensor_free(indices);
}

int THSTensor_(isContiguous)(const THSTensor *self) {
  return self->contiguous;
}

/* Internal slice operations. Buffers can be reused across calls to avoid
allocating tensors every time */

void THSTensor_(addSlice)(
  THTensor *dstBuffer, THTensor *src1Buffer, THTensor *src2Buffer,
  THTensor *dst, THTensor *src1, real value, THTensor *src2,
  long dim, long dstIdx, long src1Idx, long src2Idx) {
  if (src1->nDimension > 1) {
    THTensor_(select)(src1Buffer, src1, dim, src1Idx);
    THTensor_(select)(src2Buffer, src2, dim, src2Idx);
    THTensor_(select)(dstBuffer, dst, dim, dstIdx);
    THTensor_(cadd)(dstBuffer, src1Buffer, value, src2Buffer);
  } else {
    THTensor_fastSet1d(dst, dstIdx, THTensor_fastGet1d(src1, src1Idx) + value * THTensor_fastGet1d(src2, src2Idx));
  }
}

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

void THSTensor_(copySlice)(
  THTensor *dstBuffer, THTensor *srcBuffer,
  THTensor *dst, THTensor *src,
  long dim, long dstIdx, long srcIdx) {
  if (src->nDimension > 1) {
    THTensor_(select)(srcBuffer, src, dim, srcIdx);
    THTensor_(select)(dstBuffer, dst, dim, dstIdx);
    THTensor_(copy)(dstBuffer, srcBuffer);
  } else {
    THTensor_fastSet1d(dst, dstIdx, THTensor_fastGet1d(src, srcIdx));
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

void THSTensor_(reorder)(THSTensor *self) {
  /* TODO: We do an insertion sort here, should change to quicksort or shellsort
  */
  if (self->nnz < 2) return;
  long d, i, j, p, cmp, nDimI, nDimV, indskip, tmplong;
  real tmpreal;
  THLongTensor *indices_ = self->indices;
  THTensor *values_ = self->values;
  long *indices = THLongTensor_data(indices_);
  real *values = THTensor_(data)(values_);
  indskip = THLongTensor_size(indices_, 1); // To index indices
  nDimI = THSTensor_(nDimensionI)(self);
  nDimV = THSTensor_(nDimensionV)(self);

  THTensor *srcBuffer = THTensor_(new)();
  THTensor *dstBuffer = THTensor_(new)();
  THTensor *tmpBuffer = THSTensor_(newValuesWithSizeOf)(values_, 1);

#define IND(i, d) indices[d * indskip + i]
  for (i = 1; i < self->nnz; i++) {
    for (j = i-1; j >= 0; j--) {
      cmp = 0;
      for (d = 0; d < nDimI; d++) {
        if (IND(j+1, d) < IND(j, d))
          cmp = 1;
        if (IND(j+1, d) != IND(j, d)) break;
      }
      if (cmp) {
        THSTensor_(copySlice)(dstBuffer, srcBuffer, tmpBuffer, values_, 0, 0, j+1);
        THSTensor_(copySlice)(dstBuffer, srcBuffer, values_, values_, 0, j+1, j);
        THSTensor_(copySlice)(dstBuffer, srcBuffer, values_, tmpBuffer, 0, j, 0);
        for (d = 0; d < nDimI; d++) {
          tmplong = IND(j+1, d); IND(j+1, d) = IND(j, d); IND(j, d) = tmplong;
        }
      } else break;
    }
  }

  i = 0;
  for (j = 1; j < self->nnz; j++) {
    cmp = 1;
    // TODO: pass eps in as a parameter
    for (d = 0; d < nDimI; d++)
      if (IND(i, d) != IND(j, d)) {
        cmp = 0;
        break;
      }
    if (cmp) {
      THSTensor_(addSlice)(dstBuffer, dstBuffer, srcBuffer, values_, values_, 1, values_, 0, i, i, j);
    }
    else {
      THSTensor_(copySlice)(dstBuffer, srcBuffer, values_, values_, 0, ++i, j);
      for (d = 0; d < nDimI; d++) IND(i, d) = IND(j, d);
    }
  }
  self->nnz = i + 1;
#undef IND

  THTensor_(free)(srcBuffer);
  THTensor_(free)(dstBuffer);
  THTensor_(free)(tmpBuffer);
}

void THSTensor_(contiguous)(THSTensor *self) {
  if (self->contiguous) return;
  THSTensor_(reorder)(self);
  self->contiguous = 1;
}

void THTensor_(sparseMask)(THSTensor *r_, THTensor *t, THSTensor *mask) {
  THSTensor_(resizeAs)(r_, mask);
  if (mask->nnz == 0) {
    THSTensor_(zero)(r_);
    return;
  }
  long nDim = THTensor_(nDimension)(t);
  long nDimI = THSTensor_(nDimensionI)(mask);
  long nDimV = THSTensor_(nDimensionV)(mask);
  THLongTensor *mask_indices_ = THSTensor_(indices)(mask);
  THTensor *mask_values_ = THSTensor_(values)(mask);
  THTensor *r_values_ = THTensor_(new)();
  THTensor_(resizeAs)(r_values_, mask_values_);
  THSTensor_(move)(r_, THLongTensor_newClone(mask_indices_), r_values_);
  r_->contiguous = mask->contiguous;
  r_->nnz = mask->nnz;

  if (nDim > nDimI) {
    THTensor *srcBuffer = THTensor_(new)();
    THTensor *dstBuffer = THTensor_(new)();
    for (long i = 0; i < r_->nnz; i++) {
      THTensor_(set)(srcBuffer, t);
      for (long d = 0; d < nDimI; d++) {
        THTensor_(select)(srcBuffer, srcBuffer, 0, THTensor_fastGet2d(mask_indices_, d, i));
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
