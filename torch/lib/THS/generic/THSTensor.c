#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensor.c"
#else

/******************************************************************************
 * access methods
 ******************************************************************************/

int THSTensor_(nDimension)(const THSTensor *self)
{
  return self->nDimension;
}

long THSTensor_(size)(const THSTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 1, "dimension %d out of range of %dD tensor",
      dim+1, THSTensor_(nDimension)(self));
  return self->size[dim];
}

ptrdiff_t THSTensor_(nnz)(const THSTensor *self) {
  return self->nnz;
}

THLongStorage *THSTensor_(newSizeOf)(THSTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
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
  self->nDimension = 0;
  self->contiguous = 0;
  self->nnz = 0;
  // self->flag = TH_TENSOR_REFCOUNTED;
}

static void THSTensor_(rawResize)(THSTensor *self, int nDim, long *size) {
  // Only resize valid sizes into tensor.
  self->size = THRealloc(self->size, sizeof(long)*nDim);

  long d, nDim_ = 0;
  for (d = 0; d < nDim; d++)
    if (size[d] > 0)
      self->size[nDim_++] = size[d];
  self->nDimension = nDim_;
  self->contiguous = 0;
}

THSTensor* THSTensor_(set)(THSTensor *self, THLongTensor *indices, THTensor *values) {
  // Note: Not like torch.set, this is an internal method
  THArgCheck(THLongTensor_nDimension(indices) == 2, 1,
      "indices must be nDim x nnz");
  THArgCheck(THTensor_(nDimension)(values) == 1, 2, "values must nnz vector");
  THArgCheck(THLongTensor_size(indices, 1) == THTensor_(size)(values, 0), 1,
      "indices and values must have same nnz");
  THLongTensor_free(self->indices);
  THTensor_(free)(self->values);
  self->indices = THLongTensor_newClone(indices);
  self->values = THTensor_(newClone)(values);
  self->nnz = THTensor_(size)(values, 0);

  return self;
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
  long nDim;
  THLongTensor *ignore;

  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  THSTensor_(set)(self, indices, values);

  nDim = THLongTensor_size(indices, 0);
  if (!sizes) {
    ignore = THLongTensor_new();
    THLongTensor *computed_sizes = THLongTensor_new();
    THLongTensor_max(computed_sizes, ignore, indices, 1);
    THLongTensor_add(computed_sizes, computed_sizes, 1);
    THSTensor_(rawResize)(self, nDim, THLongTensor_data(computed_sizes));
    THLongTensor_free(computed_sizes);
    THLongTensor_free(ignore);
  }
  else {
    THSTensor_(rawResize)(self, nDim, THLongStorage_data(sizes));
  }

  return self;
}

THSTensor *THSTensor_(newWithSize)(THLongStorage *size)
{
  THSTensor *self = THAlloc(sizeof(THSTensor));
  THSTensor_(rawInit)(self);
  THSTensor_(rawResize)(self, size->size, size->data);

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
  THSTensor_(rawResize)(self, 4, size);

  return self;
}

THSTensor *THSTensor_(newClone)(THSTensor *self) {
  THSTensor *other = THSTensor_(new)();
  THSTensor_(rawResize)(other, self->nDimension, self->size);

  THSTensor_(set)(
    other,
    THLongTensor_newClone(self->indices),
    THTensor_(newClone)(self->values)
  );

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

THSTensor *THSTensor_(resize)(THSTensor *self, THLongStorage *size)
{
  THSTensor_(rawResize)(self, size->size, size->data);
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
  THSTensor_(rawResize)(self, 4, size);
  return self;
}

THTensor *THSTensor_(toDense)(THSTensor *self) {
  int d, k, index;
  ptrdiff_t nnz;
  long ndim, indskip;
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

  // Some necessary dimensions and sizes
  nnz = THSTensor_(nnz)(self);
  ndim = THSTensor_(nDimension)(self);
  sizes = storage->data;

  // These should be contiguous...
  values_ = THSTensor_(values)(self);
  indices_ = self->indices;
  values = THTensor_(data)(values_);
  indices = THLongTensor_data(indices_);
  indskip = THLongTensor_size(indices_, 1); // To index indices

  #pragma omp parallel for private(k, index)
  for (k = 0; k < nnz; k++) {
    for (d = 0, index = 0; d < ndim; d++)
      index = sizes[d] * index + indices[d * indskip + k];
    other[index] = values[k];
  }

  THTensor_(free)(values_);
  THLongStorage_free(storage);
  return other_;
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

void THSTensor_(reorder)(THSTensor *self) {
  /* TODO: We do an insertion sort here, should change to quicksort or shellsort
  */
  if (self->nnz < 2) return;
  long d, i, j, p, cmp, ndim, indskip, tmplong;
  real tmpreal;
  THLongTensor *indices_ = self->indices;
  THTensor *values_ = self->values;
  long *indices = THLongTensor_data(indices_);
  real *values = THTensor_(data)(values_);
  indskip = THLongTensor_size(indices_, 1); // To index indices
  ndim = THSTensor_(nDimension)(self);

#define IND(i, d) indices[d * indskip + i]
  for (i = 1; i < self->nnz; i++) {
    for (j = i-1; j >= 0; j--) {
      cmp = 0;
      for (d = 0; d < ndim; d++) {
        if (IND(j+1, d) < IND(j, d))
          cmp = 1;
        if (IND(j+1, d) != IND(j, d)) break;
      }
      if (cmp) {
        tmpreal = values[j+1]; values[j+1] = values[j]; values[j] = tmpreal;
        for (d = 0; d < ndim; d++) {
          tmplong = IND(j+1, d); IND(j+1, d) = IND(j, d); IND(j, d) = tmplong;
        }
      } else break;
    }
  }

  i = 0;
  for (j = 1; j < self->nnz; j++) {
    cmp = 1;
    // TODO: pass eps in as a parameter
    if (values[j] == 0) continue;
    for (d = 0; d < ndim; d++)
      if (IND(i, d) != IND(j, d)) {
        cmp = 0;
        break;
      }
    if (cmp) values[i] += values[j];
    else {
      values[++i] = values[j];
      for (d = 0; d < ndim; d++) IND(i, d) = IND(j, d);
    }
  }
  self->nnz = i + 1;
#undef IND
}

void THSTensor_(contiguous)(THSTensor *self) {
  if (self->contiguous) return;
  THSTensor_(reorder)(self);
  self->contiguous = 1;
}

void THSTensor_(free)(THSTensor *self)
{
  if(!self)
    return;
  if(THAtomicDecrementRef(&self->refcount))
  {
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
