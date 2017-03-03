#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.c"
#else

/******************************************************************************
 * access methods
 ******************************************************************************/

int THCSTensor_(nDimension)(THCState *state, const THCSTensor *self)
{
  return self->nDimensionI + self->nDimensionV;
}

int THCSTensor_(nDimensionI)(THCState *state, const THCSTensor *self)
{
  return self->nDimensionI;
}

int THCSTensor_(nDimensionV)(THCState *state, const THCSTensor *self)
{
  return self->nDimensionV;
}

long THCSTensor_(size)(THCState *state, const THCSTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimensionI + self->nDimensionV),
      1, "dimension %d out of range of %dD tensor",
      dim+1, THCSTensor_(nDimension)(state, self));
  return self->size[dim];
}

ptrdiff_t THCSTensor_(nnz)(THCState *state, const THCSTensor *self) {
  return self->nnz;
}

THLongStorage *THCSTensor_(newSizeOf)(THCState *state, THCSTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimensionI + self->nDimensionV);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

/*** TODO: watch out for memory leaks ***/
THCIndexTensor *THCSTensor_(indices)(THCState *state, const THCSTensor *self) {
  if (!self->indices) return self->indices;
  return THCIndexTensor_(newNarrow)(state, self->indices, 1, 0, self->nnz);
}

THCTensor *THCSTensor_(values)(THCState *state, const THCSTensor *self) {
  if (!self->indices) return self->values;
  return THCTensor_(newNarrow)(state, self->values, 0, 0, self->nnz);
}


/******************************************************************************
 * creation methods
 ******************************************************************************/

/*** Helper methods ***/
static void THCSTensor_(rawInit)(THCState *state, THCSTensor *self)
{
  self->size = NULL;
  self->indices = NULL;
  self->values = NULL;
  self->nDimensionI = 0;
  self->nDimensionV = 0;
  self->contiguous = 0;
  self->nnz = 0;
  // self->flag = TH_TENSOR_REFCOUNTED;
  self->refcount = 1;
}

static void THCSTensor_(rawResize)(THCState *state, THCSTensor *self, int nDimI, int nDimV, long *size) {
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

THCSTensor *THCSTensor_(set)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values) {
  THArgCheck(THCIndexTensor_(nDimension)(state, indices) == 2, 2,
      "indices must be nDim x nnz");
  THArgCheck(THCIndexTensor_(size)(state, indices, 1) == THCTensor_(size)(state, values, 0), 2,
      "indices and values must have same nnz");
  THCIndexTensor_(free)(state, self->indices);
  THCTensor_(free)(state, self->values);
  self->indices = THCIndexTensor_(newClone)(state, indices);
  self->values = THCTensor_(newClone)(state, values);
  self->nnz = THCTensor_(size)(state, values, 0);

  return self;
}

/*** end helper methods ***/

/* Empty init */
THCSTensor *THCSTensor_(new)(THCState *state)
{
  THCSTensor *self = THAlloc(sizeof(THCSTensor));
  THCSTensor_(rawInit)(state, self);
  return self;
}

/* Pointer-copy init */
THCSTensor *THCSTensor_(newWithTensor)(THCState *state, THCIndexTensor *indices, THCTensor *values)
{
  return THCSTensor_(newWithTensorAndSize)(state, indices, values, NULL);
}

THCSTensor *THCSTensor_(newWithTensorAndSize)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes)
{  // If sizes are not given, it is inferred as max index of each dim.
  long nDimI, nDimV;

  THCSTensor *self = THAlloc(sizeof(THCSTensor));
  THCSTensor_(rawInit)(state, self);
  THCSTensor_(set)(state, self, indices, values);

  nDimI = THCIndexTensor_(size)(state, indices, 0);
  nDimV = THCTensor_(nDimension)(state, values) - 1;
  if (!sizes) {
    // TODO Make it work with N-dimensional values
    THArgCheck(nDimV > 0, 3, "size must be provided when nDimV > 0");
    THCudaLongTensor *ignore, *s;
    THLongTensor *computed_sizes;
    ignore = THCudaLongTensor_new(state);
    s = THCudaLongTensor_new(state);
    THCudaLongTensor_max(state, s, ignore, indices, 1);
    THCudaLongTensor_add(state, s, s, 1);

    // TODO make sure this doesn't sync the hell out of everything
    //      Should be fine according to sam's memory manager.
    computed_sizes = THLongTensor_newWithSize(THCudaLongTensor_newSizeOf(state, s), NULL);
    THLongTensor_copyCudaLong(state, computed_sizes, s);
    THCSTensor_(rawResize)(state, self, nDimI, nDimV, THLongTensor_data(computed_sizes));

    THCudaFree(state, s);
    THCudaFree(state, ignore);
    THLongTensor_free(computed_sizes);
  }
  else {
    THArgCheck(THLongStorage_size(sizes) == nDimI + nDimV, 3,
        "number of dimensions must be nDimI + nDimV");
    THCSTensor_(rawResize)(state, self, nDimI, nDimV, THLongStorage_data(sizes));
  }

  return self;
}

THCSTensor *THCSTensor_(newWithSize)(THCState *state, THLongStorage *size)
{
  THCSTensor *self = THAlloc(sizeof(THCSTensor));
  THCSTensor_(rawInit)(state, self);
  THCSTensor_(rawResize)(state, self, size->size, 0, size->data);

  return self;
}

THCSTensor *THCSTensor_(newWithSize1d)(THCState *state, long size0)
{
  return THCSTensor_(newWithSize4d)(state, size0, -1, -1, -1);
}

THCSTensor *THCSTensor_(newWithSize2d)(THCState *state, long size0, long size1)
{
  return THCSTensor_(newWithSize4d)(state, size0, size1, -1, -1);
}

THCSTensor *THCSTensor_(newWithSize3d)(THCState *state, long size0, long size1, long size2)
{
  return THCSTensor_(newWithSize4d)(state, size0, size1, size2, -1);
}

THCSTensor *THCSTensor_(newWithSize4d)(THCState *state, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THCSTensor *self = THAlloc(sizeof(THCSTensor));
  THCSTensor_(rawInit)(state, self);
  THCSTensor_(rawResize)(state, self, 4, 0, size);

  return self;
}

THCSTensor *THCSTensor_(newClone)(THCState *state, THCSTensor *self) {
  THCSTensor *other = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, other, self->nDimensionI, self->nDimensionV, self->size);

  THCSTensor_(set)(
      state,
      other,
      THCIndexTensor_(newClone)(state, self->indices),
      THCTensor_(newClone)(state, self->values)
      );

  other->nnz = self->nnz;
  return other;
}

THCSTensor *THCSTensor_(newContiguous)(THCState *state, THCSTensor *self) {
  THCSTensor *other = THCSTensor_(newClone)(state, self);
  THCSTensor_(contiguous)(state, other);
  return other;
}

THCSTensor *THCSTensor_(newTranspose)(THCState *state, THCSTensor *self, int d1, int d2) {
  THCSTensor *other = THCSTensor_(newClone)(state, self);
  THCSTensor_(transpose)(state, other, d1, d2);
  return other;
}


/******************************************************************************
 * reshaping methods
 ******************************************************************************/

int THCSTensor_(isSameSizeAs)(THCState *state, const THCSTensor *self, const THCSTensor* src)
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

THCSTensor *THCSTensor_(resize)(THCState *state, THCSTensor *self, THLongStorage *size)
{
  THCSTensor_(rawResize)(state, self, size->size, 0, size->data);
  return self;
}

THCSTensor *THCSTensor_(resizeAs)(THCState *state, THCSTensor *self, THCSTensor *src)
{
  if(!THCSTensor_(isSameSizeAs)(state, self, src)) {
    THCSTensor_(rawResize)(state, self, src->nDimensionI, src->nDimensionV, src->size);
  }
  return self;
}

THCSTensor *THCSTensor_(resize1d)(THCState *state, THCSTensor *self, long size0)
{
  return THCSTensor_(resize4d)(state, self, size0, -1, -1, -1);
}

THCSTensor *THCSTensor_(resize2d)(THCState *state, THCSTensor *self, long size0, long size1)
{
  return THCSTensor_(resize4d)(state, self, size0, size1, -1, -1);
}

THCSTensor *THCSTensor_(resize3d)(THCState *state, THCSTensor *self, long size0, long size1, long size2)
{
  return THCSTensor_(resize4d)(state, self, size0, size1, size2, -1);
}

THCSTensor *THCSTensor_(resize4d)(THCState *state, THCSTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};
  THCSTensor_(rawResize)(state, self, 4, 0, size);
  return self;
}

int THCSTensor_(isContiguous)(THCState *state, const THCSTensor *self) {
  return self->contiguous;
}

void THCSTensor_(free)(THCState *state, THCSTensor *self)
{
  if(!self)
    return;
  if(THAtomicDecrementRef(&self->refcount))
  {
    THFree(self->size);
    THCIndexTensor_(free)(state, self->indices);
    THCTensor_(free)(state, self->values);
    THFree(self);
  }
}

void THCSTensor_(retain)(THCState *state, THCSTensor *self)
{
  THAtomicIncrementRef(&self->refcount);
}

#endif
