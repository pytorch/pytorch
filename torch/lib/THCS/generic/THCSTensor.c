#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.c"
#else

/******************************************************************************
 * access methods
 ******************************************************************************/

int THCSTensor_(nDimension)(THCState *state, const THCSTensor *self)
{
  return self->nDimension;
}

long THCSTensor_(size)(THCState *state, const THCSTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 1, "dimension %d out of range of %dD tensor",
      dim+1, THCSTensor_(nDimension)(state, self));
  return self->size[dim];
}

ptrdiff_t THCSTensor_(nnz)(THCState *state, const THCSTensor *self) {
  return self->nnz;
}

THLongStorage *THCSTensor_(newSizeOf)(THCState *state, THCSTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
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
  self->nDimension = 0;
  self->contiguous = 0;
  self->nnz = 0;
  // self->flag = TH_TENSOR_REFCOUNTED;
}

static void THCSTensor_(rawResize)(THCState *state, THCSTensor *self, int nDim, long *size) {
  // Only resize valid sizes into tensor.
  self->size = THRealloc(self->size, sizeof(long)*nDim);

  long d, nDim_ = 0;
  for (d = 0; d < nDim; d++)
    if (size[d] > 0)
      self->size[nDim_++] = size[d];
  self->nDimension = nDim_;
  self->contiguous = 0;
}

THCSTensor *THCSTensor_(set)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values) {
  THArgCheck(THCIndexTensor_(nDimension)(state, indices) == 2, 1,
      "indices must be nDim x nnz");
  THArgCheck(THCTensor_(nDimension)(state, values) == 1, 2, "values must nnz vector");
  THArgCheck(THCIndexTensor_(size)(state, indices, 1) == THCTensor_(size)(state, values, 0), 1,
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
  long nDim;

  THCSTensor *self = THAlloc(sizeof(THCSTensor));
  THCSTensor_(rawInit)(state, self);
  THCSTensor_(set)(state, self, indices, values);

  nDim = THCIndexTensor_(size)(state, indices, 0);
  if (!sizes) {
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
    THCSTensor_(rawResize)(state, self, nDim, THLongTensor_data(computed_sizes));

    THCudaFree(state, s);
    THCudaFree(state, ignore);
    THLongTensor_free(computed_sizes);
  }
  else {
    THCSTensor_(rawResize)(state, self, nDim, THLongStorage_data(sizes));
  }

  return self;
}

THCSTensor *THCSTensor_(newWithSize)(THCState *state, THLongStorage *size)
{
  THCSTensor *self = THAlloc(sizeof(THCSTensor));
  THCSTensor_(rawInit)(state, self);
  THCSTensor_(rawResize)(state, self, size->size, size->data);

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
  THCSTensor_(rawResize)(state, self, 4, size);

  return self;
}

THCSTensor *THCSTensor_(newClone)(THCState *state, THCSTensor *self) {
  THCSTensor *other = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, other, self->nDimension, self->size);

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

THCSTensor *THCSTensor_(resize)(THCState *state, THCSTensor *self, THLongStorage *size)
{
  THCSTensor_(rawResize)(state, self, size->size, size->data);
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
  THCSTensor_(rawResize)(state, self, 4, size);
  return self;
}

int THCSTensor_(isContiguous)(THCState *state, const THCSTensor *self) {
  return self->contiguous;
}

void THCSTensor_(free)(THCState *state, THCSTensor *self)
{
  if(!self)
    return;

  THCIndexTensor_(free)(state, self->indices);
  THCTensor_(free)(state, self->values);
  THFree(self);
}

#endif
