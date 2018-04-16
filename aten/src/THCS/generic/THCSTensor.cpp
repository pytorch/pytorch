#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.cpp"
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

int64_t THCSTensor_(size)(THCState *state, const THCSTensor *self, int dim)
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
THCIndexTensor *THCSTensor_(newIndices)(THCState *state, const THCSTensor *self) {
  if (self->nnz == 0) {
    // Narrows don't work on 0-length tensors
    THCIndexTensor_(retain)(state, self->indices);
    return self->indices;
  }
  return THCIndexTensor_(newNarrow)(state, self->indices, 1, 0, self->nnz);
}

THCTensor *THCSTensor_(newValues)(THCState *state, const THCSTensor *self) {
  if (self->nnz == 0) {
    THCTensor_(retain)(state, self->values);
    return self->values;
  }
  return THCTensor_(newNarrow)(state, self->values, 0, 0, self->nnz);
}


/******************************************************************************
 * creation methods
 ******************************************************************************/

/*** Helper methods ***/
static void THCSTensor_(rawInit)(THCState *state, THCSTensor *self)
{
  self->size = NULL;
  self->indices = THCIndexTensor_(new)(state);
  self->values = THCTensor_(new)(state);
  self->nDimensionI = 0;
  self->nDimensionV = 0;
  self->coalesced = 0;
  self->nnz = 0;
  // self->flag = TH_TENSOR_REFCOUNTED;
  self->refcount = 1;
}

THCSTensor* THCSTensor_(rawResize)(THCState *state, THCSTensor *self, int nDimI, int nDimV, int64_t *size) {
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
THCSTensor* THCSTensor_(_move)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values) {
  int empty = THCTensor_(nDimension)(state, values) == 0;
  if (!empty) {
    THArgCheck(THCIndexTensor_(nDimension)(state, indices) == 2, 2,
        "indices must be nDim x nnz");
    THArgCheck(THCIndexTensor_(size)(state, indices, 1) == THCTensor_(size)(state, values, 0), 2,
        "indices and values must have same nnz");
    THArgCheck(THCIndexTensor_(size)(state, indices, 0) == self->nDimensionI, 2,
        "indices has incorrect first dimension, expected %d, got %d", self->nDimensionI, THCIndexTensor_(size)(state, indices, 0));
    THArgCheck(THCTensor_(nDimension)(state, values) == self->nDimensionV + 1, 3,
        "values has incorrect number of dimensions, expected %d, got %d", self->nDimensionV + 1, THCTensor_(nDimension)(state, values));
  } else {
    THArgCheck(THCIndexTensor_(nDimension)(state, indices) == 0, 2,
        "if values is empty, indices must be empty too");
  }
  THCIndexTensor_(free)(state, self->indices);
  THCTensor_(free)(state, self->values);
  self->indices = indices;
  self->values = values;
  self->nnz = empty ? 0 : THCTensor_(size)(state, values, 0);
  self->coalesced = 0;

  return self;
}

THCSTensor* THCSTensor_(_set)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values) {
  // Note: Not like torch.set, this is an internal method
  return THCSTensor_(_move)(state, self, THCIndexTensor_(newClone)(state, indices), THCTensor_(newClone)(state, values));
}

static inline THCSTensor* THCSTensor_(_newWithDimsAndTensor)(THCState *state, int64_t nDimI, int64_t nDimV, int64_t *sizes, THCIndexTensor *indices, THCTensor *values) {
  THCSTensor *self = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, self, nDimI, nDimV, sizes);

  // NB: by default, we do NOT clone indices/values into the sparse tensor.
  // Efficient API by default!
  THCSTensor_(_move)(state, self, THCIndexTensor_(newWithTensor)(state, indices), THCTensor_(newWithTensor)(state, values));
  return self;
}

/*** end helper methods ***/

/* Empty init */
THCSTensor *THCSTensor_(new)(THCState *state)
{
  THCSTensor *self = (THCSTensor *)THAlloc(sizeof(THCSTensor));
  THCSTensor_(rawInit)(state, self);
  return self;
}

/* Pointer-copy init */
THCSTensor *THCSTensor_(newWithTensor)(THCState *state, THCIndexTensor *indices, THCTensor *values)
{
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 0, 2, indices, values));

  int64_t nDimI = THCIndexTensor_(size)(state, indices, 0);
  int64_t nDimV = THCTensor_(nDimension)(state, values) - 1;

  // TODO Make it work with N-dimensional values
  THArgCheck(nDimV > 0, 3, "size must be provided when nDimV > 0");
  THLongTensor *computed_sizes;
  THCudaLongTensor *ignore = THCudaLongTensor_new(state);
  THCIndexTensor *s = THCIndexTensor_(new)(state);
  THCIndexTensor_(max)(state, s, ignore, indices, 1, 1);
  THCIndexTensor_(add)(state, s, s, 1);

  // TODO make sure this doesn't sync the hell out of everything
  //      Should be fine according to sam's memory manager.
  THLongStorage *newSize = THCIndexTensor_(newSizeOf)(state, s);
  computed_sizes = THLongTensor_newWithSize(newSize, NULL);
  THLongStorage_free(newSize);
  THLongTensor_copyCudaLong(state, computed_sizes, s);

  THCSTensor *self = THCSTensor_(_newWithDimsAndTensor)(state, nDimI, nDimV, THLongTensor_data(computed_sizes), indices, values);
  THCIndexTensor_(free)(state, s);
  THCudaLongTensor_free(state, ignore);
  THLongTensor_free(computed_sizes);
  return self;
}

THCSTensor *THCSTensor_(newWithTensorAndSizeUnsafe)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes)
{
  if (sizes == NULL) {
    return THCSTensor_(newWithTensor)(state, indices, values);
  }
  if (THCudaLongTensor_nDimension(state, indices) == 0 && THCTensor_(nDimension)(state, values) == 0) {
    return THCSTensor_(newWithSize)(state, sizes, NULL);
  }

  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 0, 2, indices, values));

  int64_t nDimI = THCIndexTensor_(size)(state, indices, 0);
  int64_t nDimV = THCTensor_(nDimension)(state, values) - 1;

  return THCSTensor_(_newWithDimsAndTensor)(state, nDimI, nDimV, THLongStorage_data(sizes), indices, values);
}

THCSTensor *THCSTensor_(newWithTensorAndSize)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes)
{
  if (sizes == NULL) {
    return THCSTensor_(newWithTensor)(state, indices, values);
  }
  if (THCudaLongTensor_nDimension(state, indices) == 0 && THCTensor_(nDimension)(state, values) == 0) {
    return THCSTensor_(newWithSize)(state, sizes, NULL);
  }

  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 0, 2, indices, values));

  int64_t nDimI = THCIndexTensor_(size)(state, indices, 0);
  int64_t nDimV = THCTensor_(nDimension)(state, values) - 1;
  THArgCheck(THLongStorage_size(sizes) == nDimI + nDimV, 3,
      "number of dimensions must be nDimI + nDimV");

  THCudaLongTensor *max_indices = THCudaLongTensor_new(state);
  THCudaLongTensor *ignore = THCudaLongTensor_new(state);
  THCudaLongTensor_max(state, max_indices, ignore, indices, 1, 0);
  THCudaLongTensor_free(state, ignore);
  for (int d = 0; d < nDimI; d++) {
    int64_t max_index_in_dim = THCudaLongTensor_get1d(state, max_indices, d);
    int64_t dim_size = sizes->data[d];
    THArgCheck(max_index_in_dim < dim_size, 2,
        "sizes is inconsistent with indices: for dim %d, size is %lld but found index %lld",
        d, (long long)dim_size, (long long)max_index_in_dim);
  }
  for (int d = 0; d < nDimV; d++) {
    int64_t values_size = THCTensor_(size)(state, values, d + 1);
    int64_t specified_size = sizes->data[nDimI + d];
    THArgCheck(values_size <= specified_size, 2,
        "values and sizes are inconsistent: sizes[%d] is %lld but values.size(%d) is %lld",
        d + nDimI, (long long)specified_size, d + 1, (long long)values_size);
  }
  THCudaLongTensor_free(state, max_indices);

  return THCSTensor_(_newWithDimsAndTensor)(state, nDimI, nDimV, THLongStorage_data(sizes), indices, values);
}

THCSTensor *THCSTensor_(newWithSize)(THCState *state, THLongStorage *size, THLongStorage *_ignored)
{
  THCSTensor *self = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, self, size->size, 0, size->data);
  return self;
}

THCSTensor *THCSTensor_(newWithSize1d)(THCState *state, int64_t size0)
{
  int64_t size[1] = {size0};
  THCSTensor *self = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, self, 1, 0, size);
  return self;
}

THCSTensor *THCSTensor_(newWithSize2d)(THCState *state, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THCSTensor *self = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, self, 2, 0, size);
  return self;
}

THCSTensor *THCSTensor_(newWithSize3d)(THCState *state, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THCSTensor *self = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, self, 3, 0, size);
  return self;
}

THCSTensor *THCSTensor_(newWithSize4d)(THCState *state, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THCSTensor *self = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, self, 4, 0, size);
  return self;
}

THCSTensor *THCSTensor_(newClone)(THCState *state, THCSTensor *self) {
  THCSTensor *other = THCSTensor_(new)(state);
  THCSTensor_(rawResize)(state, other, self->nDimensionI, self->nDimensionV, self->size);

  THCSTensor_(_set)(state, other, self->indices, self->values);

  other->nnz = self->nnz;
  other->coalesced = self->coalesced;
  return other;
}

THCSTensor *THCSTensor_(newTranspose)(THCState *state, THCSTensor *self, int d1, int d2) {
  THCSTensor *other = THCSTensor_(newClone)(state, self);
  THCSTensor_(transpose)(state, other, d1, d2);
  return other;
}

THCTensor *THCSTensor_(newValuesWithSizeOf)(THCState *state, THCTensor *values, int64_t nnz) {
  THCTensor *new_values;
  if (THCTensor_(nDimension)(state, values) == 0) { // values tensor uninitialized
    new_values = THCTensor_(newWithSize1d)(state, nnz);
  } else {
    THLongStorage *size = THCTensor_(newSizeOf)(state, values);
    size->data[0] = nnz;
    new_values = THCTensor_(newWithSize)(state, size, NULL);
    THLongStorage_free(size);
  }
  return new_values;
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

int THCSTensor_(isSameSizeAs)(THCState *state, const THCSTensor *self, const THCSTensor* src)
{
  if (self->nDimensionI != src->nDimensionI || self->nDimensionV != src->nDimensionV)
    return 0;
  for(int d = 0; d < self->nDimensionI + self->nDimensionV; ++d) {
    if(self->size[d] != src->size[d]) {
      return 0;
    }
  }
  return 1;
}

int THCSTensor_(isSameSizeAsDense)(THCState *state, const THCSTensor *self, const THCTensor* src)
{
  if (self->nDimensionI + self->nDimensionV != src->nDimension)
    return 0;
  for(int d = 0; d < src->nDimension; ++d) {
    if(self->size[d] != src->size[d]) {
      return 0;
    }
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

THCSTensor *THCSTensor_(resize1d)(THCState *state, THCSTensor *self, int64_t size0)
{
  int64_t size[1] = {size0};
  THCSTensor_(rawResize)(state, self, 1, 0, size);
  return self;
}

THCSTensor *THCSTensor_(resize2d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THCSTensor_(rawResize)(state, self, 2, 0, size);
  return self;
}

THCSTensor *THCSTensor_(resize3d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THCSTensor_(rawResize)(state, self, 3, 0, size);
  return self;
}

THCSTensor *THCSTensor_(resize4d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THCSTensor_(rawResize)(state, self, 4, 0, size);
  return self;
}

void THCSTensor_(copy)(THCState *state, THCSTensor *self, THCSTensor *src) {
  if (self == src) return;
  THCSTensor_(rawResize)(state, self, src->nDimensionI, src->nDimensionV, src->size);
  THCSTensor_(_set)(state, self, src->indices, src->values);
  self->nnz = src->nnz;
  self->coalesced = src->coalesced;
}

int THCSTensor_(isCoalesced)(THCState *state, const THCSTensor *self) {
  return self->coalesced;
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

int THCSTensor_(checkGPU)(THCState *state, unsigned int nSparseTensors, unsigned int nTensors, ...)
{
  /* FIXME: remove this flag after any users stop using it since it is
     now superseded by the runtime option */
#ifdef DISABLE_CHECK_GPU
  return 1;
#else
  int kernelP2PEnabled =
    THCState_getKernelPeerToPeerAccessEnabled(state);

  int curDev = -1;
  unsigned int nDenseTensors = nTensors - nSparseTensors;
  THCudaCheck(cudaGetDevice(&curDev));
  va_list(args);
  va_start(args, nTensors);
  int valid = 1;
  for (unsigned int i = 0; i < nSparseTensors + nDenseTensors; i++) {
    THCSTensor *sparseTensor;
    THCTensor *denseTensor;
    if (i < nSparseTensors) {
      sparseTensor = va_arg(args, THCSTensor*);
      if (sparseTensor == NULL) {
        continue;
      }
    } else {
      denseTensor = va_arg(args, THCTensor*);
      if (denseTensor == NULL) {
        continue;
      }
    }
    int tensorDev = i < nSparseTensors ?
      THCSTensor_(getDevice)(state, sparseTensor) :
      THCTensor_(getDevice)(state, denseTensor);
    if (tensorDev == -1) {
      /* This tensor does not have GPU memory (empty) */
      continue;
    }

    if (tensorDev != curDev) {
      if (kernelP2PEnabled) {
        /* Kernel p2p access is allowed */
        /* Can `curDev` access `tensorDev` directly? */
        if (!THCState_getPeerToPeerAccess(state, curDev, tensorDev)) {
          valid = 0;
          break;
        }
      } else {
        /* No kernel p2p access allowed */
        valid = 0;
        break;
      }
    }
  }

  va_end(args);
  return valid;
#endif // DISABLE_CHECK_GPU
}

void THCTensor_(sparseMask)(THCState *state, THCSTensor *r_, THCTensor *t, THCSTensor *mask) {
  THArgCheck(mask->coalesced, 2, "mask is uncoalesced");
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 2, 3, r_, mask, t));
  if(!THCSTensor_(isSameSizeAsDense)(state, mask, t)) {
    THError("sparseMask operands have incompatible sizes");
  }
  THCSTensor_(resizeAs)(state, r_, mask);
  if (mask->nnz == 0) {
    THCSTensor_(zero)(state, r_);
    return;
  }
  THCIndexTensor *maskIndices = THCSTensor_(newIndices)(state, mask);
  THCTensor *maskValues = THCSTensor_(newValues)(state, mask);
  THCTensor *rValues = THCTensor_(new)(state);
  THCTensor_(resizeAs)(state, rValues, maskValues);
  THCSTensor_(_move)(state, r_, THCIndexTensor_(newClone)(state, maskIndices), rValues);
  r_->coalesced = mask->coalesced;
  r_->nnz = mask->nnz;

  THCudaLongTensor *indices = THCudaLongTensor_newWithSize1d(state, mask->nnz);
  THCudaLongTensor *indicesBuffer = THCudaLongTensor_new(state);

  THCudaLongTensor_zero(state, indices);
  for (int64_t d = 0; d < mask->nDimensionI; d++) {
    THCudaLongTensor_mul(state, indices, indices, mask->size[d]);
    THCudaLongTensor_select(state, indicesBuffer, maskIndices, 0, d);
    THCudaLongTensor_cadd(state, indices, indices, 1, indicesBuffer);
  }
  THLongStorage *viewSize = THLongStorage_newWithSize(1 + mask->nDimensionV);
  viewSize->data[0] = -1;
  for (int64_t d = 0; d < mask->nDimensionV; d++) {
    viewSize->data[1 + d] = mask->size[mask->nDimensionI + d];
  }
  THCTensor *t_view = THCTensor_(newView)(state, t, viewSize);
  THCTensor_(indexSelect)(state, rValues, t_view, 0, indices);

  THCudaLongTensor_free(state, indices);
  THCudaLongTensor_free(state, indicesBuffer);
  THLongStorage_free(viewSize);
  THCTensor_(free)(state, t_view);
  THCIndexTensor_(free)(state, maskIndices);
  THCTensor_(free)(state, maskValues);
}

#endif
