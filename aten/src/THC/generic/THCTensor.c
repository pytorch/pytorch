#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.c"
#else

/**** access methods ****/
THCStorage *THCTensor_(storage)(THCState *state, const THCTensor *self)
{
  return self->storage;
}

ptrdiff_t THCTensor_(storageOffset)(THCState *state, const THCTensor *self)
{
  return self->storageOffset;
}

int THCTensor_(nDimension)(THCState *state, const THCTensor *self)
{
  return self->nDimension;
}

int64_t THCTensor_(size)(THCState *state, const THCTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

int64_t THCTensor_(stride)(THCState *state, const THCTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THCTensor_(newSizeOf)(THCState *state, THCTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THCTensor_(newStrideOf)(THCState *state, THCTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

real *THCTensor_(data)(THCState *state, const THCTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THCTensor_(setFlag)(THCState *state, THCTensor *self, const char flag)
{
  self->flag |= flag;
}

void THCTensor_(clearFlag)(THCState *state, THCTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THCTensor_(rawInit)(THCState *state, THCTensor *self);


/* Empty init */
THCTensor *THCTensor_(new)(THCState *state)
{
  THCTensor *self = (THCTensor*)THAlloc(sizeof(THCTensor));
  THCTensor_(rawInit)(state, self);
  return self;
}

/* Pointer-copy init */
THCTensor *THCTensor_(newWithTensor)(THCState *state, THCTensor *tensor)
{
  THCTensor *self = (THCTensor*)THAlloc(sizeof(THCTensor));
  THCTensor_(rawInit)(state, self);
  THCTensor_(setStorageNd)(state,
                           self,
                           tensor->storage,
                           tensor->storageOffset,
                           tensor->nDimension,
                           tensor->size,
                           tensor->stride);
  return self;
}

/* Storage init */
THCTensor *THCTensor_(newWithStorage)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THCTensor *self = (THCTensor*)THAlloc(sizeof(THCTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THCTensor_(rawInit)(state, self);
  THCTensor_(setStorageNd)(state,
                           self,
                           storage,
                           storageOffset,
                           (size ? size->size : (stride ? stride->size : 0)),
                           (size ? size->data : NULL),
                           (stride ? stride->data : NULL));

  return self;
}
THCTensor *THCTensor_(newWithStorage1d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THCTensor_(newWithStorage4d)(state, storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THCTensor *THCTensor_(newWithStorage2d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THCTensor_(newWithStorage4d)(state, storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THCTensor *THCTensor_(newWithStorage3d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THCTensor_(newWithStorage4d)(state, storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THCTensor *THCTensor_(newWithStorage4d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  int64_t stride[4] = {stride0, stride1, stride2, stride3};

  THCTensor *self = (THCTensor*)THAlloc(sizeof(THCTensor));
  THCTensor_(rawInit)(state, self);
  THCTensor_(setStorageNd)(state, self, storage, storageOffset, 4, size, stride);

  return self;
}

THCTensor *THCTensor_(newWithSize)(THCState *state, THLongStorage *size, THLongStorage *stride)
{
  return THCTensor_(newWithStorage)(state, NULL, 0, size, stride);
}

THCTensor *THCTensor_(newWithSize1d)(THCState *state, int64_t size0)
{
  return THCTensor_(newWithSize4d)(state, size0, -1, -1, -1);
}

THCTensor *THCTensor_(newWithSize2d)(THCState *state, int64_t size0, int64_t size1)
{
  return THCTensor_(newWithSize4d)(state, size0, size1, -1, -1);
}

THCTensor *THCTensor_(newWithSize3d)(THCState *state, int64_t size0, int64_t size1, int64_t size2)
{
  return THCTensor_(newWithSize4d)(state, size0, size1, size2, -1);
}

THCTensor *THCTensor_(newWithSize4d)(THCState *state, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THCTensor *self = (THCTensor*)THAlloc(sizeof(THCTensor));
  THCTensor_(rawInit)(state, self);
  THCTensor_(resizeNd)(state, self, 4, size, NULL);

  return self;
}

THCTensor *THCTensor_(newClone)(THCState *state, THCTensor *self)
{
  THCTensor *tensor = THCTensor_(new)(state);
  THCTensor_(resizeAs)(state, tensor, self);
  THCTensor_(copy)(state, tensor, self);
  return tensor;
}

THCTensor *THCTensor_(newContiguous)(THCState *state, THCTensor *self)
{
  if(!THCTensor_(isContiguous)(state, self)) {
    return THCTensor_(newClone)(state, self);
  } else {
    THCTensor_(retain)(state, self);
    return self;
  }
}

THCTensor *THCTensor_(newSelect)(THCState *state, THCTensor *tensor, int dimension_, int64_t sliceIndex_)
{
  THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(select)(state, self, NULL, dimension_, sliceIndex_);
  return self;
}

THCTensor *THCTensor_(newNarrow)(THCState *state, THCTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_)
{
  THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(narrow)(state, self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THCTensor *THCTensor_(newTranspose)(THCState *state, THCTensor *tensor, int dimension1_, int dimension2_)
{
  THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(transpose)(state, self, NULL, dimension1_, dimension2_);
  return self;
}

THCTensor *THCTensor_(newUnfold)(THCState *state, THCTensor *tensor, int dimension_, int64_t size_, int64_t step_)
{
  THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(unfold)(state, self, NULL, dimension_, size_, step_);
  return self;
}

// Also sets new_stride if viewable.
//
// On a high level,
// 1. separate tensor->size into chunks of dimensions, where the dimensions are
//    ``contiguous'' in each chunk, i.e., stride[i] = size[i+1] * stride[i+1]
// 2. view_size must be able to be separated into same number of chunks, where
//    each chunk pair has matching ``numel'', i.e., number of subspaces.
static int THCTensor_(isViewable)(THCState *state, THCTensor *tensor, THLongStorage *view_size, THLongStorage *new_stride) {
  // dim indices
  int64_t tensor_d = tensor->nDimension - 1;
  if (tensor_d < 0) {
    return 1;
  }
  int64_t view_d = view_size->size - 1;
  // stride for each subspace in the chunk
  int64_t chunk_base_stride = tensor->stride[tensor_d];
  // numel in current chunk
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;
  for (; tensor_d >= 0; tensor_d--) {
    tensor_numel *= tensor->size[tensor_d];
    // if end of tensor size chunk, check view
    if ((tensor_d == 0) ||
        (tensor->size[tensor_d - 1] != 1 && tensor->stride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      while (view_d >= 0 && (view_numel < tensor_numel || view_size->data[view_d] == 1)) {
        new_stride->data[view_d] = view_numel * chunk_base_stride;
        view_numel *= view_size->data[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) {
        return 0;
      }
      if (tensor_d > 0) {
        chunk_base_stride = tensor->stride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  // check that we iterated through all view size
  return view_d == -1;
}

THCTensor *THCTensor_(newView)(THCState *state, THCTensor *tensor, THLongStorage *size)
{
  ptrdiff_t numel = THCTensor_(nElement)(state, tensor);
  THCTensor *self = THCTensor_(new)(state);
  THLongStorage *inferred_size = THLongStorage_newInferSize(size, numel);
  THLongStorage *new_stride = THLongStorage_newWithSize(size->size);
  THArgCheck(THCTensor_(isViewable)(state, tensor, inferred_size, new_stride), 2, "View size is "
    "not compatible with input tensor's size and stride (at least one dimension spans "
    "across two contiguous subspaces). Call .contiguous() before .view().");
  THCTensor_(setStorage)(state, self, tensor->storage, tensor->storageOffset, inferred_size, new_stride);
  THLongStorage_free(inferred_size);
  THLongStorage_free(new_stride);
  return self;
}

// Collapses the first two dimensions of a tensor.
// Assumes the input tensor is contiguous.
THCTensor *THCTensor_(newFoldBatchDim)(THCState *state, THCTensor *input) {
  int in_dims = THCTensor_(nDimension)(state, input);
  THArgCheck(in_dims >= 2, 1, "Tensor needs to have at least two dimensions");
  THArgCheck(THCTensor_(isContiguous)(state, input), 1,
             "Tensor must be contiguous");
  THLongStorage *newSize = THLongStorage_newWithSize(in_dims - 1);
  newSize->data[0] = THCTensor_(size)(state, input, 0) * THCTensor_(size)(state, input, 1);
  for (int i = 2; i < in_dims; i++) {
    newSize->data[i - 1] = THCTensor_(size)(state, input, i);
  }
  THCTensor *output = THCTensor_(newView)(state, input, newSize);
  THLongStorage_free(newSize);
  return output;
}

/* Resize */
void THCTensor_(resize)(THCState *state, THCTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THCTensor_(resizeNd)(state, self, size->size, size->data, (stride ? stride->data : NULL));
}

void THCTensor_(resizeAs)(THCState *state, THCTensor *self, THCTensor *src)
{
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THCTensor_(resizeNd)(state, self, src->nDimension, src->size, NULL);
}

void THCTensor_(resize1d)(THCState *state, THCTensor *tensor, int64_t size0)
{
  THCTensor_(resize4d)(state, tensor, size0, -1, -1, -1);
}

void THCTensor_(resize2d)(THCState *state, THCTensor *tensor, int64_t size0, int64_t size1)
{
  THCTensor_(resize4d)(state, tensor, size0, size1, -1, -1);
}

void THCTensor_(resize3d)(THCState *state, THCTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  THCTensor_(resize4d)(state, tensor, size0, size1, size2, -1);
}

void THCTensor_(resize4d)(THCState *state, THCTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THCTensor_(resizeNd)(state, self, 4, size, NULL);
}

void THCTensor_(resize5d)(THCState *state, THCTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
    int64_t size[5] = {size0, size1, size2, size3, size4};

  THCTensor_(resizeNd)(state, self, 5, size, NULL);
}

void THCTensor_(set)(THCState *state, THCTensor *self, THCTensor *src)
{
  if(self != src)
    THCTensor_(setStorageNd)(state,
                             self,
                             src->storage,
                             src->storageOffset,
                             src->nDimension,
                             src->size,
                             src->stride);
}

void THCTensor_(setStorage)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THCTensor_(setStorageNd)(state,
                           self,
                           storage_,
                           storageOffset_,
                           (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                           (size_ ? size_->data : NULL),
                           (stride_ ? stride_->data : NULL));
}

void THCTensor_(setStorage1d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THCTensor_(setStorage4d)(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            -1, -1,
                            -1, -1,
                            -1, -1);
}

void THCTensor_(setStorage2d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THCTensor_(setStorage4d)(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            -1, -1,
                            -1, -1);
}

void THCTensor_(setStorage3d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THCTensor_(setStorage4d)(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            size2_, stride2_,
                            -1, -1);
}

void THCTensor_(setStorage4d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THCTensor_(setStorageNd)(state, self, storage_, storageOffset_, 4, size, stride);
}


void THCTensor_(narrow)(THCState *state, THCTensor *self, THCTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 4, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 5, "out of range");

  THCTensor_(set)(state, self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THCTensor_(select)(THCState *state, THCTensor *self, THCTensor *src, int dimension, int64_t sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THCTensor_(set)(state, self, src);
  THCTensor_(narrow)(state, self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THCTensor_(transpose)(THCState *state, THCTensor *self, THCTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THCTensor_(set)(state, self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THCTensor_(unfold)(THCState *state, THCTensor *self, THCTensor *src, int dimension, int64_t size, int64_t step)
{
  int64_t *newSize;
  int64_t *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THCTensor_(set)(state, self, src);

  newSize = (int64_t*)THAlloc(sizeof(int64_t)*(self->nDimension+1));
  newStride = (int64_t*)THAlloc(sizeof(int64_t)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;
}

/* we have to handle the case where the result is a number */
void THCTensor_(squeeze)(THCState *state, THCTensor *self, THCTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THCTensor_(set)(state, self, src);

  for(d = 0; d < src->nDimension; d++)
  {
    if(src->size[d] != 1)
    {
      if(d != ndim)
      {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->nDimension > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
}

void THCTensor_(squeeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THCTensor_(set)(state, self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}

void THCTensor_(unsqueeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->nDimension), 3, "dimension out of range");
  THArgCheck(src->nDimension > 0, 3, "cannot unsqueeze empty tensor");

  THCTensor_(set)(state, self, src);

  self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*(self->nDimension+1));
  self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*(self->nDimension+1));
  self->nDimension++;
  for (d = self->nDimension-1; d > dimension; d--) {
    self->size[d] = self->size[d-1];
    self->stride[d] = self->stride[d-1];
  }
  if (dimension+1 < self->nDimension) {
    self->stride[dimension] = self->size[dimension+1] * self->stride[dimension+1];
  } else {
    self->stride[dimension] = 1;
  }
  self->size[dimension] = 1;
}

int THCTensor_(isContiguous)(THCState *state, const THCTensor *self)
{
  int64_t z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THCTensor_(isSize)(THCState *state, const THCTensor *self, const THLongStorage *dims)
{
  int d;
  if (self->nDimension != dims->size)
    return 0;

  for (d = 0; d < self->nDimension; ++d)
  {
    if (self->size[d] != dims->data[d])
      return 0;
  }
  return 1;
}

int THCTensor_(isSetTo)(THCState *state, const THCTensor *self, const THCTensor *src)
{
  if (self->storage == src->storage &&
      self->storageOffset == src->storageOffset &&
      self->nDimension == src->nDimension)
  {
    int d;
    for (d = 0; d < self->nDimension; ++d)
    {
      if (self->size[d] != src->size[d] || self->stride[d] != src->stride[d])
        return 0;
    }
    return 1;
  }
  return 0;
}

int THCTensor_(isSameSizeAs)(THCState *state, const THCTensor *self, const THCTensor* src)
{
  int d;
  if (self->nDimension != src->nDimension)
    return 0;
  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

ptrdiff_t THCTensor_(nElement)(THCState *state, const THCTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THCTensor_(retain)(THCState *state, THCTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THCTensor_(free)(THCState *state, THCTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(THAtomicDecrementRef(&self->refcount))
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THCStorage_(free)(state, self->storage);
      THFree(self);
    }
  }
}

void THCTensor_(freeCopyTo)(THCState *state, THCTensor *self, THCTensor *dst)
{
  if(self != dst)
    THCTensor_(copy)(state, dst, self);

  THCTensor_(free)(state, self);
}

/*******************************************************************************/

static void THCTensor_(rawInit)(THCState *state, THCTensor *self)
{
  self->refcount = 1;
  self->storage = THCStorage_(new)(state);
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

void THCTensor_(setStorageNd)(THCState *state, THCTensor *self, THCStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THCStorage_(free)(state, self->storage);

    if(storage)
    {
      self->storage = storage;
      THCStorage_(retain)(state, self->storage);
    }
    else
      self->storage = THCStorage_(new)(state);
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THCTensor_(resizeNd)(state, self, nDimension, size, stride);
}

void THCTensor_(resizeNd)(THCState *state, THCTensor *self, int nDimension, int64_t *size, int64_t *stride)
{
  int d;
  int nDimension_;
  ptrdiff_t totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*nDimension);
      self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THCStorage_(new)(state);
      if(totalSize+self->storageOffset > self->storage->size)
        THCStorage_(resize)(state, self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THCTensor_(set1d)(THCState *state, THCTensor *tensor, int64_t x0, real value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

real THCTensor_(get1d)(THCState *state, const THCTensor *tensor, int64_t x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THCStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THCTensor_(set2d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, real value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

real THCTensor_(get2d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THCStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THCTensor_(set3d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, real value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

real THCTensor_(get3d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THCStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THCTensor_(set4d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, real value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

real THCTensor_(get4d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THCStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

int THCTensor_(checkGPU)(THCState *state, unsigned int nTensors, ...)
{
  /* FIXME: remove this flag after any users stop using it since it is
     now superseded by the runtime option */
#ifdef DISABLE_CHECK_GPU
  return 1;
#else
  int kernelP2PEnabled =
    THCState_getKernelPeerToPeerAccessEnabled(state);

  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));
  va_list(args);
  va_start(args, nTensors);
  int valid = 1;
  for (unsigned int i = 0; i < nTensors; i++) {
    THCTensor* tensor = va_arg(args, THCTensor*);
    if (tensor == NULL) {
      continue;
    }
    int tensorDev = THCTensor_(getDevice)(state, tensor);
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

THCDescBuff THCTensor_(sizeDesc)(THCState *state, const THCTensor *tensor) {
  const int L = THC_DESC_BUFF_LEN;
  THCDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for(i = 0; i < tensor->nDimension; i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size[i]);
    if(i < tensor->nDimension-1) {
      n += snprintf(str+n, L-n, " x ");
    }
  }
  if(n < L - 2) {
    snprintf(str+n, L-n, "]");
  } else {
    snprintf(str+L-5, 5, "...]");
  }
  return buf;
}

#endif
