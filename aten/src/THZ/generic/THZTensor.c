#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensor.c"
#else

/**** access methods ****/
THZStorage *THZTensor_(storage)(const THZTensor *self)
{
  return self->storage;
}

ptrdiff_t THZTensor_(storageOffset)(const THZTensor *self)
{
  return self->storageOffset;
}

int THZTensor_(nDimension)(const THZTensor *self)
{
  return self->nDimension;
}

int64_t THZTensor_(size)(const THZTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THZTensor_(nDimension)(self));
  return self->size[dim];
}

int64_t THZTensor_(stride)(const THZTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THZTensor_(nDimension)(self));
  return self->stride[dim];
}

THLongStorage *THZTensor_(newSizeOf)(THZTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THZTensor_(newStrideOf)(THZTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

ntype *THZTensor_(data)(const THZTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THZTensor_(setFlag)(THZTensor *self, const char flag)
{
  self->flag |= flag;
}

void THZTensor_(clearFlag)(THZTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THZTensor_(rawInit)(THZTensor *self);


/* Empty init */
THZTensor *THZTensor_(new)(void)
{
  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THZTensor *THZTensor_(newWithTensor)(THZTensor *tensor)
{
  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  THZTensor_(setStorageNd)(self,
                          tensor->storage,
                          tensor->storageOffset,
                          tensor->nDimension,
                          tensor->size,
                          tensor->stride);
  return self;
}

/* Storage init */
THZTensor *THZTensor_(newWithStorage)(THZStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THZTensor *self = THAlloc(sizeof(THZTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THZTensor_(rawInit)(self);
#ifdef DEBUG
  THAssert((size ? size->size : (stride ? stride->size : 0)) <= INT_MAX);
#endif
  THZTensor_(setStorageNd)(self,
                          storage,
                          storageOffset,
                          (size ? size->size : (stride ? stride->size : 0)),
                          (size ? size->data : NULL),
                          (stride ? stride->data : NULL));

  return self;
}
THZTensor *THZTensor_(newWithStorage1d)(THZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THZTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THZTensor *THZTensor_(newWithStorage2d)(THZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THZTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THZTensor *THZTensor_(newWithStorage3d)(THZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THZTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THZTensor *THZTensor_(newWithStorage4d)(THZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  int64_t stride[4] = {stride0, stride1, stride2, stride3};

  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  THZTensor_(setStorageNd)(self, storage, storageOffset, 4, size, stride);

  return self;
}

THZTensor *THZTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride)
{
  return THZTensor_(newWithStorage)(NULL, 0, size, stride);
}

THZTensor *THZTensor_(newWithSize1d)(int64_t size0)
{
  return THZTensor_(newWithSize4d)(size0, -1, -1, -1);
}

THZTensor *THZTensor_(newWithSize2d)(int64_t size0, int64_t size1)
{
  return THZTensor_(newWithSize4d)(size0, size1, -1, -1);
}

THZTensor *THZTensor_(newWithSize3d)(int64_t size0, int64_t size1, int64_t size2)
{
  return THZTensor_(newWithSize4d)(size0, size1, size2, -1);
}

THZTensor *THZTensor_(newWithSize4d)(int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  THZTensor_(resizeNd)(self, 4, size, NULL);

  return self;
}

THZTensor *THZTensor_(newClone)(THZTensor *self)
{
  THZTensor *tensor = THZTensor_(new)();
  THZTensor_(resizeAs)(tensor, self);
  THZTensor_(copy)(tensor, self);
  return tensor;
}

THZTensor *THZTensor_(newContiguous)(THZTensor *self)
{
  if(!THZTensor_(isContiguous)(self))
    return THZTensor_(newClone)(self);
  else
  {
    THZTensor_(retain)(self);
    return self;
  }
}

THZTensor *THZTensor_(newSelect)(THZTensor *tensor, int dimension_, int64_t sliceIndex_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(select)(self, NULL, dimension_, sliceIndex_);
  return self;
}

THZTensor *THZTensor_(newNarrow)(THZTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(narrow)(self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THZTensor *THZTensor_(newTranspose)(THZTensor *tensor, int dimension1_, int dimension2_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(transpose)(self, NULL, dimension1_, dimension2_);
  return self;
}

THZTensor *THZTensor_(newUnfold)(THZTensor *tensor, int dimension_, int64_t size_, int64_t step_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(unfold)(self, NULL, dimension_, size_, step_);
  return self;
}

THZTensor *THZTensor_(newView)(THZTensor *tensor, THLongStorage *size)
{
  THArgCheck(THZTensor_(isContiguous)(tensor), 1, "input is not contiguous");
  ptrdiff_t numel = THZTensor_(nElement)(tensor);
  THZTensor *self = THZTensor_(new)();
  THLongStorage *inferred_size = THLongStorage_newInferSize(size, numel);
  THZTensor_(setStorage)(self, tensor->storage, tensor->storageOffset, inferred_size, NULL);
  THLongStorage_free(inferred_size);
  return self;
}

/* Resize */
void THZTensor_(resize)(THZTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

#ifdef DEBUG
  THAssert(size->size <= INT_MAX);
#endif
  THZTensor_(resizeNd)(self, size->size, size->data, (stride ? stride->data : NULL));
}

void THZTensor_(resizeAs)(THZTensor *self, THZTensor *src)
{
  if(!THZTensor_(isSameSizeAs)(self, src))
    THZTensor_(resizeNd)(self, src->nDimension, src->size, NULL);
}

void THZTensor_(resize1d)(THZTensor *tensor, int64_t size0)
{
  THZTensor_(resize4d)(tensor, size0, -1, -1, -1);
}

void THZTensor_(resize2d)(THZTensor *tensor, int64_t size0, int64_t size1)
{
  THZTensor_(resize4d)(tensor, size0, size1, -1, -1);
}

void THZTensor_(resize3d)(THZTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  THZTensor_(resize4d)(tensor, size0, size1, size2, -1);
}

void THZTensor_(resize4d)(THZTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THZTensor_(resizeNd)(self, 4, size, NULL);
}

void THZTensor_(resize5d)(THZTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
    int64_t size[5] = {size0, size1, size2, size3, size4};

  THZTensor_(resizeNd)(self, 5, size, NULL);
}

THZTensor* THZTensor_(newExpand)(THZTensor *tensor, THLongStorage *sizes) {
  THZTensor *result = THZTensor_(new)();
  THZTensor_(expand)(result, tensor, sizes);
  return result;
}

void THZTensor_(expand)(THZTensor *r, THZTensor *tensor, THLongStorage *sizes) {
  THArgCheck(THZTensor_(nDimension)(tensor) > 0 || THLongStorage_size(sizes) == 0, 0,
             "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THZTensor_(nDimension)(tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  int64_t *expandedSizes;
  int64_t *expandedStrides;
  char error_buffer[1024];
  int ret =
      THLongStorage_inferExpandGeometry(tensor->size, tensor->stride, THZTensor_(nDimension)(tensor),
                                        sizes, &expandedSizes, &expandedStrides, error_buffer, 1024);

  if (ret != 0) {
    THError(error_buffer);
    return;
  }

  THZTensor_(setStorageNd)(r, THZTensor_(storage)(tensor), THZTensor_(storageOffset)(tensor),
                          THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}


void THZTensor_(expandNd)(THZTensor **rets, THZTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THZTensor_(nDimension)(ops[i]) > 0, i, "can't expand empty tensor %d", i);
  }

  int64_t **op_sizes = THAlloc(sizeof(int64_t*) * count);
  int64_t *op_dims = THAlloc(sizeof(int64_t) * count);

  for (int i = 0; i < count; ++i) {
    op_sizes[i] = ops[i]->size;
    op_dims[i] = ops[i]->nDimension;
  }

  THLongStorage *sizes = THLongStorage_new();
  char error_buffer[1024];
  int ret = THLongStorage_inferSizeN(sizes,
                                     count,
                                     op_sizes,
                                     op_dims,
                                     error_buffer,
                                     1024);

  if(ret != 0) {
    THFree(op_sizes);
    THFree(op_dims);
    THLongStorage_free(sizes);
    THError(error_buffer);
    return;
  }

  for (int i = 0; i < count; ++i) {
    THZTensor_(expand)(rets[i], ops[i], sizes);
  }

  THFree(op_sizes);
  THFree(op_dims);
  THLongStorage_free(sizes);
}

void THZTensor_(set)(THZTensor *self, THZTensor *src)
{
  if(self != src)
    THZTensor_(setStorageNd)(self,
                            src->storage,
                            src->storageOffset,
                            src->nDimension,
                            src->size,
                            src->stride);
}

void THZTensor_(setStorage)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

#ifdef DEBUG
  THAssert((size_ ? size_->size : (stride_ ? stride_->size : 0)) <= INT_MAX);
#endif
  THZTensor_(setStorageNd)(self,
                          storage_,
                          storageOffset_,
                          (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                          (size_ ? size_->data : NULL),
                          (stride_ ? stride_->data : NULL));
}

void THZTensor_(setStorage1d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THZTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          -1, -1,
                          -1, -1,
                          -1, -1);
}

void THZTensor_(setStorage2d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THZTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          -1, -1,
                          -1, -1);
}

void THZTensor_(setStorage3d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THZTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          size2_, stride2_,
                          -1, -1);
}

void THZTensor_(setStorage4d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THZTensor_(setStorageNd)(self, storage_, storageOffset_, 4, size, stride);
}


void THZTensor_(narrow)(THZTensor *self, THZTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex <= src->size[dimension] - size), 4, "out of range");

  THZTensor_(set)(self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THZTensor_(select)(THZTensor *self, THZTensor *src, int dimension, int64_t sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  THZTensor_(set)(self, src);
  THZTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THZTensor_(transpose)(THZTensor *self, THZTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THZTensor_(set)(self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THZTensor_(unfold)(THZTensor *self, THZTensor *src, int dimension, int64_t size, int64_t step)
{
  int64_t *newSize;
  int64_t *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THZTensor_(set)(self, src);

  newSize = THAlloc(sizeof(int64_t)*(self->nDimension+1));
  newStride = THAlloc(sizeof(int64_t)*(self->nDimension+1));

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
void THZTensor_(squeeze)(THZTensor *self, THZTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THZTensor_(set)(self, src);

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

void THZTensor_(squeeze1d)(THZTensor *self, THZTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "dimension out of range");

  THZTensor_(set)(self, src);

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

void THZTensor_(unsqueeze1d)(THZTensor *self, THZTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->nDimension), 2, "dimension out of range");
  THArgCheck(src->nDimension > 0, 2, "cannot unsqueeze empty tensor");

  THZTensor_(set)(self, src);

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

int THZTensor_(isTransposed)(const THZTensor *self)
{
  if (THZTensor_(isContiguous)(self)) {
    return 0;
  }
  int64_t max_stride = 1;
  int64_t size_max_stride = 1;
  int64_t z = 1;
  int d;
  for (d = 0; d < self->nDimension; ++d) {
    if (self->stride[d] == 0 && self->size[d] != 1)
      return 0;
    if (self->stride[d] > max_stride) {
      max_stride = self->stride[d];
      size_max_stride = self->size[d];
    }
    z *= self->size[d];
  }
  if (z == max_stride * size_max_stride) {
    return 1;
  }
  return 0;
}

int THZTensor_(isContiguous)(const THZTensor *self)
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

int THZTensor_(isSize)(const THZTensor *self, const THLongStorage *dims)
{
  int d;
  if (self->nDimension != dims->size)
    return 0;

  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != dims->data[d])
      return 0;
  }
  return 1;
}

int THZTensor_(isSameSizeAs)(const THZTensor *self, const THZTensor* src)
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

int THZTensor_(isSetTo)(const THZTensor *self, const THZTensor* src)
{
  if (!self->storage)
    return 0;
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

ptrdiff_t THZTensor_(nElement)(const THZTensor *self)
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

void THZTensor_(retain)(THZTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THZTensor_(free)(THZTensor *self)
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
        THZStorage_(free)(self->storage);
      THFree(self);
    }
  }
}

void THZTensor_(freeCopyTo)(THZTensor *self, THZTensor *dst)
{
  if(self != dst)
    THZTensor_(copy)(dst, self);

  THZTensor_(free)(self);
}

/*******************************************************************************/

static void THZTensor_(rawInit)(THZTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

void THZTensor_(setStorageNd)(THZTensor *self, THZStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THZStorage_(free)(self->storage);

    if(storage)
    {
      self->storage = storage;
      THZStorage_(retain)(self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THZTensor_(resizeNd)(self, nDimension, size, stride);
}

void THZTensor_(resizeNd)(THZTensor *self, int nDimension, int64_t *size, int64_t *stride)
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
      self->size = THRealloc(self->size, sizeof(int64_t)*nDimension);
      self->stride = THRealloc(self->stride, sizeof(int64_t)*nDimension);
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
        self->storage = THZStorage_(new)();
      if(totalSize+self->storageOffset > self->storage->size)
        THZStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THZTensor_(set1d)(THZTensor *tensor, int64_t x0, ntype value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

ntype THZTensor_(get1d)(const THZTensor *tensor, int64_t x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THZTensor_(set2d)(THZTensor *tensor, int64_t x0, int64_t x1, ntype value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

ntype THZTensor_(get2d)(const THZTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THZTensor_(set3d)(THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, ntype value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

ntype THZTensor_(get3d)(const THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THZTensor_(set4d)(THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, ntype value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

ntype THZTensor_(get4d)(const THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

THDescBuff THZTensor_(desc)(const THZTensor *tensor) {
  const int L = TH_DESC_BUFF_LEN;
  THDescBuff buf;
  char *str = buf.str;
  int n = 0;
#define _stringify(x) #x
  n += snprintf(str, L-n, "torch." _stringify(x) "Tensor of size ");
#undef _stringify
  int i;
  for(i = 0; i < tensor->nDimension; i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size[i]);
    if(i < tensor->nDimension-1) {
      n += snprintf(str+n, L-n, "x");
    }
  }
  if(n >= L) {
    snprintf(str+L-4, 4, "...");
  }
  return buf;
}

THDescBuff THZTensor_(sizeDesc)(const THZTensor *tensor) {
  THLongStorage *size = THZTensor_(newSizeOf)((THZTensor*)tensor);
  THDescBuff buf = THLongStorage_sizeDesc(size);
  THLongStorage_free(size);
  return buf;
}

#endif
