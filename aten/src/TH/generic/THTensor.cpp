#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.cpp"
#else

/**** access methods ****/
THStorage *THTensor_(storage)(const THTensor *self)
{
  return self->storage;
}

ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
{
  return self->storageOffset;
}

int THTensor_(nDimension)(const THTensor *self)
{
  return self->nDimension;
}

int64_t THTensor_(size)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_(nDimension)(self));
  return self->size[dim];
}

int64_t THTensor_(stride)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_(nDimension)(self));
  return self->stride[dim];
}

THLongStorage *THTensor_(newSizeOf)(THTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THTensor_(newStrideOf)(THTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

real *THTensor_(data)(const THTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THTensor_(setFlag)(THTensor *self, const char flag)
{
  self->flag |= flag;
}

void THTensor_(clearFlag)(THTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THTensor_(rawInit)(THTensor *self);


/* Empty init */
THTensor *THTensor_(new)(void)
{
  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *tensor)
{
  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(setStorageNd)(self,
                          tensor->storage,
                          tensor->storageOffset,
                          tensor->nDimension,
                          tensor->size,
                          tensor->stride);
  return self;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THTensor_(rawInit)(self);
#ifdef DEBUG
  THAssert((size ? size->size : (stride ? stride->size : 0)) <= INT_MAX);
#endif
  THTensor_(setStorageNd)(self,
                          storage,
                          storageOffset,
                          (size ? size->size : (stride ? stride->size : 0)),
                          (size ? size->data : NULL),
                          (stride ? stride->data : NULL));

  return self;
}
THTensor *THTensor_(newWithStorage1d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage2d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage3d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THTensor *THTensor_(newWithStorage4d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  int64_t stride[4] = {stride0, stride1, stride2, stride3};

  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(setStorageNd)(self, storage, storageOffset, 4, size, stride);

  return self;
}

THTensor *THTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride)
{
  return THTensor_(newWithStorage)(NULL, 0, size, stride);
}

THTensor *THTensor_(newWithSize1d)(int64_t size0)
{
  return THTensor_(newWithSize4d)(size0, -1, -1, -1);
}

THTensor *THTensor_(newWithSize2d)(int64_t size0, int64_t size1)
{
  return THTensor_(newWithSize4d)(size0, size1, -1, -1);
}

THTensor *THTensor_(newWithSize3d)(int64_t size0, int64_t size1, int64_t size2)
{
  return THTensor_(newWithSize4d)(size0, size1, size2, -1);
}

THTensor *THTensor_(newWithSize4d)(int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(resizeNd)(self, 4, size, NULL);

  return self;
}

THTensor *THTensor_(newClone)(THTensor *self)
{
  THTensor *tensor = THTensor_(new)();
  THTensor_(resizeAs)(tensor, self);
  THTensor_(copy)(tensor, self);
  return tensor;
}

THTensor *THTensor_(newContiguous)(THTensor *self)
{
  if(!THTensor_(isContiguous)(self))
    return THTensor_(newClone)(self);
  else
  {
    THTensor_(retain)(self);
    return self;
  }
}

THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, int64_t sliceIndex_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(select)(self, NULL, dimension_, sliceIndex_);
  return self;
}

THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(narrow)(self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(transpose)(self, NULL, dimension1_, dimension2_);
  return self;
}

THTensor *THTensor_(newUnfold)(THTensor *tensor, int dimension_, int64_t size_, int64_t step_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(unfold)(self, NULL, dimension_, size_, step_);
  return self;
}

// Also sets new_stride if viewable.
//
// On a high level,
// 1. separate tensor->size into chunks of dimensions, where the dimensions are
//    ``contiguous'' in each chunk, i.e., stride[i] = size[i+1] * stride[i+1]
// 2. view_size must be able to be separated into same number of chunks, where
//    each chunk pair has matching ``numel'', i.e., number of subspaces.
static int THTensor_(isViewable)(THTensor *tensor, THLongStorage *view_size, THLongStorage *new_stride) {
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

THTensor *THTensor_(newView)(THTensor *tensor, THLongStorage *size)
{
  ptrdiff_t numel = THTensor_(nElement)(tensor);
  THTensor *self = THTensor_(new)();
  THLongStorage *inferred_size = THLongStorage_newInferSize(size, numel);
  THLongStorage *new_stride = THLongStorage_newWithSize(size->size);
  THArgCheck(THTensor_(isViewable)(tensor, inferred_size, new_stride), 2, "view size is "
    "not compatible with input tensor's size and stride (at least one dimension spans "
    "across two contiguous subspaces). Call .contiguous() before .view().");
  THTensor_(setStorage)(self, tensor->storage, tensor->storageOffset, inferred_size, new_stride);
  THLongStorage_free(inferred_size);
  THLongStorage_free(new_stride);
  return self;
}

/* Resize */
void THTensor_(resize)(THTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

#ifdef DEBUG
  THAssert(size->size <= INT_MAX);
#endif
  THTensor_(resizeNd)(self, size->size, size->data, (stride ? stride->data : NULL));
}

void THTensor_(resizeAs)(THTensor *self, THTensor *src)
{
  if(!THTensor_(isSameSizeAs)(self, src))
    THTensor_(resizeNd)(self, src->nDimension, src->size, NULL);
}

void THTensor_(resize1d)(THTensor *tensor, int64_t size0)
{
  THTensor_(resize4d)(tensor, size0, -1, -1, -1);
}

void THTensor_(resize2d)(THTensor *tensor, int64_t size0, int64_t size1)
{
  THTensor_(resize4d)(tensor, size0, size1, -1, -1);
}

void THTensor_(resize3d)(THTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  THTensor_(resize4d)(tensor, size0, size1, size2, -1);
}

void THTensor_(resize4d)(THTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THTensor_(resizeNd)(self, 4, size, NULL);
}

void THTensor_(resize5d)(THTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
    int64_t size[5] = {size0, size1, size2, size3, size4};

  THTensor_(resizeNd)(self, 5, size, NULL);
}

void THTensor_(set)(THTensor *self, THTensor *src)
{
  if(self != src)
    THTensor_(setStorageNd)(self,
                            src->storage,
                            src->storageOffset,
                            src->nDimension,
                            src->size,
                            src->stride);
}

void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

#ifdef DEBUG
  THAssert((size_ ? size_->size : (stride_ ? stride_->size : 0)) <= INT_MAX);
#endif
  THTensor_(setStorageNd)(self,
                          storage_,
                          storageOffset_,
                          (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                          (size_ ? size_->data : NULL),
                          (stride_ ? stride_->data : NULL));
}

void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          -1, -1,
                          -1, -1,
                          -1, -1);
}

void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          -1, -1,
                          -1, -1);
}

void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          size2_, stride2_,
                          -1, -1);
}

void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THTensor_(setStorageNd)(self, storage_, storageOffset_, 4, size, stride);
}


void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex <= src->size[dimension] - size), 4, "out of range");

  THTensor_(set)(self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THTensor_(select)(THTensor *self, THTensor *src, int dimension, int64_t sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  THTensor_(set)(self, src);
  THTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THTensor_(set)(self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension, int64_t size, int64_t step)
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

  THTensor_(set)(self, src);

  newSize = (int64_t *)THAlloc(sizeof(int64_t)*(self->nDimension+1));
  newStride = (int64_t *)THAlloc(sizeof(int64_t)*(self->nDimension+1));

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
void THTensor_(squeeze)(THTensor *self, THTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THTensor_(set)(self, src);

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

void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "dimension out of range");

  THTensor_(set)(self, src);

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

void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->nDimension), 2, "dimension out of range");
  THArgCheck(src->nDimension > 0, 2, "cannot unsqueeze empty tensor");

  THTensor_(set)(self, src);

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

int THTensor_(isTransposed)(const THTensor *self)
{
  if (THTensor_(isContiguous)(self)) {
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

int THTensor_(isContiguous)(const THTensor *self)
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

int THTensor_(isSize)(const THTensor *self, const THLongStorage *dims)
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

int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor* src)
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

int THTensor_(isSetTo)(const THTensor *self, const THTensor* src)
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

ptrdiff_t THTensor_(nElement)(const THTensor *self)
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

void THTensor_(retain)(THTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THTensor_(free)(THTensor *self)
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
        THStorage_(free)(self->storage);
      THFree(self);
    }
  }
}

void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst)
{
  if(self != dst)
    THTensor_(copy)(dst, self);

  THTensor_(free)(self);
}

/*******************************************************************************/

static void THTensor_(rawInit)(THTensor *self)
{
  self->refcount = 1;
  self->storage = THStorage_(new)();
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

void THTensor_(setStorageNd)(THTensor *self, THStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THStorage_(free)(self->storage);

    if(storage)
    {
      self->storage = storage;
      THStorage_(retain)(self->storage);
    }
    else
      self->storage = THStorage_(new)();
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THTensor_(resizeNd)(self, nDimension, size, stride);
}

void THTensor_(resizeNd)(THTensor *self, int nDimension, int64_t *size, int64_t *stride)
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
      self->size = (int64_t *)THRealloc(self->size, sizeof(int64_t)*nDimension);
      self->stride = (int64_t *)THRealloc(self->stride, sizeof(int64_t)*nDimension);
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
        self->storage = THStorage_(new)();
      if(totalSize+self->storageOffset > self->storage->size)
        THStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THTensor_(set1d)(THTensor *tensor, int64_t x0, real value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

real THTensor_(get1d)(const THTensor *tensor, int64_t x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THTensor_(set2d)(THTensor *tensor, int64_t x0, int64_t x1, real value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

real THTensor_(get2d)(const THTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THTensor_(set3d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, real value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

real THTensor_(get3d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THTensor_(set4d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, real value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

real THTensor_(get4d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

THDescBuff THTensor_(desc)(const THTensor *tensor) {
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

THDescBuff THTensor_(sizeDesc)(const THTensor *tensor) {
  THLongStorage *size = THTensor_(newSizeOf)((THTensor*)tensor);
  THDescBuff buf = THLongStorage_sizeDesc(size);
  THLongStorage_free(size);
  return buf;
}

#endif
