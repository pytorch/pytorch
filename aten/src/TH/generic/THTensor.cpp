#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.cpp"
#else

#include <new>

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
  return self->dim();
}

int THTensor_(_nDimension)(const THTensor *self)
{
  return self->_dim();
}

int64_t THTensor_(size)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_(nDimension)(self));
  return self->size[dim];
}

int64_t THTensor_(stride)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_(nDimension)(self));
  return self->stride[dim];
}

THLongStorage *THTensor_(newSizeOf)(THTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->dim());
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THTensor_(newStrideOf)(THTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->dim());
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

real *THTensor_(data)(const THTensor *self)
{
  if(self->storage)
    return (THStorage_(data)(self->storage)+self->storageOffset);
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
                          tensor->dim(),
                          tensor->size,
                          tensor->stride);
  return self;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  if(size && stride) {
    THArgCheck(size->size == stride->size, 4, "inconsistent size");
  }

  AT_CHECK(size, "size must not be null");
  THTensor_(rawInit)(self);
#ifdef DEBUG
  THAssert(size->size <= INT_MAX);
#endif
  THTensor_(setStorageNd)(self,
                          storage,
                          storageOffset,
                          size->size,
                          THLongStorage_data(size),
                          (stride ? THLongStorage_data(stride) : NULL));

  return self;
}

THTensor *THTensor_(newWithStorageIntLists)(THStorage *storage, ptrdiff_t storageOffset, at::IntList sizes, at::IntList strides) {
  AT_CHECK(sizes.size() == strides.size(), "number of sizes and strides must match");
  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(setStorageNd)(self, storage, storageOffset, sizes.size(),
                          const_cast<int64_t*>(sizes.data()), const_cast<int64_t*>(strides.data()));

  return self;
}

THTensor *THTensor_(newWithStorage1d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THTensor_(newWithStorageIntLists)(storage, storageOffset, {size0}, {stride0});
}

THTensor *THTensor_(newWithStorage2d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THTensor_(newWithStorageIntLists)(storage, storageOffset, {size0, size1}, {stride0, stride1});
}

THTensor *THTensor_(newWithStorage3d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THTensor_(newWithStorageIntLists)(storage, storageOffset, {size0, size1, size2}, {stride0, stride1, stride2});
}

THTensor *THTensor_(newWithStorage4d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  return THTensor_(newWithStorageIntLists)(storage, storageOffset,
                                          {size0, size1, size2, size3},
                                          {stride0, stride1, stride2, stride3});
}

THTensor *THTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride)
{
  return THTensor_(newWithStorage)(NULL, 0, size, stride);
}

THTensor *THTensor_(newWithSizeIntList)(at::IntList sizes) {
  THTensor *self = (THTensor *)THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(resizeNd)(self, sizes.size(), const_cast<int64_t*>(sizes.data()), nullptr);

  return self;
}

THTensor *THTensor_(newWithSize1d)(int64_t size0)
{
  return THTensor_(newWithSizeIntList)({size0});
}

THTensor *THTensor_(newWithSize2d)(int64_t size0, int64_t size1)
{
  return THTensor_(newWithSizeIntList)({size0, size1});
}

THTensor *THTensor_(newWithSize3d)(int64_t size0, int64_t size1, int64_t size2)
{
  return THTensor_(newWithSizeIntList)({size0, size1, size2});
}

THTensor *THTensor_(newWithSize4d)(int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  return THTensor_(newWithSizeIntList)({size0, size1, size2, size3});
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

THTensor *THTensor_(newView)(THTensor *tensor, THLongStorage *size)
{
  ptrdiff_t numel = THTensor_(nElement)(tensor);
  THTensor *self = THTensor_(new)();
  THLongStorage *inferred_size = THLongStorage_newInferSize(size, numel);
  auto stride = THTensor_compute_stride(at::IntList(tensor->size, tensor->dim()),
                                        at::IntList(tensor->stride, tensor->dim()),
                                        at::IntList(inferred_size->data<int64_t>(), inferred_size->size));
  THArgCheck(stride.has_value(), 2, "view size is "
    "not compatible with input tensor's size and stride (at least one dimension spans "
    "across two contiguous subspaces). Call .contiguous() before .view().");
  auto stride_value = *stride;
  THLongStorage *new_stride = THLongStorage_newWithSize(stride_value.size());
  THLongStorage_rawCopy(new_stride, stride_value.data());
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
  THTensor_(resizeNd)(self, size->size, THLongStorage_data(size), (stride ? THLongStorage_data(stride) : NULL));
}

void THTensor_(resizeAs)(THTensor *self, THTensor *src)
{
  if(!THTensor_(isSameSizeAs)(self, src))
    THTensor_(resizeNd)(self, src->dim(), src->size, NULL);
}

void THTensor_(resize1d)(THTensor *tensor, int64_t size0)
{
  int64_t size[1] = {size0};
  THTensor_(resizeNd)(tensor, 1, size, nullptr);
}

void THTensor_(resize2d)(THTensor *tensor, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THTensor_(resizeNd)(tensor, 2, size, nullptr);
}

void THTensor_(resize3d)(THTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THTensor_(resizeNd)(tensor, 3, size, nullptr);
}

void THTensor_(resize4d)(THTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THTensor_(resizeNd)(self, 4, size, nullptr);
}

void THTensor_(resize5d)(THTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
  int64_t size[5] = {size0, size1, size2, size3, size4};
  THTensor_(resizeNd)(self, 5, size, nullptr);
}

void THTensor_(set)(THTensor *self, THTensor *src)
{
  if(self != src)
    THTensor_(setStorageNd)(self,
                            src->storage,
                            src->storageOffset,
                            src->dim(),
                            src->size,
                            src->stride);
}

void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  AT_CHECK(size_, "size must not be null");
#ifdef DEBUG
  THAssert(size_ <= INT_MAX);
#endif
  THTensor_(setStorageNd)(self,
                          storage_,
                          storageOffset_,
                          size_->size,
                          THLongStorage_data(size_),
                          (stride_ ? THLongStorage_data(stride_) : NULL));
}

void THTensor_(setStorageIntLists)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                                   at::IntList sizes, at::IntList strides)
{
  AT_CHECK(sizes.size() == strides.size(), "number of sizes and strides must match");

  THTensor_(setStorageNd)(self, storage_, storageOffset_, sizes.size(),
                          const_cast<int64_t *>(sizes.data()), const_cast<int64_t *>(strides.data()));
}

void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THTensor_(setStorageIntLists)(self, storage_, storageOffset_,
                                {size0_}, {stride0_});
}

void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THTensor_(setStorageIntLists)(self, storage_, storageOffset_,
                                {size0_, size1_},
                                {stride0_, stride1_});
}

void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THTensor_(setStorageIntLists)(self, storage_, storageOffset_,
                                {size0_, size1_, size2_},
                                {stride0_, stride1_, stride2_});
}

void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THTensor_(setStorageIntLists)(self, storage_, storageOffset_, size, stride);
}


void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->dim()), 2, "out of range");
  THArgCheck( firstIndex >= 0, 3, "out of range");
#ifdef USE_TH_SIZE_ZERO_DIM
  THArgCheck( size >= 0, 4, "out of range");
#else
  THArgCheck( size > 0, 4, "out of range");
#endif
  THArgCheck(firstIndex <= src->size[dimension] - size, 4, "out of range");

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

#ifndef USE_TH_SCALAR
  THArgCheck(src->_dim() > 1, 1, "cannot select on a vector");
#endif
  THArgCheck((dimension >= 0) && (dimension < src->dim()), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  THTensor_(set)(self, src);
  THTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->dim()-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->dim_--;
}

void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->_dim()), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->_dim()), 2, "out of range");

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

  THArgCheck(!src->is_empty(), 1, "cannot unfold an empty tensor");
  THArgCheck((dimension >= 0) && (dimension < src->dim()), 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THTensor_(set)(self, src);

  newSize = (int64_t *)THAlloc(sizeof(int64_t)*(self->dim()+1));
  newStride = (int64_t *)THAlloc(sizeof(int64_t)*(self->dim()+1));

  newSize[self->dim()] = size;
  newStride[self->dim()] = self->stride[dimension];
  for(d = 0; d < self->dim(); d++)
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
  self->dim_++;
}

/* we have to handle the case where the result is a number */
void THTensor_(squeeze)(THTensor *self, THTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THTensor_(set)(self, src);

  for(d = 0; d < src->dim(); d++)
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

#ifndef USE_TH_SCALAR
  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->dim() > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
#endif
  self->dim_ = ndim;
}

void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension < src->dim()), 2, "dimension out of range");

  THTensor_(set)(self, src);

#ifdef USE_TH_SCALAR
  if(src->size[dimension] == 1)
#else
  if(src->size[dimension] == 1 && src->dim() > 1)
#endif
  {
    for(d = dimension; d < self->dim()-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->dim_--;
  }
}

void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->dim()), 2, "dimension out of range");
#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(!src->is_empty(), 2, "cannot unsqueeze empty tensor");
#endif

  THTensor_(set)(self, src);

  self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*(self->dim()+1));
  self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*(self->dim()+1));
  self->dim_++;
  for (d = self->dim()-1; d > dimension; d--) {
    self->size[d] = self->size[d-1];
    self->stride[d] = self->stride[d-1];
  }
  if (dimension+1 < self->dim()) {
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
  for (d = 0; d < self->_dim(); ++d) {
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
  for(d = self->_dim()-1; d >= 0; d--)
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
  if (self->_dim() != dims->size)
    return 0;

  for(d = 0; d < self->_dim(); ++d)
  {
    if(self->size[d] != THLongStorage_data(dims)[d])
      return 0;
  }
  return 1;
}

int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor* src)
{
  int d;
  if (self->dim() != src->dim())
    return 0;
  for(d = 0; d < self->dim(); ++d)
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
      self->_dim() == src->_dim())
  {
    int d;
    for (d = 0; d < self->_dim(); ++d)
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
  if(self->_dim() == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < self->_dim(); d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THTensor_(retain)(THTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    ++self->refcount;
}

void THTensor_(free)(THTensor *self)
{
  THTensor_free(self);
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
  new (&self->refcount) std::atomic<int>(1);
  self->storage = THStorage_(new)();
  self->storageOffset = 0;
  self->size = static_cast<int64_t *>(THAlloc(sizeof(int64_t)));
  self->stride = static_cast<int64_t *>(THAlloc(sizeof(int64_t)));
  self->size[0] = 0;
  self->stride[0] = 1;
  self->dim_ = 1;
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
  ptrdiff_t totalSize;
  bool hascorrectsize = true;

#ifndef USE_TH_SCALAR
  AT_CHECK(nDimension > 0, "resizeNd nDimension must be greater than 0");
#else
  AT_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
#endif

  for(d = 0; d < nDimension; d++)
  {
#ifndef USE_TH_SIZE_ZERO_DIM
    // we can't support this unless we have arbitrary 0-sized dimensions, but some calls to this
    // currently exist and expect a size [0] tensor to be returned.
    if (d == 0 && size[d] == 0) {
      nDimension = 1;
    } else {
      AT_CHECK(size[d] > 0, "sizes must be non-negative");
    }
#endif
    if((self->dim() > d) && (size[d] != self->size[d])) {
      hascorrectsize = false;
    }

    // NB: this used to test that stride[d] was >= 0
    if((self->dim() > d) && stride && (stride[d] != self->stride[d])) {
      hascorrectsize = false;
    }
  }

  if(nDimension != self->dim()) {
    hascorrectsize = false;
  }

  if(hascorrectsize) {
    return;
  }

  if(nDimension != self->dim())
  {
    self->size = (int64_t *)THRealloc(self->size, sizeof(int64_t)*nDimension);
    self->stride = (int64_t *)THRealloc(self->stride, sizeof(int64_t)*nDimension);
    self->dim_ = nDimension;
  }

  totalSize = 1;
  for(d = nDimension-1; d >= 0; d--)
  {
    self->size[d] = size[d];
    if(stride && (stride[d] >= 0) ) {
      self->stride[d] = stride[d];
    } else {
      if(d == nDimension-1) {
        self->stride[d] = 1;
      } else {
        // Keep stride monotonically increasing to match NumPy.
        self->stride[d] = std::max<int64_t>(self->size[d+1], 1)*self->stride[d+1];
      }
    }
    totalSize += (self->size[d]-1)*self->stride[d];
  }

  if(totalSize+self->storageOffset > 0)
  {
    if(!self->storage) {
      self->storage = THStorage_(new)();
    }
    if(totalSize+self->storageOffset > self->storage->size) {
      THStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  }
}

void THTensor_(set1d)(THTensor *tensor, int64_t x0, real value)
{
  THArgCheck(tensor->_dim() == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

real THTensor_(get1d)(const THTensor *tensor, int64_t x0)
{
  THArgCheck(tensor->_dim() == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THTensor_(set2d)(THTensor *tensor, int64_t x0, int64_t x1, real value)
{
  THArgCheck(tensor->_dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

real THTensor_(get2d)(const THTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->_dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THTensor_(set3d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, real value)
{
  THArgCheck(tensor->_dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

real THTensor_(get3d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->_dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THTensor_(set4d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, real value)
{
  THArgCheck(tensor->_dim() == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

real THTensor_(get4d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->_dim() == 4, 1, "tensor must have four dimensions");
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
  for(i = 0; i < tensor->_dim(); i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size[i]);
    if(i < tensor->_dim()-1) {
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
