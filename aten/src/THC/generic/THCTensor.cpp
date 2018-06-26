#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.cpp"
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
  return THCTensor_nDimension(state, self);
}

int THCTensor_(_nDimension)(THCState *state, const THCTensor *self)
{
  return THCTensor__nDimension(state, self);
}

int64_t THCTensor_(size)(THCState *state, const THCTensor *self, int dim)
{
  return THCTensor_size(state, self, dim);
}

int64_t THCTensor_(stride)(THCState *state, const THCTensor *self, int dim)
{
  return THCTensor_stride(state, self, dim);
}

THLongStorage *THCTensor_(newSizeOf)(THCState *state, THCTensor *self)
{
  return THCTensor_newSizeOf(state, self);
}

THLongStorage *THCTensor_(newStrideOf)(THCState *state, THCTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->dim());
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

real *THCTensor_(data)(THCState *state, const THCTensor *self)
{
  if(self->storage)
    return (THCStorage_(data)(state, self->storage)+self->storageOffset);
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
                           tensor->dim(),
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

  AT_CHECK(size, "size must not be null");
  THCTensor_(rawInit)(state, self);
  THCTensor_(setStorageNd)(state,
                           self,
                           storage,
                           storageOffset,
                           size->size,
                           THLongStorage_data(size),
                           (stride ? THLongStorage_data(stride) : NULL));

  return self;
}

THCTensor *THCTensor_(newWithStorageIntLists)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset, at::IntList sizes, at::IntList strides) {
  AT_CHECK(sizes.size() == strides.size(), "number of sizes and strides must match");
  THCTensor *self = (THCTensor *)THAlloc(sizeof(THCTensor));
  THCTensor_(rawInit)(state, self);
  THCTensor_(setStorageNd)(state, self, storage, storageOffset, sizes.size(),
                           const_cast<int64_t*>(sizes.data()), const_cast<int64_t*>(strides.data()));

  return self;
}

THCTensor *THCTensor_(newWithStorage1d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THCTensor_(newWithStorageIntLists)(state, storage, storageOffset, {size0}, {stride0});
}

THCTensor *THCTensor_(newWithStorage2d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THCTensor_(newWithStorageIntLists)(state, storage, storageOffset, {size0, size1}, {stride0, stride1});
}

THCTensor *THCTensor_(newWithStorage3d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THCTensor_(newWithStorageIntLists)(state, storage, storageOffset, {size0, size1, size2}, {stride0, stride1, stride2});
}

THCTensor *THCTensor_(newWithStorage4d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  return THCTensor_(newWithStorageIntLists)(state, storage, storageOffset,
                                            {size0, size1, size2, size3},
                                            {stride0, stride1, stride2, stride3});
}

THCTensor *THCTensor_(newWithSize)(THCState *state, THLongStorage *size, THLongStorage *stride)
{
  return THCTensor_(newWithStorage)(state, NULL, 0, size, stride);
}

THCTensor *THCTensor_(newWithSizeIntList)(THCState *state, at::IntList sizes) {
  THCTensor *self = (THCTensor *)THAlloc(sizeof(THCTensor));
  THCTensor_(rawInit)(state, self);
  THCTensor_(resizeNd)(state, self, sizes.size(), const_cast<int64_t*>(sizes.data()), nullptr);

  return self;
}

THCTensor *THCTensor_(newWithSize1d)(THCState *state, int64_t size0)
{
  return THCTensor_(newWithSizeIntList)(state, {size0});
}

THCTensor *THCTensor_(newWithSize2d)(THCState *state, int64_t size0, int64_t size1)
{
  return THCTensor_(newWithSizeIntList)(state, {size0, size1});
}

THCTensor *THCTensor_(newWithSize3d)(THCState *state, int64_t size0, int64_t size1, int64_t size2)
{
  return THCTensor_(newWithSizeIntList)(state, {size0, size1, size2});
}

THCTensor *THCTensor_(newWithSize4d)(THCState *state, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  return THCTensor_(newWithSizeIntList)(state, {size0, size1, size2, size3});
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

THCTensor *THCTensor_(newView)(THCState *state, THCTensor *tensor, THLongStorage *size)
{
  ptrdiff_t numel = THCTensor_(nElement)(state, tensor);
  THCTensor *self = THCTensor_(new)(state);
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
  THCTensor_(setStorage)(state, self, tensor->storage, tensor->storageOffset, inferred_size, new_stride);
  THLongStorage_free(inferred_size);
  THLongStorage_free(new_stride);
  return self;
}

// Collapses the first two dimensions of a tensor.
// Assumes the input tensor is contiguous.
THCTensor *THCTensor_(newFoldBatchDim)(THCState *state, THCTensor *input) {
  int in_dims = THCTensor_(_nDimension)(state, input);
  THArgCheck(in_dims >= 2, 1, "Tensor needs to have at least two dimensions");
  THArgCheck(THCTensor_(isContiguous)(state, input), 1,
             "Tensor must be contiguous");
  THLongStorage *newSize = THLongStorage_newWithSize(in_dims - 1);
  THLongStorage_data(newSize)[0] = THCTensor_(size)(state, input, 0) * THCTensor_(size)(state, input, 1);
  for (int i = 2; i < in_dims; i++) {
    THLongStorage_data(newSize)[i - 1] = THCTensor_(size)(state, input, i);
  }
  THCTensor *output = THCTensor_(newView)(state, input, newSize);
  THLongStorage_free(newSize);
  return output;
}

/* Resize */
void THCTensor_(resize)(THCState *state, THCTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THCTensor_resize(state, self, size, stride);
}

void THCTensor_(resizeAs)(THCState *state, THCTensor *self, THCTensor *src)
{
  THCTensor_resizeAs(state, self, src);
}

void THCTensor_(resize1d)(THCState *state, THCTensor *tensor, int64_t size0)
{
  int64_t size[1] = {size0};
  THCTensor_resizeNd(state, tensor, 1, size, nullptr);
}

void THCTensor_(resize2d)(THCState *state, THCTensor *tensor, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THCTensor_resizeNd(state, tensor, 2, size, nullptr);
}

void THCTensor_(resize3d)(THCState *state, THCTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THCTensor_resizeNd(state, tensor, 3, size, nullptr);
}

void THCTensor_(resize4d)(THCState *state, THCTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THCTensor_resizeNd(state, self, 4, size, nullptr);
}

void THCTensor_(resize5d)(THCState *state, THCTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
  int64_t size[5] = {size0, size1, size2, size3, size4};
  THCTensor_resizeNd(state, self, 5, size, nullptr);
}

void THCTensor_(set)(THCState *state, THCTensor *self, THCTensor *src)
{
  THCTensor_set(state, self, src);
}

void THCTensor_(setStorage)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  AT_CHECK(size_, "size must not be null");
  THCTensor_(setStorageNd)(state,
                           self,
                           storage_,
                           storageOffset_,
                           size_->size,
                           THLongStorage_data(size_),
                           (stride_ ? THLongStorage_data(stride_) : NULL));
}

void THCTensor_(setStorageIntLists)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                                    at::IntList sizes, at::IntList strides)
{
  AT_CHECK(sizes.size() == strides.size(), "number of sizes and strides must match");

  THCTensor_(setStorageNd)(state, self, storage_, storageOffset_, sizes.size(),
                           const_cast<int64_t*>(sizes.data()), const_cast<int64_t*>(strides.data()));
}

void THCTensor_(setStorage1d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THCTensor_(setStorageIntLists)(state, self, storage_, storageOffset_,
                                 {size0_}, {stride0_});
}

void THCTensor_(setStorage2d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THCTensor_(setStorageIntLists)(state, self, storage_, storageOffset_,
                                 {size0_, size1_},
                                 {stride0_, stride1_});
}

void THCTensor_(setStorage3d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THCTensor_(setStorageIntLists)(state, self, storage_, storageOffset_,
                                 {size0_, size1_, size2_},
                                 {stride0_, stride1_, stride2_});
}

void THCTensor_(setStorage4d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THCTensor_(setStorageIntLists)(state, self, storage_, storageOffset_, size, stride);
}


void THCTensor_(narrow)(THCState *state, THCTensor *self, THCTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->dim()), 3, "out of range");
  THArgCheck( firstIndex >= 0, 4, "out of range");
#ifdef USE_TH_SIZE_ZERO_DIM
  THArgCheck( size >= 0, 5, "out of range");
#else
  THArgCheck( size > 0, 5, "out of range");
#endif
  THArgCheck(firstIndex+size <= src->size[dimension], 5, "out of range");

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

#ifndef USE_TH_SCALAR
  THArgCheck(src->_dim() > 1, 1, "cannot select on a vector");
#endif
  THArgCheck((dimension >= 0) && (dimension < src->dim()), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THCTensor_(set)(state, self, src);
  THCTensor_(narrow)(state, self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->dim()-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->dim_--;
}

void THCTensor_(transpose)(THCState *state, THCTensor *self, THCTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->_dim()), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->_dim()), 2, "out of range");

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

  THArgCheck(!src->is_empty(), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->dim(), 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THCTensor_(set)(state, self, src);

  newSize = (int64_t*)THAlloc(sizeof(int64_t)*(self->dim()+1));
  newStride = (int64_t*)THAlloc(sizeof(int64_t)*(self->dim()+1));

  newSize[self->dim()] = size;
  newStride[self->dim()] = self->stride[dimension];
  for(d = 0; d < self->_dim(); d++)
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
void THCTensor_(squeeze)(THCState *state, THCTensor *self, THCTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THCTensor_(set)(state, self, src);

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
  self->dim_ = ndim;
}
#endif

void THCTensor_(squeeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  THCTensor_squeeze1d(state, self, src, dimension);
}

void THCTensor_(unsqueeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  THCTensor_unsqueeze1d(state, self, src, dimension);
}

int THCTensor_(isContiguous)(THCState *state, const THCTensor *self)
{
  return THCTensor_isContiguous(state, self);
}

int THCTensor_(isSize)(THCState *state, const THCTensor *self, const THLongStorage *dims)
{
  int d;
  if (self->dim() != dims->size)
    return 0;

  for (d = 0; d < self->dim(); ++d)
  {
    if (self->size[d] != THLongStorage_data(dims)[d])
      return 0;
  }
  return 1;
}

int THCTensor_(isSetTo)(THCState *state, const THCTensor *self, const THCTensor *src)
{
  if (self->storage == src->storage &&
      self->storageOffset == src->storageOffset &&
      self->dim() == src->dim())
  {
    int d;
    for (d = 0; d < self->dim(); ++d)
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
  if (self->dim() != src->dim())
    return 0;
  for(d = 0; d < self->dim(); ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

ptrdiff_t THCTensor_(nElement)(THCState *state, const THCTensor *self)
{
  return THCTensor_nElement(state, self);
}

void THCTensor_(retain)(THCState *state, THCTensor *self)
{
  THCTensor_retain(state, self);
}

void THCTensor_(free)(THCState *state, THCTensor *self)
{
  THCTensor_free(state, self);
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
  new (&self->refcount) std::atomic<int>(1);
  self->storage = THCStorage_(new)(state);
  self->storageOffset = 0;
  self->size = static_cast<int64_t *>(THAlloc(sizeof(int64_t)));
  self->stride = static_cast<int64_t *>(THAlloc(sizeof(int64_t)));
  self->size[0] = 0;
  self->stride[0] = 1;
  self->dim_ = 1;
  self->flag = TH_TENSOR_REFCOUNTED;
}

void THCTensor_(setStorageNd)(THCState *state, THCTensor *self, THCStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride)
{
  THCTensor_setStorageNd(state, self, storage, storageOffset, nDimension, size, stride);
}

void THCTensor_(resizeNd)(THCState *state, THCTensor *self, int nDimension, int64_t *size, int64_t *stride)
{
  THCTensor_resizeNd(state, self, nDimension, size, stride);
}

void THCTensor_(set1d)(THCState *state, THCTensor *tensor, int64_t x0, real value)
{
  THArgCheck(tensor->dim() == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

real THCTensor_(get1d)(THCState *state, const THCTensor *tensor, int64_t x0)
{
  THArgCheck(tensor->dim() == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THCStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THCTensor_(set2d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, real value)
{
  THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

real THCTensor_(get2d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THCStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THCTensor_(set3d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, real value)
{
  THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

real THCTensor_(get3d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THCStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THCTensor_(set4d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, real value)
{
  THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THCStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

real THCTensor_(get4d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
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
  for(i = 0; i < tensor->dim(); i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size[i]);
    if(i < tensor->dim()-1) {
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
