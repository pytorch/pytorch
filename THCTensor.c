#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCTensorCopy.h"

/**** access methods ****/
THCudaStorage *THCudaTensor_storage(THCState *state, const THCudaTensor *self)
{
  return self->storage;
}

long THCudaTensor_storageOffset(THCState *state, const THCudaTensor *self)
{
  return self->storageOffset;
}

int THCudaTensor_nDimension(THCState *state, const THCudaTensor *self)
{
  return self->nDimension;
}

long THCudaTensor_size(THCState *state, const THCudaTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

long THCudaTensor_stride(THCState *state, const THCudaTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THCudaTensor_newSizeOf(THCState *state, THCudaTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THCudaTensor_newStrideOf(THCState *state, THCudaTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

float *THCudaTensor_data(THCState *state, const THCudaTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THCudaTensor_setFlag(THCState *state, THCudaTensor *self, const char flag)
{
  self->flag |= flag;
}

void THCudaTensor_clearFlag(THCState *state, THCudaTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THCudaTensor_rawInit(THCState *state, THCudaTensor *self);
static void THCudaTensor_rawSet(THCState *state, THCudaTensor *self, THCudaStorage *storage, long storageOffset, int nDimension, long *size, long *stride);
static void THCudaTensor_rawResize(THCState *state, THCudaTensor *self, int nDimension, long *size, long *stride);


/* Empty init */
THCudaTensor *THCudaTensor_new(THCState *state)
{
  THCudaTensor *self = (THCudaTensor*)THAlloc(sizeof(THCudaTensor));
  THCudaTensor_rawInit(state, self);
  return self;
}

/* Pointer-copy init */
THCudaTensor *THCudaTensor_newWithTensor(THCState *state, THCudaTensor *tensor)
{
  THCudaTensor *self = (THCudaTensor*)THAlloc(sizeof(THCudaTensor));
  THCudaTensor_rawInit(state, self);
  THCudaTensor_rawSet(state,
                      self,
                      tensor->storage,
                      tensor->storageOffset,
                      tensor->nDimension,
                      tensor->size,
                      tensor->stride);
  return self;
}

/* Storage init */
THCudaTensor *THCudaTensor_newWithStorage(THCState *state, THCudaStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THCudaTensor *self = (THCudaTensor*)THAlloc(sizeof(THCudaTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THCudaTensor_rawInit(state, self);
  THCudaTensor_rawSet(state,
                      self,
                      storage,
                      storageOffset,
                      (size ? size->size : (stride ? stride->size : 0)),
                      (size ? size->data : NULL),
                      (stride ? stride->data : NULL));

  return self;
}
THCudaTensor *THCudaTensor_newWithStorage1d(THCState *state, THCudaStorage *storage, long storageOffset,
                               long size0, long stride0)
{
  return THCudaTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THCudaTensor *THCudaTensor_newWithStorage2d(THCState *state, THCudaStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THCudaTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THCudaTensor *THCudaTensor_newWithStorage3d(THCState *state, THCudaStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THCudaTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THCudaTensor *THCudaTensor_newWithStorage4d(THCState *state, THCudaStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4] = {size0, size1, size2, size3};
  long stride[4] = {stride0, stride1, stride2, stride3};

  THCudaTensor *self = (THCudaTensor*)THAlloc(sizeof(THCudaTensor));
  THCudaTensor_rawInit(state, self);
  THCudaTensor_rawSet(state, self, storage, storageOffset, 4, size, stride);

  return self;
}

THCudaTensor *THCudaTensor_newWithSize(THCState *state, THLongStorage *size, THLongStorage *stride)
{
  return THCudaTensor_newWithStorage(state, NULL, 0, size, stride);
}

THCudaTensor *THCudaTensor_newWithSize1d(THCState *state, long size0)
{
  return THCudaTensor_newWithSize4d(state, size0, -1, -1, -1);
}

THCudaTensor *THCudaTensor_newWithSize2d(THCState *state, long size0, long size1)
{
  return THCudaTensor_newWithSize4d(state, size0, size1, -1, -1);
}

THCudaTensor *THCudaTensor_newWithSize3d(THCState *state, long size0, long size1, long size2)
{
  return THCudaTensor_newWithSize4d(state, size0, size1, size2, -1);
}

THCudaTensor *THCudaTensor_newWithSize4d(THCState *state, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THCudaTensor *self = (THCudaTensor*)THAlloc(sizeof(THCudaTensor));
  THCudaTensor_rawInit(state, self);
  THCudaTensor_rawResize(state, self, 4, size, NULL);

  return self;
}

THCudaTensor *THCudaTensor_newClone(THCState *state, THCudaTensor *self)
{
  THCudaTensor *tensor = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, tensor, self);
  THCudaTensor_copy(state, tensor, self);
  return tensor;
}

THCudaTensor *THCudaTensor_newContiguous(THCState *state, THCudaTensor *self)
{
  if(!THCudaTensor_isContiguous(state, self))
    return THCudaTensor_newClone(state, self);
  else
  {
    THCudaTensor_retain(state, self);
    return self;
  }
}

THCudaTensor *THCudaTensor_newSelect(THCState *state, THCudaTensor *tensor, int dimension_, long sliceIndex_)
{
  THCudaTensor *self = THCudaTensor_newWithTensor(state, tensor);
  THCudaTensor_select(state, self, NULL, dimension_, sliceIndex_);
  return self;
}

THCudaTensor *THCudaTensor_newNarrow(THCState *state, THCudaTensor *tensor, int dimension_, long firstIndex_, long size_)
{
  THCudaTensor *self = THCudaTensor_newWithTensor(state, tensor);
  THCudaTensor_narrow(state, self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THCudaTensor *THCudaTensor_newTranspose(THCState *state, THCudaTensor *tensor, int dimension1_, int dimension2_)
{
  THCudaTensor *self = THCudaTensor_newWithTensor(state, tensor);
  THCudaTensor_transpose(state, self, NULL, dimension1_, dimension2_);
  return self;
}

THCudaTensor *THCudaTensor_newUnfold(THCState *state, THCudaTensor *tensor, int dimension_, long size_, long step_)
{
  THCudaTensor *self = THCudaTensor_newWithTensor(state, tensor);
  THCudaTensor_unfold(state, self, NULL, dimension_, size_, step_);
  return self;
}

/* Resize */
void THCudaTensor_resize(THCState *state, THCudaTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THCudaTensor_rawResize(state, self, size->size, size->data, (stride ? stride->data : NULL));
}

void THCudaTensor_resizeAs(THCState *state, THCudaTensor *self, THCudaTensor *src)
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
    THCudaTensor_rawResize(state, self, src->nDimension, src->size, NULL);
}

void THCudaTensor_resize1d(THCState *state, THCudaTensor *tensor, long size0)
{
  THCudaTensor_resize4d(state, tensor, size0, -1, -1, -1);
}

void THCudaTensor_resize2d(THCState *state, THCudaTensor *tensor, long size0, long size1)
{
  THCudaTensor_resize4d(state, tensor, size0, size1, -1, -1);
}

void THCudaTensor_resize3d(THCState *state, THCudaTensor *tensor, long size0, long size1, long size2)
{
  THCudaTensor_resize4d(state, tensor, size0, size1, size2, -1);
}

void THCudaTensor_resize4d(THCState *state, THCudaTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THCudaTensor_rawResize(state, self, 4, size, NULL);
}

void THCudaTensor_resize5d(THCState *state, THCudaTensor *self, long size0, long size1, long size2, long size3, long size4)
{
    long size[5] = {size0, size1, size2, size3, size4};

  THCudaTensor_rawResize(state, self, 5, size, NULL);
}

void THCudaTensor_set(THCState *state, THCudaTensor *self, THCudaTensor *src)
{
  if(self != src)
    THCudaTensor_rawSet(state,
                        self,
                        src->storage,
                        src->storageOffset,
                        src->nDimension,
                        src->size,
                        src->stride);
}

void THCudaTensor_setStorage(THCState *state, THCudaTensor *self, THCudaStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THCudaTensor_rawSet(state,
                      self,
                      storage_,
                      storageOffset_,
                      (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                      (size_ ? size_->data : NULL),
                      (stride_ ? stride_->data : NULL));
}

void THCudaTensor_setStorage1d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_)
{
  THCudaTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            -1, -1,
                            -1, -1,
                            -1, -1);
}

void THCudaTensor_setStorage2d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_)
{
  THCudaTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            -1, -1,
                            -1, -1);
}

void THCudaTensor_setStorage3d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_)
{
  THCudaTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            size2_, stride2_,
                            -1, -1);
}

void THCudaTensor_setStorage4d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_,
                             long size3_, long stride3_)
{

  long size[4] = {size0_, size1_, size2_, size3_};
  long stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THCudaTensor_rawSet(state, self, storage_, storageOffset_, 4, size, stride);
}


void THCudaTensor_narrow(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 4, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 5, "out of range");

  THCudaTensor_set(state, self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THCudaTensor_select(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THCudaTensor_set(state, self, src);
  THCudaTensor_narrow(state, self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THCudaTensor_transpose(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension1, int dimension2)
{
  long z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THCudaTensor_set(state, self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THCudaTensor_unfold(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THCudaTensor_set(state, self, src);

  newSize = (long*)THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = (long*)THAlloc(sizeof(long)*(self->nDimension+1));

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
void THCudaTensor_squeeze(THCState *state, THCudaTensor *self, THCudaTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THCudaTensor_set(state, self, src);

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

void THCudaTensor_squeeze1d(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THCudaTensor_set(state, self, src);

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

int THCudaTensor_isContiguous(THCState *state, const THCudaTensor *self)
{
  long z = 1;
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

int THCudaTensor_isSameSizeAs(THCState *state, const THCudaTensor *self, const THCudaTensor* src)
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

long THCudaTensor_nElement(THCState *state, const THCudaTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THCudaTensor_retain(THCState *state, THCudaTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    ++self->refcount;
}

void THCudaTensor_free(THCState *state, THCudaTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(--self->refcount == 0)
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THCudaStorage_free(state, self->storage);
      THFree(self);
    }
  }
}

void THCudaTensor_freeCopyTo(THCState *state, THCudaTensor *self, THCudaTensor *dst)
{
  if(self != dst)
    THCudaTensor_copy(state, dst, self);

  THCudaTensor_free(state, self);
}

/*******************************************************************************/

static void THCudaTensor_rawInit(THCState *state, THCudaTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

static void THCudaTensor_rawSet(THCState *state, THCudaTensor *self, THCudaStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THCudaStorage_free(state, self->storage);

    if(storage)
    {
      self->storage = storage;
      THCudaStorage_retain(state, self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THCudaTensor_rawResize(state, self, nDimension, size, stride);
}

static void THCudaTensor_rawResize(THCState *state, THCudaTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;
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
      self->size = (long*)THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = (long*)THRealloc(self->stride, sizeof(long)*nDimension);
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
        self->storage = THCudaStorage_new(state);
      if(totalSize+self->storageOffset > self->storage->size)
        THCudaStorage_resize(state, self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THCudaTensor_set1d(THCState *state, THCudaTensor *tensor, long x0, float value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

float THCudaTensor_get1d(THCState *state, const THCudaTensor *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THCudaTensor_set2d(THCState *state, THCudaTensor *tensor, long x0, long x1, float value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

float THCudaTensor_get2d(THCState *state, const THCudaTensor *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THCudaTensor_set3d(THCState *state, THCudaTensor *tensor, long x0, long x1, long x2, float value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

float THCudaTensor_get3d(THCState *state, const THCudaTensor *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THCudaTensor_set4d(THCState *state, THCudaTensor *tensor, long x0, long x1, long x2, long x3, float value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

float THCudaTensor_get4d(THCState *state, const THCudaTensor *tensor, long x0, long x1, long x2, long x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

int THCudaTensor_checkGPU(THCState *state, unsigned int nTensors, ...)
{
  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));
  va_list(args);
  va_start(args, nTensors);
  int valid = 1;
  for (unsigned int i = 0; i < nTensors; i++) {
    THCudaTensor* tensor = va_arg(args, THCudaTensor*);
    int tensorDev = THCudaTensor_getDevice(state, tensor);
    if (tensorDev != -1 && tensorDev != curDev) {
      valid = 0;
      break;
    }
  }
  va_end(args);
  return valid;
}
