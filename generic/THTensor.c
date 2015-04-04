#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.c"
#else

/**** access methods ****/
THStorage *THTensor_(storage)(const THTensor *self)
{
  return self->storage;
}

long THTensor_(storageOffset)(const THTensor *self)
{
  return self->storageOffset;
}

int THTensor_(nDimension)(const THTensor *self)
{
  return self->nDimension;
}

long THTensor_(size)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+1, THTensor_(nDimension)(self));
  return self->size[dim];
}

long THTensor_(stride)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor", dim+1,
      THTensor_(nDimension)(self));
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
static void THTensor_(rawSet)(THTensor *self, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride);
static void THTensor_(rawResize)(THTensor *self, int nDimension, long *size, long *stride);


/* Empty init */
THTensor *THTensor_(new)(void)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *tensor)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(rawSet)(self,
                    tensor->storage,
                    tensor->storageOffset,
                    tensor->nDimension,
                    tensor->size,
                    tensor->stride);
  return self;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THTensor_(rawInit)(self);
  THTensor_(rawSet)(self,
                    storage,
                    storageOffset,
                    (size ? size->size : (stride ? stride->size : 0)),
                    (size ? size->data : NULL),
                    (stride ? stride->data : NULL));

  return self;
}
THTensor *THTensor_(newWithStorage1d)(THStorage *storage, long storageOffset,
                               long size0, long stride0)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage2d)(THStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage3d)(THStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THTensor *THTensor_(newWithStorage4d)(THStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4] = {size0, size1, size2, size3};
  long stride[4] = {stride0, stride1, stride2, stride3};

  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(rawSet)(self, storage, storageOffset, 4, size, stride);

  return self;
}

THTensor *THTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride)
{
  return THTensor_(newWithStorage)(NULL, 0, size, stride);
}

THTensor *THTensor_(newWithSize1d)(long size0)
{
  return THTensor_(newWithSize4d)(size0, -1, -1, -1);
}

THTensor *THTensor_(newWithSize2d)(long size0, long size1)
{
  return THTensor_(newWithSize4d)(size0, size1, -1, -1);
}

THTensor *THTensor_(newWithSize3d)(long size0, long size1, long size2)
{
  return THTensor_(newWithSize4d)(size0, size1, size2, -1);
}

THTensor *THTensor_(newWithSize4d)(long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(rawResize)(self, 4, size, NULL);

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

THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, long sliceIndex_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(select)(self, NULL, dimension_, sliceIndex_);
  return self;
}

THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, long firstIndex_, long size_)
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

THTensor *THTensor_(newUnfold)(THTensor *tensor, int dimension_, long size_, long step_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(unfold)(self, NULL, dimension_, size_, step_);
  return self;
}

/* Resize */
void THTensor_(resize)(THTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THTensor_(rawResize)(self, size->size, size->data, (stride ? stride->data : NULL));
}

void THTensor_(resizeAs)(THTensor *self, THTensor *src)
{
  if(!THTensor_(isSameSizeAs)(self, src))
    THTensor_(rawResize)(self, src->nDimension, src->size, NULL);
}

void THTensor_(resize1d)(THTensor *tensor, long size0)
{
  THTensor_(resize4d)(tensor, size0, -1, -1, -1);
}

void THTensor_(resize2d)(THTensor *tensor, long size0, long size1)
{
  THTensor_(resize4d)(tensor, size0, size1, -1, -1);
}

void THTensor_(resize3d)(THTensor *tensor, long size0, long size1, long size2)
{
  THTensor_(resize4d)(tensor, size0, size1, size2, -1);
}

void THTensor_(resize4d)(THTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THTensor_(rawResize)(self, 4, size, NULL);
}

void THTensor_(resize5d)(THTensor *self, long size0, long size1, long size2, long size3, long size4)
{
    long size[5] = {size0, size1, size2, size3, size4};

  THTensor_(rawResize)(self, 5, size, NULL);
}

void THTensor_(set)(THTensor *self, THTensor *src)
{
  if(self != src)
    THTensor_(rawSet)(self,
                      src->storage,
                      src->storageOffset,
                      src->nDimension,
                      src->size,
                      src->stride);
}

void THTensor_(setStorage)(THTensor *self, THStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THTensor_(rawSet)(self,
                    storage_,
                    storageOffset_,
                    (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                    (size_ ? size_->data : NULL),
                    (stride_ ? stride_->data : NULL));
}

void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, long storageOffset_,
                             long size0_, long stride0_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          -1, -1,
                          -1, -1,
                          -1, -1);
}

void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          -1, -1,
                          -1, -1);
}

void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          size2_, stride2_,
                          -1, -1);
}

void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_,
                             long size3_, long stride3_)
{

  long size[4] = {size0_, size1_, size2_, size3_};
  long stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THTensor_(rawSet)(self, storage_, storageOffset_, 4, size, stride);
}


void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 4, "out of range");

  THTensor_(set)(self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THTensor_(select)(THTensor *self, THTensor *src, int dimension, long sliceIndex)
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
  long z;

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

void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THTensor_(set)(self, src);

  newSize = THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = THAlloc(sizeof(long)*(self->nDimension+1));

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

int THTensor_(isContiguous)(const THTensor *self)
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

long THTensor_(nElement)(const THTensor *self)
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
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

static void THTensor_(rawSet)(THTensor *self, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
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
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THTensor_(rawResize)(self, nDimension, size, stride);
}

static void THTensor_(rawResize)(THTensor *self, int nDimension, long *size, long *stride)
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
      self->size = THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = THRealloc(self->stride, sizeof(long)*nDimension);
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

void THTensor_(set1d)(THTensor *tensor, long x0, real value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

real THTensor_(get1d)(const THTensor *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

real THTensor_(get2d)(const THTensor *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

real THTensor_(get3d)(const THTensor *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

real THTensor_(get4d)(const THTensor *tensor, long x0, long x1, long x2, long x3)
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
    n += snprintf(str+n, L-n, "%ld", tensor->size[i]);
    if(i < tensor->nDimension-1) {
      n += snprintf(str+n, L-n, "x");
    }
  }
  if(n >= L) {
    snprintf(str+L-4+n, 4, "...");
  }
  return buf;
}

THDescBuff THTensor_(sizeDesc)(const THTensor *tensor) {
  const int L = TH_DESC_BUFF_LEN;
  THDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for(i = 0; i < tensor->nDimension; i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%ld", tensor->size[i]);
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
