#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

THDStorage *THDTensor_(storage)(const THDTensor *self)
{
  return self->storage;
}

ptrdiff_t THDTensor_(storageOffset)(const THDTensor *self)
{
  return self->storageOffset;
}

int THDTensor_(nDimension)(const THDTensor *self)
{
  return self->nDimension;
}

long THDTensor_(size)(const THDTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+1, THDTensor_(nDimension)(self));
  return self->size[dim];
}

long THDTensor_(stride)(const THDTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor", dim+1,
      THDTensor_(nDimension)(self));
  return self->stride[dim];
}

THLongStorage *THDTensor_(newSizeOf)(THDTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THDTensor_(newStrideOf)(THDTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

void THDTensor_(setFlag)(THDTensor *self, const char flag)
{
  self->flag |= flag;
}

void THDTensor_(clearFlag)(THDTensor *self, const char flag)
{
  self->flag &= ~flag;
}

static THDTensor* THDTensor_(_alloc)() {
  THDTensor* new_tensor = new THDTensor();
  std::memset(reinterpret_cast<void*>(new_tensor), 0, sizeof(THDTensor));
  new_tensor->tensor_id = THDState::s_nextId++;
  new_tensor->refcount = 1;
  new_tensor->flag = TH_TENSOR_REFCOUNTED;
  // TODO: allocate storage
  return new_tensor;
}

THDTensor* THDTensor_(new)() {
  THDTensor* tensor = THDTensor_(_alloc)();
  Type constructed_type = type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::construct,
      constructed_type,
      tensor
    ),
    THDState::s_current_worker
  );
  return tensor;
}

THDTensor* THDTensor_(newWithSize)(THLongStorage *sizes, THLongStorage *strides) {
  THDTensor* tensor = THDTensor_(_alloc)();
  Type constructed_type = type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::constructWithSize,
      constructed_type,
      tensor,
      sizes,
      strides
    ),
    THDState::s_current_worker
  );
  return tensor;
}

// taken from TH (generic/THTensor.c)
static void THDTensor_(_resize)(THDTensor *self, int nDimension, long *size, long *stride)
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
      self->size = reinterpret_cast<long*>(THRealloc(self->size, sizeof(long)*nDimension));
      self->stride = reinterpret_cast<long*>(THRealloc(self->stride, sizeof(long)*nDimension));
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

    if(totalSize + self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THDStorage_(new)();
      if(totalSize+self->storageOffset > self->storage->size)
        THDStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THDTensor_(resize)(THDTensor *tensor, THLongStorage *size, THLongStorage *stride) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::resize,
      tensor,
      size,
      stride
    ),
    THDState::s_current_worker
  );
  THDTensor_(_resize)(tensor, size->size, size->data, stride ? stride->data : nullptr);
}

void THDTensor_(resizeAs)(THDTensor *tensor, THDTensor *src) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::resizeAs,
      tensor,
      src
    ),
    THDState::s_current_worker
  );
  THDTensor_(_resize)(tensor, src->nDimension, src->size, nullptr);
}

void THDTensor_(resize1d)(THDTensor *tensor, long size0) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::resize1d,
      tensor,
      size0
    ),
    THDState::s_current_worker
  );
  THDTensor_(_resize)(tensor, 1, &size0, nullptr);
}

void THDTensor_(resize2d)(THDTensor *tensor, long size0, long size1) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::resize2d,
      tensor,
      size0,
      size1
    ),
    THDState::s_current_worker
  );
  long sizes[] = {size0, size1};
  THDTensor_(_resize)(tensor, 2, sizes, nullptr);
}

void THDTensor_(resize3d)(THDTensor *tensor, long size0, long size1, long size2) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::resize3d,
      tensor,
      size0,
      size1,
      size2
    ),
    THDState::s_current_worker
  );
  long sizes[] = {size0, size1, size2};
  THDTensor_(_resize)(tensor, 3, sizes, nullptr);
}

void THDTensor_(resize4d)(THDTensor *tensor, long size0, long size1, long size2, long size3) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::resize4d,
      tensor,
      size0,
      size1,
      size2,
      size3
    ),
    THDState::s_current_worker
  );
  long sizes[] = {size0, size1, size2, size3};
  THDTensor_(_resize)(tensor, 4, sizes, nullptr);
}

void THDTensor_(resize5d)(THDTensor *tensor, long size0, long size1, long size2, long size3, long size4_) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::resize5d,
      tensor,
      size0,
      size1,
      size2,
      size3,
      size4_
    ),
    THDState::s_current_worker
  );
  long sizes[] = {size0, size1, size2, size3, size4_};
  THDTensor_(_resize)(tensor, 5, sizes, nullptr);
}

static void THDTensor_(_set)(THDTensor *self, THDStorage *storage,
  ptrdiff_t storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THDStorage_(free)(self->storage);

    if(storage)
    {
      self->storage = storage;
      THDStorage_(retain)(self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("can't set negative storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THDTensor_(_resize)(self, nDimension, size, stride);
}

void THDTensor_(set)(THDTensor *self, THDTensor *src) {
  if (self == src)
    return;

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::set,
      self,
      src
    ),
    THDState::s_current_worker
  );
  THDTensor_(_set)(self, src->storage, src->storageOffset,
      src->nDimension, src->size, src->stride);
}

void THDTensor_(setStorage)(THDTensor *self, THDStorage *storage,
    ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::setStorage,
      self,
      storage,
      storageOffset,
      size,
      stride
    ),
    THDState::s_current_worker
  );
  if (size && stride)
    THArgCheck(size->size == stride->size, 5, "inconsistent number of sizes and strides");

  THDTensor_(_set)(
    self,
    storage,
    storageOffset,
    (size ? size->size : (stride ? stride->size : 0)),
    (size ? size->data : nullptr),
    (stride ? stride->data : nullptr)
  );
}

void THDTensor_(setStorage1d)(THDTensor *self, THDStorage *storage, ptrdiff_t storageOffset,
                            long size0, long stride0) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::setStorage1d,
      self,
      storage,
      storageOffset,
      size0,
      stride0
    ),
    THDState::s_current_worker
  );
  long size[] = {size0};
  long stride[] = {stride0};
  THDTensor_(_set)(
    self,
    storage,
    storageOffset,
    1,
    size,
    stride
  );
}

void THDTensor_(setStorage2d)(THDTensor *self, THDStorage *storage, ptrdiff_t storageOffset,
                            long size0, long stride0,
                            long size1, long stride1) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::setStorage2d,
      self,
      storage,
      storageOffset,
      size0,
      size1,
      stride0,
      stride1
    ),
    THDState::s_current_worker
  );
  long size[] = {size0, size1};
  long stride[] = {stride0, stride1};
  THDTensor_(_set)(
    self,
    storage,
    storageOffset,
    2,
    size,
    stride
  );
}

void THDTensor_(setStorage3d)(THDTensor *self, THDStorage *storage, ptrdiff_t storageOffset,
                            long size0, long stride0,
                            long size1, long stride1,
                            long size2, long stride2) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::setStorage2d,
      self,
      storage,
      storageOffset,
      size0,
      size1,
      size2,
      stride0,
      stride1,
      stride2
    ),
    THDState::s_current_worker
  );
  long size[] = {size0, size1, size2};
  long stride[] = {stride0, stride1, stride2};
  THDTensor_(_set)(
    self,
    storage,
    storageOffset,
    3,
    size,
    stride
  );
}
void THDTensor_(setStorage4d)(THDTensor *self, THDStorage *storage, ptrdiff_t storageOffset,
                                    long size0, long stride0,
                                    long size1, long stride1,
                                    long size2, long stride2,
                                    long size3, long stride3) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::setStorage2d,
      self,
      storage,
      storageOffset,
      size0,
      size1,
      size2,
      size3,
      stride0,
      stride1,
      stride2,
      stride3
    ),
    THDState::s_current_worker
  );
  long size[] = {size0, size1, size2, size3};
  long stride[] = {stride0, stride1, stride2, stride3};
  THDTensor_(_set)(
    self,
    storage,
    storageOffset,
    4,
    size,
    stride
  );
}


void THDTensor_(free)(THDTensor *tensor) {
  // TODO: refcount
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::free,
      tensor
    ),
    THDState::s_current_worker
  );
}

ptrdiff_t THDTensor_(nElement)(const THDTensor *self)
{
  if(self->nDimension == 0) return 0;

  ptrdiff_t nElement = 1;
  int d;
  for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
  return nElement;
}

void THDTensor_(narrow)(THDTensor *self, THDTensor *src, int dimension,
    long firstIndex, long size) {
  if(!src) src = self;

  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck((size > 0) && (firstIndex <= src->size[dimension] - size), 4, "out of range");

  THDTensor_(set)(self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::narrow,
      self,
      src,
      dimension,
      firstIndex,
      size
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(select)(THDTensor *self, THDTensor *src, int dimension,
    long sliceIndex) {
  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  THDTensor_(set)(self, src);
  THDTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);
  for(int d = dimension; d < self->nDimension-1; d++) {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::select,
      self,
      src,
      dimension,
      sliceIndex
    ),
    THDState::s_current_worker
  );
}

THDTensor *THDTensor_(newWithStorage1d)(THDStorage *storage_,
    ptrdiff_t storageOffset_, long size0_, long stride0_) {
  THError("newWithStorage1d not supported yet");
  return nullptr;
}

THDTensor *THDTensor_(newWithTensor)(THDTensor *tensor) {
  THError("newWithTensor not supported yet");
  return nullptr;
}


void THDTensor_(fill)(THDTensor *tensor, real value) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::fill,
      tensor,
      value
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(zeros)(THDTensor *tensor, THLongStorage *size) {
  THDTensor_(resize)(tensor, size, nullptr);
  THDTensor_(fill)(tensor, 0);
}

void THDTensor_(ones)(THDTensor *tensor, THLongStorage *size) {
  THDTensor_(resize)(tensor, size, nullptr);
  THDTensor_(fill)(tensor, 0);
}

ptrdiff_t THDTensor_(numel)(THDTensor *t) {
  return THDTensor_(nElement)(t);
}



#endif
