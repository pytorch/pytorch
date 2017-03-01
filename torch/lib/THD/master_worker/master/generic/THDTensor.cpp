#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

template<typename T>
T THDTensor_(receiveValueFromWorker)(int worker_id) {
  thpp::Type type = thpp::type_traits<real>::type;
  if (thpp::isInteger(type)) {
    IntScalar wrapped_value;
    dataChannel->receive(wrapped_value, worker_id);
    return static_cast<T>(wrapped_value.value());
  } else if (thpp::isFloat(type)) {
    FloatScalar wrapped_value;
    dataChannel->receive(wrapped_value, worker_id);
    return static_cast<T>(wrapped_value.value());
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

// taken from TH (generic/THTensor.c)
THDDescBuff THDTensor_(sizeDesc)(const THDTensor *tensor) {
  const int L = THD_DESC_BUFF_LEN;
  THDDescBuff buf;
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

THDStorage *THDTensor_(storage)(const THDTensor *self) {
  return self->storage;
}

ptrdiff_t THDTensor_(storageOffset)(const THDTensor *self) {
  return self->storageOffset;
}

int THDTensor_(nDimension)(const THDTensor *self) {
  return self->nDimension;
}

long THDTensor_(size)(const THDTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+1, THDTensor_(nDimension)(self));
  return self->size[dim];
}

long THDTensor_(stride)(const THDTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor", dim+1,
      THDTensor_(nDimension)(self));
  return self->stride[dim];
}

THLongStorage *THDTensor_(newSizeOf)(THDTensor *self) {
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THDTensor_(newStrideOf)(THDTensor *self) {
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

void THDTensor_(setFlag)(THDTensor *self, const char flag) {
  self->flag |= flag;
}

void THDTensor_(clearFlag)(THDTensor *self, const char flag) {
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
  thpp::Type constructed_type = thpp::type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorConstruct,
      constructed_type,
      tensor
    ),
    THDState::s_current_worker
  );
  return tensor;
}

THDTensor* THDTensor_(newWithSize)(THLongStorage *sizes, THLongStorage *strides) {
  THDTensor* tensor = THDTensor_(_alloc)();
  thpp::Type constructed_type = thpp::type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorConstructWithSize,
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
      Functions::tensorResize,
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
      Functions::tensorResizeAs,
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
      Functions::tensorResize1d,
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
      Functions::tensorResize2d,
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
      Functions::tensorResize3d,
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
      Functions::tensorResize4d,
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
      Functions::tensorResize5d,
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

  THDTensor_(_set)(self, src->storage, src->storageOffset,
      src->nDimension, src->size, src->stride);

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorSet,
      self,
      src
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(setStorage)(THDTensor *self, THDStorage *storage,
    ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorSetStorage,
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

void THDTensor_(setStorage1d)(THDTensor *self,
                              THDStorage *storage,
                              ptrdiff_t storageOffset,
                              long size0, long stride0) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorSetStorage1d,
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

void THDTensor_(setStorage2d)(THDTensor *self,
                              THDStorage *storage,
                              ptrdiff_t storageOffset,
                              long size0, long stride0,
                              long size1, long stride1) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorSetStorage2d,
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

void THDTensor_(setStorage3d)(THDTensor *self,
                              THDStorage *storage,
                              ptrdiff_t storageOffset,
                              long size0, long stride0,
                              long size1, long stride1,
                              long size2, long stride2) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorSetStorage2d,
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
void THDTensor_(setStorage4d)(THDTensor *self,
                              THDStorage *storage,
                              ptrdiff_t storageOffset,
                              long size0, long stride0,
                              long size1, long stride1,
                              long size2, long stride2,
                              long size3, long stride3) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorSetStorage2d,
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
      Functions::tensorFree,
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
      Functions::tensorNarrow,
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
      Functions::tensorSelect,
      self,
      src,
      dimension,
      sliceIndex
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(transpose)(THDTensor *self, THDTensor *src, int dimension1,
    int dimension2) {
  if (!src)
    src = self;

  THArgCheck((dimension1 >= 0) && dimension1 < src->nDimension, 1,
    "out of range");
  THArgCheck((dimension2 >= 0) && dimension2 < src->nDimension, 1,
    "out of range");

  THDTensor_(set)(self, src);

  if (dimension1 == dimension2)
    return;

  std::swap(self->stride[dimension1], self->stride[dimension2]);
  std::swap(self->size[dimension1], self->size[dimension2]);

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorTranspose,
      self,
      src,
      dimension1,
      dimension2
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(unfold)(THDTensor *self, THDTensor *src,
                        int dimension, long size, long step) {
  long *newSize, *newStride;
  if (!src)
    src = self;

  THArgCheck((src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck((dimension > 0) && (dimension < src->nDimension), 2,
    "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THDTensor_(set)(self, src);

  newSize = static_cast<long *>(THAlloc(sizeof(long) * (self->nDimension + 1)));
  newStride = static_cast<long *>(THAlloc(sizeof(long) * (self->nDimension + 1)));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];

  for (std::size_t d = 0; d < self->nDimension; d++) {
    if (d == dimension) {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step * self->stride[d];
    } else {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorUnfold,
      self,
      src,
      dimension,
      size,
      step
    ),
    THDState::s_current_worker
  );
}

// TODO implement
int THDTensor_(isSameSizeAs)(const THDTensor *self, const THDTensor *src) {
  throw std::runtime_error("isSameSizeAs not implemented yet");
}

void THDTensor_(gather)(THDTensor *self, THDTensor *src, int dim, THDLongTensor *index) {
  THArgCheck(dim < self->nDimension, 2, "Index dimension is out of bounds");
  THArgCheck(THDLongTensor_nDimension(index) == self->nDimension, 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(src->nDimension == self->nDimension, 4,
             "Input tensor must have same dimensions as output tensor");

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorGather,
      self,
      src,
      dim,
      index
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(scatter)(THDTensor *self, int dim, THDLongTensor *index, THDTensor *src) {
  THArgCheck(dim < self->nDimension, 2, "Index dimension is out of bounds");
  THArgCheck(THDLongTensor_nDimension(index) == self->nDimension, 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(src->nDimension == self->nDimension, 4,
             "Input tensor must have same dimensions as output tensor");

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorScatter,
      self,
      dim,
      index,
      src
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(scatterFill)(THDTensor *self, int dim, THDLongTensor *index, real val) {
  THArgCheck(dim < self->nDimension, 2, "Index dimension is out of bounds");
  THArgCheck(THDLongTensor_nDimension(index) == self->nDimension, 3,
             "Index tensor must have same dimensions as output tensor");

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorScatterFill,
      self,
      dim,
      index,
      val
    ),
    THDState::s_current_worker
  );
}

accreal THDTensor_(dot)(THDTensor *self, THDTensor *src) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorDot, self, src),
    THDState::s_current_worker
  );

  return THDTensor_(receiveValueFromWorker)<accreal>(THDState::s_current_worker);
}

real THDTensor_(minall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMinall, self),
    THDState::s_current_worker
  );

  return THDTensor_(receiveValueFromWorker)<real>(THDState::s_current_worker);
}

real THDTensor_(maxall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMaxall, self),
    THDState::s_current_worker
  );

  return THDTensor_(receiveValueFromWorker)<real>(THDState::s_current_worker);
}

accreal THDTensor_(sumall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSumall, self),
    THDState::s_current_worker
  );

  return THDTensor_(receiveValueFromWorker)<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(prodall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorProdall, self),
    THDState::s_current_worker
  );

  return THDTensor_(receiveValueFromWorker)<accreal>(THDState::s_current_worker);
}

void THDTensor_(neg)(THDTensor *self, THDTensor *src) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorNeg, self, src),
    THDState::s_current_worker
  );
}

void THDTensor_(cinv)(THDTensor *self, THDTensor *src) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCinv, self, src),
    THDState::s_current_worker
  );
}

void THDTensor_(add)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAdd, self, src, value),
    THDState::s_current_worker
  );
}

void THDTensor_(sub)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSub, self, src, value),
    THDState::s_current_worker
  );
}

void THDTensor_(mul)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMul, self, src, value),
    THDState::s_current_worker
  );
}

void THDTensor_(div)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorDiv, self, src, value),
    THDState::s_current_worker
  );
}

void THDTensor_(fmod)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorFmod, self, src, value),
    THDState::s_current_worker
  );
}

void THDTensor_(remainder)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRemainder, self, src, value),
    THDState::s_current_worker
  );
}

void THDTensor_(clamp)(THDTensor *self, THDTensor *src, real min_value, real max_value) {
  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorClamp, self, src, min_value, max_value),
    THDState::s_current_worker
  );
}

void THDTensor_(cadd)(THDTensor *self, THDTensor *src1, real value, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCadd, self, src1, src2, value),
    THDState::s_current_worker
  );
}

void THDTensor_(csub)(THDTensor *self, THDTensor *src1, real value, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCsub, self, src1, src2, value),
    THDState::s_current_worker
  );
}

void THDTensor_(cmul)(THDTensor *self, THDTensor *src1, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCmul, self, src1, src2),
    THDState::s_current_worker
  );
}

void THDTensor_(cpow)(THDTensor *self, THDTensor *src1, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCpow, self, src1, src2),
    THDState::s_current_worker
  );
}

void THDTensor_(cdiv)(THDTensor *self, THDTensor *src1, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCdiv, self, src1, src2),
    THDState::s_current_worker
  );
}

void THDTensor_(cfmod)(THDTensor *self, THDTensor *src1, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCfmod, self, src1, src2),
    THDState::s_current_worker
  );
}

void THDTensor_(cremainder)(THDTensor *self, THDTensor *src1, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCremainder, self, src1, src2),
    THDState::s_current_worker
  );
}

void THDTensor_(addcmul)(THDTensor *self, THDTensor *src1, real value, THDTensor *src2, THDTensor *src3) {
  if (self != src1) {
    THDTensor_(resizeAs)(self, src1);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAddcmul, self, src1, src2, src3, value),
    THDState::s_current_worker
  );
}

void THDTensor_(addcdiv)(THDTensor *self, THDTensor *src1, real value, THDTensor *src2, THDTensor *src3) {
  if (self != src1) {
    THDTensor_(resizeAs)(self, src1);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAddcdiv, self, src1, src2, src3, value),
    THDState::s_current_worker
  );
}

void THDTensor_(addmv)(THDTensor *self, real beta, THDTensor *src, real alpha, THDTensor *mat,  THDTensor *vec) {
  if ((mat->nDimension != 2) || (vec->nDimension != 1))
    THError("matrix and vector expected, got %dD, %dD", mat->nDimension, vec->nDimension);

  if (mat->size[1] != vec->size[0]) {
    THDDescBuff bm = THDTensor_(sizeDesc)(mat);
    THDDescBuff bv = THDTensor_(sizeDesc)(vec);
    THError("size mismatch, %s, %s", bm.str, bv.str);
  }

  if (src->nDimension != 1)
    THError("vector expected, got src: %dD", src->nDimension);

  if (src->size[0] != mat->size[0]) {
    THDDescBuff bt = THDTensor_(sizeDesc)(src);
    THDDescBuff bm = THDTensor_(sizeDesc)(mat);
    THError("size mismatch, src: %s, mat: %s", bt.str, bm.str);
  }

  if (self != src) {
    THDTensor_(resizeAs)(self, src);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAddmv, self, src, mat, vec, beta, alpha),
    THDState::s_current_worker
  );
}

void THDTensor_(addmm)(THDTensor *self, real beta, THDTensor *src, real alpha, THDTensor *mat1, THDTensor *mat2) {
  if ((mat1->nDimension != 2) || (mat2->nDimension != 2))
    THError("matrices expected, got %dD, %dD tensors", mat1->nDimension, mat2->nDimension);

  if (mat1->size[1] != mat2->size[0]) {
    THDDescBuff bm1 = THDTensor_(sizeDesc)(mat1);
    THDDescBuff bm2 = THDTensor_(sizeDesc)(mat2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if (src->nDimension != 2)
    THError("matrix expected, got %dD tensor for t", src->nDimension);

  if ((src->size[0] != mat1->size[0]) || (src->size[1] != mat2->size[1])) {
    THDDescBuff bt  = THDTensor_(sizeDesc)(src);
    THDDescBuff bm1 = THDTensor_(sizeDesc)(mat1);
    THDDescBuff bm2 = THDTensor_(sizeDesc)(mat2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if (self != src) {
    THDTensor_(resizeAs)(self, src);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAddmm, self, src, mat1, mat2, beta, alpha),
    THDState::s_current_worker
  );
}

void THDTensor_(addr)(THDTensor *self,  real beta, THDTensor *src, real alpha, THDTensor *vec1, THDTensor *vec2) {
  if ((vec1->nDimension != 1) || (vec2->nDimension != 1))
    THError("vector and vector expected, got %dD, %dD tensors", vec1->nDimension, vec2->nDimension);

  if (src->nDimension != 2)
    THError("expected matrix, got %dD tensor for t", src->nDimension);

  if ((src->size[0] != vec1->size[0]) || (src->size[1] != vec2->size[0])) {
    THDDescBuff bt  = THDTensor_(sizeDesc)(src);
    THDDescBuff bv1 = THDTensor_(sizeDesc)(vec1);
    THDDescBuff bv2 = THDTensor_(sizeDesc)(vec2);
    THError("size mismatch, src: %s, vec1: %s, vec2: %s", bt.str, bv1.str, bv2.str);
  }

  if (self != src) {
    THDTensor_(resizeAs)(self, src);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAddr, self, src, vec1, vec2, beta, alpha),
    THDState::s_current_worker
  );
}

void THDTensor_(addbmm)(THDTensor *self, real beta, THDTensor *src, real alpha, THDTensor *batch1, THDTensor *batch2) {
  THArgCheck(batch1->nDimension == 3, 1, "expected 3D tensor");
  THArgCheck(batch2->nDimension == 3, 2, "expected 3D tensor");
  THArgCheck(batch1->size[0] == batch2->size[0], 2,
             "equal number of batches expected, got %d, %d",
             batch1->size[0], batch2->size[0]);
  THArgCheck(batch1->size[2] == batch2->size[1], 2,
             "wrong matrix size, batch1: %dx%d, batch2: %dx%d",
             batch1->size[1], batch1->size[2], batch2->size[1], batch2->size[2]);

  THArgCheck(src->size[0] == batch1->size[1], 1, "output tensor of incorrect size");
  THArgCheck(src->size[1] == batch2->size[2], 1, "output tensor of incorrect size");

  if (self != src) {
    THDTensor_(resizeAs)(self, src);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAddbmm, self, src, batch1, batch2, beta, alpha),
    THDState::s_current_worker
  );
}

void THDTensor_(baddbmm)(THDTensor *self, real beta, THDTensor *src, real alpha, THDTensor *batch1, THDTensor *batch2) {
  THArgCheck(batch1->nDimension == 3, 1, "expected 3D tensor");
  THArgCheck(batch2->nDimension == 3, 2, "expected 3D tensor");
  THArgCheck(batch1->size[0] == batch2->size[0], 2,
             "equal number of batches expected, got %d, %d",
             batch1->size[0], batch2->size[0]);
  THArgCheck(batch1->size[2] == batch2->size[1], 2,
             "wrong matrix size, batch1: %dx%d, batch2: %dx%d",
             batch1->size[1], batch1->size[2], batch2->size[1], batch2->size[2]);

  THArgCheck(src->size[0] == batch1->size[0], 1, "output tensor of incorrect size");
  THArgCheck(src->size[1] == batch1->size[1], 1, "output tensor of incorrect size");
  THArgCheck(src->size[2] == batch2->size[2], 1, "output tensor of incorrect size");

  if (self != src) {
    THDTensor_(resizeAs)(self, src);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorBaddbmm, self, beta, src, alpha, batch1, batch2),
    THDState::s_current_worker
  );
}

void THDTensor_(match)(THDTensor *self, THDTensor *m1, THDTensor *m2, real gain) {
  THDTensor_(resize2d)(self, m1->size[0], m2->size[0]);
  THArgCheck(m1->size[1] == m2->size[1], 3, "m1 and m2 must have the same inner vector dim");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMatch, self, m1, m2, gain),
    THDState::s_current_worker
  );
}

void THDTensor_(max)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMax, self, indices_, src, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(min)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMin, self, indices_, src, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(kthvalue)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, long k, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 3, "dimension out of range");
  THArgCheck(k > 0 && k <= src->size[dimension], 2, "selected index out of range");

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorKthvalue, self, indices_, src, k, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(mode)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 3, "dimension out of range");

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMode, self, indices_, src, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(median)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 3, "dimension out of range");

  long t_size_dim = src->size[dimension];
  long k = (t_size_dim - 1) >> 1; /* take middle or one-before-middle element */

  THDTensor_(kthvalue)(self, indices_, src, k + 1, dimension);
}

void THDTensor_(sum)(THDTensor *self, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSum, self, src, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(prod)(THDTensor *self, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorProd, self, src, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(cumsum)(THDTensor *self, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCumsum, self, src, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(cumprod)(THDTensor *self, THDTensor *src, int dimension) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THDTensor_(resizeAs)(self, src);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCumprod, self, src, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(sign)(THDTensor *self, THDTensor *src) {
  THDTensor_(resizeAs)(self, src);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSign, self, src),
    THDState::s_current_worker
  );
}

accreal THDTensor_(trace)(THDTensor *self) {
  THArgCheck(self->nDimension == 2, 1, "expected a matrix");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTrace, self),
    THDState::s_current_worker
  );

  return THDTensor_(receiveValueFromWorker)<accreal>(THDState::s_current_worker);
}

void THDTensor_(cross)(THDTensor *self, THDTensor *src1, THDTensor *src2, int dimension) {
  if (src1->nDimension != src2->nDimension)
    THError("inconsistent tensor dimension %dD, %dD", src1->nDimension, src2->nDimension);

  for (int i = 0; i < src1->nDimension; i++) {
    if (src1->size[i] != src2->size[i]) {
      THDDescBuff ba = THDTensor_(sizeDesc)(src1);
      THDDescBuff bb = THDTensor_(sizeDesc)(src2);
      THError("inconsistent tensor sizes %s, %s", ba.str, bb.str);
    }
  }

  if (dimension < 0) {
    for (int i = 0; i < src1->nDimension; i++) {
      if (src1->size[i] == 3) {
        dimension = i;
        break;
      }
    }

    if (dimension < 0) {
      THDDescBuff ba = THDTensor_(sizeDesc)(src1);
      THError("no dimension of size 3 in a: %s", ba.str);
    }
  }

  THArgCheck(dimension >= 0 && dimension < src1->nDimension, 3, "dimension %d out of range",
      dimension + TH_INDEX_BASE);
  THArgCheck(src1->size[dimension] == 3, 3, "dimension %d does not have size 3",
      dimension + TH_INDEX_BASE);

  THDTensor_(resizeAs)(self, src1);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTrace, self, src1, src2, dimension),
    THDState::s_current_worker
  );
}

void THDTensor_(cmax)(THDTensor *self, THDTensor *src1, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCmax, self, src1, src2),
    THDState::s_current_worker
  );
}

void THDTensor_(cmin)(THDTensor *self, THDTensor *src1, THDTensor *src2) {
  THDTensor_(resizeAs)(self, src1);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCmin, self, src1, src2),
    THDState::s_current_worker
  );
}

void THDTensor_(cmaxValue)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCmaxValue, self, src, value),
    THDState::s_current_worker
  );
}

void THDTensor_(cminValue)(THDTensor *self, THDTensor *src, real value) {
  THDTensor_(resizeAs)(self, src);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCminValue, self, src, value),
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

#endif
