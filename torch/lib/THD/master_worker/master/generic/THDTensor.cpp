#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

// taken from TH (generic/THTensor.c)
THDDescBuff THDTensor_(sizeDesc)(const THDTensor *tensor) {
  const int L = THD_DESC_BUFF_LEN;
  THDDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for (i = 0; i < tensor->nDimension; i++) {
    if (n >= L) break;
    n += snprintf(str+n, L-n, "%ld", tensor->size[i]);
    if (i < tensor->nDimension-1) {
      n += snprintf(str+n, L-n, " x ");
    }
  }
  if (n < L - 2) {
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

void THDTensor_(setFlag)(THDTensor *self, char flag) {
  self->flag |= flag;
}

void THDTensor_(clearFlag)(THDTensor *self, char flag) {
  self->flag &= ~flag;
}

THDTensor *THDTensor_(new)() {
  THDTensor *tensor = THDTensor_(_alloc)();
  thpp::Type constructed_type = thpp::type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorNew,
      constructed_type,
      tensor
    ),
    THDState::s_current_worker
  );
  return tensor;
}

THDTensor *THDTensor_(newWithTensor)(THDTensor *self) {
  THDTensor *tensor = THDTensor_(_alloc)();
  THDTensor_(_set)(
    tensor,
    self->storage,
    self->storageOffset,
    self->nDimension,
    self->size,
    self->stride
  );
  thpp::Type constructed_type = thpp::type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorNewWithTensor,
      constructed_type,
      tensor,
      self
    ),
    THDState::s_current_worker
  );
  return tensor;
}

THDTensor *THDTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride) {
  THDTensor* tensor = THDTensor_(_alloc)();
  if (size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");
  long *size_cpy = size ? new long[size->size] : nullptr;
  long *stride_cpy = stride ? new long[stride->size] : nullptr;
  memcpy(size_cpy, size->data, size->size * sizeof(long));
  if (stride)
    memcpy(stride_cpy, stride->data, stride->size * sizeof(long));
  tensor->size = size_cpy;
  tensor->stride = stride_cpy;
  thpp::Type constructed_type = thpp::type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorNewWithSize,
      constructed_type,
      tensor,
      size,
      stride
    ),
    THDState::s_current_worker
  );
  return tensor;
}

THDTensor *THDTensor_(newWithSize1d)(long size0) {
  THLongStorage *size = THLongStorage_newWithSize1(size0);
  THDTensor *tensor = THDTensor_(newWithSize)(size, NULL);
  THLongStorage_free(size);
  return tensor;
}

THDTensor *THDTensor_(newWithSize2d)(long size0, long size1) {
  THLongStorage *size = THLongStorage_newWithSize2(size0, size1);
  THDTensor *tensor = THDTensor_(newWithSize)(size, NULL);
  THLongStorage_free(size);
  return tensor;
}

THDTensor *THDTensor_(newWithSize3d)(long size0, long size1, long size2) {
  THLongStorage *size = THLongStorage_newWithSize3(size0, size1, size2);
  THDTensor *tensor = THDTensor_(newWithSize)(size, NULL);
  THLongStorage_free(size);
  return tensor;
}

THDTensor *THDTensor_(newWithSize4d)(long size0, long size1, long size2, long size3) {
  THLongStorage *size = THLongStorage_newWithSize4(size0, size1, size2, size3);
  THDTensor *tensor = THDTensor_(newWithSize)(size, NULL);
  THLongStorage_free(size);
  return tensor;
}


THDTensor *THDTensor_(newWithStorage)(THDStorage *storage, ptrdiff_t storageOffset,
                                      THLongStorage *size, THLongStorage *stride) {
  THDTensor* tensor = THDTensor_(_alloc)();
  THDTensor_(_set)(
    tensor,
    storage,
    storageOffset,
    (size ? size->size : (stride ? stride->size : 0)),
    (size ? size->data : nullptr),
    (stride ? stride->data : nullptr)
  );
  thpp::Type constructed_type = thpp::type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorNewWithStorage,
      constructed_type,
      storage,
      storageOffset,
      size,
      stride
    ),
    THDState::s_current_worker
  );
  return tensor;
}

THDTensor *THDTensor_(newWithStorage1d)(THDStorage *storage, ptrdiff_t storageOffset,
                                        long size0, long stride0) {
  THLongStorage *size = THLongStorage_newWithSize1(size0);
  THLongStorage *stride = THLongStorage_newWithSize1(stride0);
  THDTensor *tensor = THDTensor_(newWithStorage)(storage, storageOffset, size, stride);
  THLongStorage_free(size);
  THLongStorage_free(stride);
  return tensor;
}

THDTensor *THDTensor_(newWithStorage2d)(THDStorage *storage, ptrdiff_t storageOffset,
                                        long size0, long stride0, long size1, long stride1) {
  THLongStorage *size = THLongStorage_newWithSize2(size0, size1);
  THLongStorage *stride = THLongStorage_newWithSize2(stride0, stride1);
  THDTensor *tensor = THDTensor_(newWithStorage)(storage, storageOffset, size, stride);
  THLongStorage_free(size);
  THLongStorage_free(stride);
  return tensor;
}

THDTensor *THDTensor_(newWithStorage3d)(THDStorage *storage, ptrdiff_t storageOffset,
                                        long size0, long stride0, long size1, long stride1,
                                        long size2, long stride2) {
  THLongStorage *size = THLongStorage_newWithSize3(size0, size1, size2);
  THLongStorage *stride = THLongStorage_newWithSize3(stride0, stride1, stride2);
  THDTensor *tensor = THDTensor_(newWithStorage)(storage, storageOffset, size, stride);
  THLongStorage_free(size);
  THLongStorage_free(stride);
  return THDTensor_(newWithStorage)(storage, storageOffset, size, stride);
}

THDTensor *THDTensor_(newWithStorage4d)(THDStorage *storage, ptrdiff_t storageOffset,
                                        long size0, long stride0, long size1, long stride1,
                                        long size2, long stride2, long size3, long stride3) {
  THLongStorage *size = THLongStorage_newWithSize4(size0, size1, size2, size3);
  THLongStorage *stride = THLongStorage_newWithSize4(stride0, stride1, stride2, stride3);
  THDTensor *tensor = THDTensor_(newWithStorage)(storage, storageOffset, size, stride);
  THLongStorage_free(size);
  THLongStorage_free(stride);
  return THDTensor_(newWithStorage)(storage, storageOffset, size, stride);
}

THDTensor *THDTensor_(newClone)(THDTensor *self) {
  THDTensor *clone = THDTensor_(_alloc)();
  THDTensor_(_resize)(clone, self->nDimension, self->size, self->stride);
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorNewClone,
      clone,
      self
    ),
    THDState::s_current_worker
  );
  return clone;
}

THDTensor *THDTensor_(newContiguous)(THDTensor *self) {
  if (!THDTensor_(isContiguous)(self)) {
    return THDTensor_(newClone)(self);
  } else {
    THDTensor_(retain)(self);
    return self;
  }
}

THDTensor *THDTensor_(newSelect)(THDTensor *tensor, int dimension, long sliceIndex) {
  THDTensor *self = THDTensor_(newWithTensor)(tensor);
  THDTensor_(select)(self, NULL, dimension, sliceIndex);
  return self;
}

THDTensor *THDTensor_(newNarrow)(THDTensor *tensor, int dimension,
                                 long firstIndex, long size) {
  THDTensor *self = THDTensor_(newWithTensor)(tensor);
  THDTensor_(narrow)(self, NULL, dimension, firstIndex, size);
  return self;
}

THDTensor *THDTensor_(newTranspose)(THDTensor *tensor, int dimension1, int dimension2) {
  THDTensor *self = THDTensor_(newWithTensor)(tensor);
  THDTensor_(transpose)(self, NULL, dimension1, dimension2);
  return self;
}

THDTensor *THDTensor_(newUnfold)(THDTensor *tensor, int dimension, long size, long step) {
  THDTensor *self = THDTensor_(newWithTensor)(tensor);
  THDTensor_(unfold)(self, NULL, dimension, size, step);
  return self;
}

THDTensor *THDTensor_(newView)(THDTensor *tensor, THLongStorage *size) {
  // TODO
  THError("newView not implemented");
  return nullptr;
}

THDTensor *THDTensor_(newExpand)(THDTensor *tensor, THLongStorage *size) {
  // TODO
  THError("newExpand not implemented");
  return nullptr;
}

void THDTensor_(resize)(THDTensor *tensor, THLongStorage *size, THLongStorage *stride) {
  THArgCheck(size != NULL, 2, "invalid size");
  if (stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

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
  THDTensor_(_resize2d)(tensor, size0, size1);
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
  THDTensor_(_resize3d)(tensor, size0, size1, size2);
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
  THDTensor_(_resize4d)(tensor, size0, size1, size2, size3);
}

void THDTensor_(resize5d)(THDTensor *tensor, long size0, long size1, long size2, long size3, long size4) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorResize5d,
      tensor,
      size0,
      size1,
      size2,
      size3,
      size4
    ),
    THDState::s_current_worker
  );
  THDTensor_(_resize5d)(tensor, size0, size1, size2, size3, size4);
}

real THDTensor_(get1d)(const THDTensor *tensor, long x0)
{
  // TODO
  THError("get1d not supported!");
  //THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  //THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  //return THDStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
  return 0;
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
                            ptrdiff_t storageOffset, THLongStorage *size,
                            THLongStorage *stride) {
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

void THDTensor_(narrow)(THDTensor *self, THDTensor *src, int dimension,
    long firstIndex, long size) {
  if (!src) src = self;

  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck((size > 0) && (firstIndex <= src->size[dimension] - size), 4, "out of range");

  THDTensor_(set)(self, src);

  if (firstIndex > 0)
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

void THDTensor_(select)(THDTensor *self, THDTensor *src, int dimension, long sliceIndex) {
  if (!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  THDTensor_(set)(self, src);
  THDTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);
  for (int d = dimension; d < self->nDimension-1; d++) {
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

  newSize = new long[self->nDimension + 1];
  newStride = new long[self->nDimension + 1];

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


  delete[] self->size;
  delete[] self->stride;

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

void THDTensor_(squeeze)(THDTensor *self, THDTensor *src) {
  int ndim = 0;

  if (!src)
    src = self;

  THDTensor_(set)(self, src);

  for (std::size_t d = 0; d < src->nDimension; d++) {
    if (src->size[d] != 1) {
      if (d != ndim) {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if (ndim == 0 && src->nDimension > 0) {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
  masterCommandChannel->sendMessage(
      packMessage(Functions::tensorSqueeze, self, src),
      THDState::s_current_worker
  );
}

void THDTensor_(squeeze1d)(THDTensor *self, THDTensor *src, int dimension) {
  if (!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "dimension out of range");

  THDTensor_(set)(self, src);

  if (src->size[dimension] == 1 && src->nDimension > 1) {
    for (std::size_t d = dimension; d < self->nDimension-1; d++) {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
  masterCommandChannel->sendMessage(
      packMessage(Functions::tensorSqueeze1d, self, src),
      THDState::s_current_worker
  );
}

void THDTensor_(unsqueeze1d)(THDTensor *self, THDTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->nDimension), 3, "dimension out of range");
  THArgCheck(src->nDimension > 0, 3, "cannot unsqueeze empty tensor");

  THDTensor_(set)(self, src);

  self->size = (long*)THRealloc(self->size, sizeof(long)*(self->nDimension+1));
  self->stride = (long*)THRealloc(self->stride, sizeof(long)*(self->nDimension+1));
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

int THDTensor_(isContiguous)(const THDTensor *self) {
  long z = 1;
  for (std::ptrdiff_t d = self->nDimension - 1; d >= 0; d--) {
    if (self->size[d] != 1) {
      if (self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THDTensor_(isSameSizeAs)(const THDTensor *self, const THDTensor *src) {
  if (self->nDimension != src->nDimension)
    return 0;
  for (std::size_t d = 0; d < self->nDimension; d++)
    if (self->size[d] != src->size[d])
      return 0;
  return 1;
}

int THDTensor_(isSetTo)(const THDTensor *self, const THDTensor *src) {
  if (!self->storage)
    return 0;
  if (self->storage == src->storage &&
      self->storageOffset == src->storageOffset &&
      self->nDimension == src->nDimension) {
    for (std::size_t d = 0; d < self->nDimension; d++) {
      if (self->size[d] != src->size[d] || self->stride[d] != src->stride[d])
        return 0;
    }
    return 1;
  }
  return 0;
}

int THDTensor_(isSize)(const THDTensor *self, const THLongStorage *dims) {
  if (self->nDimension != dims->size)
    return 0;
  for (std::size_t d = 0; d < self->nDimension; d++)
    if (self->size[d] != dims->data[d])
      return 0;
  return 1;
}

ptrdiff_t THDTensor_(nElement)(const THDTensor *self) {
  if (self->nDimension == 0) {
    return 0;
  } else {
    ptrdiff_t nElement = 1;
    for (std::size_t d = 0; d < self->nDimension; d++) {
      nElement *= self->size[d];
    }
    return nElement;
  }
}

void THDTensor_(retain)(THDTensor *tensor) {
  if (tensor->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&tensor->refcount);
}

void THDTensor_(free)(THDTensor *tensor) {
  if (!tensor)
    return;

  // TODO: check refcounted flag?
  if (THAtomicDecrementRef(&tensor->refcount)) {
    delete[] tensor->size;
    delete[] tensor->stride;
    masterCommandChannel->sendMessage(
        packMessage(
          Functions::tensorFree,
          tensor
          ),
        THDState::s_current_worker
    );
    THDStorage_(free)(tensor->storage);
  }
}

accreal THDTensor_(dot)(THDTensor *self, THDTensor *src) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorDot, self, src),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

real THDTensor_(minall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMinall, self),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<real>(THDState::s_current_worker);
}

real THDTensor_(maxall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMaxall, self),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<real>(THDState::s_current_worker);
}

accreal THDTensor_(sumall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSumall, self),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(prodall)(THDTensor *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorProdall, self),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
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

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
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

#endif
