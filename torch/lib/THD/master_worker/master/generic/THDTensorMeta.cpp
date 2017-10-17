#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorMeta.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

// taken from TH (generic/THTensor.c)
// with a little fixes done so as to allocate
// and free memory the way it is done in THDTensor
static void THDTensor_(_resize)(THDTensor *self, int nDimension, int64_t *size, int64_t *stride) {
  int nDimension_;
  ptrdiff_t totalSize;
  bool hasRequiredSize = true;

  nDimension_ = 0;
  for (std::size_t d = 0; d < nDimension; d++) {
    if (size[d] > 0) {
      nDimension_++;
      if ((self->nDimension > d) && (size[d] != self->size[d]))
        hasRequiredSize = false;
      if ((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hasRequiredSize = false;
    } else {
      break;
    }
  }
  nDimension = nDimension_;

  if (nDimension != self->nDimension)
    hasRequiredSize = false;

  if (hasRequiredSize)
    return;

  if (nDimension > 0) {
    if (nDimension != self->nDimension) {
      delete[] self->size;
      delete[] self->stride;
      self->size = new int64_t[nDimension];
      self->stride = new int64_t[nDimension];
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for (std::ptrdiff_t d = self->nDimension - 1; d >= 0; d--) {
      self->size[d] = size[d];
      if (stride && (stride[d] >= 0)) {
        self->stride[d] = stride[d];
      } else {
        if (d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if (totalSize + self->storageOffset > 0) {
      if (!self->storage)
        self->storage = THDStorage_(new)();
      if (totalSize + self->storageOffset > self->storage->size)
        THDStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  } else {
    self->nDimension = 0;
  }
}

void THDTensor_(_resize2d)(THDTensor *tensor, int64_t size0, int64_t size1) {
  int64_t sizes[] = {size0, size1};
  THDTensor_(_resize)(tensor, 2, sizes, nullptr);
}

void THDTensor_(_resize3d)(THDTensor *tensor, int64_t size0, int64_t size1, int64_t size2) {
  int64_t sizes[] = {size0, size1, size2};
  THDTensor_(_resize)(tensor, 2, sizes, nullptr);
}

void THDTensor_(_resize4d)(THDTensor *tensor, int64_t size0, int64_t size1, int64_t size2, int64_t size3) {
  int64_t sizes[] = {size0, size1, size2, size3};
  THDTensor_(_resize)(tensor, 2, sizes, nullptr);
}

void THDTensor_(_resize5d)(THDTensor *tensor, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4) {
  int64_t sizes[] = {size0, size1, size2, size3, size4};
  THDTensor_(_resize)(tensor, 2, sizes, nullptr);
}

void THDTensor_(_squeeze1d)(THDTensor *self, THDTensor *src, int dimension) {
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
}

static void THDTensor_(_set)(THDTensor *self, THDStorage *storage,
                             ptrdiff_t storageOffset, int nDimension,
                             int64_t *size, int64_t *stride) {
  /* storage */
  if (self->storage != storage) {
    if (self->storage)
      THDStorage_(free)(self->storage);

    if (storage) {
      self->storage = storage;
      THDStorage_(retain)(self->storage);
    } else {
      self->storage = NULL;
    }
  }

  /* storageOffset */
  if (storageOffset < 0)
    THError("can't set negative storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THDTensor_(_resize)(self, nDimension, size, stride);
}

static THDTensor *THDTensor_(_alloc)() {
  THDTensor *new_tensor = new THDTensor();
  std::memset(reinterpret_cast<void*>(new_tensor), 0, sizeof(THDTensor));
  new_tensor->size = nullptr;
  new_tensor->stride = nullptr;
  new_tensor->nDimension = 0;

  new_tensor->storage = nullptr;
  new_tensor->storageOffset = 0;

  new_tensor->refcount = 1;
  new_tensor->flag = TH_TENSOR_REFCOUNTED;

  new_tensor->tensor_id = THDState::s_nextId++;
  return new_tensor;
}

#endif
