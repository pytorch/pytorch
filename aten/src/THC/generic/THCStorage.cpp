#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.cpp"
#else

real* THCStorage_(data)(THCState *state, const at::CUDAStorageImpl *self)
{
  return self->data<real>();
}

ptrdiff_t THCStorage_(size)(THCState *state, const at::CUDAStorageImpl *self)
{
  return self->size();
}

int THCStorage_(elementSize)(THCState *state)
{
  return sizeof(real);
}

void THCStorage_(set)(THCState *state, at::CUDAStorageImpl *self, ptrdiff_t index, real value)
{
  THArgCheck((index >= 0) && (index < self->size()), 2, "index out of bounds");
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(self->data<real>() + index, &value, sizeof(real),
                              cudaMemcpyHostToDevice,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
}

real THCStorage_(get)(THCState *state, const at::CUDAStorageImpl *self, ptrdiff_t index)
{
  THArgCheck((index >= 0) && (index < self->size()), 2, "index out of bounds");
  real value;
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(&value, self->data<real>() + index, sizeof(real),
                              cudaMemcpyDeviceToHost, stream));
  THCudaCheck(cudaStreamSynchronize(stream));
  return value;
}

at::CUDAStorageImpl* THCStorage_(new)(THCState *state)
{
  return THCStorage_(newWithSize)(state, 0);
}

at::CUDAStorageImpl* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size)
{
  return THCStorage_(newWithAllocator)(
    state, size,
    state->cudaDeviceAllocator,
    state->cudaDeviceAllocator->state);
}

at::CUDAStorageImpl* THCStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                          THCDeviceAllocator* allocator,
                                          void* allocatorContext)
{
  THArgCheck(size >= 0, 2, "invalid size");
  int device;
  THCudaCheck(cudaGetDevice(&device));

  at::CUDAStorageImpl *storage = (at::CUDAStorageImpl*)THAlloc(sizeof(THCStorage));
  memset(storage, 0, sizeof(THCStorage));
  new (&storage->refcount) std::atomic<int>(1);
  char flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  void * data;

  if(size > 0)
  {
    // update heap *before* attempting malloc, to free space for the malloc
    cudaError_t err =
      (*allocator->malloc)(allocatorContext,
                           (void**)&(data),
                           size * sizeof(real),
                           THCState_getCurrentStream(state));
    if(err != cudaSuccess){
      free(storage);
    }
    THCudaCheck(err);
  } else {
    data = NULL;
  }
  return new at::CUDAStorageImpl(data, size, flag, allocator, allocatorContext, device);
}

at::CUDAStorageImpl* THCStorage_(newWithSize1)(THCState *state, real data0)
{
  at::CUDAStorageImpl *self = THCStorage_(newWithSize)(state, 1);
  THCStorage_(set)(state, self, 0, data0);
  return self;
}

at::CUDAStorageImpl* THCStorage_(newWithSize2)(THCState *state, real data0, real data1)
{
  at::CUDAStorageImpl *self = THCStorage_(newWithSize)(state, 2);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  return self;
}

at::CUDAStorageImpl* THCStorage_(newWithSize3)(THCState *state, real data0, real data1, real data2)
{
  at::CUDAStorageImpl *self = THCStorage_(newWithSize)(state, 3);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  return self;
}

at::CUDAStorageImpl* THCStorage_(newWithSize4)(THCState *state, real data0, real data1, real data2, real data3)
{
  at::CUDAStorageImpl *self = THCStorage_(newWithSize)(state, 4);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  THCStorage_(set)(state, self, 3, data3);
  return self;
}

at::CUDAStorageImpl* THCStorage_(newWithMapping)(THCState *state, const char *fileName, ptrdiff_t size, int isShared)
{
  THError("not available yet for THCStorage");
  return NULL;
}

at::CUDAStorageImpl* THCStorage_(newWithData)(THCState *state, real *data, ptrdiff_t size)
{
  return THCStorage_(newWithDataAndAllocator)(state, data, size,
                                              state->cudaDeviceAllocator,
                                              state->cudaDeviceAllocator->state);
}

at::CUDAStorageImpl* THCStorage_(newWithDataAndAllocator)(
  THCState *state, real *data, ptrdiff_t size,
  THCDeviceAllocator *allocator, void *allocatorContext) {
  char flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  int device;
  if (data) {
    struct cudaPointerAttributes attr;
    THCudaCheck(cudaPointerGetAttributes(&attr, data));
    device = attr.device;
  } else {
    THCudaCheck(cudaGetDevice(&device));
  }
  return new at::CUDAStorageImpl(data, size, flag, allocator, allocatorContext, device);
}

void THCStorage_(setFlag)(THCState *state, at::CUDAStorageImpl *storage, const char flag)
{
  storage->flag() |= flag;
}

void THCStorage_(clearFlag)(THCState *state, at::CUDAStorageImpl *storage, const char flag)
{
  storage->flag() &= ~flag;
}

void THCStorage_(retain)(THCState *state, at::CUDAStorageImpl *self)
{
  if(self && (self->flag() & TH_STORAGE_REFCOUNTED))
    self->refcount++;
}

int THCStorage_(retainIfLive)(THCState *state, at::CUDAStorageImpl *storage)
{
  // TODO: Check if THC_STORAGE_REFCOUNTED?
  int refcount = storage->refcount.load();
  while (refcount > 0) {
    if (storage->refcount.compare_exchange_strong(refcount, refcount + 1)) {
      return 1;
    }
    refcount = storage->refcount.load();
  }
  return 0;
}

void THCStorage_(free)(THCState *state, at::CUDAStorageImpl *self)
{
  if(!(self->flag() & TH_STORAGE_REFCOUNTED))
    return;

  if (--self->refcount == 0)
  {
    if(self->flag() & TH_STORAGE_FREEMEM) {
      THCudaCheck(
        (*self->allocator()->free)(self->allocatorContext(), self->data<real>()));
    }
    if(self->flag() & TH_STORAGE_VIEW) {
      THCStorage_(free)(state, self->view());
    }
    self->refcount.~atomic<int>();
    THFree(self);
  }
}
#endif
