#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.cpp"
#else

real* THCStorage_(data)(THCState *state, const THCStorage *self)
{
  return self->data<real>();
}

ptrdiff_t THCStorage_(size)(THCState *state, const THCStorage *self)
{
  return THStorage_size(self);
}

int THCStorage_(elementSize)(THCState *state)
{
  return sizeof(real);
}

void THCStorage_(set)(THCState *state, THCStorage *self, ptrdiff_t index, real value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(THCStorage_(data)(state, self) + index, &value, sizeof(real),
                              cudaMemcpyHostToDevice,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
}

real THCStorage_(get)(THCState *state, const THCStorage *self, ptrdiff_t index)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  real value;
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(&value, THCStorage_(data)(state, self) + index, sizeof(real),
                              cudaMemcpyDeviceToHost, stream));
  THCudaCheck(cudaStreamSynchronize(stream));
  return value;
}

THCStorage* THCStorage_(new)(THCState *state)
{
  return THCStorage_new(state, at::CTypeToScalarType<real>::to());
}

THCStorage* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size)
{
  return THCStorage_newWithSize(state, at::CTypeToScalarType<real>::to(), size);
}

THCStorage* THCStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                          THCDeviceAllocator* allocator,
                                          void* allocatorContext)
{
  return THCStorage_newWithAllocator(state, at::CTypeToScalarType<real>::to(),
                                     size, allocator, allocatorContext);
}

THCStorage* THCStorage_(newWithSize1)(THCState *state, real data0)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 1);
  THCStorage_(set)(state, self, 0, data0);
  return self;
}

THCStorage* THCStorage_(newWithSize2)(THCState *state, real data0, real data1)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 2);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  return self;
}

THCStorage* THCStorage_(newWithSize3)(THCState *state, real data0, real data1, real data2)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 3);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  return self;
}

THCStorage* THCStorage_(newWithSize4)(THCState *state, real data0, real data1, real data2, real data3)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 4);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  THCStorage_(set)(state, self, 3, data3);
  return self;
}

THCStorage* THCStorage_(newWithMapping)(THCState *state, const char *fileName, ptrdiff_t size, int isShared)
{
  THError("not available yet for THCStorage");
  return NULL;
}

THCStorage* THCStorage_(newWithData)(THCState *state, real *data, ptrdiff_t size)
{
  return THCStorage_newWithData(state, at::CTypeToScalarType<real>::to(), data, size);
}

THCStorage* THCStorage_(newWithDataAndAllocator)(
  THCState *state, real *data, ptrdiff_t size,
  THCDeviceAllocator *allocator, void *allocatorContext) {
  return THCStorage_newWithDataAndAllocator(state, at::CTypeToScalarType<real>::to(), data, size, allocator, allocatorContext);
}

void THCStorage_(setFlag)(THCState *state, THCStorage *storage, const char flag)
{
  THStorage_setFlag(storage, flag);
}

void THCStorage_(clearFlag)(THCState *state, THCStorage *storage, const char flag)
{
  THStorage_clearFlag(storage, flag);
}

void THCStorage_(retain)(THCState *state, THCStorage *self)
{
  THStorage_retain(self);
}

int THCStorage_(retainIfLive)(THCState *state, THCStorage *storage)
{
  return THStorage_retainIfLive(storage);
}

void THCStorage_(free)(THCState *state, THCStorage *self)
{
  THCStorage_free(state, self);
}
#endif
