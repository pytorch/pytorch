#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.c"
#else

real* THCStorage_(data)(THCState *state, const THCStorage *self)
{
  return self->data;
}

long THCStorage_(size)(THCState *state, const THCStorage *self)
{
  return self->size;
}

int THCStorage_(elementSize)(THCState *state)
{
  return sizeof(real);
}

void THCStorage_(set)(THCState *state, THCStorage *self, long index, hostreal _value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  real value = hostrealToReal(_value);
  THCudaCheck(cudaMemcpy(self->data + index, &value, sizeof(real), cudaMemcpyHostToDevice));
}

hostreal THCStorage_(get)(THCState *state, const THCStorage *self, long index)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
#ifndef THC_REAL_IS_HALF
  real value;
  THCudaCheck(cudaMemcpy(&value, self->data + index, sizeof(real), cudaMemcpyDeviceToHost));
  return realToHostreal(value);
#else
  float *ret_d;
  float ret;
  THCudaCheck(THCudaMalloc(state, (void**)&ret_d, sizeof(float)));
  THCHalf2Float(state, ret_d, self->data + index, 1);
  THCudaCheck(cudaMemcpy(&ret, ret_d, sizeof(float), cudaMemcpyDeviceToHost));
  THCudaFree(state, ret_d);
  return ret;
#endif
}

THCStorage* THCStorage_(new)(THCState *state)
{
  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  storage->data = NULL;
  storage->size = 0;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THCStorage* THCStorage_(newWithSize)(THCState *state, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
  {
    THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));

    // update heap *before* attempting malloc, to free space for the malloc
    THCHeapUpdate(state, size * sizeof(real));
    cudaError_t err =
      THCudaMalloc(state, (void**)&(storage->data), size * sizeof(real));
    if(err != cudaSuccess){
      THCHeapUpdate(state, -size * sizeof(real));
    }
    THCudaCheck(err);

    storage->size = size;
    storage->refcount = 1;
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    return storage;
  }
  else
  {
    return THCStorage_(new)(state);
  }
}

THCStorage* THCStorage_(newWithSize1)(THCState *state, hostreal data0)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 1);
  THCStorage_(set)(state, self, 0, data0);
  return self;
}

THCStorage* THCStorage_(newWithSize2)(THCState *state, hostreal data0, hostreal data1)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 2);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  return self;
}

THCStorage* THCStorage_(newWithSize3)(THCState *state, hostreal data0, hostreal data1, hostreal data2)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 3);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  return self;
}

THCStorage* THCStorage_(newWithSize4)(THCState *state, hostreal data0, hostreal data1, hostreal data2, hostreal data3)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 4);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  THCStorage_(set)(state, self, 3, data3);
  return self;
}

THCStorage* THCStorage_(newWithMapping)(THCState *state, const char *fileName, long size, int isShared)
{
  THError("not available yet for THCStorage");
  return NULL;
}

THCStorage* THCStorage_(newWithData)(THCState *state, real *data, long size)
{
  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

void THCStorage_(setFlag)(THCState *state, THCStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THCStorage_(clearFlag)(THCState *state, THCStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THCStorage_(retain)(THCState *state, THCStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&self->refcount);
}

void THCStorage_(free)(THCState *state, THCStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (THAtomicDecrementRef(&self->refcount))
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCHeapUpdate(state, -self->size * sizeof(real));
      THCudaCheck(THCudaFree(state, self->data));
    }
    THFree(self);
  }
}
#endif
