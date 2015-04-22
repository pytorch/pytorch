#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.c"
#else

real* THStorage_(data)(const THStorage *self)
{
  return self->data;
}

long THStorage_(size)(const THStorage *self)
{
  return self->size;
}

int THStorage_(elementSize)()
{
  return sizeof(real);
}

THStorage* THStorage_(new)(void)
{
  return THStorage_(newWithSize)(0);
}

THStorage* THStorage_(newWithSize)(long size)
{
  return THStorage_(newWithAllocator)(size, &THDefaultAllocator, NULL);
}

THStorage* THStorage_(newWithAllocator)(long size,
                                        THAllocator *allocator,
                                        void *allocatorContext)
{
  THStorage *storage = THAlloc(sizeof(THStorage));
  storage->data = allocator->malloc(allocatorContext, sizeof(real)*size);
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  return storage;
}

THStorage* THStorage_(newWithMapping)(const char *filename, long size, int shared)
{
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, shared);

  THStorage *storage = THStorage_(newWithAllocator)(size,
                                                    &THMapAllocator,
                                                    ctx);

  if(size <= 0)
    storage->size = THMapAllocatorContext_size(ctx)/sizeof(real);

  THStorage_(clearFlag)(storage, TH_STORAGE_RESIZABLE);

  return storage;
}

THStorage* THStorage_(newWithSize1)(real data0)
{
  THStorage *self = THStorage_(newWithSize)(1);
  self->data[0] = data0;
  return self;
}

THStorage* THStorage_(newWithSize2)(real data0, real data1)
{
  THStorage *self = THStorage_(newWithSize)(2);
  self->data[0] = data0;
  self->data[1] = data1;
  return self;
}

THStorage* THStorage_(newWithSize3)(real data0, real data1, real data2)
{
  THStorage *self = THStorage_(newWithSize)(3);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  return self;
}

THStorage* THStorage_(newWithSize4)(real data0, real data1, real data2, real data3)
{
  THStorage *self = THStorage_(newWithSize)(4);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  self->data[3] = data3;
  return self;
}

void THStorage_(setFlag)(THStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THStorage_(clearFlag)(THStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THStorage_(retain)(THStorage *storage)
{
  if(storage && (storage->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&storage->refcount);
}

void THStorage_(free)(THStorage *storage)
{
  if(!storage)
    return;

  if((storage->flag & TH_STORAGE_REFCOUNTED) && (THAtomicGet(&storage->refcount) > 0))
  {
    if(THAtomicDecrementRef(&storage->refcount))
    {
      if(storage->flag & TH_STORAGE_FREEMEM)
        storage->allocator->free(storage->allocatorContext, storage->data);
      THFree(storage);
    }
  }
}

THStorage* THStorage_(newWithData)(real *data, long size)
{
  return THStorage_(newWithDataAndAllocator)(data, size,
                                             &THDefaultAllocator, NULL);
}

THStorage* THStorage_(newWithDataAndAllocator)(real* data, long size,
                                               THAllocator* allocator,
                                               void* allocatorContext) {
  THStorage *storage = THAlloc(sizeof(THStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  return storage;
}

void THStorage_(resize)(THStorage *storage, long size)
{
  if(storage->flag & TH_STORAGE_RESIZABLE)
  {
    storage->data = storage->allocator->realloc(
        storage->allocatorContext,
        storage->data,
        sizeof(real)*size);
    storage->size = size;
  }
}

void THStorage_(fill)(THStorage *storage, real value)
{
  long i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = value;
}

void THStorage_(set)(THStorage *self, long idx, real value)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  self->data[idx] = value;
}

real THStorage_(get)(const THStorage *self, long idx)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  return self->data[idx];
}

#endif
