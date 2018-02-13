#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.c"
#else

real* THStorage_(data)(const THStorage *self)
{
  return self->data;
}

ptrdiff_t THStorage_(size)(const THStorage *self)
{
  return self->size;
}

size_t THStorage_(elementSize)()
{
  return sizeof(real);
}

THStorage* THStorage_(new)(void)
{
  return THStorage_(newWithSize)(0);
}

THStorage* THStorage_(newWithSize)(ptrdiff_t size)
{
  return THStorage_(newWithAllocator)(size, &THDefaultAllocator, NULL);
}

THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
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

THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, flags);

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
      if(storage->flag & TH_STORAGE_FREEMEM) {
        storage->allocator->free(storage->allocatorContext, storage->data);
      }
      if(storage->flag & TH_STORAGE_VIEW) {
        THStorage_(free)(storage->view);
      }
      THFree(storage);
    }
  }
}

THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size)
{
  return THStorage_(newWithDataAndAllocator)(data, size,
                                             &THDefaultAllocator, NULL);
}

THStorage* THStorage_(newWithDataAndAllocator)(real* data, ptrdiff_t size,
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

void THStorage_(resize)(THStorage *storage, ptrdiff_t size)
{
  if(storage->flag & TH_STORAGE_RESIZABLE)
  {
    if(storage->allocator->realloc == NULL) {
      /* case when the allocator does not have a realloc defined */
      real *old_data = storage->data;
      ptrdiff_t old_size = storage->size;
      if (size == 0) {
        storage->data = NULL;
      } else {
        storage->data = storage->allocator->malloc(
            storage->allocatorContext,
            sizeof(real)*size);
      }
      storage->size = size;
      if (old_data != NULL) {
        ptrdiff_t copy_size = old_size;
        if (storage->size < copy_size) {
          copy_size = storage->size;
        }
        if (copy_size > 0) {
          memcpy(storage->data, old_data, sizeof(real)*copy_size);
        }
        storage->allocator->free(storage->allocatorContext, old_data);
      }
    } else {
      storage->data = storage->allocator->realloc(
              storage->allocatorContext,
              storage->data,
              sizeof(real)*size);
      storage->size = size;
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}

void THStorage_(fill)(THStorage *storage, real value)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = value;
}

void THStorage_(set)(THStorage *self, ptrdiff_t idx, real value)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  self->data[idx] = value;
}

real THStorage_(get)(const THStorage *self, ptrdiff_t idx)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  return self->data[idx];
}

void THStorage_(swap)(THStorage *storage1, THStorage *storage2)
{
#define SWAP(val) { val = storage1->val; storage1->val = storage2->val; storage2->val = val; }
    real *data;
    ptrdiff_t size;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;

    SWAP(data);
    SWAP(size);
    SWAP(flag);
    // don't swap refcount!
    SWAP(allocator);
    SWAP(allocatorContext);
    SWAP(view);
#undef SWAP
}

#endif
