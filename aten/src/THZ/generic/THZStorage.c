#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorage.c"
#else

ntype* THZStorage_(data)(const THZStorage *self)
{
  return self->data;
}

ptrdiff_t THZStorage_(size)(const THZStorage *self)
{
  return self->size;
}

size_t THZStorage_(elementSize)()
{
  return sizeof(ntype);
}

THZStorage* THZStorage_(new)(void)
{
  return THZStorage_(newWithSize)(0);
}

THZStorage* THZStorage_(newWithSize)(ptrdiff_t size)
{
  return THZStorage_(newWithAllocator)(size, &THDefaultAllocator, NULL);
}

THZStorage* THZStorage_(newWithAllocator)(ptrdiff_t size,
                                        THAllocator *allocator,
                                        void *allocatorContext)
{
  THZStorage *storage = THAlloc(sizeof(THZStorage));
  storage->data = allocator->malloc(allocatorContext, sizeof(ntype)*size);
  storage->size = size;
  storage->refcount = 1;
  storage->flag = THZ_STORAGE_REFCOUNTED | THZ_STORAGE_RESIZABLE | THZ_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  return storage;
}

THZStorage* THZStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, flags);

  THZStorage *storage = THZStorage_(newWithAllocator)(size,
                                                    &THMapAllocator,
                                                    ctx);

  if(size <= 0)
    storage->size = THMapAllocatorContext_size(ctx)/sizeof(ntype);

  THZStorage_(clearFlag)(storage, THZ_STORAGE_RESIZABLE);

  return storage;
}

THZStorage* THZStorage_(newWithSize1)(ntype data0)
{
  THZStorage *self = THZStorage_(newWithSize)(1);
  self->data[0] = data0;
  return self;
}

THZStorage* THZStorage_(newWithSize2)(ntype data0, ntype data1)
{
  THZStorage *self = THZStorage_(newWithSize)(2);
  self->data[0] = data0;
  self->data[1] = data1;
  return self;
}

THZStorage* THZStorage_(newWithSize3)(ntype data0, ntype data1, ntype data2)
{
  THZStorage *self = THZStorage_(newWithSize)(3);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  return self;
}

THZStorage* THZStorage_(newWithSize4)(ntype data0, ntype data1, ntype data2, ntype data3)
{
  THZStorage *self = THZStorage_(newWithSize)(4);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  self->data[3] = data3;
  return self;
}

void THZStorage_(setFlag)(THZStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THZStorage_(clearFlag)(THZStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THZStorage_(retain)(THZStorage *storage)
{
  if(storage && (storage->flag & THZ_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&storage->refcount);
}

void THZStorage_(free)(THZStorage *storage)
{
  if(!storage)
    return;

  if((storage->flag & THZ_STORAGE_REFCOUNTED) && (THAtomicGet(&storage->refcount) > 0))
  {
    if(THAtomicDecrementRef(&storage->refcount))
    {
      if(storage->flag & THZ_STORAGE_FREEMEM) {
        storage->allocator->free(storage->allocatorContext, storage->data);
      }
      if(storage->flag & THZ_STORAGE_VIEW) {
        THZStorage_(free)(storage->view);
      }
      THFree(storage);
    }
  }
}

THZStorage* THZStorage_(newWithData)(ntype *data, ptrdiff_t size)
{
  return THZStorage_(newWithDataAndAllocator)(data, size,
                                             &THDefaultAllocator, NULL);
}

THZStorage* THZStorage_(newWithDataAndAllocator)(ntype* data, ptrdiff_t size,
                                               THAllocator* allocator,
                                               void* allocatorContext) {
  THZStorage *storage = THAlloc(sizeof(THZStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = THZ_STORAGE_REFCOUNTED | THZ_STORAGE_RESIZABLE | THZ_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  return storage;
}

void THZStorage_(resize)(THZStorage *storage, ptrdiff_t size)
{
  if(storage->flag & THZ_STORAGE_RESIZABLE)
  {
    if(storage->allocator->realloc == NULL) {
      /* case when the allocator does not have a realloc defined */
      ntype *old_data = storage->data;
      ptrdiff_t old_size = storage->size;
      if (size == 0) {
  storage->data = NULL;
      } else {
  storage->data = storage->allocator->malloc(
               storage->allocatorContext,
               sizeof(ntype)*size);
      }
      storage->size = size;
      if (old_data != NULL) {
  ptrdiff_t copy_size = old_size;
  if (storage->size < copy_size) {
    copy_size = storage->size;
  }
  if (copy_size > 0) {
    memcpy(storage->data, old_data, sizeof(ntype)*copy_size);
  }
  storage->allocator->free(storage->allocatorContext, old_data);
      }
    } else {
      storage->data = storage->allocator->realloc(
              storage->allocatorContext,
              storage->data,
              sizeof(ntype)*size);
      storage->size = size;
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}

void THZStorage_(fill)(THZStorage *storage, ntype value)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = value;
}

void THZStorage_(set)(THZStorage *self, ptrdiff_t idx, ntype value)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  self->data[idx] = value;
}

ntype THZStorage_(get)(const THZStorage *self, ptrdiff_t idx)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  return self->data[idx];
}

void THZStorage_(swap)(THZStorage *storage1, THZStorage *storage2)
{
#define SWAP(val) { val = storage1->val; storage1->val = storage2->val; storage2->val = val; }
    ntype *data;
    ptrdiff_t size;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THZStorage *view;

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
