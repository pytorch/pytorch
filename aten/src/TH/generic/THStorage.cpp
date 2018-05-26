#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.cpp"
#else

#include <new>

real* THStorage_(data)(const at::StorageImpl *self)
{
  return (real*)self->data_ptr();
}

ptrdiff_t THStorage_(size)(const at::StorageImpl *self)
{
  return self->size();
}

size_t THStorage_(elementSize)()
{
  return sizeof(real);
}

at::StorageImpl * THStorage_(new)(void)
{
  return THStorage_(newWithSize)(0);
}

at::StorageImpl * THStorage_(newWithSize)(ptrdiff_t size)
{
  return THStorage_(newWithAllocator)(size, &THDefaultAllocator, NULL);
}

at::StorageImpl * THStorage_(newWithAllocator)(ptrdiff_t size,
                                        THAllocator *allocator,
                                        void *allocatorContext)
{
  real *data = static_cast<real*>(allocator->malloc(allocatorContext, sizeof(real)*size));
  char flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return new at::StorageImpl(data, size, flag, allocator, allocatorContext);
}

at::StorageImpl * THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  THAllocator *allocator = &THMapAllocator;
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, flags);
  real *data = static_cast<real*>(allocator->malloc(ctx, sizeof(real)*size));
  if(size <= 0)
     size = THMapAllocatorContext_size(ctx)/sizeof(real);
  char flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_FREEMEM;
  return new at::StorageImpl(data, size, flag, &THMapAllocator, ctx);
}

at::StorageImpl * THStorage_(newWithSize1)(real data0)
{
  at::StorageImpl *self = THStorage_(newWithSize)(1);
  self->data<real>()[0] = data0;
  return self;
}

at::StorageImpl * THStorage_(newWithSize2)(real data0, real data1)
{
  at::StorageImpl *self = THStorage_(newWithSize)(2);
  self->data<real>()[0] = data0;
  self->data<real>()[1] = data1;
  return self;
}

at::StorageImpl * THStorage_(newWithSize3)(real data0, real data1, real data2)
{
  at::StorageImpl *self = THStorage_(newWithSize)(3);
  self->data<real>()[0] = data0;
  self->data<real>()[1] = data1;
  self->data<real>()[2] = data2;
  return self;
}

at::StorageImpl * THStorage_(newWithSize4)(real data0, real data1, real data2, real data3)
{
  at::StorageImpl *self = THStorage_(newWithSize)(4);
  self->data<real>()[0] = data0;
  self->data<real>()[1] = data1;
  self->data<real>()[2] = data2;
  self->data<real>()[3] = data3;
  return self;
}

void THStorage_(setFlag)(at::StorageImpl *storage, const char flag)
{
  storage->flag() |= flag;
}

void THStorage_(clearFlag)(at::StorageImpl *storage, const char flag)
{
  storage->flag() &= ~flag;
}

void THStorage_(retain)(at::StorageImpl *storage)
{
  if(storage && (storage->flag() & TH_STORAGE_REFCOUNTED))
    ++storage->refcount;
}

int THStorage_(retainIfLive)(at::StorageImpl *storage)
{
  // TODO: Check if TH_STORAGE_REFCOUNTED?
  int refcount = storage->refcount.load();
  while (refcount > 0) {
    if (storage->refcount.compare_exchange_strong(refcount, refcount + 1)) {
      return 1;
    }
    refcount = storage->refcount.load();
  }
  return 0;
}

void THStorage_(free)(at::StorageImpl *storage)
{
  if(!storage)
    return;

  if((storage->flag() & TH_STORAGE_REFCOUNTED) && (storage->refcount.load() > 0))
  {
    if(--storage->refcount == 0)
    {
      if(storage->flag() & TH_STORAGE_FREEMEM) {
        storage->allocator()->free(storage->allocatorContext(), storage->data_ptr());
      }
      if(storage->flag() & TH_STORAGE_VIEW) {
        THStorage_(free)(storage->view());
      }
      storage->refcount.~atomic<int>();
      THFree(storage);
    }
  }
}

at::StorageImpl * THStorage_(newWithData)(real *data, ptrdiff_t size)
{
  return THStorage_(newWithDataAndAllocator)(data, size,
                                             &THDefaultAllocator, NULL);
}

at::StorageImpl * THStorage_(newWithDataAndAllocator)(real* data, ptrdiff_t size,
                                               THAllocator* allocator,
                                               void* allocatorContext) {
  return new at::StorageImpl(data, size, TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM,
                             allocator, allocatorContext);
}

void THStorage_(resize)(at::StorageImpl *storage, ptrdiff_t size)
{
  storage->resize(size, sizeof(real));
  /*if(storage->flag & TH_STORAGE_RESIZABLE)
  {
    if(storage->allocator()->realloc == NULL) {
      // case when the allocator does not have a realloc defined
      real *old_data = storage->data<real>();
      ptrdiff_t old_size = storage->size();
      if (size == 0) {
        storage->data = NULL;
      } else {
        storage->data = static_cast<real*>(storage->allocator()->malloc(
            storage->allocatorContext(),
            sizeof(real)*size));
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
        storage->allocator()->free(storage->allocatorContext(), old_data);
      }
    } else {
      storage->data = static_cast<real*>(storage->allocator()->realloc(
              storage->allocatorContext,
              storage->data,
              sizeof(real)*size));
      storage->size = size;
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }*/
}

void THStorage_(fill)(at::StorageImpl *storage, real value)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size(); i++)
    THStorage_(data)(storage)[i] = value;
}

void THStorage_(set)(at::StorageImpl *self, ptrdiff_t idx, real value)
{
  THArgCheck((idx >= 0) && (idx < self->size()), 2, "out of bounds");
  THStorage_(data)(self)[idx] = value;
}

real THStorage_(get)(const at::StorageImpl *self, ptrdiff_t idx)
{
  THArgCheck((idx >= 0) && (idx < self->size()), 2, "out of bounds");
  return THStorage_(data)(self)[idx];
}

void THStorage_(swap)(at::StorageImpl *storage1, at::StorageImpl *storage2)
{
  storage1->swap(storage2);
}

#endif
