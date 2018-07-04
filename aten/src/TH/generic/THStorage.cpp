#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.cpp"
#else

#include <new>

real* THStorage_(data)(const THStorage *self)
{
  return self->data<real>();
}

ptrdiff_t THStorage_(size)(const THStorage *self)
{
  return THStorage_size(self);
}

size_t THStorage_(elementSize)()
{
  return sizeof(real);
}

THStorage* THStorage_(new)(void)
{
  return THStorage_new(at::CTypeToScalarType<th::from_type<real>>::to());
}

THStorage* THStorage_(newWithSize)(ptrdiff_t size)
{
  return THStorage_newWithSize(at::CTypeToScalarType<th::from_type<real>>::to(), size);
}

THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
                                        THAllocator *allocator,
                                        void *allocatorContext)
{
  return THStorage_newWithAllocator(at::CTypeToScalarType<th::from_type<real>>::to(), size, allocator, allocatorContext);
}


THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  return THStorage_newWithMapping(at::CTypeToScalarType<th::from_type<real>>::to(), filename, size, flags);
}

THStorage* THStorage_(newWithSize1)(real data0)
{
  THStorage *self = THStorage_(newWithSize)(1);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  return self;
}

THStorage* THStorage_(newWithSize2)(real data0, real data1)
{
  THStorage *self = THStorage_(newWithSize)(2);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  return self;
}

THStorage* THStorage_(newWithSize3)(real data0, real data1, real data2)
{
  THStorage *self = THStorage_(newWithSize)(3);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  data[2] = data2;
  return self;
}

THStorage* THStorage_(newWithSize4)(real data0, real data1, real data2, real data3)
{
  THStorage *self = THStorage_(newWithSize)(4);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  data[2] = data2;
  data[3] = data3;
  return self;
}

void THStorage_(setFlag)(THStorage *storage, const char flag)
{
  THStorage_setFlag(storage, flag);
}

void THStorage_(clearFlag)(THStorage *storage, const char flag)
{
  THStorage_clearFlag(storage, flag);
}

void THStorage_(retain)(THStorage *storage)
{
  THStorage_retain(storage);
}

int THStorage_(retainIfLive)(THStorage *storage)
{
  return THStorage_retainIfLive(storage);
}

void THStorage_(free)(THStorage *storage)
{
  THStorage_free(storage);
}

THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size)
{
  return THStorage_newWithData(at::CTypeToScalarType<th::from_type<real>>::to(), data, size);
}

THStorage* THStorage_(newWithDataAndAllocator)(real* data, ptrdiff_t size,
                                               THAllocator* allocator,
                                               void* allocatorContext) {
  return THStorage_newWithDataAndAllocator(at::CTypeToScalarType<th::from_type<real>>::to(), data, size, allocator, allocatorContext);
}

void THStorage_(resize)(THStorage *storage, ptrdiff_t size)
{
  return THStorage_resize(storage, size);
}

void THStorage_(fill)(THStorage *storage, real value)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size; i++)
    THStorage_(data)(storage)[i] = value;
}

void THStorage_(set)(THStorage *self, ptrdiff_t idx, real value)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  THStorage_(data)(self)[idx] = value;
}

real THStorage_(get)(const THStorage *self, ptrdiff_t idx)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  return THStorage_(data)(self)[idx];
}

void THStorage_(swap)(THStorage *storage1, THStorage *storage2)
{
  THStorage_swap(storage1, storage2);
}

#endif
