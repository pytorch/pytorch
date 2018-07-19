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
  THStorage* storage = new THStorage(
      at::CTypeToScalarType<th::from_type<real>>::to(),
      0,
      getTHDefaultAllocator(),
      TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE);
  return storage;
}

THStorage* THStorage_(newWithSize)(ptrdiff_t size)
{
  THStorage* storage = new THStorage(
      at::CTypeToScalarType<th::from_type<real>>::to(),
      size,
      getTHDefaultAllocator(),
      TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE);
  return storage;
}

THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
                                        at::Allocator *allocator)
{
  THStorage* storage = new THStorage(
      at::CTypeToScalarType<th::from_type<real>>::to(),
      size,
      allocator,
      TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE);
  return storage;
}


THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  auto scalar_type = at::CTypeToScalarType<th::from_type<real>>::to();
  size_t actual_size = -1;
  THStorage* storage = new THStorage(
      scalar_type,
      size,
      THMapAllocator::makeDataPtr(
          filename, flags, size * at::elementSize(scalar_type), &actual_size),
      /* allocator */ nullptr,
      TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE);

  if (size <= 0) {
    storage->size = actual_size / at::elementSize(scalar_type);
  }

  THStorage_clearFlag(storage, TH_STORAGE_RESIZABLE);

  return storage;
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

void THStorage_(free)(THStorage *storage)
{
  THStorage_free(storage);
}

THStorage* THStorage_(newWithDataAndAllocator)(at::DataPtr&& data, ptrdiff_t size,
                                               at::Allocator* allocator) {
  THStorage* storage = new THStorage(
      at::CTypeToScalarType<th::from_type<real>>::to(),
      size,
      std::move(data),
      allocator,
      TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE);
  return storage;
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
