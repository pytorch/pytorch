#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorage.cpp"
#else

#include <new>

scalar_t* THStorage_(data)(const THStorage *self)
{
#if defined(THQUANTIZED)
  return reinterpret_cast<scalar_t*>(self->data<quantized_t>());
#else
  return self->data<scalar_t>();
#endif
}

ptrdiff_t THStorage_(size)(const THStorage *self)
{
  return THStorage_size(self);
}

size_t THStorage_(elementSize)()
{
  return sizeof(scalar_t);
}

THStorage* THStorage_(new)(void)
{
#ifdef THQUANTIZED
  return THStorage_new(caffe2::TypeMeta::Make<quantized_t>());
#else
  return THStorage_new(caffe2::TypeMeta::Make<scalar_t>());
#endif
}

THStorage* THStorage_(newWithSize)(ptrdiff_t size)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
#ifdef THQUANTIZED
      caffe2::TypeMeta::Make<quantized_t>(),
#else
      caffe2::TypeMeta::Make<scalar_t>(),
#endif
      size,
      getTHDefaultAllocator(),
      true).release();
  return storage;
}

THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
                                        at::Allocator *allocator)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
#ifdef THQUANTIZED
      caffe2::TypeMeta::Make<quantized_t>(),
#else
      caffe2::TypeMeta::Make<scalar_t>(),
#endif
      size,
      allocator,
      true).release();
  return storage;
}


THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  auto type_meta = caffe2::TypeMeta::Make<scalar_t>();
  size_t actual_size = -1;
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      type_meta,
      size,
      THMapAllocator::makeDataPtr(
          filename, flags, size * type_meta.itemsize(), &actual_size),
      /* allocator */ nullptr,
      false).release();

  if (size <= 0) {
    storage->set_numel(actual_size / type_meta.itemsize());
  }

  return storage;
}

THStorage* THStorage_(newWithSize1)(scalar_t data0)
{
  THStorage *self = THStorage_(newWithSize)(1);
  scalar_t *data = THStorage_(data)(self);
  data[0] = data0;
  return self;
}

THStorage* THStorage_(newWithSize2)(scalar_t data0, scalar_t data1)
{
  THStorage *self = THStorage_(newWithSize)(2);
  scalar_t *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  return self;
}

THStorage* THStorage_(newWithSize3)(scalar_t data0, scalar_t data1, scalar_t data2)
{
  THStorage *self = THStorage_(newWithSize)(3);
  scalar_t *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  data[2] = data2;
  return self;
}

THStorage* THStorage_(newWithSize4)(scalar_t data0, scalar_t data1, scalar_t data2, scalar_t data3)
{
  THStorage *self = THStorage_(newWithSize)(4);
  scalar_t *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  data[2] = data2;
  data[3] = data3;
  return self;
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
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
#ifdef THQUANTIZED
      caffe2::TypeMeta::Make<quantized_t>(),
#else
      caffe2::TypeMeta::Make<scalar_t>(),
#endif
      size,
      std::move(data),
      allocator,
      allocator != nullptr).release();
  return storage;
}

void THStorage_(resize)(THStorage *storage, ptrdiff_t size)
{
  return THStorage_resize(storage, size);
}

void THStorage_(fill)(THStorage *storage, scalar_t value)
{
  ptrdiff_t i;
  for(i = 0; i < storage->numel(); i++)
    THStorage_(data)(storage)[i] = value;
}

void THStorage_(set)(THStorage *self, ptrdiff_t idx, scalar_t value)
{
  THArgCheck((idx >= 0) && (idx < self->numel()), 2, "out of bounds");
  THStorage_(data)(self)[idx] = value;
}

scalar_t THStorage_(get)(const THStorage *self, ptrdiff_t idx)
{
  THArgCheck((idx >= 0) && (idx < self->numel()), 2, "out of bounds");
  return THStorage_(data)(self)[idx];
}

void THStorage_(swap)(THStorage *storage1, THStorage *storage2)
{
  std::swap(*storage1, *storage2);
}

#endif
