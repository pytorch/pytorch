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

size_t THStorage_(elementSize)()
{
  return sizeof(scalar_t);
}

THStorage* THStorage_(new)(void)
{
  return THStorage_new();
}

THStorage* THStorage_(newWithSize)(ptrdiff_t size)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
#ifdef THQUANTIZED
                           size * sizeof(quantized_t),
#else
                           size * sizeof(scalar_t),
#endif
                           getTHDefaultAllocator(),
                           true)
                           .release();
  return storage;
}

THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
                                        at::Allocator *allocator)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
#ifdef THQUANTIZED
                           size * sizeof(quantized_t),
#else
                           size * sizeof(scalar_t),
#endif
                           allocator,
                           true)
                           .release();
  return storage;
}


THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  size_t actual_size = -1;
  THStorage* storage =
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size * sizeof(scalar_t),
          THMapAllocator::makeDataPtr(
              filename, flags, size * sizeof(scalar_t), &actual_size),
          /* allocator */ nullptr,
          false)
          .release();

  if (size <= 0) {
    storage->set_nbytes(actual_size);
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
                           c10::StorageImpl::use_byte_size_t(),
#ifdef THQUANTIZED
                           size * sizeof(quantized_t),
#else
                           size * sizeof(scalar_t),
#endif
                           std::move(data),
                           allocator,
                           allocator != nullptr)
                           .release();
  return storage;
}

void THStorage_(resizeBytes)(THStorage* storage, ptrdiff_t size_bytes) {
  return THStorage_resizeBytes(storage, size_bytes);
}

void THStorage_(fill)(THStorage *storage, scalar_t value)
{
  auto type_meta = caffe2::TypeMeta::Make<scalar_t>();
  size_t numel = storage->nbytes() / type_meta.itemsize();
  for (size_t i = 0; i < numel; i++)
    THStorage_(data)(storage)[i] = value;
}

void THStorage_(set)(THStorage *self, ptrdiff_t idx, scalar_t value)
{
  auto type_meta = caffe2::TypeMeta::Make<scalar_t>();
  size_t numel = self->nbytes() / type_meta.itemsize();
  THArgCheck((idx >= 0) && (idx < numel), 2, "out of bounds");
  THStorage_(data)(self)[idx] = value;
}

scalar_t THStorage_(get)(const THStorage *self, ptrdiff_t idx)
{
  auto type_meta = caffe2::TypeMeta::Make<scalar_t>();
  size_t numel = self->nbytes() / type_meta.itemsize();
  THArgCheck((idx >= 0) && (idx < numel), 2, "out of bounds");
  return THStorage_(data)(self)[idx];
}

void THStorage_(swap)(THStorage *storage1, THStorage *storage2)
{
  std::swap(*storage1, *storage2);
}

#endif
