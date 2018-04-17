#include "ATen/CPULongStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPULongStorage::CPULongStorage(Context* context):
    storage(THLongStorage_new()), context(context) {}

CPULongStorage::CPULongStorage(Context* context, THLongStorage* storage):
    storage(storage), context(context) {}

CPULongStorage::CPULongStorage(Context* context, std::size_t storage_size)
  : storage(THLongStorage_newWithSize( storage_size)), context(context) {}

#if false
static cudaError_t call_deleter(void * ctx, void * data) {
  auto fnptr = (std::function<void(void*)>*) ctx;
  (*fnptr)(data);
  delete fnptr;
  return cudaSuccess;
}
static THCDeviceAllocator storage_deleter = {
  nullptr,
  nullptr,
  call_deleter,
  nullptr,
  nullptr,
};
static cudaError_t wrapped_alloc(void * ctx, void** result, size_t size, cudaStream_t stream) {
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->retain();
  *result = ac->allocate(size);
  return cudaSuccess;
}
static cudaError_t wrapped_free(void * ctx, void * data) {
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->deallocate(data);
  ac->release();
  return cudaSuccess;
}
static THCDeviceAllocator wrapped_allocator = {
  wrapped_alloc,
  nullptr,
  wrapped_free,
  nullptr,
  nullptr,
};
#else
static void call_deleter(void * ctx, void * data) {
  auto fnptr = (std::function<void(void*)>*) ctx;
  (*fnptr)(data);
  delete fnptr;
}
static THAllocator storage_deleter = {
  nullptr,
  nullptr,
  call_deleter,
};
static void* wrapped_alloc(void * ctx, ptrdiff_t size) {
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->retain();
  return ac->allocate(size);
}
static void wrapped_free(void * ctx, void * data) {
  auto ac = static_cast<detail::AllocatorRetainable*>(ctx);
  ac->deallocate(data);
  ac->release();
}
static THAllocator wrapped_allocator = {
  wrapped_alloc,
  nullptr,
  wrapped_free,
};
#endif

CPULongStorage::CPULongStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THLongStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THLongStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPULongStorage::CPULongStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THLongStorage_newWithDataAndAllocator(
     static_cast<int64_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THLongStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPULongStorage::~CPULongStorage() {
  THLongStorage_free( storage);
}

std::size_t CPULongStorage::elementSize() const {
  return sizeof(int64_t);
}

std::size_t CPULongStorage::size() const {
  return storage->size;
}

void* CPULongStorage::data() {
  return storage->data;
}

const void* CPULongStorage::data() const {
  return storage->data;
}

auto CPULongStorage::retain() -> CPULongStorage& {
  THLongStorage_retain( storage);
  return *this;
}

auto CPULongStorage::free() -> CPULongStorage& {
  THLongStorage_free( storage);
  return *this;
}

void* CPULongStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THLongStorage_retain( storage);
  }
  return storage;
}

auto CPULongStorage::resize(int64_t new_size) -> CPULongStorage& {
  THLongStorage_resize( storage, new_size);
  return *this;
}

auto CPULongStorage::fill(Scalar value) -> CPULongStorage& {
  THLongStorage_fill( storage, long(value.toLong()));
  return *this;
}

auto CPULongStorage::set(std::size_t ind, Scalar value) -> CPULongStorage& {
  THLongStorage_set( storage, ind, long(value.toLong()));
  return *this;
}

auto CPULongStorage::fast_set(std::size_t ind, Scalar value) -> CPULongStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPULongStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int64_t>(int64_t(THLongStorage_get( storage, ind)));
}

auto CPULongStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int64_t>(int64_t(storage->data[ind]));
}

void CPULongStorage::set_flag(char flag) {
  THLongStorage_setFlag( storage, flag);
}

void CPULongStorage::clear_flag(char flag) {
  THLongStorage_clearFlag( storage, flag);
}

int CPULongStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPULongStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Long);
}

const char * CPULongStorage::toString() const {
  return "CPULongStorage";
}

const char * CPULongStorage::typeString() {
  return "CPULongType";
}

}
