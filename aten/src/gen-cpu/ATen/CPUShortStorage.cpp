#include "ATen/CPUShortStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUShortStorage::CPUShortStorage(Context* context):
    storage(THShortStorage_new()), context(context) {}

CPUShortStorage::CPUShortStorage(Context* context, THShortStorage* storage):
    storage(storage), context(context) {}

CPUShortStorage::CPUShortStorage(Context* context, std::size_t storage_size)
  : storage(THShortStorage_newWithSize( storage_size)), context(context) {}

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

CPUShortStorage::CPUShortStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THShortStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THShortStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUShortStorage::CPUShortStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THShortStorage_newWithDataAndAllocator(
     static_cast<int16_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THShortStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUShortStorage::~CPUShortStorage() {
  THShortStorage_free( storage);
}

std::size_t CPUShortStorage::elementSize() const {
  return sizeof(int16_t);
}

std::size_t CPUShortStorage::size() const {
  return storage->size;
}

void* CPUShortStorage::data() {
  return storage->data;
}

const void* CPUShortStorage::data() const {
  return storage->data;
}

auto CPUShortStorage::retain() -> CPUShortStorage& {
  THShortStorage_retain( storage);
  return *this;
}

auto CPUShortStorage::free() -> CPUShortStorage& {
  THShortStorage_free( storage);
  return *this;
}

void* CPUShortStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THShortStorage_retain( storage);
  }
  return storage;
}

auto CPUShortStorage::resize(int64_t new_size) -> CPUShortStorage& {
  THShortStorage_resize( storage, new_size);
  return *this;
}

auto CPUShortStorage::fill(Scalar value) -> CPUShortStorage& {
  THShortStorage_fill( storage, (value.toShort()));
  return *this;
}

auto CPUShortStorage::set(std::size_t ind, Scalar value) -> CPUShortStorage& {
  THShortStorage_set( storage, ind, (value.toShort()));
  return *this;
}

auto CPUShortStorage::fast_set(std::size_t ind, Scalar value) -> CPUShortStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPUShortStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int16_t>((THShortStorage_get( storage, ind)));
}

auto CPUShortStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int16_t>((storage->data[ind]));
}

void CPUShortStorage::set_flag(char flag) {
  THShortStorage_setFlag( storage, flag);
}

void CPUShortStorage::clear_flag(char flag) {
  THShortStorage_clearFlag( storage, flag);
}

int CPUShortStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPUShortStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Short);
}

const char * CPUShortStorage::toString() const {
  return "CPUShortStorage";
}

const char * CPUShortStorage::typeString() {
  return "CPUShortType";
}

}
