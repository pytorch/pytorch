#include "ATen/CPUHalfStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUHalfStorage::CPUHalfStorage(Context* context):
    storage(THHalfStorage_new()), context(context) {}

CPUHalfStorage::CPUHalfStorage(Context* context, THHalfStorage* storage):
    storage(storage), context(context) {}

CPUHalfStorage::CPUHalfStorage(Context* context, std::size_t storage_size)
  : storage(THHalfStorage_newWithSize( storage_size)), context(context) {}

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

CPUHalfStorage::CPUHalfStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THHalfStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THHalfStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUHalfStorage::CPUHalfStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THHalfStorage_newWithDataAndAllocator(
     static_cast<THHalf*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THHalfStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUHalfStorage::~CPUHalfStorage() {
  THHalfStorage_free( storage);
}

std::size_t CPUHalfStorage::elementSize() const {
  return sizeof(Half);
}

std::size_t CPUHalfStorage::size() const {
  return storage->size;
}

void* CPUHalfStorage::data() {
  return storage->data;
}

const void* CPUHalfStorage::data() const {
  return storage->data;
}

auto CPUHalfStorage::retain() -> CPUHalfStorage& {
  THHalfStorage_retain( storage);
  return *this;
}

auto CPUHalfStorage::free() -> CPUHalfStorage& {
  THHalfStorage_free( storage);
  return *this;
}

void* CPUHalfStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THHalfStorage_retain( storage);
  }
  return storage;
}

auto CPUHalfStorage::resize(int64_t new_size) -> CPUHalfStorage& {
  THHalfStorage_resize( storage, new_size);
  return *this;
}

auto CPUHalfStorage::fill(Scalar value) -> CPUHalfStorage& {
  THHalfStorage_fill( storage, HalfFix<THHalf,Half>(value.toHalf()));
  return *this;
}

auto CPUHalfStorage::set(std::size_t ind, Scalar value) -> CPUHalfStorage& {
  THHalfStorage_set( storage, ind, HalfFix<THHalf,Half>(value.toHalf()));
  return *this;
}

auto CPUHalfStorage::fast_set(std::size_t ind, Scalar value) -> CPUHalfStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPUHalfStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<Half>(HalfFix<Half,THHalf>(THHalfStorage_get( storage, ind)));
}

auto CPUHalfStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<Half>(HalfFix<Half,THHalf>(storage->data[ind]));
}

void CPUHalfStorage::set_flag(char flag) {
  THHalfStorage_setFlag( storage, flag);
}

void CPUHalfStorage::clear_flag(char flag) {
  THHalfStorage_clearFlag( storage, flag);
}

int CPUHalfStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPUHalfStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Half);
}

const char * CPUHalfStorage::toString() const {
  return "CPUHalfStorage";
}

const char * CPUHalfStorage::typeString() {
  return "CPUHalfType";
}

}
