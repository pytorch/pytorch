#include "ATen/CPUIntStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUIntStorage::CPUIntStorage(Context* context):
    storage(THIntStorage_new()), context(context) {}

CPUIntStorage::CPUIntStorage(Context* context, THIntStorage* storage):
    storage(storage), context(context) {}

CPUIntStorage::CPUIntStorage(Context* context, std::size_t storage_size)
  : storage(THIntStorage_newWithSize( storage_size)), context(context) {}

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

CPUIntStorage::CPUIntStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THIntStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THIntStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUIntStorage::CPUIntStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THIntStorage_newWithDataAndAllocator(
     static_cast<int32_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THIntStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUIntStorage::~CPUIntStorage() {
  THIntStorage_free( storage);
}

std::size_t CPUIntStorage::elementSize() const {
  return sizeof(int);
}

std::size_t CPUIntStorage::size() const {
  return storage->size;
}

void* CPUIntStorage::data() {
  return storage->data;
}

const void* CPUIntStorage::data() const {
  return storage->data;
}

auto CPUIntStorage::retain() -> CPUIntStorage& {
  THIntStorage_retain( storage);
  return *this;
}

auto CPUIntStorage::free() -> CPUIntStorage& {
  THIntStorage_free( storage);
  return *this;
}

void* CPUIntStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THIntStorage_retain( storage);
  }
  return storage;
}

auto CPUIntStorage::resize(int64_t new_size) -> CPUIntStorage& {
  THIntStorage_resize( storage, new_size);
  return *this;
}

auto CPUIntStorage::fill(Scalar value) -> CPUIntStorage& {
  THIntStorage_fill( storage, (value.toInt()));
  return *this;
}

auto CPUIntStorage::set(std::size_t ind, Scalar value) -> CPUIntStorage& {
  THIntStorage_set( storage, ind, (value.toInt()));
  return *this;
}

auto CPUIntStorage::fast_set(std::size_t ind, Scalar value) -> CPUIntStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPUIntStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int>((THIntStorage_get( storage, ind)));
}

auto CPUIntStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int>((storage->data[ind]));
}

void CPUIntStorage::set_flag(char flag) {
  THIntStorage_setFlag( storage, flag);
}

void CPUIntStorage::clear_flag(char flag) {
  THIntStorage_clearFlag( storage, flag);
}

int CPUIntStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPUIntStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Int);
}

const char * CPUIntStorage::toString() const {
  return "CPUIntStorage";
}

const char * CPUIntStorage::typeString() {
  return "CPUIntType";
}

}
