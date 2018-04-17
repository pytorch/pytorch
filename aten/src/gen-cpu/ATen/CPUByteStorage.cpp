#include "ATen/CPUByteStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUByteStorage::CPUByteStorage(Context* context):
    storage(THByteStorage_new()), context(context) {}

CPUByteStorage::CPUByteStorage(Context* context, THByteStorage* storage):
    storage(storage), context(context) {}

CPUByteStorage::CPUByteStorage(Context* context, std::size_t storage_size)
  : storage(THByteStorage_newWithSize( storage_size)), context(context) {}

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

CPUByteStorage::CPUByteStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THByteStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THByteStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUByteStorage::CPUByteStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THByteStorage_newWithDataAndAllocator(
     static_cast<uint8_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THByteStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUByteStorage::~CPUByteStorage() {
  THByteStorage_free( storage);
}

std::size_t CPUByteStorage::elementSize() const {
  return sizeof(uint8_t);
}

std::size_t CPUByteStorage::size() const {
  return storage->size;
}

void* CPUByteStorage::data() {
  return storage->data;
}

const void* CPUByteStorage::data() const {
  return storage->data;
}

auto CPUByteStorage::retain() -> CPUByteStorage& {
  THByteStorage_retain( storage);
  return *this;
}

auto CPUByteStorage::free() -> CPUByteStorage& {
  THByteStorage_free( storage);
  return *this;
}

void* CPUByteStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THByteStorage_retain( storage);
  }
  return storage;
}

auto CPUByteStorage::resize(int64_t new_size) -> CPUByteStorage& {
  THByteStorage_resize( storage, new_size);
  return *this;
}

auto CPUByteStorage::fill(Scalar value) -> CPUByteStorage& {
  THByteStorage_fill( storage, (value.toByte()));
  return *this;
}

auto CPUByteStorage::set(std::size_t ind, Scalar value) -> CPUByteStorage& {
  THByteStorage_set( storage, ind, (value.toByte()));
  return *this;
}

auto CPUByteStorage::fast_set(std::size_t ind, Scalar value) -> CPUByteStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPUByteStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<uint8_t>((THByteStorage_get( storage, ind)));
}

auto CPUByteStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<uint8_t>((storage->data[ind]));
}

void CPUByteStorage::set_flag(char flag) {
  THByteStorage_setFlag( storage, flag);
}

void CPUByteStorage::clear_flag(char flag) {
  THByteStorage_clearFlag( storage, flag);
}

int CPUByteStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPUByteStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Byte);
}

const char * CPUByteStorage::toString() const {
  return "CPUByteStorage";
}

const char * CPUByteStorage::typeString() {
  return "CPUByteType";
}

}
