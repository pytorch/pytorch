#include "ATen/CPUFloatStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUFloatStorage::CPUFloatStorage(Context* context):
    storage(THFloatStorage_new()), context(context) {}

CPUFloatStorage::CPUFloatStorage(Context* context, THFloatStorage* storage):
    storage(storage), context(context) {}

CPUFloatStorage::CPUFloatStorage(Context* context, std::size_t storage_size)
  : storage(THFloatStorage_newWithSize( storage_size)), context(context) {}

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

CPUFloatStorage::CPUFloatStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THFloatStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THFloatStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUFloatStorage::CPUFloatStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THFloatStorage_newWithDataAndAllocator(
     static_cast<float*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THFloatStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUFloatStorage::~CPUFloatStorage() {
  THFloatStorage_free( storage);
}

std::size_t CPUFloatStorage::elementSize() const {
  return sizeof(float);
}

std::size_t CPUFloatStorage::size() const {
  return storage->size;
}

void* CPUFloatStorage::data() {
  return storage->data;
}

const void* CPUFloatStorage::data() const {
  return storage->data;
}

auto CPUFloatStorage::retain() -> CPUFloatStorage& {
  THFloatStorage_retain( storage);
  return *this;
}

auto CPUFloatStorage::free() -> CPUFloatStorage& {
  THFloatStorage_free( storage);
  return *this;
}

void* CPUFloatStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THFloatStorage_retain( storage);
  }
  return storage;
}

auto CPUFloatStorage::resize(int64_t new_size) -> CPUFloatStorage& {
  THFloatStorage_resize( storage, new_size);
  return *this;
}

auto CPUFloatStorage::fill(Scalar value) -> CPUFloatStorage& {
  THFloatStorage_fill( storage, (value.toFloat()));
  return *this;
}

auto CPUFloatStorage::set(std::size_t ind, Scalar value) -> CPUFloatStorage& {
  THFloatStorage_set( storage, ind, (value.toFloat()));
  return *this;
}

auto CPUFloatStorage::fast_set(std::size_t ind, Scalar value) -> CPUFloatStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPUFloatStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<float>((THFloatStorage_get( storage, ind)));
}

auto CPUFloatStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<float>((storage->data[ind]));
}

void CPUFloatStorage::set_flag(char flag) {
  THFloatStorage_setFlag( storage, flag);
}

void CPUFloatStorage::clear_flag(char flag) {
  THFloatStorage_clearFlag( storage, flag);
}

int CPUFloatStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPUFloatStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Float);
}

const char * CPUFloatStorage::toString() const {
  return "CPUFloatStorage";
}

const char * CPUFloatStorage::typeString() {
  return "CPUFloatType";
}

}
