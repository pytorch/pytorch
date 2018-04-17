#include "ATen/CPUDoubleStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUDoubleStorage::CPUDoubleStorage(Context* context):
    storage(THDoubleStorage_new()), context(context) {}

CPUDoubleStorage::CPUDoubleStorage(Context* context, THDoubleStorage* storage):
    storage(storage), context(context) {}

CPUDoubleStorage::CPUDoubleStorage(Context* context, std::size_t storage_size)
  : storage(THDoubleStorage_newWithSize( storage_size)), context(context) {}

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

CPUDoubleStorage::CPUDoubleStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THDoubleStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THDoubleStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUDoubleStorage::CPUDoubleStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THDoubleStorage_newWithDataAndAllocator(
     static_cast<double*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THDoubleStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUDoubleStorage::~CPUDoubleStorage() {
  THDoubleStorage_free( storage);
}

std::size_t CPUDoubleStorage::elementSize() const {
  return sizeof(double);
}

std::size_t CPUDoubleStorage::size() const {
  return storage->size;
}

void* CPUDoubleStorage::data() {
  return storage->data;
}

const void* CPUDoubleStorage::data() const {
  return storage->data;
}

auto CPUDoubleStorage::retain() -> CPUDoubleStorage& {
  THDoubleStorage_retain( storage);
  return *this;
}

auto CPUDoubleStorage::free() -> CPUDoubleStorage& {
  THDoubleStorage_free( storage);
  return *this;
}

void* CPUDoubleStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THDoubleStorage_retain( storage);
  }
  return storage;
}

auto CPUDoubleStorage::resize(int64_t new_size) -> CPUDoubleStorage& {
  THDoubleStorage_resize( storage, new_size);
  return *this;
}

auto CPUDoubleStorage::fill(Scalar value) -> CPUDoubleStorage& {
  THDoubleStorage_fill( storage, (value.toDouble()));
  return *this;
}

auto CPUDoubleStorage::set(std::size_t ind, Scalar value) -> CPUDoubleStorage& {
  THDoubleStorage_set( storage, ind, (value.toDouble()));
  return *this;
}

auto CPUDoubleStorage::fast_set(std::size_t ind, Scalar value) -> CPUDoubleStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPUDoubleStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<double>((THDoubleStorage_get( storage, ind)));
}

auto CPUDoubleStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<double>((storage->data[ind]));
}

void CPUDoubleStorage::set_flag(char flag) {
  THDoubleStorage_setFlag( storage, flag);
}

void CPUDoubleStorage::clear_flag(char flag) {
  THDoubleStorage_clearFlag( storage, flag);
}

int CPUDoubleStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPUDoubleStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Double);
}

const char * CPUDoubleStorage::toString() const {
  return "CPUDoubleStorage";
}

const char * CPUDoubleStorage::typeString() {
  return "CPUDoubleType";
}

}
