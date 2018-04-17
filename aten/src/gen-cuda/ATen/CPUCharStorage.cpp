#include "ATen/CPUCharStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUCharStorage::CPUCharStorage(Context* context):
    storage(THCharStorage_new()), context(context) {}

CPUCharStorage::CPUCharStorage(Context* context, THCharStorage* storage):
    storage(storage), context(context) {}

CPUCharStorage::CPUCharStorage(Context* context, std::size_t storage_size)
  : storage(THCharStorage_newWithSize( storage_size)), context(context) {}

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

CPUCharStorage::CPUCharStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCharStorage_newWithAllocator( size, &wrapped_allocator, ctx);
  ctx->release();
  THCharStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUCharStorage::CPUCharStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCharStorage_newWithDataAndAllocator(
     static_cast<int8_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCharStorage_clearFlag( storage, TH_STORAGE_RESIZABLE);
}

CPUCharStorage::~CPUCharStorage() {
  THCharStorage_free( storage);
}

std::size_t CPUCharStorage::elementSize() const {
  return sizeof(int8_t);
}

std::size_t CPUCharStorage::size() const {
  return storage->size;
}

void* CPUCharStorage::data() {
  return storage->data;
}

const void* CPUCharStorage::data() const {
  return storage->data;
}

auto CPUCharStorage::retain() -> CPUCharStorage& {
  THCharStorage_retain( storage);
  return *this;
}

auto CPUCharStorage::free() -> CPUCharStorage& {
  THCharStorage_free( storage);
  return *this;
}

void* CPUCharStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCharStorage_retain( storage);
  }
  return storage;
}

auto CPUCharStorage::resize(int64_t new_size) -> CPUCharStorage& {
  THCharStorage_resize( storage, new_size);
  return *this;
}

auto CPUCharStorage::fill(Scalar value) -> CPUCharStorage& {
  THCharStorage_fill( storage, (value.toChar()));
  return *this;
}

auto CPUCharStorage::set(std::size_t ind, Scalar value) -> CPUCharStorage& {
  THCharStorage_set( storage, ind, (value.toChar()));
  return *this;
}

auto CPUCharStorage::fast_set(std::size_t ind, Scalar value) -> CPUCharStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CPUCharStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int8_t>((THCharStorage_get( storage, ind)));
}

auto CPUCharStorage::fast_get(std::size_t ind) -> Scalar {
  if(false)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int8_t>((storage->data[ind]));
}

void CPUCharStorage::set_flag(char flag) {
  THCharStorage_setFlag( storage, flag);
}

void CPUCharStorage::clear_flag(char flag) {
  THCharStorage_clearFlag( storage, flag);
}

int CPUCharStorage::getDevice() const {
  throw std::runtime_error("CPU storage has no device"); //storage->device;
}

Type& CPUCharStorage::type() const {
  return context->getType(Backend::CPU,ScalarType::Char);
}

const char * CPUCharStorage::toString() const {
  return "CPUCharStorage";
}

const char * CPUCharStorage::typeString() {
  return "CPUCharType";
}

}
