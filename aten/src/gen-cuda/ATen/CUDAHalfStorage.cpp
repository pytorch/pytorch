#include "ATen/CUDAHalfStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAHalfStorage::CUDAHalfStorage(Context* context):
    storage(THCudaHalfStorage_new(context->thc_state)), context(context) {}

CUDAHalfStorage::CUDAHalfStorage(Context* context, THCudaHalfStorage* storage):
    storage(storage), context(context) {}

CUDAHalfStorage::CUDAHalfStorage(Context* context, std::size_t storage_size)
  : storage(THCudaHalfStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

#if true
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

CUDAHalfStorage::CUDAHalfStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaHalfStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaHalfStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAHalfStorage::CUDAHalfStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaHalfStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<half*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaHalfStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAHalfStorage::~CUDAHalfStorage() {
  THCudaHalfStorage_free(context->thc_state,  storage);
}

std::size_t CUDAHalfStorage::elementSize() const {
  return sizeof(Half);
}

std::size_t CUDAHalfStorage::size() const {
  return storage->size;
}

void* CUDAHalfStorage::data() {
  return storage->data;
}

const void* CUDAHalfStorage::data() const {
  return storage->data;
}

auto CUDAHalfStorage::retain() -> CUDAHalfStorage& {
  THCudaHalfStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDAHalfStorage::free() -> CUDAHalfStorage& {
  THCudaHalfStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDAHalfStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaHalfStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDAHalfStorage::resize(int64_t new_size) -> CUDAHalfStorage& {
  THCudaHalfStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDAHalfStorage::fill(Scalar value) -> CUDAHalfStorage& {
  THCudaHalfStorage_fill(context->thc_state,  storage, HalfFix<__half,Half>(value.toHalf()));
  return *this;
}

auto CUDAHalfStorage::set(std::size_t ind, Scalar value) -> CUDAHalfStorage& {
  THCudaHalfStorage_set(context->thc_state,  storage, ind, HalfFix<__half,Half>(value.toHalf()));
  return *this;
}

auto CUDAHalfStorage::fast_set(std::size_t ind, Scalar value) -> CUDAHalfStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDAHalfStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<Half>(HalfFix<Half,__half>(THCudaHalfStorage_get(context->thc_state,  storage, ind)));
}

auto CUDAHalfStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<Half>(HalfFix<Half,__half>(storage->data[ind]));
}

void CUDAHalfStorage::set_flag(char flag) {
  THCudaHalfStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDAHalfStorage::clear_flag(char flag) {
  THCudaHalfStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDAHalfStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDAHalfStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Half);
}

const char * CUDAHalfStorage::toString() const {
  return "CUDAHalfStorage";
}

const char * CUDAHalfStorage::typeString() {
  return "CUDAHalfType";
}

}
