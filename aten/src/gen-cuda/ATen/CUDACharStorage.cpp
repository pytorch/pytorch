#include "ATen/CUDACharStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDACharStorage::CUDACharStorage(Context* context):
    storage(THCudaCharStorage_new(context->thc_state)), context(context) {}

CUDACharStorage::CUDACharStorage(Context* context, THCudaCharStorage* storage):
    storage(storage), context(context) {}

CUDACharStorage::CUDACharStorage(Context* context, std::size_t storage_size)
  : storage(THCudaCharStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

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

CUDACharStorage::CUDACharStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaCharStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaCharStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDACharStorage::CUDACharStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaCharStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<int8_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaCharStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDACharStorage::~CUDACharStorage() {
  THCudaCharStorage_free(context->thc_state,  storage);
}

std::size_t CUDACharStorage::elementSize() const {
  return sizeof(int8_t);
}

std::size_t CUDACharStorage::size() const {
  return storage->size;
}

void* CUDACharStorage::data() {
  return storage->data;
}

const void* CUDACharStorage::data() const {
  return storage->data;
}

auto CUDACharStorage::retain() -> CUDACharStorage& {
  THCudaCharStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDACharStorage::free() -> CUDACharStorage& {
  THCudaCharStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDACharStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaCharStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDACharStorage::resize(int64_t new_size) -> CUDACharStorage& {
  THCudaCharStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDACharStorage::fill(Scalar value) -> CUDACharStorage& {
  THCudaCharStorage_fill(context->thc_state,  storage, (value.toChar()));
  return *this;
}

auto CUDACharStorage::set(std::size_t ind, Scalar value) -> CUDACharStorage& {
  THCudaCharStorage_set(context->thc_state,  storage, ind, (value.toChar()));
  return *this;
}

auto CUDACharStorage::fast_set(std::size_t ind, Scalar value) -> CUDACharStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDACharStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int8_t>((THCudaCharStorage_get(context->thc_state,  storage, ind)));
}

auto CUDACharStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int8_t>((storage->data[ind]));
}

void CUDACharStorage::set_flag(char flag) {
  THCudaCharStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDACharStorage::clear_flag(char flag) {
  THCudaCharStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDACharStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDACharStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Char);
}

const char * CUDACharStorage::toString() const {
  return "CUDACharStorage";
}

const char * CUDACharStorage::typeString() {
  return "CUDACharType";
}

}
