#include "ATen/CUDAByteStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAByteStorage::CUDAByteStorage(Context* context):
    storage(THCudaByteStorage_new(context->thc_state)), context(context) {}

CUDAByteStorage::CUDAByteStorage(Context* context, THCudaByteStorage* storage):
    storage(storage), context(context) {}

CUDAByteStorage::CUDAByteStorage(Context* context, std::size_t storage_size)
  : storage(THCudaByteStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

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

CUDAByteStorage::CUDAByteStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaByteStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaByteStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAByteStorage::CUDAByteStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaByteStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<uint8_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaByteStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAByteStorage::~CUDAByteStorage() {
  THCudaByteStorage_free(context->thc_state,  storage);
}

std::size_t CUDAByteStorage::elementSize() const {
  return sizeof(uint8_t);
}

std::size_t CUDAByteStorage::size() const {
  return storage->size;
}

void* CUDAByteStorage::data() {
  return storage->data;
}

const void* CUDAByteStorage::data() const {
  return storage->data;
}

auto CUDAByteStorage::retain() -> CUDAByteStorage& {
  THCudaByteStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDAByteStorage::free() -> CUDAByteStorage& {
  THCudaByteStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDAByteStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaByteStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDAByteStorage::resize(int64_t new_size) -> CUDAByteStorage& {
  THCudaByteStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDAByteStorage::fill(Scalar value) -> CUDAByteStorage& {
  THCudaByteStorage_fill(context->thc_state,  storage, (value.toByte()));
  return *this;
}

auto CUDAByteStorage::set(std::size_t ind, Scalar value) -> CUDAByteStorage& {
  THCudaByteStorage_set(context->thc_state,  storage, ind, (value.toByte()));
  return *this;
}

auto CUDAByteStorage::fast_set(std::size_t ind, Scalar value) -> CUDAByteStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDAByteStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<uint8_t>((THCudaByteStorage_get(context->thc_state,  storage, ind)));
}

auto CUDAByteStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<uint8_t>((storage->data[ind]));
}

void CUDAByteStorage::set_flag(char flag) {
  THCudaByteStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDAByteStorage::clear_flag(char flag) {
  THCudaByteStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDAByteStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDAByteStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Byte);
}

const char * CUDAByteStorage::toString() const {
  return "CUDAByteStorage";
}

const char * CUDAByteStorage::typeString() {
  return "CUDAByteType";
}

}
