#include "ATen/CUDAIntStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAIntStorage::CUDAIntStorage(Context* context):
    storage(THCudaIntStorage_new(context->thc_state)), context(context) {}

CUDAIntStorage::CUDAIntStorage(Context* context, THCudaIntStorage* storage):
    storage(storage), context(context) {}

CUDAIntStorage::CUDAIntStorage(Context* context, std::size_t storage_size)
  : storage(THCudaIntStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

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

CUDAIntStorage::CUDAIntStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaIntStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaIntStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAIntStorage::CUDAIntStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaIntStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<int32_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaIntStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAIntStorage::~CUDAIntStorage() {
  THCudaIntStorage_free(context->thc_state,  storage);
}

std::size_t CUDAIntStorage::elementSize() const {
  return sizeof(int);
}

std::size_t CUDAIntStorage::size() const {
  return storage->size;
}

void* CUDAIntStorage::data() {
  return storage->data;
}

const void* CUDAIntStorage::data() const {
  return storage->data;
}

auto CUDAIntStorage::retain() -> CUDAIntStorage& {
  THCudaIntStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDAIntStorage::free() -> CUDAIntStorage& {
  THCudaIntStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDAIntStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaIntStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDAIntStorage::resize(int64_t new_size) -> CUDAIntStorage& {
  THCudaIntStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDAIntStorage::fill(Scalar value) -> CUDAIntStorage& {
  THCudaIntStorage_fill(context->thc_state,  storage, (value.toInt()));
  return *this;
}

auto CUDAIntStorage::set(std::size_t ind, Scalar value) -> CUDAIntStorage& {
  THCudaIntStorage_set(context->thc_state,  storage, ind, (value.toInt()));
  return *this;
}

auto CUDAIntStorage::fast_set(std::size_t ind, Scalar value) -> CUDAIntStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDAIntStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int>((THCudaIntStorage_get(context->thc_state,  storage, ind)));
}

auto CUDAIntStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int>((storage->data[ind]));
}

void CUDAIntStorage::set_flag(char flag) {
  THCudaIntStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDAIntStorage::clear_flag(char flag) {
  THCudaIntStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDAIntStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDAIntStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Int);
}

const char * CUDAIntStorage::toString() const {
  return "CUDAIntStorage";
}

const char * CUDAIntStorage::typeString() {
  return "CUDAIntType";
}

}
