#include "ATen/CUDAFloatStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAFloatStorage::CUDAFloatStorage(Context* context):
    storage(THCudaStorage_new(context->thc_state)), context(context) {}

CUDAFloatStorage::CUDAFloatStorage(Context* context, THCudaStorage* storage):
    storage(storage), context(context) {}

CUDAFloatStorage::CUDAFloatStorage(Context* context, std::size_t storage_size)
  : storage(THCudaStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

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

CUDAFloatStorage::CUDAFloatStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAFloatStorage::CUDAFloatStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<float*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAFloatStorage::~CUDAFloatStorage() {
  THCudaStorage_free(context->thc_state,  storage);
}

std::size_t CUDAFloatStorage::elementSize() const {
  return sizeof(float);
}

std::size_t CUDAFloatStorage::size() const {
  return storage->size;
}

void* CUDAFloatStorage::data() {
  return storage->data;
}

const void* CUDAFloatStorage::data() const {
  return storage->data;
}

auto CUDAFloatStorage::retain() -> CUDAFloatStorage& {
  THCudaStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDAFloatStorage::free() -> CUDAFloatStorage& {
  THCudaStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDAFloatStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDAFloatStorage::resize(int64_t new_size) -> CUDAFloatStorage& {
  THCudaStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDAFloatStorage::fill(Scalar value) -> CUDAFloatStorage& {
  THCudaStorage_fill(context->thc_state,  storage, (value.toFloat()));
  return *this;
}

auto CUDAFloatStorage::set(std::size_t ind, Scalar value) -> CUDAFloatStorage& {
  THCudaStorage_set(context->thc_state,  storage, ind, (value.toFloat()));
  return *this;
}

auto CUDAFloatStorage::fast_set(std::size_t ind, Scalar value) -> CUDAFloatStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDAFloatStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<float>((THCudaStorage_get(context->thc_state,  storage, ind)));
}

auto CUDAFloatStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<float>((storage->data[ind]));
}

void CUDAFloatStorage::set_flag(char flag) {
  THCudaStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDAFloatStorage::clear_flag(char flag) {
  THCudaStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDAFloatStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDAFloatStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Float);
}

const char * CUDAFloatStorage::toString() const {
  return "CUDAFloatStorage";
}

const char * CUDAFloatStorage::typeString() {
  return "CUDAFloatType";
}

}
