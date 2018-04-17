#include "ATen/CUDADoubleStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDADoubleStorage::CUDADoubleStorage(Context* context):
    storage(THCudaDoubleStorage_new(context->thc_state)), context(context) {}

CUDADoubleStorage::CUDADoubleStorage(Context* context, THCudaDoubleStorage* storage):
    storage(storage), context(context) {}

CUDADoubleStorage::CUDADoubleStorage(Context* context, std::size_t storage_size)
  : storage(THCudaDoubleStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

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

CUDADoubleStorage::CUDADoubleStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaDoubleStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaDoubleStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDADoubleStorage::CUDADoubleStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaDoubleStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<double*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaDoubleStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDADoubleStorage::~CUDADoubleStorage() {
  THCudaDoubleStorage_free(context->thc_state,  storage);
}

std::size_t CUDADoubleStorage::elementSize() const {
  return sizeof(double);
}

std::size_t CUDADoubleStorage::size() const {
  return storage->size;
}

void* CUDADoubleStorage::data() {
  return storage->data;
}

const void* CUDADoubleStorage::data() const {
  return storage->data;
}

auto CUDADoubleStorage::retain() -> CUDADoubleStorage& {
  THCudaDoubleStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDADoubleStorage::free() -> CUDADoubleStorage& {
  THCudaDoubleStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDADoubleStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaDoubleStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDADoubleStorage::resize(int64_t new_size) -> CUDADoubleStorage& {
  THCudaDoubleStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDADoubleStorage::fill(Scalar value) -> CUDADoubleStorage& {
  THCudaDoubleStorage_fill(context->thc_state,  storage, (value.toDouble()));
  return *this;
}

auto CUDADoubleStorage::set(std::size_t ind, Scalar value) -> CUDADoubleStorage& {
  THCudaDoubleStorage_set(context->thc_state,  storage, ind, (value.toDouble()));
  return *this;
}

auto CUDADoubleStorage::fast_set(std::size_t ind, Scalar value) -> CUDADoubleStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDADoubleStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<double>((THCudaDoubleStorage_get(context->thc_state,  storage, ind)));
}

auto CUDADoubleStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<double>((storage->data[ind]));
}

void CUDADoubleStorage::set_flag(char flag) {
  THCudaDoubleStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDADoubleStorage::clear_flag(char flag) {
  THCudaDoubleStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDADoubleStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDADoubleStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Double);
}

const char * CUDADoubleStorage::toString() const {
  return "CUDADoubleStorage";
}

const char * CUDADoubleStorage::typeString() {
  return "CUDADoubleType";
}

}
