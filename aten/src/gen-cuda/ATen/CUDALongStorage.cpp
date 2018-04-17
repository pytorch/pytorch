#include "ATen/CUDALongStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDALongStorage::CUDALongStorage(Context* context):
    storage(THCudaLongStorage_new(context->thc_state)), context(context) {}

CUDALongStorage::CUDALongStorage(Context* context, THCudaLongStorage* storage):
    storage(storage), context(context) {}

CUDALongStorage::CUDALongStorage(Context* context, std::size_t storage_size)
  : storage(THCudaLongStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

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

CUDALongStorage::CUDALongStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaLongStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaLongStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDALongStorage::CUDALongStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaLongStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<int64_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaLongStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDALongStorage::~CUDALongStorage() {
  THCudaLongStorage_free(context->thc_state,  storage);
}

std::size_t CUDALongStorage::elementSize() const {
  return sizeof(int64_t);
}

std::size_t CUDALongStorage::size() const {
  return storage->size;
}

void* CUDALongStorage::data() {
  return storage->data;
}

const void* CUDALongStorage::data() const {
  return storage->data;
}

auto CUDALongStorage::retain() -> CUDALongStorage& {
  THCudaLongStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDALongStorage::free() -> CUDALongStorage& {
  THCudaLongStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDALongStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaLongStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDALongStorage::resize(int64_t new_size) -> CUDALongStorage& {
  THCudaLongStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDALongStorage::fill(Scalar value) -> CUDALongStorage& {
  THCudaLongStorage_fill(context->thc_state,  storage, long(value.toLong()));
  return *this;
}

auto CUDALongStorage::set(std::size_t ind, Scalar value) -> CUDALongStorage& {
  THCudaLongStorage_set(context->thc_state,  storage, ind, long(value.toLong()));
  return *this;
}

auto CUDALongStorage::fast_set(std::size_t ind, Scalar value) -> CUDALongStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDALongStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int64_t>(int64_t(THCudaLongStorage_get(context->thc_state,  storage, ind)));
}

auto CUDALongStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int64_t>(int64_t(storage->data[ind]));
}

void CUDALongStorage::set_flag(char flag) {
  THCudaLongStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDALongStorage::clear_flag(char flag) {
  THCudaLongStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDALongStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDALongStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Long);
}

const char * CUDALongStorage::toString() const {
  return "CUDALongStorage";
}

const char * CUDALongStorage::typeString() {
  return "CUDALongType";
}

}
