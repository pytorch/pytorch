#include "ATen/CUDAShortStorage.h"
#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAShortStorage::CUDAShortStorage(Context* context):
    storage(THCudaShortStorage_new(context->thc_state)), context(context) {}

CUDAShortStorage::CUDAShortStorage(Context* context, THCudaShortStorage* storage):
    storage(storage), context(context) {}

CUDAShortStorage::CUDAShortStorage(Context* context, std::size_t storage_size)
  : storage(THCudaShortStorage_newWithSize(context->thc_state,  storage_size)), context(context) {}

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

CUDAShortStorage::CUDAShortStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator)
  : storage(nullptr),
    context(context) {
  auto ctx = new detail::AllocatorRetainable(std::move(allocator));
  storage = THCudaShortStorage_newWithAllocator(context->thc_state,  size, &wrapped_allocator, ctx);
  ctx->release();
  THCudaShortStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAShortStorage::CUDAShortStorage(Context* context,
  void * data, std::size_t size, const std::function<void(void*)> & deleter)
  : storage(THCudaShortStorage_newWithDataAndAllocator(context->thc_state, 
     static_cast<int16_t*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    THCudaShortStorage_clearFlag(context->thc_state,  storage, TH_STORAGE_RESIZABLE);
}

CUDAShortStorage::~CUDAShortStorage() {
  THCudaShortStorage_free(context->thc_state,  storage);
}

std::size_t CUDAShortStorage::elementSize() const {
  return sizeof(int16_t);
}

std::size_t CUDAShortStorage::size() const {
  return storage->size;
}

void* CUDAShortStorage::data() {
  return storage->data;
}

const void* CUDAShortStorage::data() const {
  return storage->data;
}

auto CUDAShortStorage::retain() -> CUDAShortStorage& {
  THCudaShortStorage_retain(context->thc_state,  storage);
  return *this;
}

auto CUDAShortStorage::free() -> CUDAShortStorage& {
  THCudaShortStorage_free(context->thc_state,  storage);
  return *this;
}

void* CUDAShortStorage::unsafeGetTH(bool retain) const {
  if (retain) {
    THCudaShortStorage_retain(context->thc_state,  storage);
  }
  return storage;
}

auto CUDAShortStorage::resize(int64_t new_size) -> CUDAShortStorage& {
  THCudaShortStorage_resize(context->thc_state,  storage, new_size);
  return *this;
}

auto CUDAShortStorage::fill(Scalar value) -> CUDAShortStorage& {
  THCudaShortStorage_fill(context->thc_state,  storage, (value.toShort()));
  return *this;
}

auto CUDAShortStorage::set(std::size_t ind, Scalar value) -> CUDAShortStorage& {
  THCudaShortStorage_set(context->thc_state,  storage, ind, (value.toShort()));
  return *this;
}

auto CUDAShortStorage::fast_set(std::size_t ind, Scalar value) -> CUDAShortStorage& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto CUDAShortStorage::get(std::size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<int16_t>((THCudaShortStorage_get(context->thc_state,  storage, ind)));
}

auto CUDAShortStorage::fast_get(std::size_t ind) -> Scalar {
  if(true)
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<int16_t>((storage->data[ind]));
}

void CUDAShortStorage::set_flag(char flag) {
  THCudaShortStorage_setFlag(context->thc_state,  storage, flag);
}

void CUDAShortStorage::clear_flag(char flag) {
  THCudaShortStorage_clearFlag(context->thc_state,  storage, flag);
}

int CUDAShortStorage::getDevice() const {
  return storage->device; //storage->device;
}

Type& CUDAShortStorage::type() const {
  return context->getType(Backend::CUDA,ScalarType::Short);
}

const char * CUDAShortStorage::toString() const {
  return "CUDAShortStorage";
}

const char * CUDAShortStorage::typeString() {
  return "CUDAShortType";
}

}
