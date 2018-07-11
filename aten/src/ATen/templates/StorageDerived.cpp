#include "ATen/${Storage}.h"

// ${generated_comment}

#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
$extra_cuda_headers

namespace at {

${Storage}::${Storage}(Context* context):
    storage(${THStorage}_new(${state})), context(context) {}

${Storage}::${Storage}(Context* context, THStorage* storage):
    storage(storage), context(context) {}

${Storage}::${Storage}(Context* context, size_t storage_size)
  : storage(${THStorage}_newWithSize(${state,} storage_size)), context(context) {}

#if ${isCUDA}
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
  auto ac = static_cast<Allocator*>(ctx);
  *result = ac->allocate(size);
  return cudaSuccess;
}
static cudaError_t wrapped_free(void * ctx, void * data) {
  auto ac = static_cast<Allocator*>(ctx);
  ac->deallocate(data);
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
  auto ac = static_cast<Allocator*>(ctx);
  return ac->allocate(size);
}
static void wrapped_free(void * ctx, void * data) {
  auto ac = static_cast<Allocator*>(ctx);
  ac->deallocate(data);
}
static THAllocator wrapped_allocator = {
  wrapped_alloc,
  nullptr,
  wrapped_free,
};
#endif

${Storage}::${Storage}(Context* context, size_t size, Allocator* allocator)
  : storage(nullptr),
    context(context) {
  storage = ${THStorage}_newWithAllocator(${state,} size, &wrapped_allocator, allocator);
  ${THStorage}_clearFlag(${state,} storage, TH_STORAGE_RESIZABLE);
}

${Storage}::${Storage}(Context* context,
  void * data, size_t size, const std::function<void(void*)> & deleter)
  : storage(${THStorage}_newWithDataAndAllocator(${state,}
     static_cast<${THScalarType}*>(data), size,
     &storage_deleter,
     new std::function<void(void*)>(deleter)
    )),
    context(context) {
    ${THStorage}_clearFlag(${state,} storage, TH_STORAGE_RESIZABLE);
}

${Storage}::~${Storage}() {
  ${THStorage}_free(${state,} storage);
}

size_t ${Storage}::elementSize() const {
  return sizeof(${ScalarType});
}

size_t ${Storage}::size() const {
  return storage->size;
}

void* ${Storage}::data() {
  return storage->data_ptr;
}

const void* ${Storage}::data() const {
  return storage->data_ptr;
}

auto ${Storage}::retain() -> ${Storage}& {
  ${THStorage}_retain(${state,} storage);
  return *this;
}

auto ${Storage}::free() -> ${Storage}& {
  ${THStorage}_free(${state,} storage);
  return *this;
}

void* ${Storage}::unsafeGetTH(bool retain) const {
  if (retain) {
    ${THStorage}_retain(${state,} storage);
  }
  return storage;
}

auto ${Storage}::resize(int64_t new_size) -> ${Storage}& {
  ${THStorage}_resize(${state,} storage, new_size);
  return *this;
}

auto ${Storage}::fill(Scalar value) -> ${Storage}& {
  ${THStorage}_fill(${state,} storage, ${to_th_type}(value.to${ScalarName}()));
  return *this;
}

auto ${Storage}::set(size_t ind, Scalar value) -> ${Storage}& {
  ${THStorage}_set(${state,} storage, ind, ${to_th_type}(value.to${ScalarName}()));
  return *this;
}

auto ${Storage}::fast_set(size_t ind, Scalar value) -> ${Storage}& {
  throw std::runtime_error("unsupported operation 'fast_set'");
}

auto ${Storage}::get(size_t ind) -> Scalar {
  // static cast to fix  long -> int64_t issues
  return static_cast<${ScalarType}>(${to_at_type}(${THStorage}_get(${state,} storage, ind)));
}

auto ${Storage}::fast_get(size_t ind) -> Scalar {
  if(${isCUDA})
    throw std::runtime_error("unsupported operation 'fast_get'");
  return static_cast<${ScalarType}>(${to_at_type}(storage->unsafe_data<${THScalarType}>()[ind]));
}

void ${Storage}::set_flag(char flag) {
  ${THStorage}_setFlag(${state,} storage, flag);
}

void ${Storage}::clear_flag(char flag) {
  ${THStorage}_clearFlag(${state,} storage, flag);
}

int ${Storage}::getDevice() const {
  ${storage_device} //storage->device;
}

Type& ${Storage}::type() const {
  return context->getType(Backend::${Backend},ScalarType::${ScalarName});
}

const char * ${Storage}::toString() const {
  return "${Storage}";
}

const char * ${Storage}::typeString() {
  return "${Type}";
}

}
