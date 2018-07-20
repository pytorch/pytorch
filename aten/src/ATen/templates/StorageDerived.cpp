#include "ATen/${Storage}.h"

// ${generated_comment}

#include "ATen/Half.h"
#include "ATen/Allocator.h"

#include "ATen/Config.h"
$extra_cuda_headers

namespace at {

${Storage}::${Storage}():
    Storage(${THStorage}_new(${state})) {}

${Storage}::${Storage}(THStorage* storage):
    Storage(storage) {}

${Storage}::${Storage}(size_t storage_size)
  : Storage(${THStorage}_newWithSize(${state,} storage_size)) {}

${Storage}::${Storage}(size_t size, Allocator* allocator)
  : Storage(nullptr) {
  storage = ${THStorage}_newWithAllocator(${state,} size, allocator);
  ${THStorage}_clearFlag(${state,} storage, TH_STORAGE_RESIZABLE);
}

// TODO: Take in Device as an input to the std::function constructor

#if ${isCUDA}
static int getPointerDevice(void* ptr) {
  struct cudaPointerAttributes attr;
  THCudaCheck(cudaPointerGetAttributes(&attr, ptr));
  return attr.device;
}
#endif

${Storage}::${Storage}(
  void * data, size_t size, const std::function<void(void*)> & deleter)
  : Storage(${THStorage}_newWithDataAndAllocator(${state,}
      InefficientStdFunctionContext::makeDataPtr(data, deleter,
#if ${isCUDA}
      Device(kCUDA, getPointerDevice(data))
#else
      kCPU
#endif
       ), size,
     /* allocator */ nullptr
    )) {
    ${THStorage}_clearFlag(${state,} storage, TH_STORAGE_RESIZABLE);
}

${Storage}::~${Storage}() { }

size_t ${Storage}::elementSize() const {
  return sizeof(${ScalarType});
}

Type& ${Storage}::type() const {
  return globalContext().getType(Backend::${Backend},ScalarType::${ScalarName});
}

const char * ${Storage}::typeString() {
  return "${Type}";
}

}
