#include "ATen/${Storage}.h"

// ${generated_comment}

#include <ATen/Half.h>
#include <ATen/Allocator.h>

#include <ATen/Config.h>
$extra_cuda_headers

namespace at {

${Storage}::${Storage}() :
    Storage(
      Backend::${Backend},
      ScalarType::${ScalarName},
      0,
#if ${isCUDA}
      globalContext().getTHCState()->cudaDeviceAllocator,
#else
      getTHDefaultAllocator(),
#endif
      TH_STORAGE_RESIZABLE)
 {}

${Storage}::${Storage}(size_t storage_size) : 
    Storage(
      Backend::${Backend},
      ScalarType::${ScalarName},
      storage_size,
#if ${isCUDA}
      globalContext().getTHCState()->cudaDeviceAllocator,
#else
      getTHDefaultAllocator(),
#endif
      TH_STORAGE_RESIZABLE) {}

${Storage}::${Storage}(size_t size, Allocator* allocator)
  : Storage(
  Backend::${Backend},
  ScalarType::${ScalarName}, size, allocator, TH_STORAGE_RESIZABLE) {}

// TODO: Take in Device as an input to the std::function constructor

#if ${isCUDA}
static int getPointerDevice(void* ptr) {
  struct cudaPointerAttributes attr;
  THCudaCheck(cudaPointerGetAttributes(&attr, ptr));
  return attr.device;
}
#endif

${Storage}::${Storage}(
  void * data, size_t size, const std::function<void(void*)> & deleter) :
  Storage(
      Backend::${Backend},
      ScalarType::${ScalarName},
      size,
      InefficientStdFunctionContext::makeDataPtr(data, deleter,
#if ${isCUDA}
      Device(kCUDA, getPointerDevice(data))
#else
      kCPU
#endif
       ),
     /* allocator */ nullptr,
     TH_STORAGE_RESIZABLE
    ) {}

${Storage}::~${Storage}() { }

const char * ${Storage}::typeString() {
  return "${Type}";
}

}
