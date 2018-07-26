#include "ATen/${Storage}.h"

// ${generated_comment}

#include "ATen/Half.h"
#include "ATen/Allocator.h"
#include <ATen/Context.h>

#include "ATen/Config.h"
$extra_cuda_headers

namespace at {

${Storage}::${Storage}()
  : Storage(new StorageImpl(
      ScalarType::${ScalarName}, 
      0,
#if ${isCUDA}
      globalContext().getTHCState()->cudaDeviceAllocator,
#else
      getTHDefaultAllocator(),
#endif
      true)) {}

${Storage}::${Storage}(size_t size)
  : Storage(new StorageImpl(
      ScalarType::${ScalarName}, 
      size,
#if ${isCUDA}
      globalContext().getTHCState()->cudaDeviceAllocator,
#else
      getTHDefaultAllocator(),
#endif
      true)) {}

${Storage}::${Storage}(size_t size, Allocator* allocator)
  : Storage(new StorageImpl(
      ScalarType::${ScalarName}, 
      size,
      allocator,
      true)) {}

// TODO: Take in Device as an input to the std::function constructor

#if ${isCUDA}
static int getPointerDevice(void* ptr) {
  struct cudaPointerAttributes attr;
  THCudaCheck(cudaPointerGetAttributes(&attr, ptr));
  return attr.device;
}
#endif

${Storage}::${Storage}(
  void * data, 
  size_t size, 
  const std::function<void(void*)> & deleter)
  : Storage(new StorageImpl(
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
      true)) {}
}
