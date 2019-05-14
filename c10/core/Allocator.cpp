#include <c10/core/Allocator.h>

namespace c10 {

static void deleteInefficientStdFunctionContext(void* ptr) {
  delete static_cast<InefficientStdFunctionContext*>(ptr);
}

at::DataPtr InefficientStdFunctionContext::makeDataPtr(
    void* ptr,
    const std::function<void(void*)>& deleter,
    Device device) {
  return {ptr,
          new InefficientStdFunctionContext({ptr, deleter}),
          &deleteInefficientStdFunctionContext,
          device};
}

C10_API at::Allocator* allocator_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];

void SetAllocator(at::DeviceType t, at::Allocator* alloc) {
  allocator_array[static_cast<int>(t)] = alloc;
}

at::Allocator* GetAllocator(const at::DeviceType& t) {
  auto* alloc = allocator_array[static_cast<int>(t)];
  AT_ASSERTM(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

} // namespace c10
