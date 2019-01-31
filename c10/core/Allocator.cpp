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

} // namespace c10

namespace caffe2 {

C10_API at::Allocator* allocator_array[static_cast<int>(
    at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

/*
 * Used for custom thread local allocators, i.e. those used during inference
 * runs that preallocates a contiguous chunk of memory needed for the run
 */
C10_API thread_local at::Allocator* local_allocator_array[static_cast<int>(
    at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

void SetAllocator(at::DeviceType t, at::Allocator* alloc) {
  allocator_array[static_cast<int>(t)] = alloc;
}

void SetLocalAllocator(at::DeviceType t, at::Allocator* alloc) {
  local_allocator_array[static_cast<int>(t)] = alloc;
}

at::Allocator* GetLocalAllocator(const at::DeviceType& t) {
  auto* local_alloc = local_allocator_array[static_cast<int>(t)];
  return local_alloc;
}

at::Allocator* GetAllocator(const at::DeviceType& t) {
  auto* local_alloc = local_allocator_array[static_cast<int>(t)];
  if (local_alloc) {
    return local_alloc;
  }
  auto* alloc = allocator_array[static_cast<int>(t)];
  AT_ASSERTM(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

} // namespace caffe2
