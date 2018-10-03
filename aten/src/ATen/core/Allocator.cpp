#include <ATen/core/Allocator.h>

namespace at {

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

} // namespace at

namespace caffe2 {

std::mutex& GetAllocatorArrayMutex() {
  static std::mutex mutex;
  return mutex;
}

CAFFE2_API std::unique_ptr<at::Allocator> allocator_array[static_cast<int>(
    at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];
// AllocatorArray& GetAllocatorArray() {
//   static AllocatorArray array;
//   return array;
// }

void SetAllocator(at::DeviceType t, at::Allocator* alloc) {
  // auto& array = GetAllocatorArray();
  auto& mutex = GetAllocatorArrayMutex();
  std::lock_guard<std::mutex> guard(mutex);
  auto& uniq_ptr = allocator_array[static_cast<int>(t)];
  uniq_ptr.reset(alloc);
}

} // namespace caffe2
