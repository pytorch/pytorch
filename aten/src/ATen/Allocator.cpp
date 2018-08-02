#include <ATen/Allocator.h>

namespace at {

static void deleteInefficientStdFunctionContext(void* ptr) {
  delete static_cast<InefficientStdFunctionContext*>(ptr);
}

at::DataPtr
InefficientStdFunctionContext::makeDataPtr(void* ptr, const std::function<void(void*)>& deleter, Device device) {
  return {ptr, new InefficientStdFunctionContext({ptr, deleter}), &deleteInefficientStdFunctionContext, device};
}

static void deleteInefficientSharedPtrContext(void* ptr) {
  delete static_cast<InefficientSharedPtrContext*>(ptr);
}

at::DataPtr
InefficientSharedPtrContext::makeDataPtr(void* ptr, const std::shared_ptr<void> shared_ptr, Device device) {
  return {ptr, new InefficientSharedPtrContext(shared_ptr), &deleteInefficientSharedPtrContext, device};
}

} // namespace at
