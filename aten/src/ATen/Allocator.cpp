#include <ATen/Allocator.h>

namespace at {

static void deleteInefficientStdFunctionContext(void* ptr) {
  delete static_cast<InefficientStdFunctionContext*>(ptr);
}

at::DataPtr
InefficientStdFunctionContext::makeDataPtr(void* ptr, const std::function<void(void*)>& deleter, Device device) {
  return {ptr, new InefficientStdFunctionContext({ptr, deleter}), &deleteInefficientStdFunctionContext, device};
}

} // namespace at
