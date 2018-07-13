#include <ATen/Allocator.h>

namespace at {

void deleteNothing(void*) {}
SupervisorPtr nonOwningSupervisorPtr() {
  return {nullptr, &deleteNothing};
}

static void deleteInefficientStdFunctionSupervisor(void* ptr) {
  delete static_cast<InefficientStdFunctionSupervisor*>(ptr);
}

at::DevicePtr
InefficientStdFunctionSupervisor::makeDevicePtr(void* ptr, const std::function<void(void*)>& deleter, Device device) {
  return {ptr, SupervisorPtr{new InefficientStdFunctionSupervisor({ptr, deleter}), &deleteInefficientStdFunctionSupervisor}, device};
}

} // namespace at
