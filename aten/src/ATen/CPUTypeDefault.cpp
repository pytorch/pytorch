#include <ATen/CPUTypeDefault.h>

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>

namespace at {

Allocator* CPUTypeDefault::allocator() const {
  return getCPUAllocator();
}

Device CPUTypeDefault::getDeviceFromPtr(void * data) const {
  return DeviceType::CPU;
}

std::unique_ptr<Generator> CPUTypeDefault::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(&at::globalContext()));
}

} // namespace at
