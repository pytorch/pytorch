//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSHooks.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/mps/MPSAllocatorInterface.h>

namespace at {
namespace mps {

void MPSHooks::initMPS() const {
  C10_LOG_API_USAGE_ONCE("aten.init.mps");
  // TODO: initialize MPS devices and streams here
}

bool MPSHooks::hasMPS() const {
  return at::mps::is_available();
}

bool MPSHooks::isOnMacOS13orNewer() const {
  return at::mps::is_macos_13_or_newer();
}

Allocator* MPSHooks::getMPSDeviceAllocator() const {
  return at::mps::GetMPSAllocator();
}

const Generator& MPSHooks::getDefaultMPSGenerator() const {
  return at::mps::detail::getDefaultMPSGenerator();
}

void MPSHooks::deviceSynchronize() const {
  at::mps::device_synchronize();
}

void MPSHooks::emptyCache() const {
  at::mps::getIMPSAllocator()->emptyCache();
}

size_t MPSHooks::getCurrentAllocatedMemory() const {
  return at::mps::getIMPSAllocator()->getCurrentAllocatedMemory();
}

size_t MPSHooks::getDriverAllocatedMemory() const {
  return at::mps::getIMPSAllocator()->getDriverAllocatedMemory();
}

void MPSHooks::setMemoryFraction(double ratio) const {
  at::mps::getIMPSAllocator()->setHighWatermarkRatio(ratio);
}

using at::MPSHooksRegistry;
using at::RegistererMPSHooksRegistry;

REGISTER_MPS_HOOKS(MPSHooks);

} // namespace mps
} // namespace at
