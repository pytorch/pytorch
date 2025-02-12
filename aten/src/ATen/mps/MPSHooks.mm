//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/mps/MPSHooks.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>
#include <c10/util/Logging.h>

namespace at::mps {

void MPSHooks::init() const {
  C10_LOG_API_USAGE_ONCE("aten.init.mps");
  // TODO: initialize MPS devices and streams here
}

bool MPSHooks::hasMPS() const {
  return at::mps::is_available();
}

bool MPSHooks::isOnMacOSorNewer(unsigned major, unsigned minor) const {
  switch (major) {
    case 15:
      switch (minor) {
        case 0:
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
        case 1:
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_1_PLUS);
        default:
          TORCH_WARN("Can't check whether running on 15.", minor, "+ returning one for 15.1+");
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_1_PLUS);
      }
    case 14:
      switch (minor) {
        case 0:
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS);
        case 4:
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_4_PLUS);
        default:
          TORCH_WARN("Can't check whether running on 14.", minor, "+ returning one for 14.4+");
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_4_PLUS);
      }
    case 13:
      switch (minor) {
        case 0:
          return true;
        case 1:
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_1_PLUS);
        case 2:
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS);
        case 3:
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
        default:
          TORCH_WARN("Can't check whether running on 13.", minor, "+ returning one for 13.3+");
          return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
      }
    default:
      TORCH_WARN("Checking for unexpected MacOS ", major, ".", minor, " returning false");
      return false;
  }
}

Allocator* MPSHooks::getMPSDeviceAllocator() const {
  return at::mps::GetMPSAllocator();
}

const Generator& MPSHooks::getDefaultGenerator([[maybe_unused]] DeviceIndex device_index) const {
  return at::mps::detail::getDefaultMPSGenerator();
}

Generator MPSHooks::getNewGenerator([[maybe_unused]] DeviceIndex device_index) const {
  return make_generator<at::MPSGeneratorImpl>();
}

void MPSHooks::deviceSynchronize() const {
  at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

void MPSHooks::commitStream() const {
  at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT);
}

void* MPSHooks::getCommandBuffer() const {
  return at::mps::getDefaultMPSStream()->commandBuffer();
}

void* MPSHooks::getDispatchQueue() const {
  return at::mps::getDefaultMPSStream()->queue();
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

size_t MPSHooks::getRecommendedMaxMemory() const {
  return at::mps::getIMPSAllocator()->getRecommendedMaxMemory();
}

void MPSHooks::setMemoryFraction(double ratio) const {
  at::mps::getIMPSAllocator()->setHighWatermarkRatio(ratio);
}

void MPSHooks::profilerStartTrace(const std::string& mode, bool waitUntilCompleted) const {
  at::mps::getMPSProfiler().StartTrace(mode, waitUntilCompleted);
}

void MPSHooks::profilerStopTrace() const {
  at::mps::getMPSProfiler().StopTrace();
}

uint32_t MPSHooks::acquireEvent(bool enable_timing) const {
  return at::mps::getMPSEventPool()->acquireEvent(enable_timing);
}

void MPSHooks::releaseEvent(uint32_t event_id) const {
  at::mps::getMPSEventPool()->releaseEvent(event_id);
}

void MPSHooks::recordEvent(uint32_t event_id) const {
  at::mps::getMPSEventPool()->recordEvent(event_id, /* syncEvent*/ true);
}

void MPSHooks::waitForEvent(uint32_t event_id) const {
  at::mps::getMPSEventPool()->waitForEvent(event_id, /* syncEvent*/ true);
}

void MPSHooks::synchronizeEvent(uint32_t event_id) const {
  at::mps::getMPSEventPool()->synchronizeEvent(event_id);
}

bool MPSHooks::queryEvent(uint32_t event_id) const {
  return at::mps::getMPSEventPool()->queryEvent(event_id);
}

double MPSHooks::elapsedTimeOfEvents(uint32_t start_event_id, uint32_t end_event_id) const {
  return at::mps::getMPSEventPool()->elapsedTime(start_event_id, end_event_id);
}

bool MPSHooks::isPinnedPtr(const void* data) const {
  return at::mps::isMPSPinnedPtr(data);
}

Allocator* MPSHooks::getPinnedMemoryAllocator() const {
  return at::mps::getIMPSAllocator(true);
}

using at::MPSHooksRegistry;
using at::RegistererMPSHooksRegistry;

REGISTER_MPS_HOOKS(MPSHooks);

} // namespace at::mps
