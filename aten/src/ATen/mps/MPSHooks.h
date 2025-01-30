//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/Generator.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/mps/MPSEvent.h>
#include <optional>

namespace at::mps {

// The real implementation of MPSHooksInterface
struct MPSHooks : public at::MPSHooksInterface {
  MPSHooks(at::MPSHooksArgs) {}
  void init() const override;

  // MPSDevice interface
  bool hasMPS() const override;
  bool isOnMacOSorNewer(unsigned major, unsigned minor) const override;

  // MPSGeneratorImpl interface
  const Generator& getDefaultGenerator(
      DeviceIndex device_index = -1) const override;
  Generator getNewGenerator(DeviceIndex device_index = -1) const override;

  // MPSStream interface
  void deviceSynchronize() const override;
  void commitStream() const override;
  void* getCommandBuffer() const override;
  void* getDispatchQueue() const override;

  // MPSAllocator interface
  Allocator* getMPSDeviceAllocator() const override;
  void emptyCache() const override;
  size_t getCurrentAllocatedMemory() const override;
  size_t getDriverAllocatedMemory() const override;
  size_t getRecommendedMaxMemory() const override;
  void setMemoryFraction(double ratio) const override;
  bool isPinnedPtr(const void* data) const override;
  Allocator* getPinnedMemoryAllocator() const override;

  // MPSProfiler interface
  void profilerStartTrace(const std::string& mode, bool waitUntilCompleted)
      const override;
  void profilerStopTrace() const override;

  // MPSEvent interface
  uint32_t acquireEvent(bool enable_timing) const override;
  void releaseEvent(uint32_t event_id) const override;
  void recordEvent(uint32_t event_id) const override;
  void waitForEvent(uint32_t event_id) const override;
  void synchronizeEvent(uint32_t event_id) const override;
  bool queryEvent(uint32_t event_id) const override;
  double elapsedTimeOfEvents(uint32_t start_event_id, uint32_t end_event_id)
      const override;

  // Compatibility with Accelerator API
  bool hasPrimaryContext(DeviceIndex device_index) const override {
    // When MPS is available, it is always in use for the one device.
    return true;
  }
};

} // namespace at::mps
