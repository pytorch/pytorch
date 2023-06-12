//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/Generator.h>
#include <c10/util/Optional.h>

namespace at { namespace mps {

// The real implementation of MPSHooksInterface
struct MPSHooks : public at::MPSHooksInterface {
  MPSHooks(at::MPSHooksArgs) {}
  void initMPS() const override;

  // MPSDevice interface
  bool hasMPS() const override;
  bool isOnMacOS13orNewer(unsigned minor) const override;

  // MPSGeneratorImpl interface
  const Generator& getDefaultMPSGenerator() const override;

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
  void setMemoryFraction(double ratio) const override;
  void profilerStartTrace(const std::string& mode, bool waitUntilCompleted) const override;
  void profilerStopTrace() const override;
};

}} // at::mps
