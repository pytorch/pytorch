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
  bool hasMPS() const override;
  Allocator* getMPSDeviceAllocator() const override;
  const Generator& getDefaultMPSGenerator() const override;
};

}} // at::mps
