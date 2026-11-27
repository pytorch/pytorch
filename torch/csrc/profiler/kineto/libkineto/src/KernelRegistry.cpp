/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "KernelRegistry.h"

namespace KINETO_NAMESPACE {

// Static singleton instance:
KernelRegistry* KernelRegistry::singleton() {
  static KernelRegistry instance;
  return &instance;
}

void KernelRegistry::recordKernel(
    uint32_t deviceId,
    const std::string& kernelName,
    uint64_t correlationId) {
  std::scoped_lock guard(mutex_);
  deviceKernelMap_[deviceId].emplace_back(kernelName, correlationId);
}

/// Return kernel information for the n'th hernel of a specific device with
/// 'deviceId'.
std::optional<KernelRegistry::KernelInfoTy> KernelRegistry::getKernelInfo(
    uint32_t deviceId,
    size_t idx) const {
  std::scoped_lock guard(mutex_);

  auto it = deviceKernelMap_.find(deviceId);
  if (it != deviceKernelMap_.end()) {
    const std::vector<KernelRegistry::KernelInfoTy>& kernels = it->second;
    if (idx < kernels.size()) {
      return std::make_optional(kernels[idx]);
    }
  }
  return std::nullopt;
}

/// Return the number of kernels recorded for a specific device with
/// 'deviceId'.
size_t KernelRegistry::getNumKernels(uint32_t deviceId) const {
  std::scoped_lock guard(mutex_);

  auto it = deviceKernelMap_.find(deviceId);
  if (it != deviceKernelMap_.end()) {
    return it->second.size();
  } else {
    return 0;
  }
}

void KernelRegistry::clear() {
  std::scoped_lock guard(mutex_);
  deviceKernelMap_.clear();
}

} // namespace KINETO_NAMESPACE
