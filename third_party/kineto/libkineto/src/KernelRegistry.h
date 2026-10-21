/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace KINETO_NAMESPACE {

// Class to track kernel launches across different profiling mechanisms.
class KernelRegistry {
 public:
  using KernelInfoTy = std::pair<std::string, uint64_t>;

  // Get the singleton instance.
  static KernelRegistry* singleton();

  // Record a kernel launch for a specific device.
  void recordKernel(
      uint32_t deviceId,
      const std::string& kernelName,
      uint64_t correlationId);

  /// Return kernel information for the n'th hernel of a specific device with
  /// 'deviceId'.
  std::optional<KernelRegistry::KernelInfoTy> getKernelInfo(
      uint32_t deviceId,
      size_t idx) const;

  /// Return the number of kernels recorded for a specific device with
  /// 'deviceId'.
  size_t getNumKernels(uint32_t deviceId) const;

  // Clear all recorded kernels.
  void clear();

 private:
  // Private constructor to enforce singleton pattern.
  KernelRegistry() = default;

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::vector<KernelInfoTy>> deviceKernelMap_;
};

} // namespace KINETO_NAMESPACE
