/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <cupti.h>

namespace KINETO_NAMESPACE {

// Domain of the CUPTI callback
enum class CallbackDomain : uint8_t {
  RUNTIME, // CUDA Runtime API (cudaXxx functions)
  DRIVER, // CUDA Driver API (cuXxx functions)
};

// CuptiCbidRegistry: Central registry for CUPTI callback ID metadata
//
// This class provides a single source of truth for all CUPTI callback ID
// properties, replacing scattered hardcoded checks throughout the codebase.
//
// The registry is initialized lazily on first access via instance().
// All callback IDs are registered in the constructor with their properties.
//
// Usage:
//   auto& registry = CuptiCbidRegistry::instance();
//   if (registry.requiresFlowCorrelation(CallbackDomain::RUNTIME, cbid)) {
//     // Create flow correlation
//   }
//
class CuptiCbidRegistry {
 public:
  // Get the singleton instance (lazy initialization on first call)
  static CuptiCbidRegistry& instance();

  // Disable copy/move
  CuptiCbidRegistry(const CuptiCbidRegistry&) = delete;
  CuptiCbidRegistry& operator=(const CuptiCbidRegistry&) = delete;
  CuptiCbidRegistry(CuptiCbidRegistry&&) = delete;
  CuptiCbidRegistry& operator=(CuptiCbidRegistry&&) = delete;

  // Check if a callback ID requires flow correlation (CPU->GPU arrows in trace)
  [[nodiscard]] bool requiresFlowCorrelation(
      CallbackDomain domain,
      uint32_t cbid);

  // Check if a callback ID is blocklisted (should be filtered from traces)
  [[nodiscard]] bool isBlocklisted(CallbackDomain domain, uint32_t cbid);

  // Check if a callback ID is registered
  [[nodiscard]] bool isRegistered(CallbackDomain domain, uint32_t cbid);

 private:
  CuptiCbidRegistry();
  ~CuptiCbidRegistry() = default;

  // Properties stored per callback ID
  struct CbidProperties {
    bool requiresFlowCorrelation;
    bool isBlocklisted;
  };

  // Range of callback IDs (for memory operations)
  struct CbidRange {
    uint32_t startCbid;
    uint32_t endCbid; // inclusive
    bool requiresFlowCorrelation;
    bool isBlocklisted;
  };

  // Register a callback ID
  void registerCbid(
      CallbackDomain domain,
      uint32_t cbid,
      bool requiresFlowCorrelation,
      bool isBlocklisted);

  // Register a range of callback IDs
  void registerCbidRange(
      CallbackDomain domain,
      uint32_t startCbid,
      uint32_t endCbid,
      bool requiresFlowCorrelation,
      bool isBlocklisted);

  // Storage per domain
  std::unordered_map<uint32_t, CbidProperties> runtimeCallbacks_;
  std::unordered_map<uint32_t, CbidProperties> driverCallbacks_;

  // Ranges for callback IDs that use range-based matching
  std::vector<std::pair<CallbackDomain, CbidRange>> cbidRanges_;

  // Helper to get the appropriate map for domain
  [[nodiscard]] std::unordered_map<uint32_t, CbidProperties>& getMapForDomain(
      CallbackDomain domain);
};

} // namespace KINETO_NAMESPACE
