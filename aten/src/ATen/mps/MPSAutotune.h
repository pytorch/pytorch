#pragma once

#include <c10/macros/Macros.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace at::mps {

struct MPSAutotuneTensorInfo {
  std::string name;
  std::string dtype;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t storage_offset = 0;
};

struct MPSAutotuneCandidateResult {
  std::string config;
  std::string kernel;
  double median_us = 0.0;
  int samples = 0;
  bool active = false;
};

struct MPSAutotuneRecord {
  uint64_t sequence = 0;
  std::string event;
  std::string operation;
  std::string phase;
  std::string config;
  std::string kernel;
  std::vector<MPSAutotuneTensorInfo> tensors;
  std::vector<std::pair<std::string, std::string>> attributes;
  std::vector<std::string> candidates;
  std::vector<MPSAutotuneCandidateResult> results;
};

struct MPSAutotuneSnapshot {
  std::vector<MPSAutotuneRecord> records;
  size_t dropped = 0;
};

TORCH_API bool isMPSAutotuneTraceEnabled();
TORCH_API void startMPSAutotuneTrace(size_t max_entries);
TORCH_API MPSAutotuneSnapshot stopMPSAutotuneTrace(
    bool wait_for_callbacks);
TORCH_API void recordMPSAutotuneEvent(
    MPSAutotuneRecord record,
    bool retained = false);
TORCH_API bool retainMPSAutotuneTrace();
TORCH_API void releaseMPSAutotuneTrace();

TORCH_API std::optional<std::string> getMPSAutotuneOverride(
    std::string_view operation);
TORCH_API void setMPSAutotuneOverride(
    const std::string& operation,
    std::optional<std::string> config);

TORCH_API uint64_t getMPSAutotuneCacheGeneration();
TORCH_API void clearMPSAutotuneCaches();

} // namespace at::mps
