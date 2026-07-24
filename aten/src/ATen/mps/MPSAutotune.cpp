#include <ATen/mps/MPSAutotune.h>

#include <c10/util/Exception.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace at::mps {
namespace {

struct MPSAutotuneTraceState {
  std::atomic<bool> recording{false};
  std::mutex mutex;
  std::deque<MPSAutotuneRecord> records;
  size_t max_entries = 0;
  size_t dropped = 0;
  uint64_t sequence = 0;
};

MPSAutotuneTraceState& traceState() {
  static auto* state = new MPSAutotuneTraceState();
  return *state;
}

struct MPSAutotuneOverrides {
  std::mutex mutex;
  std::unordered_map<std::string, std::string> map;
  std::atomic<size_t> count{0};
};

MPSAutotuneOverrides& overrideState() {
  static auto* state = new MPSAutotuneOverrides();
  return *state;
}

std::atomic<uint64_t> cache_generation{0};

} // namespace

bool isMPSAutotuneTraceEnabled() {
  return traceState().recording.load(std::memory_order_acquire);
}

void startMPSAutotuneTrace(size_t max_entries) {
  TORCH_CHECK(max_entries > 0, "max_entries must be greater than zero");
  auto& state = traceState();
  std::unique_lock<std::mutex> guard(state.mutex);
  TORCH_CHECK(
      !state.recording.load(), "an MPS autotune trace is already active");
  state.records.clear();
  state.max_entries = max_entries;
  state.dropped = 0;
  state.sequence = 0;
  state.recording.store(true, std::memory_order_release);
}

MPSAutotuneSnapshot stopMPSAutotuneTrace() {
  auto& state = traceState();
  std::unique_lock<std::mutex> guard(state.mutex);
  TORCH_CHECK(state.recording.load(), "no MPS autotune trace is active");
  state.recording.store(false, std::memory_order_release);
  return {{state.records.begin(), state.records.end()}, state.dropped};
}

void recordMPSAutotuneEvent(MPSAutotuneRecord record) {
  auto& state = traceState();
  if (!state.recording.load(std::memory_order_acquire)) {
    return;
  }
  std::lock_guard<std::mutex> guard(state.mutex);
  if (!state.recording.load(std::memory_order_relaxed)) {
    return;
  }
  record.sequence = ++state.sequence;
  if (state.records.size() == state.max_entries) {
    state.records.pop_front();
    ++state.dropped;
  }
  state.records.push_back(std::move(record));
}

std::optional<std::string> getMPSAutotuneOverride(
    std::string_view operation) {
  auto& state = overrideState();
  if (state.count.load(std::memory_order_acquire) == 0) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> guard(state.mutex);
  const auto it = state.map.find(std::string(operation));
  if (it == state.map.end()) {
    return std::nullopt;
  }
  return it->second;
}

void setMPSAutotuneOverride(
    const std::string& operation,
    std::optional<std::string> config) {
  TORCH_CHECK(!operation.empty(), "operation must not be empty");
  auto& state = overrideState();
  std::lock_guard<std::mutex> guard(state.mutex);
  if (config.has_value()) {
    TORCH_CHECK(!config->empty(), "config must not be empty");
    state.map.insert_or_assign(operation, std::move(*config));
  } else {
    state.map.erase(operation);
  }
  state.count.store(state.map.size(), std::memory_order_release);
}

uint64_t getMPSAutotuneCacheGeneration() {
  return cache_generation.load(std::memory_order_acquire);
}

void clearMPSAutotuneCaches() {
  cache_generation.fetch_add(1, std::memory_order_acq_rel);
}

} // namespace at::mps
