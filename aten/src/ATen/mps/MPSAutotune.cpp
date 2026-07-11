#include <ATen/mps/MPSAutotune.h>

#include <c10/util/Exception.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace at::mps {
namespace {

struct MPSAutotuneTraceState {
  std::atomic<bool> enabled{false};
  std::atomic<bool> recording{false};
  std::mutex mutex;
  std::condition_variable completed;
  std::deque<MPSAutotuneRecord> records;
  size_t max_entries = 0;
  size_t dropped = 0;
  uint64_t sequence = 0;
  size_t pending_callbacks = 0;
};

MPSAutotuneTraceState& traceState() {
  // Metal completion handlers can still drain during static teardown.
  static auto* state = new MPSAutotuneTraceState();
  return *state;
}

thread_local std::unordered_map<std::string, std::string> autotune_overrides;
std::atomic<uint64_t> cache_generation{0};

} // namespace

bool isMPSAutotuneTraceEnabled() {
  return traceState().enabled.load(std::memory_order_acquire);
}

void startMPSAutotuneTrace(size_t max_entries) {
  TORCH_CHECK(max_entries > 0, "max_entries must be greater than zero");
  auto& state = traceState();
  std::lock_guard<std::mutex> guard(state.mutex);
  TORCH_CHECK(
      !state.recording.load() && state.pending_callbacks == 0,
      "an MPS autotune trace is already active");
  state.records.clear();
  state.max_entries = max_entries;
  state.dropped = 0;
  state.sequence = 0;
  state.recording.store(true, std::memory_order_release);
  state.enabled.store(true, std::memory_order_release);
}

MPSAutotuneSnapshot stopMPSAutotuneTrace(bool wait_for_callbacks) {
  auto& state = traceState();
  std::unique_lock<std::mutex> guard(state.mutex);
  TORCH_CHECK(state.recording.load(), "no MPS autotune trace is active");
  state.enabled.store(false, std::memory_order_release);
  if (wait_for_callbacks) {
    state.completed.wait(
        guard, [&] { return state.pending_callbacks == 0; });
  }
  state.recording.store(false, std::memory_order_release);
  return {{state.records.begin(), state.records.end()}, state.dropped};
}

void recordMPSAutotuneEvent(MPSAutotuneRecord record, bool retained) {
  auto& state = traceState();
  if (!state.recording.load(std::memory_order_acquire) ||
      (!retained && !state.enabled.load(std::memory_order_acquire))) {
    return;
  }
  std::lock_guard<std::mutex> guard(state.mutex);
  if (!state.recording.load(std::memory_order_relaxed) ||
      (!retained && !state.enabled.load(std::memory_order_relaxed))) {
    return;
  }
  record.sequence = ++state.sequence;
  if (state.records.size() == state.max_entries) {
    state.records.pop_front();
    ++state.dropped;
  }
  state.records.push_back(std::move(record));
}

bool retainMPSAutotuneTrace() {
  auto& state = traceState();
  if (!state.enabled.load(std::memory_order_acquire)) {
    return false;
  }
  std::lock_guard<std::mutex> guard(state.mutex);
  if (!state.enabled.load(std::memory_order_relaxed)) {
    return false;
  }
  ++state.pending_callbacks;
  return true;
}

void releaseMPSAutotuneTrace() {
  auto& state = traceState();
  std::lock_guard<std::mutex> guard(state.mutex);
  TORCH_INTERNAL_ASSERT(state.pending_callbacks > 0);
  if (--state.pending_callbacks == 0) {
    state.completed.notify_all();
  }
}

std::optional<std::string> getMPSAutotuneOverride(
    std::string_view operation) {
  for (const auto& [name, config] : autotune_overrides) {
    if (std::string_view(name) == operation) {
      return config;
    }
  }
  return std::nullopt;
}

void setMPSAutotuneOverride(
    const std::string& operation,
    std::optional<std::string> config) {
  TORCH_CHECK(!operation.empty(), "operation must not be empty");
  if (config.has_value()) {
    TORCH_CHECK(!config->empty(), "config must not be empty");
    autotune_overrides.insert_or_assign(operation, std::move(*config));
  } else {
    autotune_overrides.erase(operation);
  }
}

uint64_t getMPSAutotuneCacheGeneration() {
  return cache_generation.load(std::memory_order_acquire);
}

void clearMPSAutotuneCaches() {
  cache_generation.fetch_add(1, std::memory_order_acq_rel);
}

} // namespace at::mps
