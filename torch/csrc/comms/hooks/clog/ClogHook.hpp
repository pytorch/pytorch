// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <torch/csrc/comms/RemovableHandle.hpp>
#include <torch/csrc/comms/TorchCommHooks.hpp>
#include <torch/csrc/comms/TorchCommTypes.hpp>
#include <torch/csrc/comms/hooks/common/SignatureBuilder.hpp>
#include <torch/csrc/comms/hooks/common/ThreadSafeLogFile.hpp>

namespace torch::comms {

// Forward declarations
class TorchComm;

using WorkId = uint64_t;
inline constexpr WorkId kWorkIdInvalid = std::numeric_limits<WorkId>::max();

template <typename K, typename V>
class ThreadSafeMap {
 public:
  void insert(const K& key, V value) {
    std::lock_guard<std::mutex> lock(mutex_);
    map_[key] = std::move(value);
  }

  V findOrInsert(const K& key, const V& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.try_emplace(key, value).first->second;
  }

  std::optional<V> find(const K& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::optional<V> findAndErase(const K& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return std::nullopt;
    }
    V val = std::move(it->second);
    map_.erase(it);
    return val;
  }

  template <typename F>
  void insertOrModify(const K& key, F&& fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    fn(map_[key]);
  }

  template <typename E>
  void valuePush(const K& key, E&& element) {
    static_assert(
        std::is_same_v<V, std::vector<typename std::decay_t<E>>>,
        "valuePush requires a vector value type");
    std::lock_guard<std::mutex> lock(mutex_);
    map_[key].push_back(std::forward<E>(element));
  }

  template <typename E>
  bool valueRemove(const K& key, const E& element) {
    static_assert(
        std::is_same_v<V, std::vector<E>>,
        "valueRemove requires a vector value type");
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return false;
    }
    auto& vec = it->second;
    auto ev_it = std::find(vec.begin(), vec.end(), element);
    if (ev_it == vec.end()) {
      return false;
    }
    vec.erase(ev_it);
    if (vec.empty()) {
      map_.erase(it);
    }
    return true;
  }

  template <typename F>
  void forEach(F&& fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [key, value] : map_) {
      fn(key, value);
    }
  }

 private:
  using Map = std::unordered_map<K, V>;
  Map map_;
  std::mutex mutex_;
};

// ClogHook: Hook-based logger for torchcomms collective operations.
//
// Registers pre-hook and post-hook callbacks on communicators to log
// collective operation signatures and lifecycle events to a pipe-delimited
// log file.
//
// Usage:
//   auto logger = std::make_shared<ClogHook>(
//       "/tmp/clog.log", {"ALL"});
//   logger->registerWithComm(comm_a);
//   logger->registerWithComm(comm_b);
//
// Constructor parameters:
//   output         - File path for log output
//   events         - Events to log: Q, S, E, W, ALL, NONE
//   verbose        - Optional fields: buffers
//
// Log format:
//   Non-graph:     C<id>|<event>|+<ts>
//   Graph capture: G<gid>|C<id>|<event>|+<ts>
//   Graph replay:  G<gid>|R<rid>|C<id>|<event>|+<ts>
//
class ClogHook : public std::enable_shared_from_this<ClogHook> {
 public:
  ClogHook(
      const std::string& output,
      const std::vector<std::string>& events,
      const std::vector<std::string>& verbose = {});

  ~ClogHook();

  ClogHook(const ClogHook&) = delete;
  ClogHook& operator=(const ClogHook&) = delete;
  ClogHook(ClogHook&&) = delete;
  ClogHook& operator=(ClogHook&&) = delete;

  // Register this hook with a communicator. Captures the comm name
  // and registers pre/post hooks for logging.
  void registerWithComm(std::shared_ptr<TorchComm> comm);

 private:
  void registerHooks(std::shared_ptr<TorchComm> comm);

  // Pre-hook: log collective signature and enqueue event.
  void onPreHook(
      const std::string& comm_name,
      int device_index,
      size_t op_id,
      const PreHookArgs& args);

  // Post-hook: register work lifecycle hooks for S/E/W events.
  void onPostHook(
      const std::string& comm_name,
      size_t op_id,
      const PostHookArgs& args);

  // -- Formatting helpers --
  static double now();

  void logEvent(uint64_t corr_id, std::string_view event);
  void logGraphEvent(
      uint64_t graph_id,
      uint64_t corr_id,
      std::string_view event);

  // Internal emitter: dedup, enqueue event, work mapping.
  static constexpr uint64_t kNoGraphCapture = UINT64_MAX;

  WorkId logCollective(
      std::string_view comm_name,
      std::string sig_body,
      bool async_op,
      void* stream,
      uint64_t graph_id);

  // Lifecycle event callbacks (registered on work objects via post-hook).
  void logLifecycleEvent(WorkId work_id, std::string_view event);

  struct GraphCaptureInfo {
    void* stream{nullptr};
    uint64_t graph_id{kNoGraphCapture};
  };
  static GraphCaptureInfo getGraphCaptureInfo(int device_index);

  // Resolve a stream from the replay hook to ClogHook's stream key.
  // If the stream is known (user stream), return it directly.
  // If unknown (internal comm stream), return the comm's fake stream.
  void* resolveReplayStream(
      uint64_t graph_id,
      const std::string& comm_name,
      void* stream);

  bool log_buffers_{false};
  bool log_lifecycle_{false};
  double base_ts_{0.0};
  std::atomic<uint64_t> next_corr_id_{1};
  std::atomic<uint64_t> next_work_id_{1};

  ThreadSafeLogFile log_file_;

  // Signature dedup: sig key (includes comm name) -> corr_id
  ThreadSafeMap<std::string, uint64_t> sig_map_;

  // work_id -> (corr_id, graph_id) for lifecycle event logging
  struct WorkInfo {
    uint64_t corr_id{};
    uint64_t graph_id{kNoGraphCapture};
    std::string comm_name;
  };
  ThreadSafeMap<uint64_t, WorkInfo> work_corr_map_;

  // work_id -> remaining expected events (entry auto-erased when vector
  // becomes empty via valueRemove)
  ThreadSafeMap<uint64_t, std::vector<std::string>> work_events_map_;

  // op_id -> work_id mapping (pre-hook stores, post-hook consumes)
  ThreadSafeMap<size_t, WorkId> op_to_work_;

  // Per-graph, per-stream ordered list of correlation IDs (built during
  // capture). Streams are either real user streams or fake streams
  // representing comm-internal async streams.
  // Used by the graph replay hook to map
  // (graph_id, stream, index) -> corr_id.
  struct GraphCollective {
    std::string comm_name;
    uint64_t corr_id;
  };
  using StreamCollectives =
      std::unordered_map<void*, std::vector<GraphCollective>>;
  ThreadSafeMap<uint64_t, StreamCollectives> graph_collectives_;

  // comm_name -> fake stream pointer for async ops during graph capture.
  // Fake streams are (void*)(-1), (void*)(-2), etc.
  ThreadSafeMap<std::string, void*> comm_fake_streams_;
  std::atomic<uintptr_t> next_fake_stream_{1};

  void* getFakeStream(const std::string& comm_name);

  // Graph replay hook callback — registered with each comm to receive
  // replay events from the backend's watchdog thread.
  void onGraphReplayEvent(
      const std::string& comm_name,
      uint64_t graph_id,
      uint64_t replay_id,
      void* stream,
      size_t collective_index,
      std::string_view event);

  // Registration tracking for cleanup (main thread only, no lock needed)
  std::atomic<size_t> active_comm_count_{0};

  struct CommRegistration {
    std::weak_ptr<TorchComm> comm;
  };
  std::vector<CommRegistration> registrations_;
};

} // namespace torch::comms
