#pragma once
#include <c10/util/Optional.h>
#include <mutex>

namespace torch {
namespace distributed {
namespace rpc {
class RemoteProfiler {
 public:
  // Retrieves the lazily-initialized RemoteProfiler singleton instance.
  static RemoteProfiler& getInstance();
  // Sets the current, thread-local profiling key.
  void setCurrentKey(const std::string key);
  // Unsets the current key and writes the given key to the profilerEvents_ map.
  void writeKey();
  // Sets a value for the profiling key. Used when the client is processing a
  // received response with profiling metadata.
  void setValue(const std::string& key, const std::string value);
  // Retrieves the currently set profiling key.
  std::string getCurrentProfilingKey();
  // Retrieves a mapping of profling keys to their associated remote events.
  std::unordered_map<std::string, std::string> getProfiledEvents();

 private:
  RemoteProfiler();
  ~RemoteProfiler() = default;
  RemoteProfiler(const RemoteProfiler& other) = delete;
  RemoteProfiler operator=(const RemoteProfiler& other) = delete;
  RemoteProfiler(RemoteProfiler&&) = delete;
  RemoteProfiler& operator=(RemoteProfiler&&) = delete;
  std::unordered_map<std::string, std::string> profilerEvents_;
  static thread_local c10::optional<std::string> currentThreadLocalKey_;
  // A mutex to guard access to profilerEvents, since multiple threads may
  // attempt to read/write them at the same time.
  std::mutex profilerEventsMutex_;
};
} // namespace rpc
} // namespace distributed
} // namespace torch
