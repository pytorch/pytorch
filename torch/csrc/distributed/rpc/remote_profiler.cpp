// Copyright 2004-present Facebook. All Rights Reserved.

#include <torch/csrc/distributed/rpc/remote_profiler.h>
#include <tensorpipe/core/message.h>
#include <torch/csrc/distributed/rpc/remote_profiler.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch {
namespace distributed {
namespace rpc {
thread_local c10::optional<std::string> RemoteProfiler::currentThreadLocalKey_ =
    c10::nullopt;
/*static */ RemoteProfiler& RemoteProfiler::getInstance() {
  static RemoteProfiler* handler = new RemoteProfiler();
  return *handler;
}

void RemoteProfiler::setCurrentKey(const std::string key) {
  // We should not allow overriding the current key, it needs to be committed
  // with writeKey() explicitly first.
  if (RemoteProfiler::currentThreadLocalKey_) {
    TORCH_CHECK(
        false,
        "Cannot call RemoteProfiler::setCurrentKey when current key is already set.");
  }
  currentThreadLocalKey_ = std::move(key);
}

std::string RemoteProfiler::getCurrentProfilingKey() {
  TORCH_CHECK(
      RemoteProfiler::currentThreadLocalKey_,
      "Must set currentThreadLocalKey_ before calling getCurrentProfilingKey");
  return *currentThreadLocalKey_;
}

std::unordered_map<std::string, std::string> RemoteProfiler::
    getProfiledEvents() {
  std::lock_guard<std::mutex> guard(profilerEventsMutex_);
  return profilerEvents_;
}

void RemoteProfiler::writeKey() {
  TORCH_CHECK(
      RemoteProfiler::currentThreadLocalKey_,
      "Must set current key with setCurrentKey.");
  std::lock_guard<std::mutex> guard(profilerEventsMutex_);
  profilerEvents_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(std::string(*currentThreadLocalKey_)),
      std::forward_as_tuple(std::string("")));
  // Since we have committed this key to the in-memory map, we can now allow a
  // newly-set current key.
  currentThreadLocalKey_ = c10::nullopt;
}

void RemoteProfiler::setValue(const std::string& key, const std::string value) {
  std::lock_guard<std::mutex> guard(profilerEventsMutex_);
  auto it = profilerEvents_.find(key);
  TORCH_CHECK(
      it != profilerEvents_.end(),
      c10::str("Remote profiling key ", key, " not found in map"));
  it->second = std::move(value);
}

RemoteProfiler::RemoteProfiler() {}
} // namespace rpc
} // namespace distributed
} // namespace torch
