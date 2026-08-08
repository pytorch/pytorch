/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "Config.h"

namespace libkineto {
class LibkinetoApi;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
class IDaemonConfigLoader;

class ConfigLoader {
 public:
  static ConfigLoader& instance();

  enum ConfigKind { ActivityProfiler = 0, NumConfigKinds };

  struct ConfigHandler {
    virtual ~ConfigHandler() = default;
    virtual bool canAcceptConfig() = 0;
    // Returns true if the handler accepted the config for scheduling, false if
    // it declined. Acceptance means the request was queued, NOT that profiling
    // will run: a queued request can still be dropped later on the profiler
    // thread (e.g. canStart() fails, or GPU buffers overflow during warmup).
    virtual bool acceptConfig(const Config& cfg) = 0;
  };

  void addHandler(ConfigKind kind, ConfigHandler* handler) {
    std::scoped_lock lock(updateThreadMutex_);
    handlers_[kind].push_back(handler);
    startThread();
  }

  void removeHandler(ConfigKind kind, ConfigHandler* handler) {
    std::scoped_lock lock(updateThreadMutex_);
    auto it =
        std::find(handlers_[kind].begin(), handlers_[kind].end(), handler);
    if (it != handlers_[kind].end()) {
      handlers_[kind].erase(it);
    }
  }

  void notifyHandlers(const Config& cfg) {
    std::scoped_lock lock(updateThreadMutex_);
    for (auto& key_val : handlers_) {
      for (ConfigHandler* handler : key_val.second) {
        handler->acceptConfig(cfg);
      }
    }
  }

  bool canHandlerAcceptConfig(ConfigKind kind) {
    std::scoped_lock lock(updateThreadMutex_);
    for (ConfigHandler* handler : handlers_[kind]) {
      if (!handler->canAcceptConfig()) {
        return false;
      }
    }
    return true;
  }

  std::unique_ptr<Config> getConfigCopy() {
    std::scoped_lock lock(configLock_);
    return config_->clone();
  }

  bool hasNewConfig(const Config& oldConfig);

  static void setDaemonConfigLoaderFactory(
      std::function<std::unique_ptr<IDaemonConfigLoader>()> factory);

  std::string getConfString();

  // Stops and joins the background config-update thread; a no-op if it was
  // never started. Exposed so a test that drives the singleton can tear the
  // thread down deterministically before the test process exits, rather than
  // leaving the join to run during static destruction.
  void stopUpdateThread();

  // Test-only. Returns how many iterations the background poll thread has
  // completed. A test that installs a fake daemon config loader can start the
  // thread and wait for this count to advance, then deterministically observe
  // the effects of a known number of real poll iterations. Loaded with acquire
  // ordering so that observing an advanced count also makes that iteration's
  // writes visible to the observer.
  [[nodiscard]] uint64_t updateThreadLoopCountForTesting() const {
    return updateThreadLoopCount_.load(std::memory_order_acquire);
  }

  // Test-only. Blocks until the poll thread's iteration count reaches target,
  // or the timeout elapses; returns true if the count was reached.
  bool waitForUpdateThreadLoopCountForTesting(
      uint64_t target,
      std::chrono::milliseconds timeout);

  // Test-only. Drops the cached daemon config loader so the next poll rebuilds
  // it from the currently registered factory. The loader is a member of this
  // process-wide singleton, so a test that injected a fake via the factory must
  // clear it (after stopping the thread) or a later test would reuse a loader
  // pointing at destroyed test state.
  void resetDaemonConfigLoaderForTesting();

  // Test-only. Clears the cached base config so the next poll thread sees the
  // base config as changed and rebuilds the daemon config loader from the
  // registered factory. The base config is a member of this process-wide
  // singleton and is reloaded only on change, so without this reset a later
  // test in the same process reuses an earlier test's base config, never
  // rebuilds the loader from its factory, and never reads the fake daemon. Call
  // after stopping the thread.
  void resetBaseConfigForTesting();

  // Test-only. Returns the on-demand poll interval the background thread is
  // currently using, taken from the loaded base config. Lets a test size a
  // timeout to the live cadence instead of assuming the default.
  std::chrono::seconds onDemandConfigUpdateIntervalForTesting();

 private:
  ConfigLoader();
  ~ConfigLoader();

  IDaemonConfigLoader* daemonConfigLoader();

  void startThread();
  void stopThread();
  void updateConfigThread();
  void updateBaseConfig();

  // Create configuration when receiving request from a daemon
  void configureFromDaemon(
      std::chrono::time_point<std::chrono::system_clock> now,
      Config& config);

  std::string readOnDemandConfigFromDaemon(
      std::chrono::time_point<std::chrono::system_clock> now);

  const char* customConfigFileName();

  std::mutex configLock_;
  std::unique_ptr<Config> config_;
  std::unique_ptr<IDaemonConfigLoader> daemonConfigLoader_;
  std::map<ConfigKind, std::vector<ConfigHandler*>> handlers_;

  std::chrono::seconds configUpdateIntervalSecs_;
  std::chrono::seconds onDemandConfigUpdateIntervalSecs_;
  std::unique_ptr<std::thread> updateThread_;
  std::condition_variable updateThreadCondVar_;
  std::mutex updateThreadMutex_;
  std::atomic_bool stopFlag_{false};

  // Incremented at the end of each updateConfigThread() iteration. Test-only
  // observation point; see updateThreadLoopCountForTesting(). loopCountCondVar_
  // is notified on each increment so a test can block until a target count.
  std::atomic<uint64_t> updateThreadLoopCount_{0};
  std::mutex loopCountMutex_;
  std::condition_variable loopCountCondVar_;
};

} // namespace KINETO_NAMESPACE
