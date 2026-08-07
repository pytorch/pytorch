/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "src/ConfigLoader.h"
#include "src/DaemonConfigLoader.h"
#include "src/ThrowUtil.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace KINETO_NAMESPACE {
namespace {

struct NoopConfigHandler final : ConfigLoader::ConfigHandler {
  bool canAcceptConfig() override {
    return true;
  }
  bool acceptConfig(const Config& /*cfg*/) override {
    return true;
  }
};

// One-shot handshake the poll thread raises once it reaches readOnDemandConfig.
// post() is idempotent so the poll loop firing it on repeated iterations is
// fine.
struct OnDemandSignal {
  void post() {
    {
      std::scoped_lock lock(mutex_);
      reached_ = true;
    }
    cv_.notify_all();
  }

  bool waitFor(std::chrono::seconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return reached_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool reached_ = false;
};

// Throws out of every daemon call the poll thread makes. readBaseConfig() feeds
// the base-config branch; readOnDemandConfig() feeds the on-demand branch and
// posts onDemandReached so the test can observe that the loop got there.
struct ThrowingDaemonConfigLoader final : IDaemonConfigLoader {
  explicit ThrowingDaemonConfigLoader(
      std::shared_ptr<OnDemandSignal> onDemandReached)
      : onDemandReached_(std::move(onDemandReached)) {}

  [[noreturn]] std::string readBaseConfig() override {
    KINETO_THROW(std::runtime_error, "injected base-config failure");
  }

  [[noreturn]] std::string readOnDemandConfig(bool /*activities*/) override {
    onDemandReached_->post();
    KINETO_THROW(std::runtime_error, "injected on-demand failure");
  }

  void setCommunicationFabric(bool /*enabled*/) override {}

 private:
  std::shared_ptr<OnDemandSignal> onDemandReached_;
};

// Drives the process-global ConfigLoader singleton and sets KINETO_CONFIG,
// which configFileName() reads once and memoizes. It assumes a fresh process
// (tpx runs each gtest case in its own), so the singleton starts clean here.
//
// The poll thread runs updateBaseConfig()/configureFromDaemon() on a bare
// std::thread, where an escaped exception calls std::terminate() and aborts the
// whole process -- the fleet SIGABRT this guard fixed. Assert generically that
// an exception from the config work is swallowed and the loop keeps running,
// without reproducing any specific failure site.
TEST(ConfigLoaderPollThreadExceptionTest, ConfigUpdateExceptionDoesNotCrash) {
  // Non-existent path so updateBaseConfig() falls through to the daemon loader.
  ::setenv("KINETO_CONFIG", "/tmp/nonexistent_libkineto_test.conf", 1);

  auto onDemandReached = std::make_shared<OnDemandSignal>();
  ConfigLoader::setDaemonConfigLoaderFactory([onDemandReached] {
    return std::make_unique<ThrowingDaemonConfigLoader>(onDemandReached);
  });

  NoopConfigHandler handler;
  // Stops and joins the poll thread before `handler` is destroyed, on every
  // exit path including an ASSERT early-return, so the thread can never touch
  // the freed stack handler. Declared after `handler` so it runs before
  // handler's destructor.
  struct ThreadStopper {
    ConfigLoader::ConfigHandler* handler;
    ~ThreadStopper() {
      ConfigLoader::instance().stopUpdateThread();
      ConfigLoader::instance().removeHandler(
          ConfigLoader::ConfigKind::ActivityProfiler, handler);
      ConfigLoader::setDaemonConfigLoaderFactory(nullptr);
    }
  } threadStopper{&handler};

  // Starts the poll thread; its first iteration fires both branches
  // immediately.
  ConfigLoader::instance().addHandler(
      ConfigLoader::ConfigKind::ActivityProfiler, &handler);

  // readBaseConfig() throws first. Reaching readOnDemandConfig() -- which posts
  // this signal -- is only possible if the base-config branch caught the
  // exception and the loop continued. Deterministic handshake, no sleeps.
  // readOnDemandConfig() then throws too, so the clean join in ThreadStopper
  // (no std::terminate) also shows the on-demand branch caught it.
  ASSERT_TRUE(onDemandReached->waitFor(std::chrono::seconds(30)))
      << "poll thread did not survive the base-config exception";
}

} // namespace
} // namespace KINETO_NAMESPACE
