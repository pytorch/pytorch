/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/Config.h"
#include "src/ConfigLoader.h"
#include "src/DaemonConfigLoader.h"

using namespace KINETO_NAMESPACE;

namespace {

// Records how ConfigLoader dispatches to a handler and lets a test control
// whether canAcceptConfig() accepts. When the real poll thread is running,
// set canAcceptResult before starting it and do not mutate these fields while
// it runs.
struct RecordingConfigHandler : ConfigLoader::ConfigHandler {
  bool canAcceptResult{true};
  int acceptCalls{0};
  const Config* lastAcceptedConfig{nullptr};

  // Copied at accept time. The daemon-poll path dispatches a Config that lives
  // only for the poll iteration, so lastAcceptedConfig dangles afterward; a
  // value copy of a parsed field stays valid for assertions.
  std::string lastRequestTraceID;

  bool canAcceptConfig() override {
    return canAcceptResult;
  }

  bool acceptConfig(const Config& cfg) override {
    ++acceptCalls;
    lastAcceptedConfig = &cfg;
    lastRequestTraceID = cfg.requestTraceID();
    return true;
  }
};

// Canned daemon responses and recorded queries. Owned by the test; the fake
// below holds a reference to it. The poll thread writes the recorded fields, so
// a test reads them only after stopping (joining) the thread.
struct DaemonPollProbe {
  std::string onDemandConfig;
  int readOnDemandCalls{0};
  bool lastActivitiesRequested{false};
};

// Stands in for the dynolog IPC config source, so the real background poll
// thread runs against canned configs with no daemon.
class FakeDaemonConfigLoader : public IDaemonConfigLoader {
 public:
  explicit FakeDaemonConfigLoader(DaemonPollProbe& probe) : probe_(probe) {}

  std::string readBaseConfig() override {
    return "";
  }

  std::string readOnDemandConfig(bool activities) override {
    ++probe_.readOnDemandCalls;
    probe_.lastActivitiesRequested = activities;
    return probe_.onDemandConfig;
  }

  void setCommunicationFabric(bool /*enabled*/) override {}

 private:
  DaemonPollProbe& probe_;
};

// Drives the ConfigLoader singleton. Two styles of test live here:
//   * Handler fan-out tests call notifyHandlers()/canHandlerAcceptConfig()
//     directly and install no daemon factory, so the background poll thread
//     (started by addHandler) reads nothing and never calls the handlers.
//   * Daemon-poll tests install a FakeDaemonConfigLoader factory, start the
//     real poll thread, and wait on the thread's iteration counter to observe
//     real poll iterations deterministically.
// The singleton persists across tests, so TearDown stops the thread, drops the
// injected loader, clears the factory, and unregisters handlers.
class ConfigLoaderTest : public ::testing::Test {
 protected:
  static ConfigLoader& loader() {
    return ConfigLoader::instance();
  }

  // Installs a fake daemon config source. Call before starting the poll thread
  // (before registering a handler) so the thread's first iteration uses it.
  static void installFakeDaemon(DaemonPollProbe& probe) {
    ConfigLoader::setDaemonConfigLoaderFactory(
        [&probe]() { return std::make_unique<FakeDaemonConfigLoader>(probe); });
  }

  // Blocks until the thread's iteration counter reaches target or the timeout
  // elapses. Returns false on timeout.
  [[nodiscard]] static bool waitForLoopCount(
      uint64_t target,
      std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    return loader().waitForUpdateThreadLoopCountForTesting(target, timeout);
  }

  void registerHandler(
      ConfigLoader::ConfigKind kind,
      ConfigLoader::ConfigHandler* handler) {
    loader().addHandler(kind, handler);
    registered_.emplace_back(kind, handler);
  }

  // Registers handler as an ActivityProfiler handler (which starts the poll
  // thread) and blocks until the thread has completed loops iterations.
  // Returns false if that count is not reached before the timeout.
  [[nodiscard]] bool startPollingAndWait(
      RecordingConfigHandler& handler,
      uint64_t loops) {
    const uint64_t base = loader().updateThreadLoopCountForTesting();
    registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &handler);
    return waitForLoopCount(base + loops);
  }

  void TearDown() override {
    // Join the poll thread first, so nothing touches the loader afterward.
    loader().stopUpdateThread();

    // Drop the injected loader and factory: both capture this test's probe,
    // which does not outlive the test.
    loader().resetDaemonConfigLoaderForTesting();
    ConfigLoader::setDaemonConfigLoaderFactory(nullptr);

    // Clear the cached base config. The singleton reloads the base config only
    // when it changes, so once an earlier test primes it (to a local config
    // file, if present) a later test in the same process would see no change,
    // never rebuild the daemon loader from its factory, and never poll the
    // fake. gtest_discover_tests masks this by running each test in its own
    // process.
    loader().resetBaseConfigForTesting();

    // removeHandler is a no-op for a handler already removed by the test, so
    // double removal is safe.
    for (const auto& [kind, handler] : registered_) {
      loader().removeHandler(kind, handler);
    }
    registered_.clear();
  }

 private:
  std::vector<std::pair<ConfigLoader::ConfigKind, ConfigLoader::ConfigHandler*>>
      registered_;
};

// ---- Handler fan-out ----

// notifyHandlers() forwards the config to every registered handler across all
// config kinds, passing through the same config object.
TEST_F(ConfigLoaderTest, NotifyHandlersForwardsConfigToAllRegisteredHandlers) {
  RecordingConfigHandler activityA;
  RecordingConfigHandler activityB;
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &activityA);
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &activityB);

  Config cfg;
  loader().notifyHandlers(cfg);

  EXPECT_EQ(activityA.acceptCalls, 1);
  EXPECT_EQ(activityB.acceptCalls, 1);
  EXPECT_EQ(activityA.lastAcceptedConfig, &cfg);
  EXPECT_EQ(activityB.lastAcceptedConfig, &cfg);
}

// A removed handler no longer receives configs.
TEST_F(ConfigLoaderTest, RemoveHandlerStopsDispatch) {
  RecordingConfigHandler handler;
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &handler);

  Config first;
  loader().notifyHandlers(first);
  EXPECT_EQ(handler.acceptCalls, 1);

  loader().removeHandler(ConfigLoader::ConfigKind::ActivityProfiler, &handler);

  Config second;
  loader().notifyHandlers(second);
  EXPECT_EQ(handler.acceptCalls, 1); // unchanged: no longer registered
}

// canHandlerAcceptConfig() is true only when every handler of that kind
// accepts.
TEST_F(ConfigLoaderTest, CanHandlerAcceptConfigRequiresAllHandlersOfKind) {
  const auto kind = ConfigLoader::ConfigKind::ActivityProfiler;
  RecordingConfigHandler first;
  RecordingConfigHandler second;
  registerHandler(kind, &first);
  registerHandler(kind, &second);

  EXPECT_TRUE(loader().canHandlerAcceptConfig(kind));

  second.canAcceptResult = false;
  EXPECT_FALSE(loader().canHandlerAcceptConfig(kind));
}

// canHandlerAcceptConfig() accepts vacuously for a config kind that has no
// registered handlers.
TEST_F(ConfigLoaderTest, CanHandlerAcceptConfigVacuouslyTrueWithNoHandlers) {
  const auto kind = ConfigLoader::ConfigKind::ActivityProfiler;
  EXPECT_TRUE(loader().canHandlerAcceptConfig(kind));
}

// ---- Real daemon poll thread ----
//
// These tests run the actual updateConfigThread() against a fake daemon and use
// the iteration counter to synchronize. They tolerate a pre-existing local
// config file (/etc/libkineto.conf or $KINETO_CONFIG) on the test system: each
// test starts from a cleared base config (see TearDown), so the poll thread
// always re-detects a base-config change and rebuilds the fake daemon loader.

// The poll thread reads the on-demand config from the daemon, parses it, and
// dispatches it to registered handlers, requesting activities while the handler
// can accept.
TEST_F(ConfigLoaderTest, ThreadPollsDaemonAndDispatchesOnDemandConfig) {
  DaemonPollProbe probe;
  probe.onDemandConfig = "REQUEST_TRACE_ID=daemon-trace-42\n";
  installFakeDaemon(probe);

  RecordingConfigHandler handler;
  ASSERT_TRUE(startPollingAndWait(handler, /*loops=*/1));
  loader().stopUpdateThread();

  EXPECT_GE(probe.readOnDemandCalls, 1);
  EXPECT_TRUE(probe.lastActivitiesRequested);
  EXPECT_GE(handler.acceptCalls, 1);
  EXPECT_EQ(handler.lastRequestTraceID, "daemon-trace-42");
}

// The thread re-polls the on-demand config on its update cadence, so over a
// couple of intervals it reads the daemon more than once.
//
// This runs at the real cadence, so it takes a few seconds. It cannot be sped
// up by injecting a smaller ON_DEMAND_CONFIG_UPDATE_INTERVAL_SECS via the fake
// daemon's base config: updateBaseConfig reads the local config file first and
// uses the daemon base config only when the local read is empty, so a present
// local config (as on the test hosts) shadows the fake. If that local-first
// ordering is ever inverted (see the TODO in updateBaseConfig), injecting a
// faster interval would let this test finish without the wait.
TEST_F(ConfigLoaderTest, ThreadRepeatedlyPollsOnDemandConfig) {
  DaemonPollProbe probe;
  probe.onDemandConfig = "REQUEST_TRACE_ID=x\n";
  installFakeDaemon(probe);

  RecordingConfigHandler handler;
  const uint64_t base = loader().updateThreadLoopCountForTesting();
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &handler);

  // The first poll is immediate; after it, the thread's on-demand interval
  // reflects the loaded base config. Size the wait for the second poll to that
  // live interval (with slack) so a host that configured a larger interval does
  // not cause a spurious timeout.
  ASSERT_TRUE(waitForLoopCount(base + 1));
  const auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(
      loader().onDemandConfigUpdateIntervalForTesting() * 3 +
      std::chrono::seconds(5));
  ASSERT_TRUE(waitForLoopCount(base + 2, timeout));
  loader().stopUpdateThread();

  EXPECT_GE(probe.readOnDemandCalls, 2);
}

// An empty on-demand response (no config posted) dispatches nothing.
TEST_F(ConfigLoaderTest, ThreadDropsEmptyOnDemandConfig) {
  DaemonPollProbe probe; // onDemandConfig defaults empty
  installFakeDaemon(probe);

  RecordingConfigHandler handler;
  ASSERT_TRUE(startPollingAndWait(handler, /*loops=*/1));
  loader().stopUpdateThread();

  EXPECT_GE(probe.readOnDemandCalls, 1);
  EXPECT_EQ(handler.acceptCalls, 0);
}

// The thread requests an activities config only while its handlers can accept
// one; a busy handler suppresses the activities request.
TEST_F(ConfigLoaderTest, ThreadSuppressesActivitiesRequestWhenHandlerBusy) {
  DaemonPollProbe probe;
  probe.onDemandConfig = "REQUEST_TRACE_ID=x\n";
  installFakeDaemon(probe);

  RecordingConfigHandler handler;
  handler.canAcceptResult = false; // set before the thread starts
  ASSERT_TRUE(startPollingAndWait(handler, /*loops=*/1));
  loader().stopUpdateThread();

  EXPECT_GE(probe.readOnDemandCalls, 1);
  EXPECT_FALSE(probe.lastActivitiesRequested);
}

// With no daemon factory installed (the default on non-daemon hosts), the poll
// thread reads no on-demand config and dispatches nothing.
TEST_F(ConfigLoaderTest, ThreadWithoutDaemonFactoryDispatchesNothing) {
  RecordingConfigHandler handler;
  ASSERT_TRUE(startPollingAndWait(handler, /*loops=*/1));
  loader().stopUpdateThread();

  EXPECT_EQ(handler.acceptCalls, 0);
}

} // namespace
