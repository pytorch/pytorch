/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>

#include "include/libkineto.h"
#include "src/ConfigLoader.h"
#include "src/DaemonConfigLoader.h"

namespace KINETO_NAMESPACE {
namespace {

// Stands in for a platform's own daemon config loader and reads no config.
struct NoopDaemonConfigLoader final : IDaemonConfigLoader {
  std::string readBaseConfig() override {
    return "";
  }

  std::string readOnDemandConfig(bool /*activities*/) override {
    return "";
  }

  void setCommunicationFabric(bool /*enabled*/) override {}
};

// Uses the process-global libkineto::api(), so it needs a fresh process.
TEST(InitTest, UseDaemonInitializesRegisteredProfiler) {
  ::setenv("KINETO_USE_DAEMON", "1", 1);
  ConfigLoader::setDaemonConfigLoaderFactory(
      [] { return std::make_unique<NoopDaemonConfigLoader>(); });
  auto& api = libkineto::api();
  libkineto_init(/*cpuOnly=*/true, /*logOnError=*/true);
  EXPECT_TRUE(api.isProfilerInitialized());
  const auto* profiler = &api.activityProfiler();
  libkineto_init(/*cpuOnly=*/true, /*logOnError=*/true);
  EXPECT_EQ(&api.activityProfiler(), profiler);
}

} // namespace
} // namespace KINETO_NAMESPACE
