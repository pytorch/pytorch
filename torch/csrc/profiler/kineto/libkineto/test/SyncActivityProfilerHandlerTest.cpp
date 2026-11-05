/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>

#include "include/Config.h"
#include "src/GenericActivityProfiler.h"
#include "src/SyncActivityProfilerHandler.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

TEST(SyncActivityProfilerHandler, Cancel) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  SyncActivityProfilerHandler handler(profiler);

  // Cancel when inactive is a no-op
  EXPECT_FALSE(handler.isSyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isSyncActive());

  // Cancel after prepareTrace
  Config cfg;
  cfg.validate(system_clock::now());
  handler.prepareTrace(cfg);
  EXPECT_TRUE(handler.isSyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isSyncActive());
}
