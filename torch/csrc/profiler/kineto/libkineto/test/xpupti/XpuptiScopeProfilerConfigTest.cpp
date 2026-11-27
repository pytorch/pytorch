/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "src/plugin/xpupti/XpuptiScopeProfilerConfig.h"
#include "src/plugin/xpupti/XpuptiScopeProfilerApi.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <gtest/gtest.h>

#include <array>

namespace KN = KINETO_NAMESPACE;

class XpuptiScopeProfilerConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    KN::XpuptiScopeProfilerConfig::registerFactory();
  }
};

TEST_F(XpuptiScopeProfilerConfigTest, ConfigureProfiler) {
  KN::Config cfg;
  std::vector<std::string> metrics = {
      "metric1",
      "metric2",
      "metric3",
  };
  auto metricsConfigStr =
      fmt::format("XPUPTI_PROFILER_METRICS = {}", fmt::join(metrics, ","));

  EXPECT_TRUE(cfg.parse(metricsConfigStr));
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_ENABLE_PER_KERNEL = true"));
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_MAX_SCOPES = 314159"));

  const KN::XpuptiScopeProfilerConfig& xpupti_cfg =
      KN::XpuptiScopeProfilerConfig::get(cfg);

  EXPECT_EQ(xpupti_cfg.activitiesXpuptiMetrics(), metrics);
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerPerKernel(), true);
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerMaxScopes(), 314159);
}

TEST_F(XpuptiScopeProfilerConfigTest, ScopesDefaults) {
  KN::Config cfg, cfg_auto;

  // do not set max scopes in config, check defaults are sane
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_METRICS = metric1"));
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_ENABLE_PER_KERNEL = false"));

  cfg.setClientDefaults();

  EXPECT_TRUE(cfg_auto.parse("XPUPTI_PROFILER_METRICS = metric2"));
  EXPECT_TRUE(cfg_auto.parse("XPUPTI_PROFILER_ENABLE_PER_KERNEL = true"));

  cfg_auto.setClientDefaults();

  int user_scopes, auto_scopes;

  user_scopes =
      KN::XpuptiScopeProfilerConfig::get(cfg).xpuptiProfilerMaxScopes();
  auto_scopes =
      KN::XpuptiScopeProfilerConfig::get(cfg_auto).xpuptiProfilerMaxScopes();

  EXPECT_EQ(user_scopes, 10);
  EXPECT_EQ(auto_scopes, 1500);
}

TEST_F(XpuptiScopeProfilerConfigTest, DevicesDefaultEmpty) {
  KN::Config cfg;
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_METRICS = metric1"));
  // No XPUPTI_PROFILER_DEVICES set -> empty means "all devices".
  const KN::XpuptiScopeProfilerConfig& xpupti_cfg =
      KN::XpuptiScopeProfilerConfig::get(cfg);
  EXPECT_TRUE(xpupti_cfg.xpuptiProfilerDevices().empty());
}

TEST_F(XpuptiScopeProfilerConfigTest, DevicesParsedList) {
  KN::Config cfg;
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_METRICS = metric1"));
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_DEVICES = 0, 2, 3"));
  const KN::XpuptiScopeProfilerConfig& xpupti_cfg =
      KN::XpuptiScopeProfilerConfig::get(cfg);
  const std::vector<int> expected{0, 2, 3};
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerDevices(), expected);
}

TEST_F(XpuptiScopeProfilerConfigTest, DevicesSingle) {
  KN::Config cfg;
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_DEVICES = 1"));
  const KN::XpuptiScopeProfilerConfig& xpupti_cfg =
      KN::XpuptiScopeProfilerConfig::get(cfg);
  const std::vector<int> expected{1};
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerDevices(), expected);
}

TEST_F(XpuptiScopeProfilerConfigTest, DevicesEmptyTokensSkipped) {
  KN::Config cfg;
  // Doubled and trailing commas produce empty tokens: they must be skipped,
  // not parsed as index 0 nor rejected as invalid integers.
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_DEVICES = 0, ,2,"));
  const KN::XpuptiScopeProfilerConfig& xpupti_cfg =
      KN::XpuptiScopeProfilerConfig::get(cfg);
  const std::vector<int> expected{0, 2};
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerDevices(), expected);
}

TEST_F(XpuptiScopeProfilerConfigTest, DevicesDuplicatesDropped) {
  KN::Config cfg;
  // Duplicate indices are collapsed, preserving first-seen order.
  EXPECT_TRUE(cfg.parse("XPUPTI_PROFILER_DEVICES = 0,0,2,2,0"));
  const KN::XpuptiScopeProfilerConfig& xpupti_cfg =
      KN::XpuptiScopeProfilerConfig::get(cfg);
  const std::vector<int> expected{0, 2};
  EXPECT_EQ(xpupti_cfg.xpuptiProfilerDevices(), expected);
}

TEST_F(XpuptiScopeProfilerConfigTest, SelectDeviceHandlesSubset) {
  // Fabricate 4 distinct opaque handles; the helper never dereferences them.
  std::array<pti_device_handle_t, 4> handles{
      reinterpret_cast<pti_device_handle_t>(0x10),
      reinterpret_cast<pti_device_handle_t>(0x20),
      reinterpret_cast<pti_device_handle_t>(0x30),
      reinterpret_cast<pti_device_handle_t>(0x40)};
  const std::vector<int> indices{0, 2, 3};
  auto out = KN::selectDeviceHandles(handles, indices);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], handles[0]);
  EXPECT_EQ(out[1], handles[2]);
  EXPECT_EQ(out[2], handles[3]);
}

TEST_F(XpuptiScopeProfilerConfigTest, SelectDeviceHandlesOutOfRangeThrows) {
  std::array<pti_device_handle_t, 2> handles{
      reinterpret_cast<pti_device_handle_t>(0x10),
      reinterpret_cast<pti_device_handle_t>(0x20)};
  const std::vector<int> indices{0, 5};
  EXPECT_THROW(KN::selectDeviceHandles(handles, indices), std::runtime_error);
}

TEST_F(XpuptiScopeProfilerConfigTest, SelectDeviceHandlesNegativeThrows) {
  std::array<pti_device_handle_t, 2> handles{
      reinterpret_cast<pti_device_handle_t>(0x10),
      reinterpret_cast<pti_device_handle_t>(0x20)};
  const std::vector<int> indices{-1};
  EXPECT_THROW(KN::selectDeviceHandles(handles, indices), std::runtime_error);
}
