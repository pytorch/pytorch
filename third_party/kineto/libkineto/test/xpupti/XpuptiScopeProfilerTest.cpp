/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiTestUtilities.h"

#include "src/plugin/xpupti/XpuptiActivityProfiler.h"
#include "src/plugin/xpupti/XpuptiProfilerMacros.h"
#include "src/plugin/xpupti/XpuptiScopeProfilerConfig.h"

#include <libkineto.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <gtest/gtest.h>

namespace KN = KINETO_NAMESPACE;

class XpuptiScopeProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    KN::XpuptiScopeProfilerConfig::registerFactory();
  }
};

void RunTest(std::string_view perKernel, unsigned maxScopes) {
  KN::Config cfg;

  std::vector<std::string_view> metrics = {
      "GpuTime",
      "GpuCoreClocks",
      "AvgGpuCoreFrequencyMHz",
      "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION",
      "XVE_ACTIVE",
      "XVE_STALL"};

  EXPECT_TRUE(cfg.parse(
      fmt::format("XPUPTI_PROFILER_METRICS = {}", fmt::join(metrics, ","))));
  EXPECT_TRUE(cfg.parse(
      fmt::format("XPUPTI_PROFILER_ENABLE_PER_KERNEL = {}", perKernel)));
  EXPECT_TRUE(
      cfg.parse(fmt::format("XPUPTI_PROFILER_MAX_SCOPES = {}", maxScopes)));

  std::set<KN::ActivityType> activities{
      KN::ActivityType::GPU_MEMCPY,
      KN::ActivityType::GPU_MEMSET,
      KN::ActivityType::CONCURRENT_KERNEL,
      KN::ActivityType::EXTERNAL_CORRELATION,
      KN::ActivityType::XPU_RUNTIME,
      KN::ActivityType::XPU_SCOPE_PROFILER};

  std::vector<std::string_view> expectedActivities = {
      "urEnqueueMemBufferWrite",
      "urEnqueueMemBufferWrite",
      "urEnqueueMemBufferWrite",
      "Memcpy M2D",
      "Memcpy M2D",
      "Memcpy M2D",
      "urEnqueueKernelLaunch",
      "Run(sycl::_V1::queue, ...)",
      "urEnqueueMemBufferRead",
      "Memcpy D2M",
      "metrics: Run(sycl::_V1::queue, ...)",
      "metrics",
      "metrics"};

  std::vector<std::string_view> expectedTypes = {
      "xpu_runtime",
      "xpu_runtime",
      "xpu_runtime",
      "gpu_memcpy",
      "gpu_memcpy",
      "gpu_memcpy",
      "xpu_runtime",
      "kernel",
      "xpu_runtime",
      "gpu_memcpy",
      "kernel",
      "xpu_scope_profiler",
      "xpu_scope_profiler"};

  std::exception_ptr eptr;

  try {
    constexpr unsigned repeatCount = 5;
    RunProfilerTest(
        metrics,
        activities,
        cfg,
        repeatCount,
        std::move(expectedActivities),
        std::move(expectedTypes));
  } catch (...) {
    eptr = std::current_exception();
  }

  bool expectThrow = (perKernel == "false");

  if (expectThrow) {
    EXPECT_THROW(
        try {
          if (eptr) {
            std::rethrow_exception(eptr);
          }
        } catch (const std::runtime_error& e) {
          static bool isVerbose = IsEnvVerbose();
          if (isVerbose) {
            std::cout << "std::runtime_error = " << e.what() << std::endl;
          }
          throw;
        },
        std::runtime_error);
  } else {
    if (eptr) {
      std::rethrow_exception(eptr);
    }
  }
}

/////////////////////////////////////////////////////////////////////

TEST_F(XpuptiScopeProfilerTest, PerKernelScope) {
  RunTest("true", 314);
}

TEST_F(XpuptiScopeProfilerTest, UserScope) {
  RunTest("false", 159);
}
