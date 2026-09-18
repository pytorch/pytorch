/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/Config.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <chrono>
#include <ctime>

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

TEST(ParseTest, Whitespace) {
  Config cfg;
  // Check that various types of whitespace is ignored
  EXPECT_TRUE(cfg.parse(""));
  EXPECT_TRUE(cfg.parse(" "));
  EXPECT_TRUE(cfg.parse("\t"));
  EXPECT_TRUE(cfg.parse("\n"));
  EXPECT_TRUE(cfg.parse("    "));
  EXPECT_TRUE(cfg.parse("\t \n  \t\t\n\n"));
  // Only the above characters are supported
  EXPECT_FALSE(cfg.parse("\r\n"));
}

TEST(ParseTest, Comment) {
  Config cfg;
  // Anything following a '#' should be ignored, up to a newline
  EXPECT_TRUE(cfg.parse("# comment"));
  EXPECT_TRUE(cfg.parse("   # ~!@#$"));
  EXPECT_TRUE(cfg.parse("\t#abc"));
  EXPECT_TRUE(cfg.parse("###\n##"));
  EXPECT_TRUE(cfg.parse("ACTIVITIES_WARMUP_ITERATIONS=1 ##ok"));
  EXPECT_TRUE(cfg.parse("ACTIVITIES_WARMUP_ITERATIONS=1 ## FOO=2"));
  // Whatever appears before the comment must be valid format
  EXPECT_FALSE(cfg.parse("util ## not ok"));
  EXPECT_FALSE(cfg.parse("## ok \n blah # not OK"));
  // Check that a comment does not affect config parsing
  EXPECT_TRUE(cfg.parse(
      "ACTIVITIES_WARMUP_ITERATIONS = 1 # Warm up for one iteration"));
  EXPECT_EQ(cfg.activitiesWarmupIterations(), 1);
}

TEST(ParseTest, Format) {
  Config cfg;
  // The basic format is just "name = value"; unknown names are tolerated.
  // A line with no '=' is invalid, an empty value is allowed, a leading '='
  // is invalid, and only one setting is allowed per line.
  EXPECT_FALSE(cfg.parse("foo"));
  EXPECT_TRUE(cfg.parse("foo="));
  EXPECT_FALSE(cfg.parse("=foo="));
  EXPECT_TRUE(cfg.parse("foo=1,2,3"));
  // Only one setting per line
  EXPECT_FALSE(cfg.parse("foo = 1,2,3 ; bar = 4,5,6"));
}

TEST(ParseTest, DefaultActivityTypes) {
  Config cfg;
  cfg.validate(std::chrono::system_clock::now());
  auto default_activities = defaultActivityTypes();
  EXPECT_EQ(
      cfg.selectedActivityTypes(),
      std::set<ActivityType>(
          default_activities.begin(), default_activities.end()));
}

TEST(ParseTest, ActivityTypes) {
  Config cfg;
  EXPECT_FALSE(cfg.parse("ACTIVITY_TYPES"));
  EXPECT_TRUE(cfg.parse("ACTIVITY_TYPES="));
  EXPECT_FALSE(cfg.parse("=ACTIVITY_TYPES="));

  EXPECT_EQ(
      cfg.selectedActivityTypes(),
      std::set<ActivityType>(
          {ActivityType::CPU_OP,
           ActivityType::CPU_INSTANT_EVENT,
           ActivityType::PYTHON_FUNCTION,
           ActivityType::USER_ANNOTATION,
           ActivityType::GPU_USER_ANNOTATION,
           ActivityType::GPU_MEMCPY,
           ActivityType::GPU_MEMSET,
           ActivityType::CONCURRENT_KERNEL,
           ActivityType::EXTERNAL_CORRELATION,
           ActivityType::OVERHEAD,
           ActivityType::CUDA_RUNTIME,
           ActivityType::CUDA_DRIVER,
           ActivityType::CUDA_SYNC,
           ActivityType::CUDA_EVENT,
           ActivityType::MTIA_RUNTIME,
           ActivityType::MTIA_INSIGHT,
           ActivityType::MTIA_CCP_EVENTS,
           ActivityType::MTIA_COUNTERS}));

  Config cfg2;
  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES=gpu_memcpy,gpu_MeMsEt,kernel"));
  EXPECT_EQ(
      cfg2.selectedActivityTypes(),
      std::set<ActivityType>(
          {ActivityType::GPU_MEMCPY,
           ActivityType::GPU_MEMSET,
           ActivityType::CONCURRENT_KERNEL}));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES = cuda_Runtime,"));
  EXPECT_EQ(
      cfg2.selectedActivityTypes(),
      std::set<ActivityType>({ActivityType::CUDA_RUNTIME}));

  // parse() returns false for an unknown activity name
  EXPECT_FALSE(cfg2.parse("ACTIVITY_TYPES = memcopy,cuda_runtime"));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES = cpu_op"));
  EXPECT_EQ(
      cfg2.selectedActivityTypes(),
      std::set<ActivityType>({ActivityType::CPU_OP}));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES = xpu_Runtime"));
  EXPECT_EQ(
      cfg2.selectedActivityTypes(),
      std::set<ActivityType>({ActivityType::XPU_RUNTIME}));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES = xpu_scope_profiler"));
  EXPECT_EQ(
      cfg2.selectedActivityTypes(),
      std::set<ActivityType>({ActivityType::XPU_SCOPE_PROFILER}));

  EXPECT_TRUE(
      cfg2.parse("ACTIVITY_TYPES=privateuse1_Runtime,privateuse1_driver"));
  EXPECT_EQ(
      cfg2.selectedActivityTypes(),
      std::set<ActivityType>(
          {ActivityType::PRIVATEUSE1_RUNTIME,
           ActivityType::PRIVATEUSE1_DRIVER}));
}

TEST(ParseTest, PerformanceMetrics) {
  Config cfg;
  EXPECT_TRUE(cfg.performanceMetricNames().empty());
  EXPECT_EQ(cfg.performanceMetricsDeviceId(), -1);

  EXPECT_TRUE(
      cfg.parse("PERFORMANCE_METRICS_DEVICE_ID=2\n"
                "PERFORMANCE_METRICS="
                "sm__cycles_active.avg,dram__bytes_read.sum"));

  EXPECT_EQ(
      cfg.performanceMetricNames(),
      std::vector<std::string>(
          {"sm__cycles_active.avg", "dram__bytes_read.sum"}));
  EXPECT_EQ(cfg.performanceMetricsDeviceId(), 2);

  auto clone = cfg.clone();
  EXPECT_EQ(clone->performanceMetricNames(), cfg.performanceMetricNames());
  EXPECT_EQ(clone->performanceMetricsDeviceId(), 2);
}

TEST(ParseTest, ProfileStartTime) {
  Config cfg;
  system_clock::time_point now = system_clock::now();
  int64_t tgood_ms =
      duration_cast<milliseconds>(now.time_since_epoch()).count();
  EXPECT_TRUE(cfg.parse(fmt::format("PROFILE_START_TIME = {}", tgood_ms)));

  // Pass given PROFILE_START_TIME = 0, a timestamp is assigned.
  tgood_ms = 0;
  EXPECT_TRUE(cfg.parse(fmt::format("PROFILE_START_TIME = {}", tgood_ms)));

  // Fail given PROFILE_START_TIME older than kMaxRequestAge from now.
  int64_t tbad_ms =
      duration_cast<milliseconds>((now - seconds(15)).time_since_epoch())
          .count();
  EXPECT_FALSE(cfg.parse(fmt::format("PROFILE_START_TIME = {}", tbad_ms)));
}

TEST(ParseTest, RequestTraceIds) {
  Config cfg;
  EXPECT_TRUE(cfg.parse("REQUEST_TRACE_ID=XYZ"));
  EXPECT_EQ(cfg.requestTraceID(), "XYZ");
  EXPECT_TRUE(cfg.parse("REQUEST_GROUP_TRACE_ID=ABC"));
  EXPECT_EQ(cfg.requestGroupTraceID(), "ABC");
}

// Trusted base config may set any trace path.
TEST(ParseTest, BaseConfigLogFileUnrestricted) {
  Config cfg;
  EXPECT_TRUE(cfg.parse("ACTIVITIES_LOG_FILE=/home/user/custom/trace.json"));
  EXPECT_EQ(cfg.activitiesLogFile(), "/home/user/custom/trace.json");
}

// On-demand path under the allowed dir is accepted.
TEST(ParseTest, OnDemandLogFileAllowed) {
  Config cfg;
  cfg.setOnDemand(true);
  EXPECT_TRUE(cfg.parse("ACTIVITIES_LOG_FILE=/tmp/my_trace.json"));
  EXPECT_EQ(cfg.activitiesLogFile(), "/tmp/my_trace.json");
}

// On-demand path outside the allowed dir (or with traversal) falls back.
TEST(ParseTest, OnDemandLogFileRejectedOutsideAllowedDir) {
  Config cfg;
  cfg.setOnDemand(true);
  const std::string original = cfg.activitiesLogFile();

  EXPECT_TRUE(cfg.parse("ACTIVITIES_LOG_FILE=/etc/cron.d/payload"));
  EXPECT_EQ(cfg.activitiesLogFile(), original);
  EXPECT_EQ(cfg.activitiesLogFile().rfind("/tmp/", 0), 0u);

  EXPECT_TRUE(cfg.parse("ACTIVITIES_LOG_FILE=/tmp/../etc/cron.d/payload"));
  EXPECT_EQ(cfg.activitiesLogFile(), original);
}
