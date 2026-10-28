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
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "include/Config.h"
#include "src/AsyncActivityProfilerHandler.h"
#include "src/GenericActivityProfiler.h"
#include "test/MockCpuActivityBuffer.h"
#include "test/TestUtils.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono;
using libkineto::test::createTempTraceFile;
using libkineto::test::logUrlToPath;
using libkineto::test::MockCpuActivityBuffer;
using libkineto::test::TempTraceFile;

namespace {

// Drives the on-demand async path end to end and asserts a real collected op
// reaches the trace file. We provide a timestamp based config and run it
// through AsyncActivityProfilerHandler's configure()/performRunLoopStep(), but
// with a clock the test controls. We inject a CPU op mid-collection.
//
// Everything runs on the test thread with a controlled clock; we never start
// AsyncActivityProfilerHandler's profiling thread. We do this to make a
// deterministic test with no race conditions.
//
// Note that this test is CPU-only.
TEST(AsyncE2ECpuTraceTest, OnDemandConfigCollectsCpuOpIntoTraceFile) {
  GenericActivityProfiler profiler(/*cpuOnly=*/true);
  AsyncActivityProfilerHandler handler(profiler);

  const TempTraceFile traceFile =
      createTempTraceFile("kineto_async_e2e_", ".json");
  const std::string traceId = "async-e2e-cpu-trace";
  const std::string opName = "async-e2e-cpu-op";

  // A controlled clock. start is warmup+1s ahead of base, so canStart() passes
  // for real and [start, end] is a live collection window.
  constexpr auto kWarmup = seconds(1);
  constexpr auto kDuration = seconds(1);
  const auto base = system_clock::now();
  const auto start = base + kWarmup + seconds(1);
  const auto end = start + kDuration;

  Config cfg;
  cfg.setOnDemand(true);
  ASSERT_TRUE(cfg.parse(fmt::format(
      "REQUEST_TRACE_ID={}\n"
      "PROFILE_START_TIME={}\n"
      "ACTIVITIES_WARMUP_PERIOD_SECS={}\n"
      "ACTIVITIES_DURATION_SECS={}\n"
      "ACTIVITIES_LOG_FILE={}\n",
      traceId,
      duration_cast<milliseconds>(start.time_since_epoch()).count(),
      kWarmup.count(),
      kDuration.count(),
      traceFile.path())));
  const std::string tracePath = logUrlToPath(cfg.activitiesLogUrl());
  ASSERT_FALSE(tracePath.empty());

  // The produced file has the pid inserted before .json, a name TempTraceFile
  // does not own, so remove it ourselves when the test ends.
  struct FileRemover {
    std::string path;
    ~FileRemover() {
      if (!path.empty()) {
        std::remove(path.c_str());
      }
    }
  } fileRemover{tracePath};

  // WaitForRequest -> Warmup
  handler.configure(cfg, base);
  ASSERT_TRUE(handler.isAsyncActive());

  // Warmup -> CollectTrace
  handler.performRunLoopStep(start, start);

  // Inject a CPU op while collecting.
  const int64_t startNs =
      duration_cast<nanoseconds>(start.time_since_epoch()).count();
  const int64_t endNs =
      duration_cast<nanoseconds>(end.time_since_epoch()).count();
  auto ops = std::make_unique<MockCpuActivityBuffer>(startNs, endNs);
  ops->addOp(opName, startNs, startNs + 1000, /*correlation=*/1);
  profiler.transferCpuTrace(std::move(ops));

  // CollectTrace -> ProcessTrace
  handler.performRunLoopStep(end, end);

  // ProcessTrace -> WaitForRequest. Finalizes and writes the file.
  handler.performRunLoopStep(end, end);
  ASSERT_FALSE(handler.isAsyncActive());

  // The trace file carries our request id and the op we collected.
  std::ifstream file(tracePath);
  ASSERT_TRUE(file.good()) << "trace file not found: " << tracePath;
  const std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  const nlohmann::json data = nlohmann::json::parse(jsonStr);

  ASSERT_TRUE(data.contains("trace_id"));
  EXPECT_EQ(data["trace_id"].get<std::string>(), traceId);

  ASSERT_TRUE(data.contains("traceEvents"));
  bool foundOp = false;
  for (const auto& event : data["traceEvents"]) {
    if (event.value("name", std::string{}) == opName) {
      foundOp = true;
      break;
    }
  }
  EXPECT_TRUE(foundOp) << "collected CPU op '" << opName << "' not found in "
                       << tracePath;
}

} // namespace
